from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Optional

import pytorch_lightning as pl
import torch
from diffusers import AutoencoderKL
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy
from PIL import Image, ImageFile
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms

from mimogpt.infer.infer_utils import parse_args_from_yaml
from mimogpt.models.selftok.image_tokenizer import ImageTokenizer
from mimogpt.models.selftok.sd3.sd3_impls import SD3LatentFormat

ImageFile.LOAD_TRUNCATED_IMAGES = True

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def str2bool(v: str) -> bool:
    if isinstance(v, bool):
        return v
    v = v.lower()
    if v in {"1", "true", "t", "yes", "y"}:
        return True
    if v in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {v}")


class RecursiveImageDataset(Dataset):
    def __init__(self, root: str, image_size: int, train: bool = True):
        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(f"Dataset root does not exist: {self.root}")

        self.paths = sorted(
            p for p in self.root.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS
        )
        if not self.paths:
            raise RuntimeError(f"No images found under: {self.root}")

        if train:
            self.transform = transforms.Compose([
                transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x * 2.0 - 1.0),
                ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x * 2.0 - 1.0),
            ])

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        path = self.paths[idx]
        with Image.open(path) as img:
            img = img.convert("RGB")
            return self.transform(img)


class SelftokDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_root: str,
        image_size: int,
        batch_size: int,
        num_workers: int,
        val_root: Optional[str] = None,
        val_split: float = 0.01,
    ):
        super().__init__()
        self.train_root = train_root
        self.val_root = val_root
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split

    def setup(self, stage: Optional[str] = None) -> None:
        if self.val_root:
            self.train_dataset = RecursiveImageDataset(self.train_root, self.image_size, train=True)
            self.val_dataset = RecursiveImageDataset(self.val_root, self.image_size, train=False)
        else:
            full = RecursiveImageDataset(self.train_root, self.image_size, train=True)
            n_val = max(1, int(len(full) * self.val_split))
            n_train = len(full) - n_val
            generator = torch.Generator().manual_seed(42)
            self.train_dataset, self.val_dataset = random_split(full, [n_train, n_val], generator=generator)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            drop_last=False,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            drop_last=False,
        )


class SelftokLightningModule(pl.LightningModule):
    def __init__(
        self,
        cfg,
        sd3_pretrained: str,
        token_lr: float,
        dit_lr: float,
        weight_decay: float,
        warmup_ratio: float,
        precision_mode: str = "bf16-mixed",
    ):
        super().__init__()
        self.cfg = cfg
        self.token_lr = token_lr
        self.dit_lr = dit_lr
        self.weight_decay = weight_decay
        self.warmup_ratio = warmup_ratio
        self.precision_mode = precision_mode

        self.tokenizer = ImageTokenizer(**cfg.tokenizer.params)

        vae_dtype = torch.float32
        if precision_mode == "bf16-mixed":
            vae_dtype = torch.bfloat16
        elif precision_mode == "16-mixed":
            vae_dtype = torch.float16

        self.vae = AutoencoderKL.from_pretrained(sd3_pretrained, subfolder="vae")
        self.vae.to(dtype=vae_dtype)
        self.vae.requires_grad_(False)
        self.vae.eval()

        self.latent_format = SD3LatentFormat()

    def _to_latents(self, batch: torch.Tensor | dict) -> torch.Tensor:
        if isinstance(batch, dict):
            if "latent" in batch:
                return batch["latent"].float()
            images = batch["image"]
        else:
            images = batch

        images = images.to(dtype=next(self.vae.parameters()).dtype)
        with torch.no_grad():
            posterior = self.vae.encode(images, return_dict=False)[0]
            latents = posterior.mode()
            latents = self.latent_format.process_in(latents)
        return latents.float()

    def training_step(self, batch, batch_idx):
        latents = self._to_latents(batch)
        loss, log_dict = self.tokenizer(latents)
        bs = latents.shape[0]

        self.log(
            "train/loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist = (self.trainer is not None and self.trainer.world_size > 1),
            batch_size=bs,
        )
        for k, v in log_dict.items():
            if k in ["loss"]:
                continue
            if k.endswith("_list"):
                v = torch.tensor(v, device=loss.device).float().mean()
            self.log(
                f"train/{k}",
                v,
                on_step=True,
                on_epoch=False,
                prog_bar=False,
                logger=True,
                sync_dist = (self.trainer is not None and self.trainer.world_size > 1),
                batch_size=bs,
            )
        return loss

    def validation_step(self, batch, batch_idx):
        latents = self._to_latents(batch)
        loss, log_dict = self.tokenizer(latents)
        bs = latents.shape[0]

        self.log(
            "val/loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist = (self.trainer is not None and self.trainer.world_size > 1),
            batch_size=bs,
        )
        for k, v in log_dict.items():
            if k in ["loss"]:
                continue
            if k.endswith("_list"):
                v = torch.tensor(v, device=loss.device).float().mean()
            self.log(
                f"val/{k}",
                v,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                logger=True,
                sync_dist = (self.trainer is not None and self.trainer.world_size > 1),
                batch_size=bs,
            )
        return loss

    def configure_optimizers(self):
        dit_params = []
        token_params = []

        for name, p in self.tokenizer.named_parameters():
            if not p.requires_grad:
                continue
            if name.startswith("model."):
                dit_params.append(p)
            else:
                token_params.append(p)

        param_groups = []
        if token_params:
            param_groups.append({"params": token_params, "lr": self.token_lr})
        if dit_params:
            param_groups.append({"params": dit_params, "lr": self.dit_lr})

        optimizer = AdamW(param_groups, betas=(0.9, 0.95), weight_decay=self.weight_decay)

        max_steps = max(1, int(self.trainer.estimated_stepping_batches))
        warmup_steps = int(max_steps * self.warmup_ratio)

        def lr_lambda(step: int) -> float:
            if warmup_steps > 0 and step < warmup_steps:
                return float(step + 1) / float(warmup_steps)
            progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    def on_save_checkpoint(self, checkpoint: dict) -> None:
        # Do not save the frozen SD3 VAE weights inside every Lightning checkpoint.
        state_dict = checkpoint.get("state_dict", {})
        keys_to_remove = [k for k in state_dict if k.startswith("vae.")]
        for k in keys_to_remove:
            state_dict.pop(k)



def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--yml-path", type=str, default="/data4/yhtang/exp/EventDDT_Private/submodules/SelftokTokenizer/configs/res256/256-eval.yml")
    parser.add_argument("--train-root", type=str, default="/data4/yhtang/exp/EventDDT_Private/submodules/SelftokTokenizer/oneimage/")
    parser.add_argument("--val-root", type=str, default="/data4/yhtang/exp/EventDDT_Private/submodules/SelftokTokenizer/oneimage/")
    parser.add_argument("--sd3-pretrained", type=str, default="/data4/yhtang/exp/EventDDT_Private/pretrain_weights/sd3-diffusers/")
    parser.add_argument("--output-dir", type=str, default="/data4/yhtang/exp/EventDDT_Private/selftok/")
    parser.add_argument("--wandb-project", type=str, default="selftok")
    parser.add_argument("--wandb-name", type=str, default=None)
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=8, help="Per-GPU batch size.")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--devices", type=int, default=[0, 1], help="Number of GPUs on this node.")
    parser.add_argument("--num-nodes", type=int, default=1)
    parser.add_argument("--precision", type=str, default=None, choices=[None, "32", "16-mixed", "bf16-mixed"])
    parser.add_argument("--accumulate-grad-batches", type=int, default=1)
    parser.add_argument("--gradient-clip-val", type=float, default=1.0)
    parser.add_argument("--val-split", type=float, default=0.01)
    parser.add_argument("--token-lr", type=float, default=None)
    parser.add_argument("--dit-lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-ratio", type=float, default=None)
    parser.add_argument("--find-unused-parameters", type=str2bool, default=True)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--save-top-k", type=int, default=3)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    cfg = parse_args_from_yaml(args.yml_path)
    pl.seed_everything(cfg.common.random_seed, workers=True)
    torch.set_float32_matmul_precision("high")

    image_size = int(cfg.tokenizer.params.image_size)
    max_epochs = args.max_epochs or int(cfg.optimize.max_epochs)

    if args.precision is not None:
        precision = args.precision
    elif bool(cfg.common.use_bf16):
        precision = "bf16-mixed"
    elif bool(cfg.common.use_fp16):
        precision = "16-mixed"
    else:
        precision = "32"

    token_lr = args.token_lr
    if token_lr is None:
        token_lr = float(cfg.optimize.lr_scheduler.token_lr)

    dit_lr = args.dit_lr
    if dit_lr is None:
        dit_lr = float(cfg.optimize.lr_scheduler.dit_lr)

    warmup_ratio = args.warmup_ratio
    if warmup_ratio is None:
        warmup_ratio = float(cfg.optimize.warmup_epochs) / float(max_epochs)

    weight_decay = args.weight_decay
    if weight_decay == 0.0 and hasattr(cfg.optimize, "weight_decay"):
        weight_decay = float(cfg.optimize.weight_decay)

    datamodule = SelftokDataModule(
        train_root=args.train_root,
        val_root=args.val_root,
        image_size=image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_split=args.val_split,
    )

    model = SelftokLightningModule(
        cfg=cfg,
        sd3_pretrained=args.sd3_pretrained,
        token_lr=token_lr,
        dit_lr=dit_lr,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        precision_mode=precision,
    )

    logger = WandbLogger(
        project=args.wandb_project,
        name=args.wandb_name,
        entity=args.wandb_entity,
        save_dir=str(Path(args.output_dir) / "wandb"),
        log_model=False,
    )
    logger.log_hyperparams(
        {
            "yml_path": args.yml_path,
            "train_root": args.train_root,
            "val_root": args.val_root,
            "sd3_pretrained": args.sd3_pretrained,
            "batch_size_per_gpu": args.batch_size,
            "devices": args.devices,
            "num_nodes": args.num_nodes,
            "precision": precision,
            "token_lr": token_lr,
            "dit_lr": dit_lr,
            "weight_decay": weight_decay,
            "warmup_ratio": warmup_ratio,
            "accumulate_grad_batches": args.accumulate_grad_batches,
            "effective_global_batch_size": args.batch_size * max(1, len(args.devices)) * max(1, args.num_nodes) * max(1, args.accumulate_grad_batches),
        }
    )

    callbacks = [
        ModelCheckpoint(
            dirpath=str(Path(args.output_dir) / "checkpoints"),
            filename="epoch{epoch:03d}-valloss{val/loss:.4f}",
            monitor="val/dm_mse",
            mode="min",
            save_last=True,
            save_top_k=args.save_top_k,
            auto_insert_metric_name=False,
            every_n_epochs=1,
        ),
        LearningRateMonitor(logging_interval="step"),
    ]

    strategy= DDPStrategy(find_unused_parameters=True) if len(args.devices) > 1 else "auto"

    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=args.devices if torch.cuda.is_available() else 1,
        num_nodes=args.num_nodes,
        strategy=strategy,
        sync_batchnorm=len(args.devices) > 1,
        precision=precision,
        max_epochs=50000, #TODO
        accumulate_grad_batches=args.accumulate_grad_batches,
        gradient_clip_val=args.gradient_clip_val,
        logger=logger,
        callbacks=callbacks,
        default_root_dir=args.output_dir,
        log_every_n_steps=10,
        check_val_every_n_epoch=50,
        num_sanity_val_steps=2,
        deterministic=False,
    )

    trainer.fit(model, datamodule=datamodule, ckpt_path=args.resume)
    if trainer.is_global_zero: # only save from the main process
        final_ckpt_path = str(Path(args.output_dir) / "checkpoints" / "final.ckpt")
        trainer.save_checkpoint(final_ckpt_path)
        print(f"Saved final checkpoint to: {final_ckpt_path}")
        import wandb
        if wandb.run is not None:
            wandb.finish()


if __name__ == "__main__":
    main()
