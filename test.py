import os
import sys
print(sys.path)
sys.path.append(".")

import argparse
from mimogpt.infer.infer_utils import parse_args_from_yaml
from torchvision import transforms
from PIL import Image
import torch
import numpy as np
from mimogpt.infer.SelftokPipeline import SelftokPipeline
from mimogpt.infer.SelftokPipeline import NormalizeToTensor
from torchvision.utils import save_image

import matplotlib.pyplot as plt

# TODO: DEBUG utils
import csv
def save_rows_to_csv(rows, path="probe_delta.csv", header=("name", "value")):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(list(header))
        writer.writerows(rows)

def _to_python_scalar(x, device, batch_size = 1):
    if x is None:
        return None

    if not torch.is_tensor(x):
        try:
            x = torch.tensor(x)
        except Exception:
            return None

    x = x.detach().float().to(device)

    if x.ndim == 0:
        return x.expand(batch_size).item() 

    if x.shape[0] == batch_size:
        if x.ndim == 1:
            return x.item() 
        return x.reshape(batch_size, -1).mean(dim=1).item() 

    if x.numel() == batch_size:
        return x.reshape(batch_size)

    return x.mean().item()

parser = argparse.ArgumentParser()
parser.add_argument("--yml-path", type=str, default="./configs/res256/256-eval.yml") # download from https://huggingface.co/stabilityai/stable-diffusion-3-medium/resolve/main/sd3_medium.safetensors?download=true, require huggingface login, you have to change the format to .pt with safetensor_to_pt.py
# parser.add_argument("--pretrained", type=str, default="/data4/yhtang/exp/EventDDT_Private/selftok/checkpoints/epoch48799-valloss0.0072.ckpt") 
parser.add_argument("--pretrained", type=str, default= "/data4/yhtang/exp/EventDDT_Private/pretrain_weights/SelftokTokenizer/tokenizer_512_ckpt.pth") 
parser.add_argument("--sd3_pretrained", type=str, default="/data4/yhtang/exp/EventDDT_Private/pretrain_weights/sd3-diffusers/") 
parser.add_argument("--data_size", type=int, default=256)

args = parser.parse_args()

cfg = parse_args_from_yaml(args.yml_path)
model = SelftokPipeline(cfg=cfg, ckpt_path=args.pretrained, sd3_path=args.sd3_pretrained, datasize=args.data_size, device='cuda')

img_transform = transforms.Compose([
    transforms.Resize(args.data_size),
    transforms.CenterCrop(args.data_size),
    NormalizeToTensor(),
])

image_paths = ['./000000.png']
images = [img_transform(Image.open(p)) for p in image_paths]
images = torch.stack(images).to('cuda')

tokens = model.encoding(images, device='cuda')
np.save('./token.npy', tokens.detach().cpu().numpy())
tokens = np.load('./token.npy')

input_images = images
    
images, debug_dict, probe_list = model.decoding(tokens, device='cuda',
                        return_debug=True, gt_images=input_images)
for b in range(len(images)):
    save_image(images[b], f"./re_{b}_{args.data_size}_2.png")


reverse_debug = debug_dict["reverse"]
# training_debug = debug_dict["training"]
raw_ts = [item["raw_t"][0].item() for item in reverse_debug]
model_ts = [item["model_t"][0].item() for item in reverse_debug]
rev_mses = [item["mse"].mean().item() for item in reverse_debug]
# TODO: DEBUG training mse
# tr_mses = [item["mse"].mean().item() for item in training_debug]
# rows = [(raw_t, model_t, rev_mse, tr_mse) 
#         for raw_t, model_t, rev_mse, tr_mse in zip(raw_ts, model_ts, rev_mses, tr_mses)]
rows = [(raw_t, model_t, rev_mse) 
        for raw_t, model_t, rev_mse in zip(raw_ts, model_ts, rev_mses)]
# write rows to csv
save_rows_to_csv(rows, path=f'./debug_{args.data_size}.csv', header=['raw_t', 'model_t', 'rev_mse', 'tr_mse'])

# TODO: DEBUG probe list
rows = []
keys = probe_list[0].keys()
rows.append(("v_gt_mean", *[reverse_debug[0]["gt_velocity"].mean().item()]*len(probe_list)))
rows.append(("v_gt_min", *[reverse_debug[0]["gt_velocity"].min().item()]*len(probe_list)))
rows.append(("v_gt_max", *[reverse_debug[0]["gt_velocity"].max().item()]*len(probe_list)))
mse_list = [torch.mean((probe["sampler.pred_v_noise"] - reverse_debug[i]["gt_velocity"].cpu()) ** 2).item() 
                    for i, probe in enumerate(probe_list)]
rows.append(("mse_vs_vgt", * mse_list))
for k in keys:
    rows.append((k, *[_to_python_scalar(probe[k], "cpu") for probe in probe_list]))

curr_idx_list = [probe["meta.curr_idx"] for probe in probe_list]
save_rows_to_csv(rows, "./probe.csv", header=("name", *curr_idx_list))


plt.figure()
plt.plot(model_ts, rev_mses, marker='o', label='reverse path')
# plt.plot(model_ts, tr_mses, marker='s', label='training path')
plt.xlabel("timestep")
plt.ylabel("mean((v_gt - pred_velocity)^2)")
plt.title("Reverse path velocity MSE")
plt.gca().invert_xaxis()
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("./reverse_path_velocity_mse_curve.png", dpi=200)
plt.close()
