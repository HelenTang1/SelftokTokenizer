# Copyright (C) 2025. Huawei Technologies Co., Ltd.  All rights reserved.

# Licensed under MIT License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://opensource.org/license/mit
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================


import sys
import os
print(sys.path)
sys.path.append(".")

import torch

# ####For running on GPU, comment out this part
# import torch_npu
# torch_npu.npu.set_compile_mode(jit_compile=False)
# from torch_npu.contrib import transfer_to_npu
# #####

from torchvision import transforms
import numpy as np
import argparse
import sys
from PIL import ImageFile
from copy import deepcopy
from collections import OrderedDict
from mimogpt.models.selftok.image_tokenizer import ImageTokenizer
ImageFile.LOAD_TRUNCATED_IMAGES = True
from mimogpt.models.selftok.sd3.sd3_impls import SDVAE, SD3LatentFormat
from mimogpt.models.selftok.sd3.rectified_flow import RectifiedFlow
from mimogpt.utils import hf_logger
from torchvision.utils import save_image
from diffusers import AutoencoderKL


def load_state(model, state_dict, prefix='',init_method = None):
    model_dict = model.state_dict()  # 当前网络结构
    # encoder.down.0.block.0.norm1.weight vae model
    # encoder.down_blocks.0.resnets.0.norm1.weight weight state
    # new_state_dict = state_dict.copy()
    # for keys in state_dict.keys():
    #     new_key = keys.replace("down_blocks","down").replace("resnets","").replace("up_blocks","down")
    #     new_state_dict[new_key] = state_dict.pop(keys)
    # state_dict = new_state_dict
    # import pdb 
    # pdb.set_trace()

    if prefix == 'model.diffusion_model.':
        excluded_keys = ['context_embedder.bias', 'context_embedder.weight']
        if init_method == 1:
            excluded_keys = ['context_embedder.bias', 'context_embedder.weight', 'final_layer.adaLN_modulation.1.bias', 'final_layer.adaLN_modulation.1.weight', 'final_layer.linear.bias', 'final_layer.linear.weight']
            pretrained_dict = {k.replace(prefix,''): v for k, v in state_dict.items() if k.replace(prefix,'') in model_dict and k.replace(prefix,'') not in excluded_keys and 'context_block' not in k}
        elif init_method == 2:
            pretrained_dict = {k.replace(prefix,''): v for k, v in state_dict.items() if k.replace(prefix,'') in model_dict and k.replace(prefix,'') not in excluded_keys and 'context_block' not in k and 'x_block.attn' not in k}
        else:
            pretrained_dict = {k.replace(prefix,''): v for k, v in state_dict.items() if k.replace(prefix,'') in model_dict and k.replace(prefix,'') not in excluded_keys and 'context_block' not in k}
    else:
        pretrained_dict = {k.replace(prefix,''): v for k, v in state_dict.items() if k.replace(prefix,'') in model_dict}

    dict_t = deepcopy(pretrained_dict)
    for key, weight in dict_t.items():
        if key in model_dict and model_dict[key].shape != dict_t[key].shape:
            pretrained_dict.pop(key)
   
    m, u = model.load_state_dict(pretrained_dict, strict=False)

    if len(m) > 0:
        hf_logger.info(" ----------------------- ")
        hf_logger.info(f"model missing keys:{m}")
        
    if len(u) > 0:
        hf_logger.info(" ----------------------- ")
        hf_logger.info(f"mode unexpected keys:{u}")

class NormalizeToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __init__(self, reshape=True):
        self.reshape = reshape

    def __call__(self, image):
        image = np.array(image).astype(np.float32)
        image = (image / 127.5 - 1.0).astype(np.float32)
        if self.reshape:
            image = np.reshape(image, (image.shape[0], image.shape[1], -1))
        image = image.transpose((2, 0, 1))
        return torch.from_numpy(image)

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        name = name.replace("module.", "")
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)

def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag

def set_sd3_vae(vae_path, device):
    vae = SDVAE(device="cpu", dtype=torch.bfloat16)
    state_dict = torch.load(vae_path, map_location='cpu')
    load_state(vae, state_dict, 'first_stage_model.')
    vae.to(device)
    vae.eval()
    return vae

def set_ema_model(model, device):
    ema = deepcopy(model).to(torch.float32)  # Create an EMA of the model for use after training
    requires_grad(ema, False)
    update_ema(ema, model, decay=0)
    ema = ema.to(device)
    ema.eval()
    return ema

def norm_ip(img, low, high):
    img.clamp_(min=low, max=high)
    img.sub_(low).div_(max(high - low, 1e-5))

def str_to_bool(value):
    """Converts a string to a boolean.

    Raises argparse.ArgumentTypeError if the value cannot be parsed as a boolean.
    """
    if isinstance(value, bool):
        return value
    if value.lower() in {'false', 'f', '0', 'no', 'n'}:
        return False
    elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
        return True
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

class SelftokPipeline():
    def __init__(self, cfg, ckpt_path, sd3_path, datasize = 256, start = 1.0, cfg_scale = 1,model_type='sd3', dtype=torch.bfloat16, ema_decoder=False, device=None):
        
        self.cfg = cfg
        self.datasize = datasize
        self.model_type = model_type
        self.dtype = dtype
        # define models
        if self.model_type == 'sd3':
            self.vae = AutoencoderKL.from_pretrained(sd3_path, subfolder="vae")
            self.vae.to(device).to(self.dtype)
        else:
            raise ValueError(f"Unsupported MODEL_TYPE: {self.model_type}. Expected 'sd3'")
        cfg.tokenizer.params.noise_schedule_config.is_eval=cfg.common.is_eval
        
        self.model = ImageTokenizer(**cfg.tokenizer.params)
        self.model.set_eval()

        self.ema_decoder = ema_decoder
        if self.ema_decoder==True:
            self.ema = set_ema_model(self.model.model, device)
            self.ema.eval()
        self.vae.eval()
        self.diti = self.model.diti
        self.K = self.diti.K
        self.count = 0
        self.count_cfg = 0
        self.start = start
        self.cfg_scale = cfg_scale
        if hasattr(cfg.tokenizer.params, "cut_of_k") and self.cfg.tokenizer.params.cut_of_k:
            self.cut_of_k = self.cfg.tokenizer.params.cut_of_k
        else:
            self.cut_of_k = None
            
       
        pretrain = ckpt_path
        
        state_dict = torch.load(pretrain, map_location="cpu")

        print(f"Loading all...")
        if self.ema_decoder==True:
            self.ema.load_state_dict(state_dict['ema_state_dict'])
        self.model.load_state_dict(state_dict, strict=False)
        
        if self.ema_decoder==True:
            self.ema.to(device)
        self.model.to(device)
        
        self._steps = 100
        self.flow = RectifiedFlow(
            self._steps, self.start, self.cut_of_k, val_schedule='uniform', shift=1.0, **cfg.tokenizer.params.noise_schedule_config,
        )

        self.cond_vary = True
        self.saved_images = 8


    def encoding(self, images, device):

        print(f"Begin encoding.")

        images = images.to(dtype=self.dtype, device=device) # Move images to GPU once per batch
        x_0 = self.vae.encode(images, return_dict=False)[0].mode()
        x_0 = SD3LatentFormat().process_in(x_0)
        
        x_0 = x_0.to(torch.float32)

        with torch.no_grad():
            _, tokens = self.model.encoder(x_0, d=None)

        print('End encoding.')
        
        return tokens
    
    def _make_fixed_probe_noise(self, shape, device, seed: int = 0):
        g = torch.Generator(device=device)
        g.manual_seed(seed)
        return torch.randn(shape, generator=g, device=device)
    
    @torch.no_grad()
    def decoding(self, idx, device, return_debug=False, gt_images=None):

        print(f"Begin decoding.")
        
        token_idx = torch.from_numpy(idx).to(device)
        B = token_idx.shape[0]
        gt_x0 = None
        if return_debug and gt_images is not None:
            gt_images = gt_images.to(dtype=self.dtype, device=device)
            gt_x0 = self.vae.encode(gt_images, return_dict=False)[0].mode()
            gt_x0 = SD3LatentFormat().process_in(gt_x0)
            gt_x0 = gt_x0.to(torch.float32)

        
        outs_q = self.model.encoder.quantizer.get_output_from_indices(token_idx)
        outs_q = outs_q.reshape(B, -1, outs_q.shape[-1])

        if self.model.encoder.post_norm:
            outs_q = self.model.encoder.final_layer_norm3(outs_q)
        
        if hasattr(self.cfg.tokenizer.params, "stages"):
            t_mapped = torch.tensor([self.flow.timestep_map[0]]*B, device=device).long()
        else:
            t_mapped = torch.tensor([(self.flow.timestep_map[0])/1000.0]*B, device=device)
        k = self.diti.to_indices(t_mapped)
        d=k
        
        enc_mask = self.model.encoder.get_encoder_mask(token_idx, d)
        attn_mask = enc_mask
        mask_v = enc_mask[..., None].expand_as(outs_q)
        encoder_hidden_states = outs_q.cuda() * mask_v.cuda()
        
        ori_hidden_states = outs_q

        model_kwargs = dict(
            encoder_hidden_states=encoder_hidden_states,
            mask=attn_mask,
            context_see_xt=True
        )
        
        latent_dim = self.datasize // 8

        # TODO: real run
        # xt = torch.randn(B, 16, latent_dim, latent_dim).to(device)
        # TODO: DEBUG
        shape = B, 16, latent_dim, latent_dim
        xt = self._make_fixed_probe_noise(shape, device=device, seed=0)

        kwargs = {}
        enc_in =xt.float()

        if self.ema_decoder==True:
            pred_x0 = self.flow.p_sample_loop(
                self.ema.model, xt.shape, xt, model_kwargs=model_kwargs,
                start_t=self._steps, cond_vary=self.cond_vary,
                diti=self.diti, encoder=self.model.encoder, x_0=enc_in,
                ori_hidden_states=ori_hidden_states,**kwargs
            )
        else: # here
            sample_out  = self.flow.p_sample_loop(
                self.model.model, xt.shape, xt, model_kwargs=model_kwargs,
                start_t=self._steps, cond_vary=self.cond_vary,
                diti=self.diti, encoder=self.model.encoder, x_0=enc_in,
                ori_hidden_states=ori_hidden_states,
                return_debug=return_debug, gt_x0=gt_x0,
                **kwargs
            )
            if return_debug:
                pred_x0, reverse_debug_list, debug_meta, probe_list = sample_out 
            else:
                pred_x0 = sample_out
            
            training_debug_list = None
            if return_debug and gt_x0 is not None:
                initial_noise = debug_meta["initial_noise"].to(device)

                # training_debug_list = self.collect_training_path_debug(
                #     gt_x0=gt_x0,
                #     initial_noise=initial_noise,
                #     token_idx=token_idx,
                #     outs_q=outs_q,
                #     device=device,
                # )

        if self.model_type == 'sd3':
            pred_x0_out = SD3LatentFormat().process_out(pred_x0)

            pred_x0_out = pred_x0_out.to(self.dtype)
            recons = self.vae.decode(pred_x0_out,return_dict=False)[0]
        
        norm_ip(recons, -1, 1)

        print('End decoding.')
        
        if return_debug:
            debug_dict = {
                "reverse": reverse_debug_list,
                # "training": training_debug_list,
            }
            return recons, debug_dict, probe_list
        return recons
    
    @torch.no_grad()
    def decoding_with_renderer(self, idx, device):
        
        print(f"Begin decoding with Renderer.")
        
        token_idx = torch.from_numpy(idx).to(device)
        B = token_idx.shape[0]

        outs_q = self.model.encoder.quantizer.get_output_from_indices(token_idx)
        outs_q = outs_q.reshape(B, -1, outs_q.shape[-1])

        if self.model.encoder.post_norm:
            outs_q = self.model.encoder.final_layer_norm3(outs_q)
        
        pred_x0, _ = self.model.model(y=None, encoder_hidden_states=outs_q)
        
        if self.model_type == 'sd3':
            pred_x0_out = SD3LatentFormat().process_out(pred_x0)
            
            pred_x0_out = pred_x0_out.to(self.dtype)
            recons = self.vae.decode(pred_x0_out)[0]
        
        norm_ip(recons, -1, 1)

        print('End decoding with Renderer.')
        
        return recons
    
    @torch.no_grad()
    def collect_training_path_debug(
        self,
        gt_x0,
        initial_noise,
        token_idx,
        outs_q,
        device,
    ):
        """
        Evaluate the model on training path:
            x_t = q_sample(x0, t, noise=initial_noise)

        Returns a list of per-step debug info.
        """

        B = gt_x0.shape[0]

        # same fixed GT velocity
        gt_velocity = initial_noise.float() - gt_x0.float()

        # same shift logic as training
        if (gt_x0.shape[2] * gt_x0.shape[3] / 4096.0) < 0.5:
            shift = 1.0
        else:
            shift = 1.878

        debug_list = []

        for raw_t_scalar in self.flow.scheduled_t:
            raw_t = torch.full((B,), raw_t_scalar.item(), device=device, dtype=gt_x0.dtype)

            # IMPORTANT:
            # token schedule uses raw_t before shift, same as training
            if self.model.k_m is None:
                t_tmp = (self.model.t2k * raw_t).clamp(0, 1.0)
                k_batch = self.diti.to_indices(t_tmp * 1000.0)
            else:
                t_tmp = (self.model.t2k * raw_t).clamp(0, 1.0)
                k_batch = self.diti.to_indices(t_tmp)

            enc_mask = self.model.encoder.get_encoder_mask(token_idx, k_batch)
            mask_v = enc_mask[..., None].expand_as(outs_q)
            encoder_hidden_states = outs_q * mask_v

            # model time uses shifted t, same as training
            model_t = self.model.diffusion.shift_t(raw_t, shift)

            model_kwargs = dict(
                encoder_hidden_states=encoder_hidden_states,
                mask=enc_mask,
                context_see_xt=True,
            )

            # training path x_t
            x_train = self.model.diffusion.q_sample(gt_x0, model_t, noise=initial_noise)

            pred_velocity, _ = self.model.model(x_train.float(), model_t, **model_kwargs)

            mse = ((gt_velocity - pred_velocity.float()) ** 2).flatten(1).mean(dim=1)

            debug_item = {
                "raw_t": raw_t.detach().cpu(),
                "model_t": model_t.detach().cpu(),
                "x": x_train.detach().cpu(),
                "pred_velocity": pred_velocity.detach().cpu(),
                "gt_velocity": gt_velocity.detach().cpu(),
                "mse": mse.detach().cpu(),
            }
            debug_list.append(debug_item)

        return debug_list



    
