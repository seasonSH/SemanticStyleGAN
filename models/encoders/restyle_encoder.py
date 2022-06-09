# MIT License

# Copyright (c) 2021 Yuval Alaluf

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# 
# This file may have been modified by Bytedance Inc. (“Bytedance Modifications”). 
# All Bytedance Modifications are Copyright 2022 Bytedance Inc.

import math
import torch
from torch import nn
from models.encoders import restyle_psp_encoders


class RestyleEncoder(nn.Module):

    def __init__(self, decoder, encoder_type="BackboneEncoder", distributed=False, local_rank=0, device="cuda"):
        super(RestyleEncoder, self).__init__()
        # Define architecturece)
        self.decoder = decoder
        self.n_styles = decoder.n_latent
        self.encoder_type = encoder_type
        self.encoder = self.set_encoder().to(device)
        self.face_pool = torch.nn.AdaptiveAvgPool2d((256, 256))
        self.latent_avg, self.image_avg = None, None
        self.device = device
        self.init_avg_image()
        # Load weights if needed
        if distributed:
            self.decoder_parallel = nn.parallel.DistributedDataParallel(
                self.decoder,
                device_ids=[local_rank],
                output_device=local_rank,
                broadcast_buffers=False,
                find_unused_parameters=True,
            )
            self.encoder_parallel = nn.parallel.DistributedDataParallel(
                self.encoder,
                device_ids=[local_rank],
                output_device=local_rank,
                broadcast_buffers=False,
            )
        else:
            self.decoder_parallel = self.decoder
            self.encoder_parallel = self.encoder

    def set_encoder(self):
        if self.encoder_type == 'BackboneEncoder':
            encoder = restyle_psp_encoders.BackboneEncoder(50, 'ir_se', self.n_styles, input_nc=6)
        elif self.encoder_type == 'ResNetBackboneEncoder':
            encoder = restyle_psp_encoders.ResNetBackboneEncoder(self.n_styles, input_nc=6)
        else:
            raise Exception(f'{self.encoder_type} is not a valid encoders')
        return encoder

    def load_weights(self, checkpoint_path, initialize=False):
        print(f'Loading ReStyle pSp from checkpoint: {checkpoint_path}')
        if initialize:
            state_dict = self.__get_encoder_checkpoint(checkpoint_path, input_nc=6)
            self.encoder.load_state_dict(state_dict, strict=False)
        else:
            ckpt = torch.load(checkpoint_path, map_location='cpu')
            state_dict = self.__get_keys(ckpt, 'encoder.')
            self.latent_avg = ckpt['latent_avg'].to(self.device).reshape(1,-1,512)[:,0]
            self.encoder.load_state_dict(state_dict, strict=True)
        self.init_avg_image()


    def forward(self, input_img, output_img=None, latent=None, randomize_noise=True, n_iter=1, return_seg=False, resize_output=True):

        batch = input_img.size(0)
        input_img = self.resize_input(input_img)
        if output_img is None:
            output_img = self.image_avg.repeat(batch,1,1,1).to(self.device)
        if latent is None:
            latent = self.latent_avg.view(1,1,512).repeat(batch,self.n_styles,1).to(self.device)
        for _ in range(n_iter):
            output_img = self.resize_input(output_img).detach()
            x = torch.cat((input_img, output_img), 1)
            residual = self.encoder_parallel(x) 
            latent = latent.detach() + residual
            output_img, output_seg = self.decoder_parallel([latent], input_is_latent=True, randomize_noise=randomize_noise)
        if resize_output:
            output_img = self.face_pool(output_img)
        if return_seg:
            return output_img, output_seg, latent
        else:
            return output_img, latent

    def resize_input(self, img):
        if img.size(2) != 256 or img.size(3) != 256:
            img = self.face_pool(img)
        return img

    def init_avg_image(self):
        with torch.no_grad():
            if self.latent_avg is None:
                self.latent_avg = self.decoder.style(torch.randn(100000,512).to(self.device)).mean(0, keepdim=True)
            codes = self.latent_avg.repeat(1, self.n_styles, 1).to(self.device)

            image_avg, _ = self.decoder([codes], input_is_latent=True, randomize_noise=False)
            self.image_avg = self.face_pool(image_avg)
            print(f"image_avg shape: {self.image_avg.shape}")


    def __get_encoder_checkpoint(self, model_path, input_nc):
        if self.encoder_type == "BackboneEncoder":
            print('Loading encoders weights from irse50!')
            encoder_ckpt = torch.load(model_path)
            # Transfer the RGB input of the irse50 network to the first 3 input channels of pSp's encoder
            if input_nc != 3:
                shape = encoder_ckpt['input_layer.0.weight'].shape
                altered_input_layer = torch.randn(shape[0], input_nc, shape[2], shape[3], dtype=torch.float32)
                altered_input_layer[:, :3, :, :] = encoder_ckpt['input_layer.0.weight']
                encoder_ckpt['input_layer.0.weight'] = altered_input_layer
            return encoder_ckpt
        else:
            print('Loading encoders weights from resnet34!')
            encoder_ckpt = torch.load(model_path)
            # Transfer the RGB input of the resnet34 network to the first 3 input channels of pSp's encoder
            if input_nc != 3:
                shape = encoder_ckpt['conv1.weight'].shape
                altered_input_layer = torch.randn(shape[0], input_nc, shape[2], shape[3], dtype=torch.float32)
                altered_input_layer[:, :3, :, :] = encoder_ckpt['conv1.weight']
                encoder_ckpt['conv1.weight'] = altered_input_layer
            mapped_encoder_ckpt = dict(encoder_ckpt)
            for p, v in encoder_ckpt.items():
                for original_name, psp_name in RESNET_MAPPING.items():
                    if original_name in p:
                        mapped_encoder_ckpt[p.replace(original_name, psp_name)] = v
                        mapped_encoder_ckpt.pop(p)
            return encoder_ckpt

    @staticmethod
    def __get_keys(d, name):
        if 'state_dict' in d:
            d = d['state_dict']
        d_filt = {k[len(name):]: v for k, v in d.items() if k[:len(name)] == name}
        return d_filt

