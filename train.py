# Copyright (C) 2022 ByteDance Inc.
# All rights reserved.
# Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).

# The software is made available under Creative Commons BY-NC-SA 4.0 license
# by ByteDance Inc. You can use, redistribute, and adapt it
# for non-commercial purposes, as long as you (a) give appropriate credit
# by citing our paper, (b) indicate any changes that you've made,
# and (c) distribute any derivative works under the same license.

# THE AUTHORS DISCLAIM ALL WARRANTIES WITH REGARD TO THIS SOFTWARE, INCLUDING ALL
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR ANY PARTICULAR PURPOSE.
# IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY SPECIAL, INDIRECT OR CONSEQUENTIAL
# DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
# WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING
# OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.

import argparse
import math
import random
import os
import sys
import time
import subprocess

import numpy as np
import torch
from torch import nn, autograd, optim
from torch.nn import functional as F
from torch.utils import data
from torchvision import transforms, utils
from torch.utils.tensorboard import SummaryWriter

from models import make_model, DualBranchDiscriminator
from utils.dataset import MaskDataset

from utils.distributed import (
    get_rank,
    synchronize,
    reduce_loss_dict,
    reduce_sum,
    get_world_size,
)

import functools
from utils.inception_utils import sample_gema, prepare_inception_metrics
from visualize.utils import color_map
import random

def data_sampler(dataset, shuffle, distributed):
    if distributed:
        return data.distributed.DistributedSampler(dataset, shuffle=shuffle)

    if shuffle:
        return data.RandomSampler(dataset)

    else:
        return data.SequentialSampler(dataset)


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(1 - decay, par2[k].data)


def sample_data(loader):
    while True:
        for batch in loader:
            yield batch


def d_logistic_loss(real_pred, fake_pred):
    real_loss = F.softplus(-real_pred)
    fake_loss = F.softplus(fake_pred)

    return real_loss.mean() + fake_loss.mean()


def d_r1_loss(real_pred, real_img, real_mask):
    grad_real_img, grad_real_mask = autograd.grad(
        outputs=real_pred.sum(), inputs=[real_img,real_mask], create_graph=True
    )
    grad_penalty_img = grad_real_img.pow(2).reshape(grad_real_img.shape[0], -1).sum(1).mean()
    grad_penalty_seg = grad_real_mask.pow(2).reshape(grad_real_mask.shape[0], -1).sum(1).mean()

    return grad_penalty_img, grad_penalty_seg


def g_nonsaturating_loss(fake_pred):
    loss = F.softplus(-fake_pred).mean()

    return loss


def g_path_regularize(fake_img, latents, mean_path_length, decay=0.01):
    noise = torch.randn_like(fake_img) / math.sqrt(
        fake_img.shape[2] * fake_img.shape[3]
    )
    grad, = autograd.grad(
        outputs=(fake_img * noise).sum(), inputs=latents, create_graph=True
    )
    path_lengths = torch.sqrt(grad.pow(2).sum(2).mean(1))

    path_mean = mean_path_length + decay * (path_lengths.mean() - mean_path_length)

    path_penalty = (path_lengths - path_mean).pow(2).mean()

    return path_penalty, path_mean.detach(), path_lengths


def make_noise(batch, latent_dim, n_noise, device):
    if n_noise == 1:
        return torch.randn(batch, latent_dim, device=device)

    noises = torch.randn(n_noise, batch, latent_dim, device=device).unbind(0)

    return noises


def mixing_noise(batch, latent_dim, prob, device):
    if prob > 0 and random.random() < prob:
        return make_noise(batch, latent_dim, 2, device)

    else:
        return [make_noise(batch, latent_dim, 1, device)]

def color_segmap(sample_seg, color_map):
    sample_seg = torch.argmax(sample_seg, dim=1)
    sample_mask = torch.zeros((sample_seg.shape[0], sample_seg.shape[1], sample_seg.shape[2], 3), dtype=torch.float)
    for key in color_map:
        sample_mask[sample_seg==key] = torch.tensor(color_map[key], dtype=torch.float)
    sample_mask = sample_mask.permute(0,3,1,2)
    return sample_mask

def save_sample_image(folder, name, sample_img, global_step, writer=None, **kwargs):
    n_sample = len(sample_img)
    utils.save_image(
            sample_img,
            os.path.join(ckpt_dir, f'{folder}/{name}_{str(global_step).zfill(6)}.jpeg'),
            nrow=int(math.ceil(n_sample ** 0.5)),
            **kwargs
    )
    if writer is not None:
        writer.add_image(name, utils.make_grid(
            sample_img,
            nrow=int(math.ceil(n_sample ** 0.5)),
            **kwargs
        ), global_step)

def train(args, ckpt_dir, loader, generator, discriminator, g_optim, d_optim, g_ema, device, writer):

    get_inception_metrics = prepare_inception_metrics(args.inception, False)
    # sample func for calculate FID
    sample_fn = functools.partial(sample_gema, g_ema=g_ema, device=device, 
                        truncation=1.0, mean_latent=None, batch_size=args.batch)

    loader = sample_data(loader)
    pbar = range(args.iter)

    mean_path_length = 0

    d_loss_val = 0
    r1_img_loss = torch.tensor(0.0, device=device)
    r1_seg_loss = torch.tensor(0.0, device=device)
    g_loss_val = 0
    path_loss = torch.tensor(0.0, device=device)
    path_lengths = torch.tensor(0.0, device=device)
    mean_path_length_avg = 0
    loss_dict = {}

    if args.distributed:
        g_module = generator.module
        d_module = discriminator.module
    else:
        g_module = generator
        d_module = discriminator
        
    accum = 0.5 ** (32 / (10 * 1000))

    sample_z = torch.randn(args.n_sample, args.latent, device=device)

    print("Start Training Iterations...")
    for idx in pbar:
        tic = time.time()
        i = idx + args.start_iter

        if i > args.iter:
            print('Done!')
            break

        real_data = next(loader)
        real_img, real_mask = real_data['image'], real_data['mask']
        real_img, real_mask = real_img.to(device), real_mask.to(device)
        
        ### Train Discriminator ###
        requires_grad(generator, False)
        requires_grad(discriminator, True)

        noise = mixing_noise(args.batch, args.latent, args.mixing, device)
        fake_img, fake_seg = generator(noise)

        fake_pred = discriminator(fake_img, fake_seg)
        real_pred = discriminator(real_img, real_mask)

        d_loss = d_logistic_loss(real_pred, fake_pred)

        loss_dict['d'] = d_loss
        loss_dict['real_score'] = real_pred.mean()
        loss_dict['fake_score'] = fake_pred.mean()

        discriminator.zero_grad()
        d_loss.backward()
        d_optim.step()

        d_regularize = i % args.d_reg_every == 0

        if d_regularize:
            real_img.requires_grad = True
            real_mask.requires_grad = True
            real_pred = discriminator(real_img, real_mask)
            r1_img_loss, r1_seg_loss = d_r1_loss(real_pred, real_img, real_mask)

            discriminator.zero_grad()
            ((args.r1_img/2*r1_img_loss+args.r1_seg/2*r1_seg_loss) * args.d_reg_every + 0 * real_pred[0]).backward()

            d_optim.step()

        loss_dict['r1_img'] = r1_img_loss
        loss_dict['r1_seg'] = r1_seg_loss


        ### Train Generator ###
        requires_grad(generator, True)
        requires_grad(discriminator, False)

        noise = mixing_noise(args.batch, args.latent, args.mixing, device)
        fake_img, fake_seg, fake_seg_coarse, _, _ = generator(noise, return_all=True)

        fake_pred = discriminator(fake_img, fake_seg)
        g_loss = g_nonsaturating_loss(fake_pred)

        # segmentation mask loss
        fake_seg_downsample = F.adaptive_avg_pool2d(fake_seg, fake_seg_coarse.shape[2:4])
        mask_loss = torch.square(fake_seg_coarse - fake_seg_downsample).mean()

        loss_dict['g'] = g_loss
        loss_dict['mask'] = mask_loss

        generator.zero_grad()
        (g_loss + args.lambda_mask * mask_loss).backward()
        g_optim.step()

        g_regularize = args.path_regularize > 0 and i % args.g_reg_every == 0

        if g_regularize:
            path_batch_size = max(1, args.batch // args.path_batch_shrink)
            with torch.no_grad():
                noise = mixing_noise(
                    path_batch_size, args.latent, args.mixing, device
                )
                noise = [g_module.style(n) for n in noise]
                latents = g_module.mix_styles(noise).clone()
            latents.requires_grad = True
            fake_img, fake_seg = generator([latents], input_is_latent=True)

            path_loss, mean_path_length, path_lengths = g_path_regularize(
                fake_img, latents, mean_path_length
            )

            generator.zero_grad()
            weighted_path_loss = args.path_regularize * args.g_reg_every * path_loss

            if args.path_batch_shrink:
                weighted_path_loss += 0 * fake_img[0, 0, 0, 0] + 0 * fake_seg[0, 0, 0, 0]

            weighted_path_loss.backward()

            g_optim.step()

            mean_path_length_avg = (
                reduce_sum(mean_path_length).item() / get_world_size()
            )

        loss_dict['path'] = path_loss
        loss_dict['path_length'] = path_lengths.mean()

        accumulate(g_ema, g_module, accum)


        ### Summarize Information ###
        loss_reduced = reduce_loss_dict(loss_dict)

        d_loss_val = loss_reduced['d'].mean().item()
        g_loss_val = loss_reduced['g'].mean().item()
        r1_img_val = loss_reduced['r1_img'].mean().item()
        r1_seg_val = loss_reduced['r1_seg'].mean().item()
        path_loss_val = loss_reduced['path'].mean().item()
        real_score_val = loss_reduced['real_score'].mean().item()
        fake_score_val = loss_reduced['fake_score'].mean().item()
        path_length_val = loss_reduced['path_length'].mean().item()
        mask_loss_val = loss_reduced['mask'].mean().item()
        batch_time = time.time() - tic

        if get_rank() == 0:
            if i% 100 == 0:
                print(
                        f"[{i:06d}] d: {d_loss_val:.4f}; g: {g_loss_val:.4f}; "
                        f"real: {real_score_val:.4f}; fake: {fake_score_val:.4f}; "
                        f"r1_img: {r1_img_val:.4f}; r1_seg: {r1_seg_val:.4f}; "
                        f"path: {path_loss_val:.4f}; mean path: {mean_path_length_avg:.4f}; "
                        f"mask: {mask_loss_val:.4f}; time: {batch_time:.2f}"
                    )

                # write to tensorboard
                if writer is not None:
                    writer.add_scalar('scores/real_score', real_score_val, global_step=i)
                    writer.add_scalar('scores/fake_score', fake_score_val, global_step=i)
                    
                    writer.add_scalar('r1/img', r1_img_val, global_step=i)
                    writer.add_scalar('r1/seg', r1_seg_val, global_step=i)

                    writer.add_scalar('path/path_loss', path_loss_val, global_step=i)
                    writer.add_scalar('path/path_length', path_length_val, global_step=i)

                    writer.add_scalar('loss/d', d_loss_val, global_step=i)
                    writer.add_scalar('loss/g', g_loss_val, global_step=i)
                    writer.add_scalar('loss/mask', mask_loss_val, global_step=i)

            if i % args.viz_every == 0:
                with torch.no_grad():
                    g_ema.eval()
                    sample_img, sample_seg, sample_seg_coarse, depths, _ = g_ema([sample_z], return_all=True)
                    sample_img = sample_img.detach().cpu()
                    sample_mask = color_segmap(sample_seg.detach().cpu(), color_map)
                    sample_mask_coarse = color_segmap(sample_seg_coarse.detach().cpu(), color_map)
                    depths = [d.detach().cpu() for d in depths]

                    os.makedirs(os.path.join(ckpt_dir, 'sample'), exist_ok=True)
                    os.makedirs(os.path.join(ckpt_dir, 'depth'), exist_ok=True)
                    save_sample_image("sample", "img", sample_img, i, writer, normalize=True, value_range=(-1,1))
                    save_sample_image("sample", "mask", sample_mask, i, writer, normalize=True, value_range=(0,255))
                    save_sample_image("sample", "mask_coarse", sample_mask_coarse, i, writer, normalize=True, value_range=(0,255))
                    for j in range(len(depths)):
                        save_sample_image("depth", f"depth_{j:02d}", depths[j], i, writer, normalize=True)

                
            if i % args.save_every == 0 and i > args.start_iter:
                print("==================Start calculating FID==================")
                IS_mean, IS_std, FID = get_inception_metrics(sample_fn, num_inception_images=10000, use_torch=False)
                print("[val] iteration {0:06d}: FID: {1:.4f}, IS_mean: {2:.4f}, IS_std: {3:.4f}".format(i, FID, IS_mean, IS_std))
                if writer is not None:
                    writer.add_scalar('metrics/FID', FID, global_step=i)
                    writer.add_scalar('metrics/IS_mean', IS_mean, global_step=i)
                    writer.add_scalar('metrics/IS_std', IS_std, global_step=i)
                
                os.makedirs(os.path.join(ckpt_dir, 'ckpt'), exist_ok=True)
                torch.save(
                    {
                        'g': g_module.state_dict(),
                        'd': d_module.state_dict(),
                        'g_ema': g_ema.state_dict(),
                        "g_optim": g_optim.state_dict(),
                        "d_optim": d_optim.state_dict(),
                        'args': args,
                    },
                    os.path.join(ckpt_dir, f'ckpt/{str(i).zfill(6)}.pt'),
                )


if __name__ == '__main__':
    device = 'cuda'

    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--inception', type=str, help='inception pkl', required=True)
    parser.add_argument('--checkpoint_dir', type=str, default='./output/')
    parser.add_argument('--ckpt', type=str, default=None)

    parser.add_argument('--iter', type=int, default=200001)
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--n_sample', type=int, default=16)
    parser.add_argument('--size', type=int, default=256)
    parser.add_argument('--r1_img', type=float, default=10)
    parser.add_argument('--r1_seg', type=float, default=1000)
    parser.add_argument('--path_regularize', type=float, default=0.5)
    parser.add_argument('--path_batch_shrink', type=int, default=2)
    parser.add_argument('--d_reg_every', type=int, default=16)
    parser.add_argument('--g_reg_every', type=int, default=4)
    parser.add_argument('--viz_every', type=int, default=2000)
    parser.add_argument('--save_every', type=int, default=10000)

    parser.add_argument('--mixing', type=float, default=0.3)
    parser.add_argument('--lr', type=float, default=0.002)
    parser.add_argument('--channel_multiplier', type=int, default=2)
    
    parser.add_argument('--seg_dim', type=int, default=13)
    parser.add_argument('--aug', action='store_true', help='augmentation')

    # Semantic StyleGAN
    parser.add_argument('--local_layers', type=int, default=10, help="number of layers in local generators")
    parser.add_argument('--base_layers', type=int, default=2, help="number of layers with shared coarse structure code")
    parser.add_argument('--depth_layers', type=int, default=6, help="number of layers before outputing pseudo-depth map")
    parser.add_argument('--local_channel', type=int, default=64, help="number of channels in local generators")
    parser.add_argument('--coarse_channel', type=int, default=512, help="number of channels in coarse feature map")
    parser.add_argument('--coarse_size', type=int, default=64, help="size of the coarse feature map and segmentation mask")
    parser.add_argument('--min_feat_size', type=int, default=16, help="size of downsampled feature map")
    parser.add_argument('--residual_refine', action="store_true", help="whether to use residual to refine the coarse mask")
    parser.add_argument('--detach_texture', action="store_true", help="whether to detach between depth layers and texture layers")
    parser.add_argument('--transparent_dims', nargs="+", default=(10,12), type=int, help="the indices of transparent classes")
    parser.add_argument('--lambda_mask', type=float, default=100.0, help="weight of the mask regularization loss")

    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--local_rank', type=int, default=0)

    args = parser.parse_args()

    # build checkpoint dir
    ckpt_dir = args.checkpoint_dir
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    writer = SummaryWriter(log_dir=ckpt_dir)

    n_gpu = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    args.n_gpu = n_gpu

    args.distributed = n_gpu > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        synchronize()

    args.latent = 512
    args.n_mlp = 8

    args.start_iter = 0

    generator = make_model(args, verbose=(args.local_rank==0)).to(device)

    discriminator = DualBranchDiscriminator(
        args.size, args.size, img_dim=3, seg_dim=args.seg_dim, channel_multiplier=args.channel_multiplier
    ).to(device)
 
    g_ema = make_model(args, verbose=(args.local_rank==0)).to(device)
    g_ema.eval()
    accumulate(g_ema, generator, 0)

    g_reg_ratio = args.g_reg_every / (args.g_reg_every + 1)
    d_reg_ratio = args.d_reg_every / (args.d_reg_every + 1)

    g_optim = optim.Adam(
        generator.parameters(),
        lr=args.lr * g_reg_ratio,
        betas=(0 ** g_reg_ratio, 0.99 ** g_reg_ratio),
    )
    d_optim = optim.Adam(
        discriminator.parameters(),
        lr=args.lr * d_reg_ratio,
        betas=(0 ** d_reg_ratio, 0.99 ** d_reg_ratio),
    )

    if args.ckpt is not None:
        print('load model:', args.ckpt)
        
        ckpt = torch.load(args.ckpt, map_location='cpu')

        ckpt_name = os.path.basename(args.ckpt)
        args.start_iter = int(os.path.splitext(ckpt_name)[0])

        generator.load_state_dict(ckpt['g'])
        discriminator.load_state_dict(ckpt['d'])
        g_ema.load_state_dict(ckpt['g_ema'])

        g_optim.load_state_dict(ckpt['g_optim'])
        d_optim.load_state_dict(ckpt['d_optim'])

    if args.distributed:
        find_unused_parameters = True
        generator = nn.parallel.DistributedDataParallel(
            generator,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
            find_unused_parameters=find_unused_parameters,
        )

        discriminator = nn.parallel.DistributedDataParallel(
            discriminator,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
            find_unused_parameters=find_unused_parameters,
        )

    dataset = MaskDataset(args.dataset, resolution=args.size, label_size=args.seg_dim, aug=args.aug)
    print("Loading train dataloader with size ", len(dataset))

    loader = data.DataLoader(
        dataset,
        batch_size=args.batch,
        sampler=data_sampler(dataset, shuffle=True, distributed=args.distributed),
        num_workers=args.num_workers//2,
        drop_last=True,
    )

    torch.backends.cudnn.benchmark = True
    
    print("Start Training...")
    train(args, ckpt_dir, loader, generator, discriminator, g_optim, d_optim, g_ema, device, writer)
