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

import os
import shutil
import argparse
import numpy as np
import imageio
import torch
from models import make_model
import matplotlib.pyplot as plt

from visualize.utils import tensor2image, tensor2seg

component_dict_celeba = {
    1:  "face",
    2:  "eye",
    3:  "eyebrow",
    4:  "mouth",
    5:  "nose",
    6:  "ear",
    7:  "hair",
    8:  "neck",
    9: "cloth",
}

def visualize_alpha(output_name, tensor):
    tensor = tensor.cpu().permute(1,2,0).numpy()
    fig = plt.matshow(tensor, cmap="inferno")
    plt.gca().set_axis_off()
    plt.gcf().set_dpi(100)
    plt.savefig(output_name, bbox_inches='tight', pad_inches=0)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('ckpt', type=str, help="path to the model checkpoint")
    parser.add_argument('--latent', type=str, default=None,
        help="path to the latent numpy")
    parser.add_argument('--outdir', type=str, default='./results/components/', 
        help="path to the output directory")
    parser.add_argument('--batch', type=int, default=8, help="batch size for inference")
    parser.add_argument("--sample", type=int, default=10,
        help="number of latent samples to be interpolated")
    parser.add_argument("--truncation", type=float, default=0.7, help="truncation ratio")
    parser.add_argument("--truncation_mean", type=int, default=10000,
        help="number of vectors to calculate mean for the truncation")
    parser.add_argument("--dataset_name", type=str, default="celeba",
        help="used for finding mapping between component (local generator) indices and names")
    parser.add_argument('--device', type=str, default="cuda", 
        help="running device for inference")
    args = parser.parse_args()

    if os.path.exists(args.outdir):
        shutil.rmtree(args.outdir)
    os.makedirs(args.outdir)

    print("Loading model ...")
    ckpt = torch.load(args.ckpt)
    model = make_model(ckpt['args'])
    model.to(args.device)
    model.eval()
    model.load_state_dict(ckpt['g_ema'])
    mean_latent = model.style(torch.randn(args.truncation_mean, model.style_dim, device=args.device)).mean(0)

    print("Generating images...")
    if args.dataset_name == "celeba":
        component_dict = component_dict_celeba
    else:
        raise ValueError("Unknown dataset name: f{args.dataset_name}")

    with torch.no_grad():
        if args.latent is None:
            styles = model.style(torch.randn(args.sample, model.style_dim, device=args.device))
            styles = args.truncation * styles + (1-args.truncation) * mean_latent.unsqueeze(0)
        else:
            styles = torch.tensor(np.load(args.latent), device=args.device).reshape(1,model.style_dim)

        for sample_index in range(styles.size(0)):
            style_inputs = styles[sample_index:sample_index+1]
            composition_mask = torch.zeros(1, model.n_local, device=args.device)
            composition_mask[:,0] = 1
            sample_outdir = args.outdir if args.latent is not None else f'{args.outdir}/{sample_index}'
            if not os.path.exists(sample_outdir):
                os.makedirs(sample_outdir)
            for component_index, component_name in component_dict.items():
                composition_mask[:,component_index] = 1
                image, seg, seg_coarse, depths, _ = model([style_inputs], input_is_latent=True, randomize_noise=False,
                                            composition_mask=composition_mask, truncation_latent=mean_latent, return_all=True)
                image = tensor2image(image)
                seg = tensor2seg(seg)
                seg_coarse = tensor2seg(seg_coarse)

                imageio.imwrite(f'{sample_outdir}/image_{component_index}_{component_name}.jpg', image[0])
                imageio.imwrite(f'{sample_outdir}/seg_{component_index}_{component_name}.jpg', seg[0])
                imageio.imwrite(f'{sample_outdir}/seg_coarse_{component_index}_{component_name}.jpg', seg_coarse[0])
                visualize_alpha(f'{sample_outdir}/depth_{component_index}_{component_name}.jpg', depths[component_index][0])