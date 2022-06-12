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
import argparse
import shutil
import numpy as np
import imageio
import time
import torch

from models import make_model
from visualize.utils import generate

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()

    parser.add_argument('ckpt', type=str, help="path to the model checkpoint")
    parser.add_argument('--outdir', type=str, default='./results/samples/', 
        help="path to the output directory")
    parser.add_argument('--batch', type=int, default=8, help="batch size for inference")
    parser.add_argument("--sample", type=int, default=20,
        help="number of samples to be generated",)
    parser.add_argument("--truncation", type=float, default=0.7, help="truncation ratio")
    parser.add_argument("--truncation_mean", type=int, default=10000,
        help="number of vectors to calculate mean for the truncation")
    parser.add_argument("--save_latent", action="store_true",
        help="whether to save the output latent codes")
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

    print("Generating images ...")
    start_time = time.time()
    with torch.no_grad():
        styles = model.style(torch.randn(args.sample, model.style_dim, device=args.device))
        styles = args.truncation * styles + (1-args.truncation) * mean_latent.unsqueeze(0)
        images, segs = generate(model, styles, mean_latent=mean_latent, batch_size=args.batch)
        for i in range(len(images)):
            imageio.imwrite(f"{args.outdir}/{str(i).zfill(6)}_img.jpg", images[i])
            imageio.imwrite(f"{args.outdir}/{str(i).zfill(6)}_seg.jpg", segs[i])
            if args.save_latent:
                np.save(f'{args.outdir}/{str(i).zfill(6)}_latent.npy', styles[i:i+1].cpu().numpy())
    print(f"Average speed: {(time.time() - start_time)/(args.sample)}s")