# Copyright (C) 2021 NVIDIA Corporation.  All rights reserved.
# Licensed under The MIT License (MIT)
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import os
import argparse
import torch
from models import make_model

import functools
from utils.inception_utils import sample_gema, prepare_inception_metrics

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser(
        description="Calculate FID score for generators",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        required=True,
        help="path to the checkpoint file",
    )
    parser.add_argument(
        "--inception",
        type=str,
        required=True,
        help="pre-calculated inception file",
    )
    parser.add_argument(
        "--batch", default=8, type=int, help="batch size for inception networks"
    )
    parser.add_argument(
        "--n_sample",
        type=int,
        default=50000,
        help="number of samples used for embedding calculation",
    )
    args = parser.parse_args()

    print("Loading model...")
    ckpt = torch.load(args.ckpt)
    g_args = ckpt['args']
    model = make_model(g_args).to(device).eval()
    model.load_state_dict(ckpt['g_ema'])
    mean_latent = model.style(torch.randn(50000,512, device=device)).mean(0)

    get_inception_metrics = prepare_inception_metrics(args.inception, False)
    sample_fn = functools.partial(sample_gema, g_ema=model, device=device, 
                    truncation=1.0, mean_latent=None, batch_size=args.batch)

    print("==================Start calculating FID==================")
    IS_mean, IS_std, FID = get_inception_metrics(sample_fn, num_inception_images=args.n_sample, use_torch=False)
    print("FID: {0:.4f}, IS_mean: {1:.4f}, IS_std: {2:.4f}".format(FID, IS_mean, IS_std))
