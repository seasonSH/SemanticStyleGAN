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

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

import argparse
from utils import inception_utils
from utils.dataset import MaskDataset, MultiResolutionDataset
import pickle

@torch.no_grad()
def extract_features(args, loader, inception, device):

    pools, logits = [], []

    for data in tqdm(loader):
        if isinstance(data, torch.Tensor):
            img = data
        else:
            img = data['image']
            
        # check img dim
        if img.shape[1] != 3:
            img = img.expand(-1,3,-1,-1)

        img = img.to(device)
        pool_val, logits_val = inception(img)
        
        pools.append(pool_val.cpu().numpy())
        logits.append(F.softmax(logits_val, dim=1).cpu().numpy())

    pools = np.concatenate(pools, axis=0)
    logits = np.concatenate(logits, axis=0)

    return pools, logits


def get_dataset(args):
    if args.dataset_type == 'mask':
        print(f"Using mask dataset: {args.path}")
        dataset = MaskDataset(args.path, resolution=args.size)
    else:
        print(f"Using image dataset: {args.path}")
        transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5), inplace=True)
                ])
        dataset = MultiResolutionDataset(args.path, transform=transform, resolution=args.size)
    return dataset

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    parser = argparse.ArgumentParser(
        description='Calculate Inception v3 features for datasets'
    )
    parser.add_argument('--size', type=int, default=256)
    parser.add_argument('--batch', default=64, type=int, help='batch size')
    parser.add_argument('--n_sample', type=int, default=None)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--image_mode', type=str, default='RGB')
    parser.add_argument('--dataset_type', type=str, choices=["mask","image"], default="mask")
    parser.add_argument('path', metavar='PATH', help='path to datset dir')

    args = parser.parse_args()

    inception = inception_utils.load_inception_net()

    dset = get_dataset(args)
    loader = DataLoader(dset, shuffle=False, batch_size=args.batch, drop_last=False, num_workers=4)

    pools, logits = extract_features(args, loader, inception, device)

    if args.n_sample is not None:
        pools = pools[: args.n_sample]
        logits = logits[: args.n_sample]

    print(f'extracted {pools.shape[0]} features')

    print('Calculating inception metrics...')
    IS_mean, IS_std = inception_utils.calculate_inception_score(logits)
    print('Training data from dataloader has IS of %5.5f +/- %5.5f' % (IS_mean, IS_std))
    print('Calculating means and covariances...')

    mean = np.mean(pools, axis=0)
    cov = np.cov(pools, rowvar=False)

    with open(args.output, 'wb') as f:
        pickle.dump({'mean': mean, 'cov': cov, 'size': args.size, 'path': args.path}, f)
