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
from glob import glob
import argparse
from io import BytesIO
import multiprocessing
from functools import partial

from PIL import Image
import lmdb
from tqdm import tqdm
from torchvision import datasets
from torchvision.transforms import functional as trans_fn


def resize_and_convert(img, size, format, resample):
    img = trans_fn.resize(img, size, resample)
    img = trans_fn.center_crop(img, size)
    buffer = BytesIO()
    img.save(buffer, format=format, quality=100)
    val = buffer.getvalue()

    return val


def resize_worker(img_file, size, use_rgb, format, resample):
    i, file = img_file
    img = Image.open(file)
    if use_rgb:
        img = img.convert("RGB")
    img = resize_and_convert(img, size, format, resample)
    return i, img

def find_images(path):
    if os.path.isfile(path):
        with open(path, "r") as f:
            files = [line.strip() for line in f.readlines()]
    else:
        files = list()
        IMAGE_EXTENSIONS = {'jpg', 'png', 'jpeg', 'webp'}
        IMAGE_EXTENSIONS = IMAGE_EXTENSIONS.union({f.upper() for f in IMAGE_EXTENSIONS})
        for ext in IMAGE_EXTENSIONS:
            files += glob(f'{path}/**/*.{ext}', recursive=True)
        files = sorted(files)
    return files

def prepare(env, files, n_worker, size, prefix, use_rgb, format, resample):
    resize_fn = partial(resize_worker, size=size, use_rgb=use_rgb, format=format, resample=resample)

    total = 0
    with env.begin(write=True) as txn:
        with multiprocessing.Pool(n_worker) as pool:
            for i, img in tqdm(pool.imap_unordered(resize_fn, enumerate(files))):
                txn.put(f"{prefix}-{str(i).zfill(7)}".encode("utf-8"), img)
                total += 1
            txn.put(f"{prefix}-length".encode("utf-8"), str(total).encode("utf-8"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess images for model training")
    parser.add_argument("--out", type=str, help="filename of the result lmdb dataset")
    parser.add_argument(
        "--size",
        type=int,
        default=256,
        help="resolutions of images for the dataset",
    )
    parser.add_argument(
        "--n_worker",
        type=int,
        default=8,
        help="number of workers for preparing dataset",
    )
    parser.add_argument("image_path", type=str, help="path to the image files")
    parser.add_argument("label_path", type=str, help="path to the label files")

    args = parser.parse_args()

    images = find_images(args.image_path)
    labels = find_images(args.label_path)
    get_key = lambda fpath: os.path.splitext(os.path.basename(fpath))[0] # Identify by basename
    label_dict = {get_key(label):label for label in labels}
    labels = [label_dict[get_key(image)] for image in images]

    print(f"Number of images: {len(images)}")
    with lmdb.open(args.out, map_size=1024 ** 4, readahead=False) as env:
        prepare(env, images, args.n_worker, args.size, 'image', use_rgb=True, format='jpeg', resample=Image.LANCZOS)
        prepare(env, labels, args.n_worker, args.size, 'label', use_rgb=False, format='png', resample=Image.NEAREST)
