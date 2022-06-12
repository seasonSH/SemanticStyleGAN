import os
import sys
import shutil
import numpy as np
from tqdm import tqdm
from imageio import imread, imwrite
from multiprocessing import Pool

dataset_path = sys.argv[1]
img_dataset_path = os.path.join(dataset_path, 'CelebA-HQ-img/')
seg_dataset_path = os.path.join(dataset_path, 'CelebAMask-HQ-mask-anno/')
img_trainset_path = os.path.join(dataset_path, 'image_train')
seg_trainset_path = os.path.join(dataset_path, 'label_train')
img_valset_path = os.path.join(dataset_path, 'image_val')
seg_valset_path = os.path.join(dataset_path, 'label_val')
colored_label = False

mapping = {
    'skin':   1,
    'eye_g':  10,
    'l_eye':  2,
    'r_eye':  2,
    'l_brow': 3,
    'r_brow': 3,
    'mouth':  4,
    'l_lip':  4,
    'u_lip':  4,
    'nose':   5,
    'hair':   7,
    'l_ear':  6,
    'r_ear':  6,
    'neck':   8,
    'neck_l': 8,
    'cloth':  9,
    'hat':    11,
    'ear_r':  12,
}

color_map = {
    0: [0, 0, 0],
    1: [239, 234, 90],
    2: [44, 105, 154],
    3: [4, 139, 168],
    4: [13, 179, 158],
    5: [131, 227, 119],
    6: [185, 231, 105],
    7: [107, 137, 198],
    8: [241, 196, 83],
    9: [242, 158, 76],
    10: [234, 114, 71],
    11: [215, 95, 155],
    12: [207, 113, 192],
    13: [159, 89, 165],
    14: [142, 82, 172],
    15: [158, 115, 200], 
    16: [116, 95, 159],
}

def process_img(i):
    if i < 28000:
        img_output_path = img_trainset_path
        seg_output_path = seg_trainset_path
    else:
        img_output_path = img_valset_path
        seg_output_path = seg_valset_path

    # Copy the image file
    input_path = os.path.join(img_dataset_path, f'{i}.jpg')
    output_path = os.path.join(img_output_path, f'{i}.jpg')
    shutil.copyfile(input_path, output_path)

    # Make a new label file
    subfolder = str(i // 2000)
    prefix = os.path.join(seg_dataset_path, subfolder, f'{i:05d}_')
    mask = np.zeros((512,512,3), dtype=np.uint8)
    success = False
    for k,v in mapping.items():
        if not os.path.exists(prefix+k+'.png'):
            continue
        mask_ = imread(prefix+k+'.png', as_gray=True) == 255
        mask[mask_] = color_map[v] if colored_label else v
        success = True
    if not success:
        print(f"No labels found for: {prefix}")
    if not colored_label:
        mask = mask[:,:,0]
    output_path = os.path.join(seg_output_path, f'{i}.png')
    imwrite(output_path, mask)

if __name__ == "__main__":

    assert os.path.isdir(img_dataset_path)
    assert os.path.isdir(seg_dataset_path)
    os.mkdir(img_trainset_path)
    os.mkdir(seg_trainset_path)
    os.mkdir(img_valset_path)
    os.mkdir(seg_valset_path)

    pool = Pool(16)
    with tqdm(total=30000) as pbar:
        for _ in pool.imap(process_img, range(30000)):
            pbar.update()
