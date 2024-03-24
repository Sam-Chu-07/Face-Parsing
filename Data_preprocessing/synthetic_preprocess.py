import os
import shutil
import pandas as pd
import os
import glob
from shutil import copyfile
from utils import make_folder
import argparse
import cv2
import numpy as np
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--sythetic_image_path', type=str,
                    default='./Synthetic')
parser.add_argument('--save_image_path', type=str,
                    default='./train_img')
parser.add_argument('--save_label_path', type=str,
                    default='./train_label')
parser.add_argument('--start_id', type=int,
                    default=20000)
def map_segmentation_labels(segmentation_result):
    label_mapping = {
        0: 0,   # BACKGROUND
        1: 1,   # SKIN
        2: 2,   # NOSE
        3: 5,   # RIGHT_EYE
        4: 4,   # LEFT_EYE
        5: 7,   # RIGHT_BROW
        6: 6,   # LEFT_BROW
        7: 9,   # RIGHT_EAR
        8: 8,   # LEFT_EAR
        9: 10,  # MOUTH_INTERIOR
        10: 11, # TOP_LIP
        11: 12, # BOTTOM_LIP
        12: 17, # NECK
        13: 13, # HAIR
        14: 1,  # BEARD
        15: 18, # CLOTHING
        16: 3,  # GLASSES
        17: 14, # HEADWEAR
        18: 0,  # FACEWEAR
        255: 0  # IGNORE
    }

    # 創建一個空白的矩陣，與 segmentation_result 相同的形狀
    mapped_result = np.zeros_like(segmentation_result)

    # 將每個像素的標籤進行映射
    for old_label, new_label in label_mapping.items():
        mapped_result[segmentation_result == old_label] = new_label

    return mapped_result


args = parser.parse_args()

label_paths = glob.glob(os.path.join(args.sythetic_image_path, "*_seg.png"))

img_paths = [i.replace('_seg', "") for i in label_paths]   
label_paths.sort()
img_paths.sort()

for i in tqdm(range(len(img_paths))):
    id = i + args.start_id
    seg_source_path = label_paths[i]
    img = img_paths[i]
    img_target_path = args.save_image_path + f'/{id}.jpg'
    seg_target_path = args.save_label_path + f'/{id}.png'
    seg = cv2.imread(seg_source_path, cv2.IMREAD_GRAYSCALE)
    trans_result = map_segmentation_labels(seg)

    copyfile(img, img_target_path)
    cv2.imwrite(seg_target_path, trans_result)
    