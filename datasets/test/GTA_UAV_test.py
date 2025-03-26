from pathlib import Path
import pandas as pd
import torch
import numpy as np
import torchvision.transforms as T
import torchvision.datasets as datasets
from PIL import Image
from torch.utils.data import Dataset
import os

import json
default_transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

BASE_PATH = '/media/whu/Largedisk/datasets/GTA-UAV-LR/'

if not Path(BASE_PATH).exists():
    raise FileNotFoundError(
        'BASE_PATH is hardcoded, please adjust to point to GTA-UAV/')

SATE_LENGTH = 24576
TILE_LENGTH = 256

def sate2loc(tile_zoom, offset, tile_x, tile_y):
    tile_pix = SATE_LENGTH / (2 ** tile_zoom)
    loc_x = (tile_pix * (tile_x+1/2+offset/TILE_LENGTH)) * 0.45
    loc_y = (tile_pix * (tile_y+1/2+offset/TILE_LENGTH)) * 0.45
    return loc_x, loc_y

def get_sate_data(root_dir):
    sate_img_dir_list = []
    sate_img_list = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            sate_img_dir_list.append(root)
            sate_img_list.append(file)
    return sate_img_dir_list, sate_img_list

class GTA_UAV_test(Dataset):
    def __init__(self,
                 pairs_meta_file,
                 which_view,
                 mode='pos',
                 query_mode='D2S',
                 pairs_sate2drone_dict=None,
                 transforms=default_transform,
                 ):
        super().__init__()
        with open(os.path.join(BASE_PATH, pairs_meta_file), 'r', encoding='utf-8') as f:
            pairs_meta_data = json.load(f)
        self.BASE_PATH = BASE_PATH
        sate_img_dir = os.path.join(BASE_PATH, 'satellite')

        self.images_path = []
        self.images_name = []
        self.images_loc_xy = []

        self.pairs_sate2drone_dict = {}
        self.pairs_drone2sate_dict = {}
        self.pairs_match_set = set()

        if which_view == 'drone':
            for pair_drone2sate in pairs_meta_data:
                drone_img_name = pair_drone2sate['drone_img_name']
                drone_img_dir = pair_drone2sate['drone_img_dir']
                drone_loc_x_y = pair_drone2sate['drone_loc_x_y']
                self.pairs_drone2sate_dict[drone_img_name] = []
                pair_sate_img_list = pair_drone2sate[f'pair_{mode}_sate_img_list']
                for pair_sate_img in pair_sate_img_list:
                    self.pairs_drone2sate_dict.setdefault(drone_img_name, []).append(pair_sate_img)
                    self.pairs_sate2drone_dict.setdefault(pair_sate_img, []).append(drone_img_name)
                    self.pairs_match_set.add((drone_img_name, pair_sate_img))
                if len(pair_sate_img_list) != 0:
                    self.images_path.append(os.path.join(BASE_PATH, drone_img_dir, drone_img_name))
                    self.images_name.append(drone_img_name)
                    self.images_loc_xy.append((drone_loc_x_y[0], drone_loc_x_y[1]))

        elif which_view == 'sate':
            if query_mode == 'D2S':
                sate_img_dir_list, sate_img_list = get_sate_data(sate_img_dir)
                for sate_img_dir, sate_img in zip(sate_img_dir_list, sate_img_list):
                    self.images_path.append(os.path.join(BASE_PATH, sate_img_dir, sate_img))
                    self.images_name.append(sate_img)

                    sate_img_name = sate_img.replace('.png', '')
                    tile_zoom, offset, tile_x, tile_y = sate_img_name.split('_')
                    tile_zoom = int(tile_zoom)
                    tile_x = int(tile_x)
                    tile_y = int(tile_y)
                    offset = int(offset)
                    self.images_loc_xy.append(sate2loc(tile_zoom, offset, tile_x, tile_y))
            else:
                sate_img_dir_list, sate_img_list = get_sate_data(sate_img_dir)
                for sate_img_dir, sate_img in zip(sate_img_dir_list, sate_img_list):
                    if sate_img not in pairs_sate2drone_dict.keys():
                        continue
                    self.images_path.append(os.path.join(BASE_PATH, sate_img_dir, sate_img))
                    self.images_name.append(sate_img)

                    sate_img_name = sate_img.replace('.png', '')
                    tile_zoom, offset, tile_x, tile_y = sate_img_name.split('_')
                    tile_zoom = int(tile_zoom)
                    tile_x = int(tile_x)
                    tile_y = int(tile_y)
                    offset = int(offset)
                    self.images_loc_xy.append(sate2loc(tile_zoom, offset, tile_x, tile_y))

        # for

        self.transforms = transforms

    def __getitem__(self, index):
        # 加载图像并应用变换
        img = Image.open(self.images_path[index]).convert("RGB")
        if self.transforms is not None:
            img = self.transforms(img)
        return img

    def __len__(self):
        return len(self.images_name)

if __name__ == "__main__":
    x = GTA_UAV_test(mode='pos',
        pairs_meta_file='cross-area-drone2sate-test.json',which_view='sate')
    y = GTA_UAV_test(mode='pos',
                     pairs_meta_file='cross-area-drone2sate-test.json', which_view='drone')
    print(len(x.images_path))