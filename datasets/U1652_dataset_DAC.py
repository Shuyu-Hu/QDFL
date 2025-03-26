import random
from pathlib import Path
from tqdm import tqdm
import numpy as np
import cv2
import torchvision.transforms as T
import copy
from torch.utils.data import Dataset
import os
import time
import albumentations as A
from PIL import Image
default_transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

BASE_PATH = '/media/whu/Largedisk/datasets/U1652/University-Release/train/'
# BASE_PATH = '/media/whu/Largedisk/datasets/SUES-200-512x512/Training/200/'

if not Path(BASE_PATH).exists():
    raise FileNotFoundError(
        'BASE_PATH is hardcoded, please adjust to point to U1652/train')


def get_data(path):
    data = {}
    for root, dirs, files in os.walk(path, topdown=False):
        for name in dirs:
            data[name] = {"path": os.path.join(root, name)}
            for _, _, files in os.walk(data[name]["path"], topdown=False):
                data[name]["files"] = files

    return data


class U1652Dataset_DAC(Dataset):
    def __init__(self,
                 sat_transform=default_transform,
                 drone_transform=default_transform,
                 sources=['satellite', 'street', 'drone', 'google'],
                 prob_flip=0.5,
                 shuffle_batch_size=24):
        super(U1652Dataset_DAC).__init__()
        self.sat_transform = sat_transform
        self.drone_transform = drone_transform

        self.sat_dict = get_data(os.path.join(BASE_PATH, 'satellite'))
        self.drone_dict = get_data(os.path.join(BASE_PATH, 'drone'))

        # use only folders that exists for both gallery and query
        self.ids = list(set(self.sat_dict.keys()).intersection(self.drone_dict.keys()))
        self.ids.sort()
        self.map_dict = {i: self.ids[i] for i in range(len(self.ids))}
        self.reverse_map_dict = {v: k for k, v in self.map_dict.items()}

        self.pairs = []

        for idx in self.ids:

            query_img = "{}/{}".format(self.sat_dict[idx]["path"],
                                       self.sat_dict[idx]["files"][0])

            gallery_path = self.drone_dict[idx]["path"]
            gallery_imgs = self.drone_dict[idx]["files"]

            label = self.reverse_map_dict[idx]

            for g in gallery_imgs:
                self.pairs.append((idx, label, query_img, "{}/{}".format(gallery_path, g)))

        self.prob_flip = prob_flip
        self.shuffle_batch_size = shuffle_batch_size

        self.samples = copy.deepcopy(self.pairs)

        self.total_img_num = len(self.sat_dict) + sum([len(v['files']) for v in self.drone_dict.values()])

    def __getitem__(self, index):

        idx, label, query_img_path, gallery_img_path = self.samples[index]

        # for query there is only one file in folder
        img_s = cv2.imread(query_img_path)
        img_s = cv2.cvtColor(img_s, cv2.COLOR_BGR2RGB)

        img_d = cv2.imread(gallery_img_path)
        img_d = cv2.cvtColor(img_d, cv2.COLOR_BGR2RGB)

        if np.random.random() < self.prob_flip:
            img_s = cv2.flip(img_s, 1)
            img_d = cv2.flip(img_d, 1)

            # image transforms
        if self.sat_transform is not None:
            if isinstance(self.sat_transform, A.BasicTransform) or isinstance(self.sat_transform, A.Compose):
                img_s = self.sat_transform(image=img_s)['image']  # 适用于 Albumentations
            elif isinstance(self.sat_transform, T.Compose):
                if isinstance(img_s, np.ndarray):
                    img_s = Image.fromarray(img_s)  # 将 NumPy 数组转换为 PIL 图像
                img_s = self.sat_transform(img_s)  # 适用于 PyTorch

            # 检查并应用 drone_transform
        if self.drone_transform is not None:
            if isinstance(self.drone_transform, A.BasicTransform) or isinstance(self.drone_transform, A.Compose):
                img_d = self.drone_transform(image=img_d)['image']  # 适用于 Albumentations
            elif isinstance(self.drone_transform, T.Compose):
                if isinstance(img_d, np.ndarray):
                    img_d = Image.fromarray(img_d)  # 将 NumPy 数组转换为 PIL 图像
                img_d = self.drone_transform(img_d)  # 适用于 PyTorch

        return img_s, img_d, idx, label


    def __len__(self):
        return len(self.sat_dict)


    def shuffle(self, ):
        '''
        custom shuffle function for unique class_id sampling in batch
        '''

        print("\nShuffle Dataset:")

        pair_pool = copy.deepcopy(self.pairs)

        # Shuffle pairs order
        random.shuffle(pair_pool)

        # Lookup if already used in epoch
        pairs_epoch = set()
        idx_batch = set()

        # buckets
        batches = []
        current_batch = []

        # counter
        break_counter = 0

        # progressbar
        pbar = tqdm()

        while True:

            pbar.update()

            if len(pair_pool) > 0:
                pair = pair_pool.pop(0)

                idx, _, _, _ = pair

                if idx not in idx_batch and pair not in pairs_epoch:

                    idx_batch.add(idx)
                    current_batch.append(pair)
                    pairs_epoch.add(pair)

                    break_counter = 0

                else:
                    # if pair fits not in batch and is not already used in epoch -> back to pool
                    if pair not in pairs_epoch:
                        pair_pool.append(pair)

                    break_counter += 1

                if break_counter >= 512:
                    break

            else:
                break

            if len(current_batch) >= self.shuffle_batch_size:
                # empty current_batch bucket to batches
                batches.extend(current_batch)
                idx_batch = set()
                current_batch = []

        pbar.close()

        # wait before closing progress bar
        time.sleep(0.3)

        self.samples = batches

        print("Original Length: {} - Length after Shuffle: {}".format(len(self.pairs), len(self.samples)))
        print("Break Counter:", break_counter)
        print("Pairs left out of last batch to avoid creating noise:", len(self.pairs) - len(self.samples))
        print("First Element ID: {} - Last Element ID: {}".format(self.samples[0][0], self.samples[-1][0]))

#
# datasets = U1652Dataset(sat_transform=default_transform, drone_transform=default_transform,
#                         sources=['satellite', 'street', 'drone'])
# datasets.shuffle()
# print(1)
