from pathlib import Path
import albumentations as A
import pandas as pd
import torch
import numpy as np
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import os

default_transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

BASE_PATH = '/media/whu/Largedisk/datasets/SUES-200-512x512/Training'

if not Path(BASE_PATH).exists():
    raise FileNotFoundError(
        'BASE_PATH is hardcoded, please adjust to point to SUES-200-512x512/Training')


class SUES200Dataset(Dataset):
    def __init__(self, sat_transform=default_transform, drone_transform=default_transform, height=200):
        super(SUES200Dataset).__init__()
        self.sat_transform = sat_transform
        self.drone_transform = drone_transform
        self.height = height
        self.data_path = os.path.join(BASE_PATH, f'{height}')

        # Prepare the dataset paths
        self.dict_path = self._create_dataset()
        self.cls_names = os.listdir(os.path.join(self.data_path, "satellite"))
        self.cls_names.sort()
        self.map_dict = {i: self.cls_names[i] for i in range(len(self.cls_names))}
        self.total_img_num =  len(self.dict_path['satellite'])+sum(len(v) for v in self.dict_path['drone'].values())

    def _create_dataset(self):
        dict_path = {}
        for source in ['satellite', 'drone']:
            dict_ = {}
            for cls_name in os.listdir(os.path.join(self.data_path, source)):
                cls_img_list = os.listdir(os.path.join(self.data_path, source, cls_name))
                img_path_list = [os.path.join(self.data_path, source, cls_name, img) for img in cls_img_list]
                dict_[cls_name] = img_path_list
            dict_path[source] = dict_
        return dict_path

    def sample_from_cls(self, name, cls_num):
        img_path = self.dict_path[name][cls_num]
        img_path = np.random.choice(img_path, 1)[0]
        img = Image.open(img_path)
        return img.convert('RGB')

    @staticmethod
    def transforms_image(img, transforms):
        if isinstance(transforms, A.Compose):
            return transforms(image=np.array(img))['image']
        elif isinstance(transforms, T.Compose):
            return transforms(img)
        else:
            raise Exception('The transform composition is not supported yet, please edit this function')

    def __getitem__(self, index):
        cls_nums = self.map_dict[index]
        # Sample and transform satellite image
        img_s = self.sample_from_cls("satellite", cls_nums)
        if self.sat_transform is not None:
            if isinstance(self.sat_transform, A.BasicTransform) or isinstance(self.sat_transform, A.Compose):
                if isinstance(img_s, Image.Image):
                    img_s = np.array(img_s)
                img_s = self.sat_transform(image=img_s)['image']
            elif isinstance(self.sat_transform, T.Compose):
                if isinstance(img_s, np.ndarray):
                    img_s = Image.fromarray(img_s)
                img_s = self.sat_transform(img_s)

        # Sample and transform drone image
        img_d = self.sample_from_cls("drone", cls_nums)
        if self.drone_transform is not None:
            if isinstance(self.drone_transform, A.BasicTransform) or isinstance(self.drone_transform, A.Compose):
                if isinstance(img_d, Image.Image):
                    img_d = np.array(img_d)
                img_d = self.drone_transform(image=img_d)['image']
            elif isinstance(self.drone_transform, T.Compose):
                if isinstance(img_d, np.ndarray):
                    img_d = Image.fromarray(img_d)
                img_d = self.drone_transform(img_d)

        return img_s, img_d, index

    def __len__(self):
        return len(self.cls_names)

# datasets = SUES200Dataset(sat_transform=default_transform,drone_transform=default_transform)
# print(1)