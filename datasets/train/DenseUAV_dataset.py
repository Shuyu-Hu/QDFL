from pathlib import Path
import albumentations as A
import pandas as pd
import torch
import numpy as np
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset
import os
#TODO:change
default_transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

BASE_PATH = '/media/whu/Largedisk/datasets/DenseUAV/train'

# if not Path(BASE_PATH).exists():
#     raise FileNotFoundError(
#         'BASE_PATH is hardcoded, please adjust to point to U1652/train')

class DenseUAVDataset(Dataset):
    def __init__(self,sat_transform=default_transform,drone_transform=default_transform,sources=['satellite','street','drone','google']):
        super(DenseUAVDataset).__init__()
        self.sat_transform = sat_transform
        self.drone_transform = drone_transform

        dict_path = {}
        for source in sources:
            dict_ = {}
            for cls_name in os.listdir(os.path.join(BASE_PATH, source)):
                cls_img_list = os.listdir(os.path.join(BASE_PATH, source, cls_name))
                img_path_list = [os.path.join(BASE_PATH, source, cls_name, img) for img in cls_img_list]
                dict_[cls_name] = img_path_list
            dict_path[source] = dict_
        cls_names = os.listdir(os.path.join(BASE_PATH, sources[0]))
        cls_names.sort()
        map_dict = {i: cls_names[i] for i in range(len(cls_names))}
        '''
        map_dict是将每一类的路径按照类索引进行键值对映射
        cls_names[0-701]
        dict_path是不同source的所有类的所有图片路径
        '''
        self.map_dict = map_dict
        self.cls_names = cls_names
        self.dict_path = dict_path
        self.total_img_num = len(dict_path['satellite'])+sum(len(key) for key in dict_path['drone'].keys())

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
            raise Exception('The transform composition is not support yet, please edit this function')

    def __getitem__(self, index):
        cls_nums = self.map_dict[index]
        img = self.sample_from_cls("satellite", cls_nums)
        img_s = self.transforms_image(img, self.sat_transform)

        img = self.sample_from_cls("drone", cls_nums)
        img_d = self.transforms_image(img, self.drone_transform)
        return img_s, img_d, index

    def __len__(self):
        return len(self.cls_names)