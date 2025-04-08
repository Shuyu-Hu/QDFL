from pathlib import Path
import pandas as pd
import torch
import numpy as np
import torchvision.transforms as T
import albumentations as A
from PIL import Image
from torch.utils.data import Dataset
import os

default_transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

BASE_PATH = '/media/whu/Largedisk/datasets/U1652/University-Release/train/'
# BASE_PATH = '/media/whu/Largedisk/datasets/SUES-200-512x512/Training/200/'

if not Path(BASE_PATH).exists():
    raise FileNotFoundError(
        'BASE_PATH is hardcoded, please adjust to point to U1652/train')

class U1652Dataset(Dataset):
    def __init__(self,sat_transform=default_transform,drone_transform=default_transform,sources=['satellite','street','drone','google']):
        super(U1652Dataset).__init__()
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

    def sample_from_cls(self,name,cls_num):
        img_path = self.dict_path[name][cls_num]
        img_path = np.random.choice(img_path,1)[0]
        img = Image.open(img_path)
        return img.convert('RGB')

    @staticmethod
    def transforms_image(img,transforms):
        if isinstance(transforms,A.Compose):
            return transforms(image=np.array(img))['image']
        elif isinstance(transforms,T.Compose):
            return transforms(img)
        else:
            raise Exception('The transform composition is not support yet, please edit this function')

    def __getitem__(self, index):
        cls_nums = self.map_dict[index]
        img = self.sample_from_cls("satellite",cls_nums)
        if self.sat_transform is not None:
            if isinstance(self.sat_transform, A.BasicTransform) or isinstance(self.sat_transform, A.Compose):
                img_s = self.sat_transform(image=img)['image']  # 适用于 Albumentations
            elif isinstance(self.sat_transform, T.Compose):
                if isinstance(img, np.ndarray):
                    img_s = Image.fromarray(img)  # 将 NumPy 数组转换为 PIL 图像
                img_s = self.sat_transform(img_s)  # 适用于 PyTorch

        img = self.sample_from_cls("street",cls_nums)
        if self.drone_transform is not None:
            if isinstance(self.drone_transform, A.BasicTransform) or isinstance(self.drone_transform, A.Compose):
                img_st = self.drone_transform(image=img)['image']  # 适用于 Albumentations
            elif isinstance(self.drone_transform, T.Compose):
                if isinstance(img, np.ndarray):
                    img_st = Image.fromarray(img)  # 将 NumPy 数组转换为 PIL 图像
                img_st = self.drone_transform(img_st)  # 适用于 PyTorch

        img = self.sample_from_cls("drone",cls_nums)
        if self.drone_transform is not None:
            if isinstance(self.drone_transform, A.BasicTransform) or isinstance(self.drone_transform, A.Compose):
                img_d = self.drone_transform(image=img)['image']  # 适用于 Albumentations
            elif isinstance(self.drone_transform, T.Compose):
                if isinstance(img, np.ndarray):
                    img_d = Image.fromarray(img)  # 将 NumPy 数组转换为 PIL 图像
                img_d = self.sat_transform(img_d)  # 适用于 PyTorch


        return img_s,img_st,img_d,index

    def __len__(self):
        return len(self.cls_names)

# datasets = U1652Dataset(sat_transform=default_transform,drone_transform=default_transform,
#                             sources=['satellite', 'street', 'drone'])
# print(1)

# class U1652Dataset(Dataset):
#     def __init__(self,sat_transform=default_transform,drone_transform=default_transform,sources=['satellite','street','drone','google']):
#         super(U1652Dataset).__init__()
#         self.sat_transform = sat_transform
#         self.drone_transform = drone_transform
#
#         dict_path = {}
#         for source in sources:
#             dict_ = {}
#             for cls_name in os.listdir(os.path.join(BASE_PATH, source)):
#                 cls_img_list = os.listdir(os.path.join(BASE_PATH, source, cls_name))
#                 img_path_list = [os.path.join(BASE_PATH, source, cls_name, img) for img in cls_img_list]
#                 dict_[cls_name] = img_path_list
#             dict_path[source] = dict_
#         cls_names = os.listdir(os.path.join(BASE_PATH, sources[0]))
#         cls_names.sort()
#         map_dict = {i: cls_names[i] for i in range(len(cls_names))}
#         '''
#         map_dict是将每一类的路径按照类索引进行键值对映射
#         cls_names[0-701]
#         dict_path是不同source的所有类的所有图片路径
#         '''
#         self.map_dict = map_dict
#         self.cls_names = cls_names
#         self.dict_path = dict_path
#         self.total_img_num = len(dict_path['satellite'])+sum(len(key) for key in dict_path['drone'].keys())
#
#     def sample_from_cls(self,name,cls_num):
#         img_path = self.dict_path[name][cls_num]
#         img_path = np.random.choice(img_path,1)[0]
#         img = Image.open(img_path)
#         return img.convert('RGB')
#
#     @staticmethod
#     def transforms_image(img,transforms):
#         if isinstance(transforms,A.Compose):
#             return transforms(image=np.array(img))['image']
#         elif isinstance(transforms,T.Compose):
#             return transforms(img)
#         else:
#             raise Exception('The transform composition is not support yet, please edit this function')
#
#     def __getitem__(self, index):
#         cls_nums = self.map_dict[index]
#         img = self.sample_from_cls("satellite",cls_nums)
#         img_s = self.transforms_image(img,self.sat_transform)
#
#         img = self.sample_from_cls("street",cls_nums)
#         img_st = self.transforms_image(img,self.drone_transform)
#
#         img = self.sample_from_cls("drone",cls_nums)
#         img_d = self.transforms_image(img,self.drone_transform)
#         return img_s,img_st,img_d,index
#
#     def __len__(self):
#         return len(self.cls_names)