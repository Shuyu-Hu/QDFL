from pathlib import Path
import pandas as pd
import torch
import numpy as np
import torchvision.transforms as T
import torchvision.datasets as datasets
from PIL import Image
from torch.utils.data import Dataset
import os

default_transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

BASE_PATH = '/media/whu/Largedisk/datasets/U1652/University-Release/test/'

if not Path(BASE_PATH).exists():
    raise FileNotFoundError(
        'BASE_PATH is hardcoded, please adjust to point to U1652/test')

class U1652_test(Dataset):
    def __init__(self, which_path, transform=None):
        super(U1652_test, self).__init__()
        self.which_path = which_path
        self.transform = transform
        self.images = []
        self.labels = []
        self.path = os.path.join(BASE_PATH,which_path)
        # 遍历目录下的所有子文件夹（每个子文件夹代表一个类）
        for class_folder in os.listdir(self.path):
            class_path = os.path.join(self.path, class_folder)
            if os.path.isdir(class_path):
                # 遍历子文件夹内的所有图像
                for image_file in os.listdir(class_path):
                    image_path = os.path.join(class_path, image_file)
                    if os.path.isfile(image_path):
                        self.images.append(image_path)
                        self.labels.append(int(class_folder))


    def __getitem__(self, index):
        # 加载图像并应用变换
        img = Image.open(self.images[index])
        if self.transform is not None:
            img = self.transform(img)
        label = self.labels[index]
        return img, label

    def __len__(self):
        return len(self.images)

if __name__ == "__main__":
    x = U1652_test('gallery_satellite')
    print(x.images.shape)