import os
from glob import glob
import numpy as np
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
import copy
from PIL import Image
import torchvision.transforms as T
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
import time
import random
import shutil
from shapely.validation import make_valid
import concurrent.futures
import itertools
import pickle
import json
from pyproj import Proj, transform

from plot_utils.PlotHeatMap import input_transform

default_transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

BASE_PATH = '/media/whu/Largedisk/datasets/VLAerial/VPR/'

if not Path(BASE_PATH).exists():
    raise FileNotFoundError(
        'BASE_PATH is hardcoded, please adjust to point to VLAerial/VPR/')

def latlon_to_utm(longitude,latitude):
    # 根据经度确定 UTM 投影带
    utm_zone = int((longitude + 180) / 6) + 1
    # 设置投影参数，北半球
    proj_utm = Proj(proj='utm', zone=utm_zone, ellps='WGS84', south=latitude < 0)
    # 转换经纬度为 UTM
    easting, northing = proj_utm(longitude, latitude)
    return easting, northing

def get_database_center_coord(lb_lon,lb_lat,tr_lon,tr_lat):
    center_longitude = (lb_lon+tr_lon)/2
    center_latitude = (lb_lat+tr_lat)/2
    return latlon_to_utm(center_longitude,center_latitude)


class AerialVL_test(Dataset):

    def __init__(self,
                 input_transform=None,
                 positive_dist_threshold=25):
        super().__init__()
        self.input_transform = input_transform
        self.positive_dist_threshold = positive_dist_threshold
        self.query_folder = os.path.join(BASE_PATH,'query_images')
        self.database_folder = os.path.join(BASE_PATH,'map_database')


        self.database_paths = sorted(glob(os.path.join(self.database_folder, "**", "*.png"), recursive=True))
        self.queries_paths = sorted(glob(os.path.join(self.query_folder, "**", "*.png"), recursive=True))
        # The format must be path/to/file/@utm_easting@utm_northing@...@.jpg
        self.database_utms = np.array(
            [get_database_center_coord(float(path.split("@")[2]), float(path.split("@")[3]), float(path.split("@")[4]), float(path.split("@")[5])) for path in self.database_paths]).astype(float)
        self.queries_utms = np.array([latlon_to_utm(float(path.split("@")[1]), float(path.split("@")[2])) for path in self.queries_paths]).astype(
            float)

        knn = NearestNeighbors(n_jobs=-1)
        knn.fit(self.database_utms)
        self.ground_truth = knn.radius_neighbors(self.queries_utms,
                                                             radius=self.positive_dist_threshold,
                                                             return_distance=False)

        self.images_paths = list(self.database_paths) + list(self.queries_paths)
        self.images_utms = list(self.database_utms) + list(self.queries_utms)

        self.database_num = len(self.database_paths)
        self.queries_num = len(self.queries_paths)

    @staticmethod
    def load_image_safe(path):
        try:
            return Image.open(path).convert('RGB')
        except (IOError, SyntaxError) as e:
            print(f"Warning: Could not load image {path} - {e}")
            return None  # 返回 None 或者一个占位图像

    def __getitem__(self, index):
        img = self.load_image_safe(self.images_paths[index])

        if self.input_transform is not None:
            if isinstance(self.input_transform, A.BasicTransform) or isinstance(self.input_transform, A.Compose):
                if isinstance(img, Image.Image):
                    img = np.array(img)
                img = self.input_transform(image=img)['image']  # 适用于 Albumentations
            elif isinstance(self.input_transform, T.Compose):
                if isinstance(img, np.ndarray):
                    img = Image.fromarray(img)  # 将 NumPy 数组转换为 PIL 图像
                img = self.input_transform(img)  # 适用于 PyTorch

        return img

    def __len__(self):
        return self.queries_num

def main():
    datasets = AerialVL_test(input_transform=None,
                            positive_dist_threshold=50)
    print(len(datasets))


if __name__ == "__main__":
    main()
