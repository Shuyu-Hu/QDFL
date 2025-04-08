from pathlib import Path
import torch
import numpy as np
import torchvision.transforms as T
import albumentations as A
from PIL import Image
from torch.utils.data import Dataset
from os.path import join
from sklearn.neighbors import NearestNeighbors
import h5py
default_transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

BASE_PATH = '/media/whu/Largedisk/datasets/thermal_h5_datasets/'

if not Path(BASE_PATH).exists():
    raise FileNotFoundError(
        'BASE_PATH is hardcoded, please adjust to point to thermal_h5_datasets')

base_transform = T.Compose(
    [
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225]),
    ]
)

class Boson_test(Dataset):
    """Dataset with images from database and queries, used for inference (testing and building cache)."""

    def __init__(
            self, query_transform=base_transform, database_transform=base_transform,
            dist_threshold=35, G_contrast=True, G_gray=True, mode="train", loading_queries=True
    ):
        super().__init__()
        self.positive_dist_threshold = dist_threshold
        self.G_gray = G_gray
        self.G_contrast = G_contrast
        self.dataset_name = BASE_PATH
        self.split = mode
        self.query_transform = query_transform
        self.database_transform = database_transform
        # self.dataset_folder = join(datasets_folder, dataset_name, "images", split)
        # if not os.path.exists(self.dataset_folder): raise FileNotFoundError(f"Folder {self.dataset_folder} does not exist")

        # Redirect datafolder path to h5
        self.database_folder_h5_path = join(
            BASE_PATH, self.split + "_database.h5"
        )
        if loading_queries:
            self.queries_folder_h5_path = join(
                BASE_PATH, self.split + "_queries.h5"
            )
        else:
            # Do not load queries when generating thermal with pix2pix
            self.queries_folder_h5_path = join(
                BASE_PATH, self.split + "_database.h5")
        database_folder_h5_df = h5py.File(self.database_folder_h5_path, "r")
        queries_folder_h5_df = h5py.File(self.queries_folder_h5_path, "r")

        # Map name to index
        self.database_name_dict = {}
        self.queries_name_dict = {}

        # Duplicated elements are removed below
        for index, database_image_name in enumerate(database_folder_h5_df["image_name"]):
            self.database_name_dict[database_image_name.decode(
                "UTF-8")] = index
        for index, queries_image_name in enumerate(queries_folder_h5_df["image_name"]):
            self.queries_name_dict[queries_image_name.decode("UTF-8")] = index

        self.database_paths = sorted(self.database_name_dict)
        self.queries_paths = sorted(self.queries_name_dict)
        # The format must be path/to/file/@utm_easting@utm_northing@...@.jpg
        self.database_utms = np.array(
            [(path.split("@")[1], path.split("@")[2])
             for path in self.database_paths]
        ).astype(np.float64)
        self.queries_utms = np.array(
            [(path.split("@")[1], path.split("@")[2])
             for path in self.queries_paths]
        ).astype(np.float64)

        # Find soft_positives_per_query, which are within val_positive_dist_threshold (deafult 25 meters)
        knn = NearestNeighbors(n_jobs=-1)
        knn.fit(self.database_utms)
        self.soft_positives_per_query = knn.radius_neighbors(
            self.queries_utms,
            radius=self.positive_dist_threshold,
            return_distance=False,
        )

        # Add database, queries prefix
        for i in range(len(self.database_paths)):
            self.database_paths[i] = "database_" + self.database_paths[i]
        for i in range(len(self.queries_paths)):
            self.queries_paths[i] = "queries_" + self.queries_paths[i]

        self.images_paths = list(self.database_paths) + \
                            list(self.queries_paths)

        self.database_num = len(self.database_paths)
        self.queries_num = len(self.queries_paths)

        # Close h5 and initialize for h5 reading in __getitem__
        self.database_folder_h5_df = None
        self.queries_folder_h5_df = None
        database_folder_h5_df.close()
        queries_folder_h5_df.close()

        self.query_transform = T.Compose(
            [
                T.Grayscale(num_output_channels=3)
                if self.G_gray
                else T.Lambda(lambda x: x),
                query_transform
            ]
        )

    def __getitem__(self, index):
        # Init
        if self.database_folder_h5_df is None:
            self.database_folder_h5_df = h5py.File(
                self.database_folder_h5_path, "r")
            self.queries_folder_h5_df = h5py.File(
                self.queries_folder_h5_path, "r")
        if self.is_index_in_queries(index):
            if self.G_contrast:
                img = self.query_transform(
                    T.functional.adjust_contrast(self._find_img_in_h5(index), contrast_factor=3))
            else:
                img = self.query_transform(
                    self._find_img_in_h5(index))
        else:
            img = self._find_img_in_h5(index)
            img = self.database_transform(img)

        return img, index

    def _find_img_in_h5(self, index, database_queries_split=None):
        # Find inside index for h5
        if database_queries_split is None:
            image_name = "_".join(self.images_paths[index].split("_")[1:])
            database_queries_split = self.images_paths[index].split("_")[0]
        else:
            if database_queries_split == "database":
                image_name = "_".join(self.database_paths[index].split("_")[1:])
            elif database_queries_split == "queries":
                image_name = "_".join(self.queries_paths[index].split("_")[1:])
            else:
                raise KeyError("Dont find correct database_queries_split!")

        if database_queries_split == "database":
            img = Image.fromarray(
                self.database_folder_h5_df["image_data"][
                    self.database_name_dict[image_name]
                ]
            )
        elif database_queries_split == "queries":
            img = Image.fromarray(
                self.queries_folder_h5_df["image_data"][
                    self.queries_name_dict[image_name]
                ]
            )
        else:
            raise KeyError("Dont find correct database_queries_split!")

        return img

    def __len__(self):
        return len(self.images_paths)

    def __repr__(self):
        return f"< {self.__class__.__name__}, {self.dataset_name} - #database: {self.database_num}; #queries: {self.queries_num} >"

    def get_positives(self):
        return self.soft_positives_per_query

    def get_hard_negatives(self):
        return self.hard_negatives_per_query

    def __del__(self):
        if (hasattr(self, "database_folder_h5_df")
                and self.database_folder_h5_df is not None):
            self.database_folder_h5_df.close()
            self.queries_folder_h5_df.close()

    def is_index_in_queries(self, index):
        if index >= self.database_num:
            return True
        else:
            return False

