import torch
import numpy as np
import pytorch_lightning as pl
from prettytable import PrettyTable
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms as T
from datasets.train.SUES_200_dataset import SUES200Dataset
from datasets.dataset_transforms.U1652_transform import get_U1652_transforms

IMAGENET_MEAN_STD = {'mean': [0.485, 0.456, 0.406],
                     'std': [0.229, 0.224, 0.225]}


class Sampler_SUES200(object):
    r"""A sampler specifically designed for the SUES200 dataset.
    This sampler provides a way to iterate over indices of dataset elements,
    ensuring that each index is repeated according to the sample_num setting.
    """

    def __init__(self, data_source, batch_size=8, sample_num=4):
        """
        Args:
            data_source (Dataset): The dataset to sample from.
            batch_size (int): The number of samples in each batch.
            sample_num (int): The number of times each index should be repeated.
        """
        self.data_len = len(data_source)
        self.batch_size = batch_size
        self.sample_num = sample_num

    def __iter__(self):
        # Create an array of indices
        indices = np.arange(0, self.data_len)
        # Shuffle the indices
        np.random.shuffle(indices)
        # Repeat each index according to sample_num
        repeated_indices = np.repeat(indices, self.sample_num, axis=0)
        return iter(repeated_indices)

    def __len__(self):
        # The length is the total number of original samples in the dataset
        return self.data_len


class SUES_200_DataModule(pl.LightningDataModule):
    def __init__(self,
                 batch_size,
                 height=200,
                 image_size=(480, 640),
                 sample_num=4,
                 num_workers=4,
                 DAC_sampling=False,
                 show_data_stats=True,
                 mean_std=IMAGENET_MEAN_STD
                 ):
        super().__init__()
        self.sample_num = sample_num
        self.DAC_sampling = DAC_sampling
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.image_size = image_size
        self.show_data_stats = show_data_stats
        self.batch_collate_fn = self.train_collate_fn
        self.height = height
        self.mean = mean_std['mean']
        self.std = mean_std['std']
        self.drone_transform, self.sat_transform, _ = get_U1652_transforms(img_size=self.image_size, padding=0)
        self.save_hyperparameters()

        self.train_loader_config = {
            'batch_size': self.batch_size,
            'num_workers': self.num_workers,
            'drop_last': True,
            'pin_memory': True,
            'collate_fn': self.batch_collate_fn}

    def train_collate_fn(self, batch):
        """
        collate_fn 函数的输入是一个 list，list 的长度是 batch size，list 中的每个元素都是 __getitem__ 得到的结果。
        对 SUES200Dataset 进行适配处理。
        """
        if self.DAC_sampling:
            img_s, img_d, ids, label = zip(*batch)
            label = torch.tensor(label, dtype=torch.int64)
            return [torch.stack(img_s, dim=0), label], (None, None), [torch.stack(img_d, dim=0), label]

        # 适应 SUES200Dataset，只处理 satellite 和 drone 数据
        img_s, img_d, ids = zip(*batch)  # 这里不再包含 img_st
        ids = torch.tensor(ids, dtype=torch.int64)

        # 返回 satellite 和 drone 数据对应的批次张量
        return [torch.stack(img_s, dim=0), ids], [torch.stack(img_d, dim=0), ids]

    def setup(self, stage=None):
        if stage=='fit':
            self.train_reload()
        if self.show_data_stats:
            self.print_stats()

    def train_reload(self):
        # Initialize datasets
        self.train_dataset = SUES200Dataset(sat_transform=self.sat_transform,
                                          drone_transform=self.drone_transform,
                                            height=self.height)

    def class_num(self):
        self.train_reload()
        return len(self.train_dataset)

    def __len__(self):
        if self.train_dataset:
            return self.train_dataset.total_img_num // self.batch_size
        else:
            raise KeyError('self.train_dataset is None')

    def train_dataloader(self):
        self.train_reload()
        self.train_loader_config['sampler'] = Sampler_SUES200(self.train_dataset, self.batch_size,
                                                                 sample_num=self.sample_num)
        return DataLoader(dataset=self.train_dataset, **self.train_loader_config)

    def get_train_dataset_length(self):
        if self.train_dataset is not None:
            return len(self.train_dataset)
        else:
            raise RuntimeError("Train dataset has not been initialized. Call setup first.")

    def print_stats(self):
        print()  # print a new line
        table = PrettyTable()
        table.field_names = ['Data', 'Value']
        table.align['Data'] = "l"
        table.align['Value'] = "l"
        table.header = False
        table.add_row(["# of source", "2"])
        table.add_row(["# of classes", f'{self.train_dataset.__len__()}'])
        table.add_row(["# of images", f'{self.train_dataset.total_img_num}'])
        print(table.get_string(title="Training Dataset"))
        print()

        table = PrettyTable()
        table.field_names = ['Data', 'Value']
        table.align['Data'] = "l"
        table.align['Value'] = "l"
        table.header = False
        table.add_row(
            ["Batch size", f"{self.batch_size}"])
        table.add_row(
            ["# of iterations", f"{self.train_dataset.__len__() // self.batch_size}"])
        table.add_row(["Image size", f"{self.image_size}"])
        print(table.get_string(title="Training config"))

if __name__ == '__main__':
    transform_train_list = [
        # T.RandomResizedCrop(size=(opt.h, opt.w), scale=(0.75,1.0), ratio=(0.75,1.3333), interpolation=3), #Image.BICUBIC)
        T.Resize((256, 256), interpolation=3),
        T.Pad(10, padding_mode='edge'),
        T.RandomCrop((256, 256)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]

    transform_train_list ={"satellite": T.Compose(transform_train_list),
                            "train":T.Compose(transform_train_list)}
    # datasets = Dataloader_University(root="/media/whu/Largedisk/datasets/U1652/University-Release/train",=transform_train_list,names=['satellite', 'drone'])
    datasets = SUES200Dataset(sat_transform=transform_train_list['satellite'],
                                    drone_transform=transform_train_list['train'])

    samper = Sampler_SUES200(datasets, 8)
    dataloader = DataLoader(datasets, batch_size=8, num_workers=0, sampler=samper, drop_last=True, shuffle=False)
    print(1)