import torch
import numpy as np
import pytorch_lightning as pl
from prettytable import PrettyTable
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms as T
from datasets.train.U1652_dataset import U1652Dataset
from datasets.train.U1652_dataset_DAC import U1652Dataset_DAC
from datasets.dataset_transforms.U1652_transform import get_U1652_transforms

IMAGENET_MEAN_STD = {'mean': [0.485, 0.456, 0.406],
                     'std': [0.229, 0.224, 0.225]}


class Sampler_University(object):
    r"""Base class for all Samplers.
    Every Sampler subclass has to provide an :meth:`__iter__` method, providing a
    way to iterate over indices of dataset elements, and a :meth:`__len__` method
    that returns the length of the returned iterators.
    .. note:: The :meth:`__len__` method isn't strictly required by
              :class:`~torch.utils.data.DataLoader`, but is expected in any
              calculation involving the length of a :class:`~torch.utils.data.DataLoader`.
    """

    def __init__(self, data_source, batchsize=8, sample_num=4):
        self.data_len = len(data_source)
        self.batchsize = batchsize
        self.sample_num = sample_num

    def __iter__(self):
        list = np.arange(0, self.data_len)
        np.random.shuffle(list)
        nums = np.repeat(list, self.sample_num, axis=0)
        return iter(nums)

    def __len__(self):
        return self.data_len


class U1652DataModule(pl.LightningModule):
    def __init__(self,
                 batch_size,
                 image_size=(480, 640),
                 sample_num=4,
                 num_workers=4,
                 DAC_sampling=False,
                 drop_last=True,
                 show_data_stats=True,
                 mean_std=IMAGENET_MEAN_STD,
                 sources=['satellite', 'drone']
                 ):
        super().__init__()
        self.batch_size = batch_size
        self.image_size = image_size
        self.sample_num = sample_num
        self.num_workers = num_workers
        self.DAC_sampling = DAC_sampling
        self.drop_last = drop_last
        self.show_data_stats = show_data_stats
        self.mean = mean_std['mean']
        self.std = mean_std['std']
        self.drone_transform, self.sat_transform, _ = get_U1652_transforms(img_size=self.image_size, padding=0)
        self.batch_collate_fn = self.train_collate_fn
        self.sources = sources
        self.save_hyperparameters()

        self.train_loader_config = {
            'batch_size': self.batch_size,
            'num_workers': self.num_workers,
            'drop_last': self.drop_last,
            'pin_memory': True,
            'collate_fn': self.batch_collate_fn}

    def train_collate_fn(self, batch):
        """
        # collate_fn这个函数的输入就是一个list，list的长度是一个batch size，list中的每个元素都是__getitem__得到的结果
        """
        if self.DAC_sampling:
            img_s, img_d, ids, label = zip(*batch)
            label = torch.tensor(label, dtype=torch.int64)
            return [torch.stack(img_s, dim=0), label], (None, None), [torch.stack(img_d, dim=0), label]
        img_s, img_st, img_d, ids = zip(*batch)
        ids = torch.tensor(ids, dtype=torch.int64)
        return [torch.stack(img_s, dim=0), ids], [torch.stack(img_st, dim=0), ids], [torch.stack(img_d, dim=0), ids]


    def class_num(self):
        self.reload()
        return len(self.train_dataset)


    def __len__(self):
        if self.train_dataset:
            return self.train_dataset.total_img_num // self.batch_size
        else:
            raise KeyError('self.train_dataset is None')


    def reload(self):
        if self.DAC_sampling:
            self.train_dataset = U1652Dataset_DAC(sat_transform=self.sat_transform,
                                                  drone_transform=self.drone_transform,
                                                  sources=self.sources)
        else:
            self.train_dataset = U1652Dataset(sat_transform=self.sat_transform,
                                              drone_transform=self.drone_transform,
                                              sources=self.sources
                                              )

    def train_dataloader(self):
        self.reload()
        if self.DAC_sampling:
            self.train_dataset.shuffle()
            return DataLoader(dataset=self.train_dataset, **self.train_loader_config)
        else:
            self.train_loader_config['sampler'] = Sampler_University(self.train_dataset, self.batch_size,
                                                                     sample_num=self.sample_num)
            return DataLoader(dataset=self.train_dataset, **self.train_loader_config)


    def print_stats(self):
        print()  # print a new line
        table = PrettyTable()
        table.field_names = ['Data', 'Value']
        table.align['Data'] = "l"
        table.align['Value'] = "l"
        table.header = False
        table.add_row(["# of source", f"{self.sources}"])
        table.add_row(["# of places", f'{self.train_dataset.__len__()}'])
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


    def setup(self, stage: str) -> None:
        if stage == 'fit':
            self.reload()
        if self.show_data_stats:
            self.print_stats()


    def get_train_dataset_length(self):
        if self.train_dataset is not None:
            return len(self.train_dataset)
        else:
            raise RuntimeError("Train dataset has not been initialized. Call setup first.")



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

    transform_train_list = {"satellite": T.Compose(transform_train_list),
                            "train": T.Compose(transform_train_list)}
    # datasets = Dataloader_University(root="/media/whu/Largedisk/datasets/U1652/University-Release/train",=transform_train_list,names=['satellite', 'drone'])
    datasets = U1652Dataset(sat_transform=transform_train_list['satellite'],
                            drone_transform=transform_train_list['train'],
                            sources=['satellite', 'street', 'drone'])
    samper = Sampler_University(datasets, 8,sample_num=1)
    dataloader = DataLoader(datasets, batch_size=24, num_workers=0, sampler=samper, drop_last=True, shuffle=False)
    for batch in enumerate(dataloader):
        # [torch.stack(img_s, dim=0), ids], [torch.stack(img_st, dim=0), ids], [torch.stack(img_d, dim=0), ids]
        print(f"{batch[1][3]}")
