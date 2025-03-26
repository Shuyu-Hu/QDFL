import os
import pprint
import sys
import torch
import cv2
import numpy as np
import torchvision
from PIL import Image
from torch.utils.data import Subset
from tqdm import tqdm
from datasets.test import U1652_test
from datasets.test import DenseUAV_test
from datasets.test import SUES200_test
from sklearn.decomposition import PCA
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms
from plModules.U1652_baseline import U1652_model
from utils import *

class Query_transforms(object):

    def __init__(self, pad=20,size=256):
        self.pad=pad
        self.size = size

    def __call__(self, img):
        if self.pad<=0:
            return img
        # img_=np.array(img).copy()
        # img_part = img_[:,0:self.pad,:]
        # img_pad = np.zeros_like(img_part,dtype=np.uint8)
        # image = np.concatenate((img_pad, img_),axis=1)
        # image = image[:,0:self.size,:]
        # image = Image.fromarray(image.astype('uint8')).convert('RGB')

        img_=np.array(img).copy()
        img_part = img_[:,0:self.pad,:]
        img_flip = cv2.flip(img_part, 1)
        image = np.concatenate((img_flip, img_),axis=1)
        image = image[:,0:self.size,:]
        image = Image.fromarray(image.astype('uint8')).convert('RGB')
        return image


def get_test_dataset_supervised(img_size=(256, 256), Qpad=0, batch_size=8, num_workers=0, which_dataset='U1652', height=200,
                     mode='sat->drone', which_weather='overexposure'):

    if mode == 'sat->drone':
        query_name = 'query_satellite'
        gallery_name = 'gallery_drone'
    elif mode == 'drone->sat':
        query_name = 'query_drone'
        gallery_name = 'gallery_satellite'
    else:
        raise Exception('error mode option')

    # data_transforms = A.Compose([A.Resize(img_size[0], img_size[1], interpolation=cv2.INTER_LINEAR_EXACT, p=1.0),
    #
    #                             # Multi-weather U1652 settings.
    #                             # iaa_weather_list[weather_id],
    #
    #                             A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    #                             ToTensorV2(),
    #                             ])
    data_query_transforms = transforms.Compose([
        transforms.Resize(img_size, interpolation=transforms.InterpolationMode.BICUBIC),
        Query_transforms(pad=Qpad, size=img_size[0]),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    data_transforms = transforms.Compose([
        transforms.Resize(img_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    if 'u1652' == which_dataset.lower():
        image_datasets = {
            'gallery_satellite': U1652_test('gallery_satellite', transform=data_transforms),
            'gallery_drone': U1652_test('gallery_drone', transform=data_transforms),
            'query_satellite': U1652_test('query_satellite', transform=data_query_transforms),
            'query_drone': U1652_test('query_drone', transform=data_query_transforms)
        }

    elif 'mw_u1652' == which_dataset.lower():
        image_datasets = {
            'gallery_satellite': MW_U1652_test('gallery_satellite',
                                               transform=get_test_transforms(img_size, which_weather='None')),
            'gallery_drone': MW_U1652_test('gallery_drone',
                                           transform=get_test_transforms(img_size, which_weather=which_weather)),
            'query_satellite': MW_U1652_test('query_satellite',
                                             transform=get_test_transforms(img_size, which_weather='None')),
            'query_drone': MW_U1652_test('query_drone',
                                         transform=get_test_transforms(img_size, which_weather=which_weather))
        }

    elif 'denseuav' in which_dataset.lower():
        if mode == 'sat->drone':
            print('Since DenseUAV only has drone->sat option, the evaluation will return D2S result')
            query_name = 'query_drone'
            gallery_name = 'gallery_satellite'
        image_datasets = {
            'gallery_satellite': DenseUAV_test('gallery_satellite', transform=data_transforms),
            'query_drone': DenseUAV_test('query_drone', transform=data_transforms)
        }

    elif 'sues' in which_dataset.lower():
        image_datasets = {
            'gallery_satellite': SUES200_test('gallery_satellite', height=f'{height}', transform=data_transforms),
            'gallery_drone': SUES200_test('gallery_drone', height=f'{height}', transform=data_transforms),
            'query_satellite': SUES200_test('query_satellite', height=f'{height}', transform=data_transforms),
            'query_drone': SUES200_test('query_drone', height=f'{height}', transform=data_transforms)
        }

    else:
        raise Exception('error dataset name')

    dataloaders = {name: DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
                   for name, dataset in image_datasets.items()}
    return dataloaders[query_name], dataloaders[gallery_name], image_datasets[query_name].images, image_datasets[
        gallery_name].images

def get_test_dataset_metric_leanring(img_size=(256, 256), batch_size=8, num_workers=0, which_dataset='U1652'):

    data_transforms = transforms.Compose([
        transforms.Resize(img_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    if 'boson' in which_dataset.lower():
        test_datasets = Boson_test(database_transform=data_transforms,query_transform=data_transforms,mode='test')
        ground_truth = test_datasets.soft_positives_per_query
        test_subsets = {
            "query_subset" : Subset(test_datasets, list(range(test_datasets.database_num, test_datasets.database_num + test_datasets.queries_num))),
            "database_subset" : Subset(test_datasets, list(range(test_datasets.database_num)))
            }
        dataloaders = {name: DataLoader(sset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
                       for name, sset in test_subsets.items()}
        return dataloaders["query_subset"], dataloaders["database_subset"], ground_truth
    elif 'gta' in which_dataset.lower():
        test_subsets = {
            "query_subset" : GTA_UAV_test(pairs_meta_file='cross-area-drone2sate-test.json',which_view='drone',transforms=data_transforms),
            "database_subset": GTA_UAV_test(pairs_meta_file='cross-area-drone2sate-test.json',which_view='sate',transforms=data_transforms)
            }
        dataloaders = {name: DataLoader(sset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
                       for name, sset in test_subsets.items()}
        extra_configs = {
            'query_list':test_subsets["query_subset"].images_name,
            'query_loc_xy_list':test_subsets["query_subset"].images_loc_xy,
            'gallery_list':test_subsets["database_subset"].images_name,
            'gallery_loc_xy_list':test_subsets["database_subset"].images_loc_xy,
            'pairs_dict':test_subsets["query_subset"].pairs_drone2sate_dict,
        }
        #generating ground truth array
        gallery_idx = {}
        for idx, gallery_img in enumerate(extra_configs['gallery_list']):
            gallery_idx[gallery_img] = idx

        gt = []
        for query_i in extra_configs['query_list']:
            pairs_list_i = extra_configs['pairs_dict'][query_i]
            lst = []
            for pair in pairs_list_i:
                lst.append(gallery_idx[pair])
            gt.append(np.array(lst))

        gt = np.array(gt, dtype=object)
        extra_configs['ground_truth'] = gt
        return dataloaders["query_subset"], dataloaders["database_subset"], extra_configs
    elif 'vpair' in which_dataset.lower():
        test_datasets = VPAir_test(input_transform=data_transforms)
        ground_truth = test_datasets.ground_truth
        test_subsets = {
            "query_subset": Subset(test_datasets, list(
                range(test_datasets.num_references, test_datasets.num_references + test_datasets.num_queries))),
            "database_subset": Subset(test_datasets, list(range(test_datasets.num_references)))
        }
        dataloaders = {name: DataLoader(sset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
                       for name, sset in test_subsets.items()}
        return dataloaders["query_subset"], dataloaders["database_subset"], ground_truth
    elif 'aerialvl' in which_dataset.lower():
        test_datasets = AerialVL_test(input_transform=data_transforms,positive_dist_threshold=50)
        ground_truth = test_datasets.ground_truth
        test_subsets = {
            "query_subset": Subset(test_datasets, list(
                range(test_datasets.database_num, test_datasets.database_num + test_datasets.queries_num))),
            "database_subset": Subset(test_datasets, list(range(test_datasets.database_num)))
        }
        dataloaders = {name: DataLoader(sset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
                       for name, sset in test_subsets.items()}
        return dataloaders["query_subset"], dataloaders["database_subset"], ground_truth
    else:
        raise Exception('error dataset name')