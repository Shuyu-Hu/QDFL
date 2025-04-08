import torch
import os
import yaml
import torch
import numpy as np
from PIL import Image
import imgaug.augmenters as iaa
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

def get_yaml_value(config_path):
    f = open(config_path, 'r', encoding="utf-8")
    t_value = yaml.load(f, Loader=yaml.FullLoader)
    f.close()
    # params = t_value[key_name]
    return t_value

class AddBlock(object):
    '''
    The gap parameter controls how much of the image is modified or replaced.
    If gap is large,
    a larger section of the image will be flipped or replaced with a solid color;
    if gap is small, the effect will be more subtle.
    '''
    def __init__(self, gap, type):
        self.gap = gap
        self.type = type
    def __call__(self, img):
        height = img.height
        img = img.crop((0, 0, height - self.gap, height))

        if self.type == "flip":
            crop = img.crop((0, 0, self.gap, height))
            crop = crop.transpose(Image.FLIP_LEFT_RIGHT)
        else:
            crop = Image.new("RGB", (self.gap, height), self.type)

        joint = Image.new("RGB", (height, height))
        joint.paste(crop, (0, 0, self.gap, height))
        joint.paste(img, (self.gap, 0, height, height))

        return joint

class Weather(object):
    def __init__(self, type):

        if type == "snow":
            self.seq = iaa.imgcorruptlike.Snow(severity=2)
        elif type == "rain":
            self.seq = iaa.Rain()
        elif type == "fog":
            self.seq = iaa.Fog()

    def __call__(self, img):
        width = img.width
        height = img.height

        img = np.array(img).reshape(1, width, height, 3)
        img = self.seq.augment_images(img)
        img = np.array(img).reshape(width, height, 3)
        img = Image.fromarray(img)

        return img

def get_SUES_200_transforms(img_size=(256,256),padding=0, train=True):
    (h,w) = img_size
    if train:
        transform_drone_list = [
            transforms.Resize((h,w), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomCrop((h,w)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]
    
        transforms_satellite_list = [
            transforms.Resize((h,w), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomCrop((h,w)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]
        return transform_drone_list, transforms_satellite_list

    transforms_test_list = [
        transforms.Resize((h,w), interpolation=transforms.InterpolationMode.BICUBIC),
        # iaa.Sequential([seq]),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]

    transforms_list = [
        transforms.Resize((h,w), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]
    
    return transforms_list, transforms_test_list

def get_uncertainties_transforms(img_size=(256,256),padding=0, gap=50):
    (h, w) = img_size
    if type in ['snow', 'rain', 'fog']:

        transforms_test_list = [
            transforms.Resize((h,w), interpolation=transforms.InterpolationMode.BICUBIC),
            Weather(type),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]
    elif type in ['flip', 'black']:

        transforms_test_list = [
            transforms.Resize((h,w), interpolation=transforms.InterpolationMode.BICUBIC),
            AddBlock(gap=gap, type=type),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]
    elif type == "normal" or "":
        transforms_test_list = [
            transforms.Resize((h,w), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]
    else:
        print("Type not designate, using default process")
        transforms_test_list = [transforms.Resize((h,w), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]

    transforms_list = [
        transforms.Resize((h,w), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]

    return transforms.Compose(transforms_list), transforms.Compose(transforms_test_list)