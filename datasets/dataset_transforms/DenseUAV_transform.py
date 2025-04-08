
import numpy as np
import PIL
from PIL.Image import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision import transforms
import random
import math
import torch
import cv2
#TODO:change
class RandomErasing(object):
    """ Randomly selects a rectangle region in an image and erases its pixels.
        'Random Erasing Data Augmentation' by Zhong et al.
        See https://arxiv.org/pdf/1708.04896.pdf
    Args:
         probability: The probability that the Random Erasing operation will be performed.
         sl: Minimum proportion of erased area against input image.
         sh: Maximum proportion of erased area against input image.
         r1: Minimum aspect ratio of erased area.
         mean: Erasing value.
    """

    def __init__(self, probability=0.5, sl=0.02, sh=0.4, r1=0.3, mean=[0.4914, 0.4822, 0.4465]):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, img):

        if random.uniform(0, 1) > self.probability:
            return img

        for attempt in range(100):
            area = img.size()[1] * img.size()[2]

            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                if img.size()[0] == 3:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                    img[1, x1:x1 + h, y1:y1 + w] = self.mean[1]
                    img[2, x1:x1 + h, y1:y1 + w] = self.mean[2]
                else:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                return img

        return img

class RotateAndCrop(object):
    def __init__(self, rate, output_size=(512, 512), rotate_range=360):
        self.rate = rate
        self.output_size = output_size
        self.rotate_range = rotate_range

    def __call__(self, img):
        img_ = np.array(img).copy()

        def getPosByAngle(img, angle):
            h, w, c = img.shape
            y_center = h // 2
            x_center = w//2
            r = h // 2
            angle_lt = angle - 45
            angle_rt = angle + 45
            angle_lb = angle + 135
            angle_rb = angle + 225
            angleList = [angle_lt, angle_rt, angle_lb, angle_rb]
            pointsList = []
            for angle in angleList:
                x1 = x_center + r * math.cos(angle * math.pi / 180)
                y1 = y_center + r * math.sin(angle * math.pi / 180)
                pointsList.append([x1, y1])
            pointsOri = np.float32(pointsList)
            pointsListAfter = np.float32(
                [[0, 0], [0, self.output_size[0]], [self.output_size[0], self.output_size[1]], [self.output_size[1], 0]])
            M = cv2.getPerspectiveTransform(pointsOri, pointsListAfter)
            res = cv2.warpPerspective(
                img, M, (self.output_size[0], self.output_size[1]))
            return res

        if np.random.random() > self.rate:
            image = img
        else:
            angle = int(np.random.random()*self.rotate_range)
            new_image = getPosByAngle(img_, angle)
            image = PIL.Image.fromarray(new_image.astype('uint8')).convert('RGB')
        return image

def get_DenseUAV_transforms(img_size=(256,256),padding=0,
                            rotate_crop:list=None,
                            random_affine:list=None,
                            color_jittering:list=None,
                            random_erasing:list=None,
                            random_erasing_prob=0.5):
    assert all(item is None or item in ['uav', 'satellite'] for item in rotate_crop), \
        "All elements in rotate_crop must be 'uav', 'satellite', or None"
    assert all(item is None or item in ['uav', 'satellite'] for item in random_affine), \
        "All elements in random_affine must be 'uav', 'satellite', or None"
    assert all(item is None or item in ['uav', 'satellite'] for item in random_erasing), \
        "All elements in random_erasing must be 'uav', 'satellite', or None"
    assert all(item is None or item in ['uav', 'satellite'] for item in color_jittering), \
        "All elements in color_jittering must be 'uav', 'satellite', or None"

    (h,w) = img_size

    transform_train_list = []
    transform_satellite_list = []
    if "uav" in rotate_crop:
        transform_train_list.append(RotateAndCrop(0.5))
    if "satellite" in rotate_crop:
        transform_satellite_list.append(RotateAndCrop(0.5))
    transform_train_list += [
        transforms.Resize((h, w), interpolation=3),
        transforms.Pad(padding, padding_mode='edge'),
        transforms.RandomHorizontalFlip(),
    ]

    transform_satellite_list += [
        transforms.Resize((h, w), interpolation=3),
        transforms.Pad(padding, padding_mode='edge'),
        transforms.RandomHorizontalFlip(),
    ]

    transform_val_list = [
        transforms.Resize(size=(h, w),
                          interpolation=3),  # Image.BICUBIC
    ]

    if "uav" in random_affine:
        transform_train_list = transform_train_list + \
                               [transforms.RandomAffine(180)]
    if "satellite" in random_affine:
        transform_satellite_list = transform_satellite_list + \
                                   [transforms.RandomAffine(180)]

    if "uav" in random_erasing:
        transform_train_list = transform_train_list + \
                               [RandomErasing(probability=random_erasing_prob)]
    if "satellite" in random_erasing:
        transform_satellite_list = transform_satellite_list + \
                                   [RandomErasing(probability=random_erasing_prob)]

    if "uav" in color_jittering:
        transform_train_list = transform_train_list + \
                               [transforms.ColorJitter(brightness=0.5, contrast=0.1, saturation=0.1,
                                                       hue=0)]
    if "satellite" in color_jittering:
        transform_satellite_list = transform_satellite_list + \
                                   [transforms.ColorJitter(brightness=0.5, contrast=0.1, saturation=0.1,
                                                           hue=0)]


    last_aug = [
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]

    transform_train_list += last_aug
    transform_satellite_list += last_aug
    transform_val_list += last_aug

    print(transform_train_list)
    print(transform_satellite_list)

    return transforms.Compose(transform_train_list), transforms.Compose(transform_satellite_list), transforms.Compose(transform_val_list)