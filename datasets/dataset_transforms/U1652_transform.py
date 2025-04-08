import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision import transforms
import random
import math
import torch
import cv2

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

def get_U1652_transforms(img_size=(256,256),padding=0):
    (h,w) = img_size

    transform_train_satellite_list = A.Compose([A.ImageCompression(quality_lower=90, quality_upper=100, p=0.5),
                                      A.Resize(img_size[0], img_size[1], interpolation=cv2.INTER_LINEAR_EXACT, p=1.0),

                                      # Multi-weather U1652 settings.
                                      # A.OneOf(iaa_weather_list, p=1.0),

                                      A.ColorJitter(brightness=0.15, contrast=0.3, saturation=0.3, hue=0.3,
                                                    always_apply=False, p=0.5),
                                      A.OneOf([
                                          A.AdvancedBlur(p=1.0),
                                          A.Sharpen(p=1.0),
                                      ], p=0.3),
                                      A.OneOf([
                                          A.GridDropout(ratio=0.4, p=1.0),
                                          A.CoarseDropout(max_holes=25,
                                                          max_height=int(0.2 * img_size[0]),
                                                          max_width=int(0.2 * img_size[0]),
                                                          min_holes=10,
                                                          min_height=int(0.1 * img_size[0]),
                                                          min_width=int(0.1 * img_size[0]),
                                                          p=1.0),
                                      ], p=0.3),
                                      A.RandomRotate90(p=1.0),
                                      A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                                      ToTensorV2(),
                                      ])

    transform_train_drone_list = A.Compose([A.ImageCompression(quality_lower=90, quality_upper=100, p=0.5),
                                        A.Resize(img_size[0], img_size[1], interpolation=cv2.INTER_LINEAR_EXACT, p=1.0),

                                        # Multi-weather U1652 settings.
                                        # A.OneOf(iaa_weather_list, p=1.0),

                                        A.ColorJitter(brightness=0.15, contrast=0.7, saturation=0.3, hue=0.3,
                                                      always_apply=False, p=0.5),
                                        A.OneOf([
                                            A.AdvancedBlur(p=1.0),
                                            A.Sharpen(p=1.0),
                                        ], p=0.3),
                                        A.OneOf([
                                            A.GridDropout(ratio=0.4, p=1.0),
                                            A.CoarseDropout(max_holes=25,
                                                            max_height=int(0.2 * img_size[0]),
                                                            max_width=int(0.2 * img_size[0]),
                                                            min_holes=10,
                                                            min_height=int(0.1 * img_size[0]),
                                                            min_width=int(0.1 * img_size[0]),
                                                            p=1.0),
                                        ], p=0.3),
                                        A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                                        ToTensorV2(),
                                        ])

    transform_val_list = A.Compose([A.Resize(h, w, interpolation=cv2.INTER_LINEAR_EXACT, p=1.0),
                                A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                                ToTensorV2(),
                                ])

    return transform_train_drone_list, transform_train_satellite_list, transform_val_list



import matplotlib.pyplot as plt
import numpy as np

def show_image(image, title="Image",convert=False):
    """显示图像的简单函数"""
    # 检查图像是否为 NumPy 数组，如果不是则转换
    if not isinstance(image, np.ndarray):
        # image = np.array(image)
        image = image['image'].permute(1,2,0).numpy()
    print(f"[DEBUG] Image shape: {image.shape}")
    print(f"[DEBUG] Image dtype: {image.dtype}")
    # 转换颜色格式并显示图像
    plt.figure(figsize=(6, 6))
    if convert:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image)
    plt.title(title)
    plt.axis('off')
    plt.show()

def test_transforms(image_path,transforms):
    # 读取图像
    original_image = cv2.imread(image_path)
    show_image(original_image, title="Original Image",convert=True)

    # 应用 transforms
    transformed = transforms(image=original_image)
    transformed_image = transformed


    # 显示转换后的图像
    show_image(transformed_image, title="Transformed Image",convert=True)

if __name__ == "__main__":
    # 替换为你要测试的图像路径
    test_image_path = '/media/whu/Largedisk/datasets/U1652/University-Release/test/4K_drone/0000/1.png'
    drone_transform, sat_transform, _ = get_U1652_transforms(img_size=(256,256), padding=0)
    test_transforms(test_image_path,drone_transform)
