import torch
from pathlib import Path
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import os


default_transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

BASE_PATH = '/media/whu/Largedisk/datasets/DenseUAV/test/'

if not Path(BASE_PATH).exists():
    raise FileNotFoundError(
        'BASE_PATH is hardcoded, please adjust to point to DenseUAV/test')

def get_parent_directory(pth_path):
    # 获取父目录路径
    parent_directory = os.path.dirname(pth_path)
    parent_directory = "/".join(parent_directory.split('/')[:-1])  # version xx path
    return parent_directory

class DenseUAV_test(Dataset):
    def __init__(self, which_path, transform=None):
        self.which_path = which_path
        self.transform = transform
        self.images = []
        self.labels = []
        self.path = os.path.join(BASE_PATH, self.which_path)
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

        self.configDict = {}
        with open(os.path.join(get_parent_directory(BASE_PATH),'Dense_GPS_ALL.txt'), "r") as F:
            context = F.readlines()
            for line in context:
                splitLineList = line.split(" ")
                self.configDict[splitLineList[0].split("/")[-2]] = [float(splitLineList[1].split("E")[-1]),
                                                               float(splitLineList[2].split("N")[-1])]

    def __getitem__(self, index):
        # 加载图像并应用变换
        img = Image.open(self.images[index])
        if self.transform is not None:
            img = self.transform(img)
        label = self.labels[index]
        return img, label

    def __len__(self):
        return len(self.images)
def main():
    datasets = DenseUAV_test(which_path='query_drone',transform=default_transform)
    dl = torch.utils.data.DataLoader(datasets,batch_size=1,num_workers=1)
    for img,label in dl:
        print(img.shape,label)


if __name__=="__main__":
    main()