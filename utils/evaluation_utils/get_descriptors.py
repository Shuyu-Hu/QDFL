import torch
from tqdm import tqdm
from sklearn.decomposition import PCA
from utils import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_descriptors_metric_learning(model, dataloader, device, pca_dim=None):
    descriptors = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Calculating descriptors...'):
            if isinstance(batch,list):
                imgs, _ = batch
            else:
                imgs = batch
            output = model(imgs.to(device)).cpu()
            descriptors.append(output)

    # Concatenate all descriptors
    descriptors = torch.cat(descriptors)

    # If PCA is specified, fit and transform the descriptors
    if pca_dim is not None:
        pca = PCA(n_components=pca_dim, svd_solver='randomized')

        # Convert to numpy for PCA
        descriptors_np = descriptors.numpy()

        # Fit PCA
        print("Fitting PCA...")
        pca.fit(descriptors_np)

        # Transform descriptors using PCA
        print("Transforming descriptors with PCA...")
        descriptors_np = pca.transform(descriptors_np)

        # Convert back to torch tensor
        descriptors = torch.from_numpy(descriptors_np)

    return descriptors

def get_descriptors_supervised_wo_label(model, dataloader, fliplr, device=device):
    def Fliplr(img):
        '''flip horizontal'''
        inv_idx = torch.arange(img.size(3) - 1, -1, -1).long()  # N x C x H x W
        img_flip = img.index_select(3, inv_idx)
        return img_flip

    descriptors = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Calculating descriptors...'):
            if isinstance(batch,list):
                imgs, _ = batch
            else:
                imgs = batch
            """
            这段代码是仿照FSRA中的extract_feature函数进行编写的，其主要思想是将图片进行左右翻转，然后提取原图和翻转图的特征并相加
            随后计算该特征的L2距离，然后除以该距离，最后将[B,512,4]展成[B,2048]的特征描述符
            """
            output = model(imgs.to(device)).cpu()
            output_norm = torch.norm(output, p=2, dim=1, keepdim=True)
            output = output.div(output_norm.expand_as(output))
            if fliplr:
                output_mirror = model(Fliplr(imgs).to(device)).cpu()
                output = output + output_mirror
                output_norm = torch.norm(output, p=2, dim=1, keepdim=True)
                output = output.div(output_norm.expand_as(output))

            output = output.view(output.size(0), -1)

            descriptors.append(output)

    descriptors = torch.cat(descriptors)
    return descriptors


# def sign_func(x:torch.Tensor):
#     return torch.sigmoid(10 * (x - 1))

def get_descriptors_supervised_with_label(model, dataloader, fliplr, device=device):
    def Fliplr(img):
        '''flip horizontal'''
        inv_idx = torch.arange(img.size(3) - 1, -1, -1).long()  # N x C x H x W
        img_flip = img.index_select(3, inv_idx)
        return img_flip

    descriptors = []
    label_list = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Calculating descriptors...'):
            imgs, labels = batch
            """
            这段代码是仿照FSRA中的extract_feature函数进行编写的，其主要思想是将图片进行左右翻转，然后提取原图和翻转图的特征并相加
            随后计算该特征的L2范数，然后除以该范数，最后将[B,512,4]展成[B,2048]的特征描述符
            """
            output = model(imgs.to(device)).cpu()
            output_norm = torch.norm(output, p=2, dim=1, keepdim=True)
            output = output.div(output_norm.expand_as(output))
            if fliplr:
                output_mirror = model(Fliplr(imgs).to(device)).cpu()
                output = output + output_mirror
                output_norm = torch.norm(output, p=2, dim=1, keepdim=True)
                output = output.div(output_norm.expand_as(output))

            output = output.view(output.size(0), -1)

            descriptors.append(output)
            label_list.append(labels)

    descriptors = torch.cat(descriptors)
    label_list = torch.cat(label_list)
    return descriptors, label_list
    # return sign_func(descriptors), label_list