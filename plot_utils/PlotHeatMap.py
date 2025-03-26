import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from plModules.U1652_baseline import U1652_model
from utils import Get_Recalls_AP_Supervised,load_config
from torchvision.transforms import ToPILImage, Resize, Compose
import torchvision.transforms as T
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def input_transform(image_size):
    return T.Compose([
                T.Resize(image_size, interpolation=T.InterpolationMode.BILINEAR),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])

def get_feature_map(model, input_tensor, layer_name, mode=None):
    assert mode in ['in', 'out'], "please input the correct option"

    def forward_hook(module, input, output):
        if mode == 'out':
            feature_maps.append(output)
        else:
            feature_maps.append(input)

    feature_maps = []
    handle = None

    # 访问模块列表中的特定层
    sub_module = model
    for name in layer_name.split('.'):
        if name.isdigit():
            sub_module = sub_module[int(name)]
        else:
            sub_module = getattr(sub_module, name)

    handle = sub_module.register_forward_hook(forward_hook)

    with torch.no_grad():
        model(input_tensor)

    handle.remove()
    return feature_maps[0]


def plot_heat_map(model=None, which_layer='backbone.model.norm', mode='in', img_path=None, img_folder=None, dim=768,
                  save=None):
    def process_and_plot(input_img, save_path=None):
        h = w = IM_SIZE[0] // 14
        input_tensor = input_transform(image_size=IM_SIZE)(input_img).unsqueeze(0).to(device='cuda')
        layer_name = which_layer
        if mode == 'in':
            feature_map = get_feature_map(model, input_tensor, layer_name, mode=mode)[0]
        else:
            feature_map = get_feature_map(model, input_tensor, layer_name, mode=mode)[1]
        # feature_map += get_feature_map(model, input_tensor, 'components.cross_attn', mode=mode)[1]
        feature_map = feature_map.reshape(feature_map.shape[0],feature_map.shape[1],h,w)
        # feature_map = feature_map.reshape(feature_map.shape[0],feature_map.shape[2],h,w)#

        ###################################Adapt fot FSRA###############################################
        # _ = model(input_tensor)
        # feature_map = model.components.feature
        # feature_map = feature_map.reshape(feature_map.shape[0], feature_map.shape[2], h, w)
        # feature_map = feature_map[:,:,:,:]

        print(f"Feature map from layer {layer_name}: {feature_map.shape}")
        if feature_map.shape[-1] == h:
            feature_map = feature_map.permute(0,2,3,1).squeeze(0)
        #[B,H,W,C]
        feature_map = feature_map.sum(2).cpu().numpy()
        # 归一化特征图以便可视化
        feature_map2 = (feature_map - feature_map.min()) / (feature_map.max() - feature_map.min())

        # 将输入图像从 PIL 图像转换为 numpy 数组
        input_image_np = np.array(input_img)

        # 调整注意力图大小以匹配输入图像尺寸
        resize_transform = Compose([Resize(input_image_np.shape[:2])])
        feature_map_resized = resize_transform(ToPILImage()(torch.tensor(feature_map2).unsqueeze(0))).convert("L")
        feature_map_resized_np = np.array(feature_map_resized)
        feature_map_resized_np = 255 - feature_map_resized_np

        # 将热力图应用于特征图
        heatmap = cv2.applyColorMap(feature_map_resized_np, cv2.COLORMAP_JET)

        # 将热力图叠加到输入图像上
        overlay = cv2.addWeighted(input_image_np, 0.4, heatmap, 0.6, 0)

        # 叠加图像
        fig, ax = plt.subplots(1, 2, figsize=(8,4))
        ax[0].imshow(input_image_np)
        ax[0].axis('off')
        ax[1].imshow(overlay)
        ax[1].axis('off')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, bbox_inches='tight', pad_inches=0)

        plt.show()

    if img_folder is not None:
        save = os.path.join(save, which_layer)
        if save and not os.path.exists(save):
            os.makedirs(save)
        for filename in os.listdir(img_folder):
            if filename.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'gif')):
                img_path = os.path.join(img_folder, filename)
                input_img = Image.open(img_path)
                save_path = os.path.join(save, filename) if save else None
                process_and_plot(input_img, save_path)
    elif img_path is not None:
        input_img = Image.open(img_path)
        save = os.path.join(save, which_layer)
        save_path = os.path.join(save, os.path.basename(img_path)) if save else None
        if save and not os.path.exists(save):
            os.makedirs(save)
        process_and_plot(input_img, save_path)
    else:
        raise ValueError("Either img_path or img_folder must be provided.")

def load_network(configs,checkpoint):
    save_filename = checkpoint
    model = U1652_model(**configs)
    # print('Load the model from %s'%save_filename)
    try:
        model.load_state_dict(torch.load(save_filename)['state_dict'])
    except:
        model.load_state_dict(torch.load(save_filename))
    model.eval()
    return model.to(device)


if __name__ == "__main__":
    '''
    want to plot query heatmap, it is neccessary to tune the function inside the plot_heat_map
    '''
    IM_SIZE = (224,224)
    configs = load_config('../model_configs/dino_b_QDFL.yaml')['model_configs']
    pth_path = '/home/whu/Documents/codespace/learn_lightning/Drone_Sat_Geo_Localization/Current_SOTA/DINO_QDFL_U1652.pth'
    model = load_network(configs,pth_path)
    plot_heat_map(model=model, which_layer='components.AQEU.cross_attn', mode='out', img_folder='/home/whu/Pictures/test/',
                  save='/home/whu/Pictures/result/')
    '''
    QDFES:
        components.hw_fusion
        components.AQEU.fc
        components.AQEU.coarse.self_attn0
        components.AQEU.coarse.cross_attn
        components.AQEU.coarse.norm_x
        components.AQEU.cross_attn
        components.boq1.cross_attn
        components.boq1.self_attn
    FSRA:
        backbone.norm, out
        components.feature
    LPN:
        backbone.model.layer4, out
    SDPL:
        backbone.model.layer4, out
    CCR:
        components.ADIB_layer, in
    '''

    ##For FSRA, we use a manual method to plot the feature map
