from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from scipy.spatial.distance import pdist, squareform
import seaborn as sns

from utils.metrics import gen_gt
import torch
from Supervised_evaluate import load_network,get_test_dataset,get_descriptors
from utils.commons import load_config

Device = torch.device('cuda')

def main():
    img_size = (224, 224)
    batch_size = 192
    which_dataset = 'sues200'
    height = 200
    mode = 'sat->drone'
    configs = load_config('/home/whu/Documents/codespace/learn_lightning/Drone_Sat_Geo_Localization/model_configs/dino_b_DF.yaml')['model_configs']
    pth_path = '/home/whu/Documents/codespace/learn_lightning/Drone_Sat_Geo_Localization/LOGS/dinov2_vitb14_DF/lightning_logs/pcgrad/checkpoints/epoch=160_time=150622.ckpt'

    model = load_network(configs, pth_path)
    Qdataloader, DBdataloader, Q_path, DB_path = get_test_dataset(img_size=img_size, Qpad=0, batch_size=batch_size,
                                                                  num_workers=4, which_dataset=which_dataset,
                                                                  height=f"{height}", mode=mode)

    Qdesc, Qlabel = get_descriptors(model, Qdataloader, fliplr=True, device=Device)
    DBdesc, DBlabel = get_descriptors(model, DBdataloader, fliplr=True, device=Device)
    Qlabel, DBlabel = Qlabel.tolist(), DBlabel.tolist()
    Qdesc, DBdesc = Qdesc.numpy(), DBdesc.numpy()
    # All_desc = np.concatenate((Qdesc,DBdesc),axis=0)
    # All_label = np.concatenate((Qlabel,DBlabel),axis=0)
    if mode=='sat->drone':
        sampled_Qdesc,sampled_Qlabel = Qdesc[:500],Qlabel[:500]
        gt = gen_gt(Qlabel,DBlabel)[:500]
        sampled_gt_list = []

        for c,sub_gt in enumerate(gt):
            sampled_gt_list.append(sub_gt[:4])

        sampled_gt_list = np.array(sampled_gt_list, dtype=np.int32).flatten()

        sampled_DBdesc, sampled_DBlabel = DBdesc[sampled_gt_list], np.array(DBlabel)[sampled_gt_list]
    else:
        pass
    All_desc = np.concatenate((sampled_Qdesc,sampled_DBdesc),axis=0)
    All_label = np.concatenate((sampled_Qlabel,sampled_DBlabel),axis=0)


    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(All_desc)
    # 可视化带标签的 PCA 结果
    plt.figure(figsize=(10, 8))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=All_label, cmap='viridis', marker='o')
    # plt.legend(handles=scatter.legend_elements()[0], labels=set(All_label))
    plt.title('PCA Visualization of Sampled Descriptors with Labels')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.show()

if __name__ == "__main__":
    main()