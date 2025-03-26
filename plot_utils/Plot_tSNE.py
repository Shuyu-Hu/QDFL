from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from scipy.spatial.distance import pdist, squareform
import seaborn as sns

from utils.metrics import gen_gt
import torch
from Supervised_evaluate import load_network_supervised,get_test_dataset_supervised,get_descriptors_supervised_with_label
from utils.commons import load_config

Device = torch.device('cuda')

def main():
    img_size = (224, 224)
    batch_size = 192
    which_dataset = 'u1652'
    height = 200
    mode = 'sat->drone'
    configs = load_config('../model_configs/dino_b_QDFL.yaml')['model_configs']
    pth_path = '/home/whu/Documents/codespace/learn_lightning/Drone_Sat_Geo_Localization/Current_SOTA/DINO_QDFL_U1652.pth'

    model = load_network_supervised(configs, pth_path)
    Qdataloader, DBdataloader, Q_path, DB_path = get_test_dataset_supervised(img_size=img_size, Qpad=0, batch_size=batch_size,
                                                                  num_workers=4, which_dataset=which_dataset,
                                                                  height=f"{height}", mode=mode)

    Qdesc, Qlabel = get_descriptors_supervised_with_label(model, Qdataloader, fliplr=True, device=Device)
    DBdesc, DBlabel = get_descriptors_supervised_with_label(model, DBdataloader, fliplr=True, device=Device)
    Qlabel, DBlabel = Qlabel.tolist(), DBlabel.tolist()
    Qdesc, DBdesc = Qdesc.numpy(), DBdesc.numpy()
    # All_desc = np.concatenate((Qdesc,DBdesc),axis=0)
    # All_label = np.concatenate((Qlabel,DBlabel),axis=0)
    if mode=='sat->drone':
        sampled_Qdesc,sampled_Qlabel = Qdesc[:40],Qlabel[:40]
        gt = gen_gt(Qlabel,DBlabel)[:40]
        sampled_gt_list = []

        for c,sub_gt in enumerate(gt):
            sampled_gt_list.append(sub_gt[:40])

        sampled_gt_list = np.array(sampled_gt_list, dtype=np.int32).flatten()

        sampled_DBdesc, sampled_DBlabel = DBdesc[sampled_gt_list], np.array(DBlabel)[sampled_gt_list]
    else:
        pass
    All_desc = np.concatenate((sampled_Qdesc,sampled_DBdesc),axis=0)
    All_label = np.concatenate((sampled_Qlabel,sampled_DBlabel),axis=0)

    # ===============================================2D===============================================
    # 计算 t-SNE 嵌入
    tsne = TSNE(n_components=2, perplexity=5, random_state=0)
    X_tsne = tsne.fit_transform(All_desc)

    # 可视化 t-SNE 结果
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=All_label, cmap='viridis', marker='.')
    plt.legend(handles=scatter.legend_elements()[0])
    plt.title('t-SNE Visualization')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.show()
    # ===============================================3D===============================================
    # 计算 t-SNE 嵌入
    # tsne = TSNE(n_components=3, perplexity=30, random_state=0)
    # X_tsne = tsne.fit_transform(All_desc)
    #
    # # 可视化 t-SNE 结果
    # fig = plt.figure(figsize=(10, 8))
    # ax = fig.add_subplot(111, projection='3d')
    # scatter = ax.scatter(X_tsne[:, 0], X_tsne[:, 1], X_tsne[:, 2], c=All_label, cmap='viridis', marker='o')
    #
    # # 添加图例
    # legend1 = ax.legend(*scatter.legend_elements(), title="Classes")
    # ax.add_artist(legend1)
    #
    # # 添加标题和轴标签
    # ax.set_title('3D t-SNE Visualization')
    # ax.set_xlabel('t-SNE 1')
    # ax.set_ylabel('t-SNE 2')
    # ax.set_zlabel('t-SNE 3')
    #
    # plt.show()

    # 计算并可视化距离矩阵
    # dist_matrix = squareform(pdist(All_desc))
    # plt.figure(figsize=(12, 10))
    # sns.heatmap(dist_matrix, cmap='viridis')
    # plt.title('Distance Matrix')
    # plt.show()


if __name__ == "__main__":
    main()