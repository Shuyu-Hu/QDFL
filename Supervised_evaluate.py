from utils.evaluation_utils.get_descriptors import get_descriptors_supervised_with_label
from utils.evaluation_utils.load_network import load_network_supervised
from utils.evaluation_utils.get_test_datasets import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def evaluate_supervised_learning(which_dataset='U1652', height=200, mode='drone->sat', fliplr=False, img_size=(256, 256), batch_size=8,
             configs=None, pth_path=None, evaluate_type='Euclidean', save_img_path=False):
    model = load_network_supervised(configs, pth_path)
    Qdataloader, DBdataloader, Q_path, DB_path = get_test_dataset_supervised(img_size=img_size, Qpad=0, batch_size=batch_size,
                                                                  num_workers=4, which_dataset=which_dataset,
                                                                  height=f"{height}", mode=mode, which_weather='Snow')

    Qdesc, Qlabel = get_descriptors_supervised_with_label(model, Qdataloader, fliplr, device)
    DBdesc, DBlabel = get_descriptors_supervised_with_label(model, DBdataloader, fliplr, device)
    print(Qdesc.shape, Qlabel.shape)
    print(DBdesc.shape, DBlabel.shape)
    print(f'Current Evaluate mode is {mode}')
    if 'denseuav' in which_dataset.lower():
        gt = [[] for _ in range(len(Qdataloader.dataset.labels))]
        label_to_indices = {}
        for i, label in enumerate(DBdataloader.dataset.labels):
            if label not in label_to_indices:
                label_to_indices[label] = []
            label_to_indices[label].append(i)

        # 遍历qlabel，将对应的dblabel索引添加到gt列表中
        for q_index, q_label in enumerate(Qdataloader.dataset.labels):
            if q_label in label_to_indices:
                gt[q_index] = label_to_indices[q_label]

        gt = np.array([np.array(lst) for lst in gt])
        loc_xy_list = Qdataloader.dataset.configDict
        Get_Recalls_AP_SDM_Dist_DenseUAV(r_list=DBdesc.numpy(),
                                        q_list=Qdesc.numpy(),
                                        Recall_k_values=[1, 5, 10, 25, 50],
                                        SDM_k_values=[1, 3, 5, 10],
                                        Dist_k_values=[1, 3, 5, 10],
                                        gt=gt,
                                        q_img_path=Q_path,
                                        db_img_path=DB_path,
                                        loc_xy_list=loc_xy_list,
                                        type=evaluate_type,
                                        print_results=True,
                                        faiss_gpu=False)
    else:
        if save_img_path:
            img_path_list = np.concatenate((np.array(DB_path, dtype=object), np.array(Q_path, dtype=object)))
            out_csv = rf'./LOGS/csv_output/{configs["backbone_name"]}_{configs["components_name"]}_{which_dataset}_{mode}.csv'
            Get_Recalls_AP_With_Output_Supervised(DBdesc.numpy(), Qdesc.numpy(), r_lable=DBlabel.tolist(),
                                                  q_lable=Qlabel.tolist(), k_values=[1, 5, 10, 25, 50],
                                                  img_path_list=img_path_list, output_path=out_csv, type=evaluate_type,
                                                  print_results=True, faiss_gpu=False, dataset_name=which_dataset)
        else:
            Get_Recalls_AP_Supervised(DBdesc.numpy(), Qdesc.numpy(), r_lable=DBlabel.tolist(), q_lable=Qlabel.tolist(),
                                      k_values=[1, 5, 10, 25, 50], type=evaluate_type, print_results=True, faiss_gpu=False,
                                      dataset_name=which_dataset)


def evaluate_all_supervised_learning(datasets_configs, base_configs, fliplr=True, img_size=(512, 512), batch_size=128, pth_path=None,
                 save_img_path=False):
    default_height = [200]
    save_path_flag = save_img_path

    def get_parent_directory(pth_path):
        # 获取父目录路径
        parent_directory = os.path.dirname(pth_path)
        parent_directory = "/".join(parent_directory.split('/')[:-1])  # version xx path
        return parent_directory

    # output_file = f'performance_{base_configs["model_configs"]["backbone_name"]}_{base_configs["model_configs"]["components_name"]}.txt'
    output_file = f'performance_{base_configs["backbone_name"]}_{base_configs["components_name"]}.txt'
    output_file = os.path.join(get_parent_directory(pth_path), output_file)

    # 打开文件以写入模式
    with open(output_file, 'w') as f:
        # 将 base_configs 参数打印到文件开头
        f.write("Base Configurations:\n")
        f.write(f"Test Image Size:{img_size}\n")
        f.write(pprint.pformat(base_configs))  # 格式化打印 base_configs
        f.write("\n\n")  # 添加换行以分隔配置和接下来的输出

        for dataset_name, modes in datasets_configs.items():
            if isinstance(modes, tuple):
                heights, mode_list = modes
            else:
                heights, mode_list = default_height, modes

            for height in heights:
                for mode in mode_list:

                    # 捕获 evaluate 函数的输出
                    original_stdout = sys.stdout
                    sys.stdout = f  # 将输出重定向到文件

                    try:
                        # 调用 evaluate 函数并将结果写入文件
                        f.write(f"Evaluating {dataset_name} with height={height} and mode={mode}\n")
                        evaluate_supervised_learning(
                            which_dataset=dataset_name,
                            height=height,
                            mode=mode,
                            fliplr=fliplr,
                            img_size=img_size,
                            batch_size=batch_size,
                            configs=base_configs,
                            pth_path=pth_path,
                            save_img_path=save_path_flag
                        )
                    finally:
                        sys.stdout = original_stdout  # 还原标准输出到控制台

if __name__ == '__main__':
    configs = load_config('./model_configs/dino_b_QDFL.yaml')["model_configs"]

    pth_path = '/home/whu/Documents/codespace/QDFL/LOGS/dinov2_vitb14_QDFL/lightning_logs/version_0/checkpoints/last.ckpt'

    datasets_configs = {
                        'U1652': ['sat->drone', 'drone->sat'],
                        # 'DenseUAV':['drone->sat'],
                        'SUES200':([150, 200, 250, 300],['sat->drone','drone->sat'])
                        }

    # single
    # evaluate_supervised_learning(which_dataset= 'denseuav',
    #          height=150,
    #          mode='drone->sat',
    #          fliplr=True,
    #          img_size=(224,224),
    #          batch_size=1,
    #          configs=configs,
    #          pth_path = pth_path,
    #          evaluate_type='eu',
    #          save_img_path=False)

    # test all
    evaluate_all_supervised_learning(datasets_configs=datasets_configs,
                 base_configs=configs,
                 fliplr=True,
                 img_size=(280,280),
                 batch_size=192,
                 pth_path=pth_path,
                 save_img_path=False
                 )