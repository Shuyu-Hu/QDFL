import math

import faiss
import pandas as pd
import torch
from prettytable import PrettyTable
import numpy as np
from torchmetrics import AveragePrecision

from sympy import print_tree


def gen_gt(q_lable: list, r_lable: list):
    gt = [[] for _ in range(len(q_lable))]
    # 创建一个字典，用于存储dblabel中每个唯一标签对应的索引列表
    label_to_indices = {}
    for i, label in enumerate(r_lable):
        if label not in label_to_indices:
            label_to_indices[label] = []
        label_to_indices[label].append(i)

    # 遍历qlabel，将对应的dblabel索引添加到gt列表中
    for q_index, q_label in enumerate(q_lable):
        if q_label in label_to_indices:
            gt[q_index] = label_to_indices[q_label]

    return np.array([np.array(lst) for lst in gt], dtype=object)

def compute_AP(index, good_index, junk_index):
    ap = 0
    if good_index.size == 0:  # if empty
        return ap

    # remove junk_index
    mask = np.in1d(index, junk_index, invert=True)
    index = index[mask]

    # find good_index index
    ngood = len(good_index)
    mask = np.in1d(index, good_index)
    rows_good = np.argwhere(mask == True)
    rows_good = rows_good.flatten()

    if rows_good.size == 0:  # Ensure rows_good is not empty
        return ap

    for i in range(min(ngood, len(rows_good))):#optimize function, improve stability
    # for i in range(ngood):
        d_recall = 1.0 / ngood
        precision = (i + 1) * 1.0 / (rows_good[i] + 1)
        if i >= len(rows_good):
            print(f"Index {i} is out of bounds for rows_good of length {len(rows_good)}")
        if rows_good[i] != 0:
            old_precision = i * 1.0 / rows_good[i]
        else:
            old_precision = 1.0
        ap += d_recall * (old_precision + precision) / 2

    return ap


def sdm(query_loc, sdmk_list, gallery_loc_xy_list, s=5e3):
    query_x, query_y = query_loc

    sdm_list = []

    for k in sdmk_list:
        sdm_nom = 0.0
        sdm_den = 0.0
        for i in range(k):
            gallery_x, gallery_y = gallery_loc_xy_list[i]
            d = np.sqrt((query_x - gallery_x)**2 + (query_y - gallery_y)**2)
            # d = euclideanDistance(query_loc,gallery_loc_xy_list[i])
            sdm_nom += (k - i) / np.exp(s * d)
            sdm_den += (k - i)
        sdm_list.append(sdm_nom/sdm_den)
    return sdm_list

def get_dis(query_loc, disk_list, gallery_loc_xy_list):
    query_x, query_y = query_loc
    dis_list = []
    for k in disk_list:
        dis_sum = 0.0
        for i in range(k):
            gallery_x, gallery_y = gallery_loc_xy_list[i]
            # dis = np.sqrt((query_x - gallery_x)**2 + (query_y - gallery_y)**2)
            dis = latlog2meter(query_y,query_x,gallery_y,gallery_x)
            dis_sum += dis
        dis_list.append(dis_sum / k)

    return dis_list

def latlog2meter(lata, loga, latb, logb):
    # log 纬度 lat 经度
    # EARTH_RADIUS = 6371.0
    EARTH_RADIUS =6378.137
    PI = math.pi
    # // 转弧度
    lat_a = lata * PI / 180
    lat_b = latb * PI / 180
    a = lat_a - lat_b
    b = loga * PI / 180 - logb * PI / 180
    dis = 2 * math.asin(
        math.sqrt(math.pow(math.sin(a / 2), 2) + math.cos(lat_a) * math.cos(lat_b) * math.pow(math.sin(b / 2), 2)))

    distance = EARTH_RADIUS * dis * 1000
    return distance

def sdm_GTA(query_loc, sdmk_list, index, gallery_loc_xy_list, s=0.001):
    query_x, query_y = query_loc

    sdm_list = []

    for k in sdmk_list:
        sdm_nom = 0.0
        sdm_den = 0.0
        for i in range(k):
            idx = index[i]
            gallery_x, gallery_y = gallery_loc_xy_list[idx]
            d = np.sqrt((query_x - gallery_x)**2 + (query_y - gallery_y)**2)
            sdm_nom += (k - i) / np.exp(s * d)
            sdm_den += (k - i)
        sdm_list.append(sdm_nom/sdm_den)
    return sdm_list


def get_dis_GTA(query_loc, disk_list, index, gallery_loc_xy_list):
    query_x, query_y = query_loc
    dis_list = []
    for k in disk_list:
        dis_sum = 0.0
        for i in range(k):
            idx = index[i]
            gallery_x, gallery_y = gallery_loc_xy_list[idx]
            dis = np.sqrt((query_x - gallery_x)**2 + (query_y - gallery_y)**2)
            dis_sum += dis
        dis_list.append(dis_sum / k)

    return dis_list

def Get_Recalls_AP_Supervised(r_list: np.array, q_list: np.array, r_lable: list, q_lable: list, k_values: list, gt: np.array=None, type='Euclidean', print_results=True, faiss_gpu=False,
                                 dataset_name='U1652'):
    embed_size = r_list.shape[1]
    if r_lable and q_lable:
        """
            这个函数主要是对查询和数据库的索引进行比对，需要输入gt的格式形如[[1],[2,3],...]，即每个查询对应的数据库图片。
        """
        gt = gen_gt(q_lable,r_lable)

    if 'eu' in type.lower():
        #是否使用faiss_gpu
        if faiss_gpu:
            res = faiss.StandardGpuResources()#收集标准GPU资源
            flat_config = faiss.GpuIndexFlatConfig()
            flat_config.useFloat16 = True
            flat_config.device = 0
            faiss_index = faiss.GpuIndexFlatL2(res, embed_size, flat_config)#初始化示例
        else:
            faiss_index = faiss.IndexFlatL2(embed_size)#初始化cpu实例

        # Add references
        faiss_index.add(r_list)#将索引加入faiss_index

        # Search for queries in the index
        # _, predictions = faiss_index.search(q_list, max(k_values))#在建立的索引中寻找每一个检索对应的前k个匹配数据库索引
        _, predictions = faiss_index.search(q_list, len(r_list))#在建立的索引中寻找每一个检索对应的前k个匹配数据库索引
    elif 'cos' in type.lower():
        # Normalize the vectors to unit length for cosine similarity
        def normalize(vectors):
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            return vectors / norms

        # Normalize reference and query lists
        r_list_normalized = normalize(r_list)
        q_list_normalized = normalize(q_list)

        # Check if GPU is enabled
        if faiss_gpu:
            res = faiss.StandardGpuResources()  # Collect standard GPU resources
            flat_config = faiss.GpuIndexFlatConfig()
            flat_config.useFloat16 = True
            flat_config.device = 0
            # Use GpuIndexFlatIP for inner product (cosine similarity)
            faiss_index = faiss.GpuIndexFlatIP(res, embed_size, flat_config)  # Initialize GPU instance
        else:
            # Use IndexFlatIP for inner product (cosine similarity)
            faiss_index = faiss.IndexFlatIP(embed_size)  # Initialize CPU instance

        # Add normalized references to the index
        faiss_index.add(r_list_normalized)  # Add the index to faiss_index

        # Search for queries in the index
        # For cosine similarity, larger values indicate more similarity
        _, predictions = faiss_index.search(q_list_normalized, len(r_list))
    else:
        raise KeyError(f"{type} not implemented!")

    # Start calculating recall_at_k
    #这一段是为了统计R@K
    correct_at_k = np.zeros(len(k_values))
    aps = []
    for q_idx, pred in enumerate(predictions):
        for i, n in enumerate(k_values):
            if np.any(np.in1d(pred[:n], gt[q_idx])):
                correct_at_k[i:] += 1
                break
        good_index = np.array(gt[q_idx]).reshape(len(gt[q_idx]),1)
        junk_index = np.argwhere(np.array(r_lable)==-1)
        ap = compute_AP(pred, good_index, junk_index)
        aps.append(ap)

    correct_at_k = correct_at_k / len(predictions)
    d = {k: v for (k, v) in zip(k_values, correct_at_k)}

    if print_results:
        print()  # print a new line
        table = PrettyTable()
        table.field_names = ['K'] + [str(k) for k in k_values]
        table.add_row(['Recall@K'] + [f'{100 * v:.2f}' for v in correct_at_k])

        map_table = PrettyTable()
        map_table.field_names = ['Metric', 'AP']
        map_table.add_row(['Value', f'{100 * np.average(aps):.2f}'])

        print(table.get_string(title=f"Recall on {dataset_name}"))
        print(map_table.get_string(title=f"Other Performances on {dataset_name}"))
        # Print time measurements
    return d, predictions

def Get_Recalls_AP_With_Output_Supervised(r_list: np.array, q_list: np.array, r_lable: list, q_lable: list, k_values: list, gt: np.array=None, img_path_list=None, output_path=None, type='Euclidean', print_results=True, faiss_gpu=False,
                                 dataset_name='U1652'):
    embed_size = r_list.shape[1]
    if r_lable and q_lable:
        """
            这个函数主要是对查询和数据库的索引进行比对，需要输入gt的格式形如[[1],[2,3],...]，即每个查询对应的数据库图片。
        """
        gt = gen_gt(q_lable,r_lable)

    if 'eu' in type.lower():
        #是否使用faiss_gpu
        if faiss_gpu:
            res = faiss.StandardGpuResources()#收集标准GPU资源
            flat_config = faiss.GpuIndexFlatConfig()
            flat_config.useFloat16 = True
            flat_config.device = 0
            faiss_index = faiss.GpuIndexFlatL2(res, embed_size, flat_config)#初始化示例
        else:
            faiss_index = faiss.IndexFlatL2(embed_size)#初始化cpu实例

        # Add references
        faiss_index.add(r_list)#将索引加入faiss_index

        # Search for queries in the index
        # _, predictions = faiss_index.search(q_list, max(k_values))#在建立的索引中寻找每一个检索对应的前k个匹配数据库索引
        _, predictions = faiss_index.search(q_list, len(r_list))#在建立的索引中寻找每一个检索对应的前k个匹配数据库索引
    elif 'cos' in type.lower():
        # Normalize the vectors to unit length for cosine similarity
        def normalize(vectors):
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            return vectors / norms

        # Normalize reference and query lists
        r_list_normalized = normalize(r_list)
        q_list_normalized = normalize(q_list)

        # Check if GPU is enabled
        if faiss_gpu:
            res = faiss.StandardGpuResources()  # Collect standard GPU resources
            flat_config = faiss.GpuIndexFlatConfig()
            flat_config.useFloat16 = True
            flat_config.device = 0
            # Use GpuIndexFlatIP for inner product (cosine similarity)
            faiss_index = faiss.GpuIndexFlatIP(res, embed_size, flat_config)  # Initialize GPU instance
        else:
            # Use IndexFlatIP for inner product (cosine similarity)
            faiss_index = faiss.IndexFlatIP(embed_size)  # Initialize CPU instance

        # Add normalized references to the index
        faiss_index.add(r_list_normalized)  # Add the index to faiss_index

        # Search for queries in the index
        # For cosine similarity, larger values indicate more similarity
        _, predictions = faiss_index.search(q_list_normalized, len(r_list))
    else:
        raise KeyError(f"{type} not implemented!")

    # Start calculating recall_at_k
    #这一段是为了统计R@K
    correct_at_k = np.zeros(len(k_values))
    aps = []
    results = []
    for q_idx, pred in enumerate(predictions):
        correct_flag = False
        dbrecall = []
        for i, n in enumerate(k_values):
            dbrecall = [img_path_list[i] for i in pred[:3]]
            if np.any(np.in1d(pred[:n], gt[q_idx])):
                correct_at_k[i:] += 1
                if n == 1:  # Check if R@1
                    correct_flag = True
                    results.append((img_path_list[len(r_list) + q_idx], 'true' if correct_flag else 'false', *dbrecall))
                    dbrecall = []
                break
        if len(dbrecall) != 0:
            results.append((img_path_list[len(r_list) + q_idx], 'true' if correct_flag else 'false', *dbrecall))
        good_index = np.array(gt[q_idx]).reshape(len(gt[q_idx]),1)
        junk_index = np.argwhere(np.array(r_lable)==-1)
        ap = compute_AP(pred, good_index, junk_index)
        aps.append(ap)

    correct_at_k = correct_at_k / len(predictions)
    d = {k: v for (k, v) in zip(k_values, correct_at_k)}

    df = pd.DataFrame(results, columns=['Query_Image', 'Result', '1ST', '2ND', '3RD'])
    df.to_csv(output_path, index=False)

    if print_results:
        print()  # print a new line
        table = PrettyTable()
        table.field_names = ['K'] + [str(k) for k in k_values]
        table.add_row(['Recall@K'] + [f'{100 * v:.2f}' for v in correct_at_k])

        map_table = PrettyTable()
        map_table.field_names = ['Metric', 'AP']
        map_table.add_row(['Value', f'{100 * np.average(aps):.2f}'])

        print(table.get_string(title=f"Recall on {dataset_name}"))
        print(map_table.get_string(title=f"Other Performances on {dataset_name}"))
        # Print time measurements
    return d, predictions


def Get_Recalls_AP_Metric_Learning(r_list: np.array, q_list: np.array, k_values: list, gt: np.array, type='Euclidean', print_results=True, faiss_gpu=False,
                                 dataset_name='U1652'):
    embed_size = r_list.shape[1]
    if 'eu' in type.lower():
        #是否使用faiss_gpu
        if faiss_gpu:
            res = faiss.StandardGpuResources()#收集标准GPU资源
            flat_config = faiss.GpuIndexFlatConfig()
            flat_config.useFloat16 = True
            flat_config.device = 0
            faiss_index = faiss.GpuIndexFlatL2(res, embed_size, flat_config)#初始化示例
        else:
            faiss_index = faiss.IndexFlatL2(embed_size)#初始化cpu实例

        # Add references
        faiss_index.add(r_list)#将索引加入faiss_index

        # Search for queries in the index
        _, predictions = faiss_index.search(q_list, max(k_values))#在建立的索引中寻找每一个检索对应的前k个匹配数据库索引
        # _, predictions = faiss_index.search(q_list, len(r_list))#在建立的索引中寻找每一个检索对应的前k个匹配数据库索引
    elif 'cos' in type.lower():
        # Normalize the vectors to unit length for cosine similarity
        def normalize(vectors):
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            return vectors / norms

        # Normalize reference and query lists
        r_list_normalized = normalize(r_list)
        q_list_normalized = normalize(q_list)

        # Check if GPU is enabled
        if faiss_gpu:
            res = faiss.StandardGpuResources()  # Collect standard GPU resources
            flat_config = faiss.GpuIndexFlatConfig()
            flat_config.useFloat16 = True
            flat_config.device = 0
            # Use GpuIndexFlatIP for inner product (cosine similarity)
            faiss_index = faiss.GpuIndexFlatIP(res, embed_size, flat_config)  # Initialize GPU instance
        else:
            # Use IndexFlatIP for inner product (cosine similarity)
            faiss_index = faiss.IndexFlatIP(embed_size)  # Initialize CPU instance

        # Add normalized references to the index
        faiss_index.add(r_list_normalized)  # Add the index to faiss_index

        # Search for queries in the index
        # For cosine similarity, larger values indicate more similarity
        _, predictions = faiss_index.search(q_list_normalized, max(k_values))
        # _, predictions = faiss_index.search(q_list_normalized, len(r_list))
    else:
        raise KeyError(f"{type} not implemented!")

    # Start calculating recall_at_k
    correct_at_k = np.zeros(len(k_values))
    aps = []
    for q_idx, pred in enumerate(predictions):
        for i, n in enumerate(k_values):
            if np.any(np.in1d(pred[:n], gt[q_idx])):
                correct_at_k[i:] += 1
                break
        good_index = torch.tensor(gt[q_idx]).reshape(len(gt[q_idx]), 1)
        ap = compute_AP(pred, good_index, np.array([]))
        aps.append(ap)

    correct_at_k = correct_at_k / len(predictions)
    d = {k: v for (k, v) in zip(k_values, correct_at_k)}

    if print_results:
        print()  # print a new line
        table = PrettyTable()
        table.field_names = ['K'] + [str(k) for k in k_values]
        table.add_row(['Recall@K'] + [f'{100 * v:.2f}' for v in correct_at_k])
        print(table.get_string(title=f"Performances on {dataset_name} (%)"))
        print()
        table = PrettyTable()
        table.field_names = ['Metric', 'AP']
        table.add_row(['Value', f'{100 * np.average(aps):.2f}'])
        print(table.get_string(title=f"AP Performance on {dataset_name} (%)"))
    return d, predictions


def Get_Recalls_AP_SDM_DIST_Metric_Learning(r_list: np.array, q_list: np.array, k_values: list, gt: np.array, type='Euclidean', print_results=True, faiss_gpu=False,
                                 dataset_name='U1652'):
    embed_size = r_list.shape[1]
    if 'eu' in type.lower():
        #是否使用faiss_gpu
        if faiss_gpu:
            res = faiss.StandardGpuResources()#收集标准GPU资源
            flat_config = faiss.GpuIndexFlatConfig()
            flat_config.useFloat16 = True
            flat_config.device = 0
            faiss_index = faiss.GpuIndexFlatL2(res, embed_size, flat_config)#初始化示例
        else:
            faiss_index = faiss.IndexFlatL2(embed_size)#初始化cpu实例

        # Add references
        faiss_index.add(r_list)#将索引加入faiss_index

        # Search for queries in the index
        _, predictions = faiss_index.search(q_list, max(k_values))#在建立的索引中寻找每一个检索对应的前k个匹配数据库索引
        # _, predictions = faiss_index.search(q_list, len(r_list))#在建立的索引中寻找每一个检索对应的前k个匹配数据库索引
    elif 'cos' in type.lower():
        # Normalize the vectors to unit length for cosine similarity
        def normalize(vectors):
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            return vectors / norms

        # Normalize reference and query lists
        r_list_normalized = normalize(r_list)
        q_list_normalized = normalize(q_list)

        # Check if GPU is enabled
        if faiss_gpu:
            res = faiss.StandardGpuResources()  # Collect standard GPU resources
            flat_config = faiss.GpuIndexFlatConfig()
            flat_config.useFloat16 = True
            flat_config.device = 0
            # Use GpuIndexFlatIP for inner product (cosine similarity)
            faiss_index = faiss.GpuIndexFlatIP(res, embed_size, flat_config)  # Initialize GPU instance
        else:
            # Use IndexFlatIP for inner product (cosine similarity)
            faiss_index = faiss.IndexFlatIP(embed_size)  # Initialize CPU instance

        # Add normalized references to the index
        faiss_index.add(r_list_normalized)  # Add the index to faiss_index

        # Search for queries in the index
        # For cosine similarity, larger values indicate more similarity
        _, predictions = faiss_index.search(q_list_normalized, max(k_values))
        # _, predictions = faiss_index.search(q_list_normalized, len(r_list))
    else:
        raise KeyError(f"{type} not implemented!")

    # Start calculating recall_at_k
    correct_at_k = np.zeros(len(k_values))
    aps = []
    for q_idx, pred in enumerate(predictions):
        for i, n in enumerate(k_values):
            if np.any(np.in1d(pred[:n], gt[q_idx])):
                correct_at_k[i:] += 1
                break
        good_index = torch.tensor(gt[q_idx]).reshape(len(gt[q_idx]), 1)
        ap = compute_AP(pred, good_index, np.array([]))
        aps.append(ap)

    correct_at_k = correct_at_k / len(predictions)
    d = {k: v for (k, v) in zip(k_values, correct_at_k)}

    if print_results:
        print()  # print a new line
        table = PrettyTable()
        table.field_names = ['K'] + [str(k) for k in k_values]
        table.add_row(['Recall@K'] + [f'{100 * v:.2f}' for v in correct_at_k])
        print(table.get_string(title=f"Performances on {dataset_name} (%)"))
        print()
        table = PrettyTable()
        table.field_names = ['Metric', 'AP']
        table.add_row(['Value', f'{100 * np.average(aps):.2f}'])
        print(table.get_string(title=f"AP Performance on {dataset_name} (%)"))
    return d, predictions

def Get_Recalls_AP_SDM_Dist_GTA(r_list: np.array,
                                q_list: np.array,
                                Recall_k_values: list,
                                SDM_k_values: list,
                                Dist_k_values: list,
                                query_loc_xy_list,
                                gallery_loc_xy_list,
                                gt,
                                type='Euclidean',
                                print_results=True,
                                faiss_gpu=False):
    embed_size = r_list.shape[1]
    if 'eu' in type.lower():
        # 是否使用faiss_gpu
        if faiss_gpu:
            res = faiss.StandardGpuResources()  # 收集标准GPU资源
            flat_config = faiss.GpuIndexFlatConfig()
            flat_config.useFloat16 = True
            flat_config.device = 0
            faiss_index = faiss.GpuIndexFlatL2(res, embed_size, flat_config)  # 初始化示例
        else:
            faiss_index = faiss.IndexFlatL2(embed_size)  # 初始化cpu实例

        # Add references
        faiss_index.add(r_list)  # 将索引加入faiss_index

        # Search for queries in the index
        _, predictions = faiss_index.search(q_list, max(max(Recall_k_values),100))  # 在建立的索引中寻找每一个检索对应的前k个匹配数据库索引
        # _, predictions = faiss_index.search(q_list, len(r_list))#在建立的索引中寻找每一个检索对应的前k个匹配数据库索引
    elif 'cos' in type.lower():
        # Normalize the vectors to unit length for cosine similarity
        def normalize(vectors):
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            return vectors / norms

        # Normalize reference and query lists
        r_list_normalized = normalize(r_list)
        q_list_normalized = normalize(q_list)

        # Check if GPU is enabled
        if faiss_gpu:
            res = faiss.StandardGpuResources()  # Collect standard GPU resources
            flat_config = faiss.GpuIndexFlatConfig()
            flat_config.useFloat16 = True
            flat_config.device = 0
            # Use GpuIndexFlatIP for inner product (cosine similarity)
            faiss_index = faiss.GpuIndexFlatIP(res, embed_size, flat_config)  # Initialize GPU instance
        else:
            # Use IndexFlatIP for inner product (cosine similarity)
            faiss_index = faiss.IndexFlatIP(embed_size)  # Initialize CPU instance

        # Add normalized references to the index
        faiss_index.add(r_list_normalized)  # Add the index to faiss_index

        # Search for queries in the index
        # For cosine similarity, larger values indicate more similarity
        _, predictions = faiss_index.search(q_list_normalized, max(Recall_k_values))
        # _, predictions = faiss_index.search(q_list_normalized, len(r_list))
    else:
        raise KeyError(f"{type} not implemented!")

    correct_at_k = np.zeros(len(Recall_k_values))
    sdm_list = []
    dis_list = []
    aps = []
    for q_idx, pred in enumerate(predictions):
        for i, n in enumerate(Recall_k_values):
            if np.any(np.in1d(pred[:n], gt[q_idx])):
                correct_at_k[i:] += 1
                break
        sdm_list.append(sdm_GTA(query_loc_xy_list[q_idx], SDM_k_values, pred, gallery_loc_xy_list))
        dis_list.append(get_dis_GTA(query_loc_xy_list[q_idx], Dist_k_values, pred, gallery_loc_xy_list))

        good_index = torch.tensor(gt[q_idx]).reshape(len(gt[q_idx]), 1)
        ap = compute_AP(pred, good_index, np.array([]))
        aps.append(ap)

    correct_at_k = correct_at_k / len(predictions)
    d = {k: v for (k, v) in zip(Recall_k_values, correct_at_k)}
    sdm_list = np.mean(np.array(sdm_list), axis=0)
    dis_list = np.mean(np.array(dis_list), axis=0)

    if print_results:
        print()  # print a new line
        table = PrettyTable()
        table.field_names = ['K'] + [str(k) for k in Recall_k_values]
        table.add_row(['Recall@K'] + [f'{100 * v:.2f}' for v in correct_at_k])
        print(table.get_string(title=f"Recall@K Performances on GTA_UAV (%)"))
        print()
        table = PrettyTable()
        table.field_names = ['K'] + [str(k) for k in SDM_k_values]
        table.add_row(['SDM@K'] + [f'{100 * v:.2f}' for v in sdm_list])
        print(table.get_string(title=f"SDM@K Performances on GTA_UAV (%)"))
        print()
        table = PrettyTable()
        table.field_names = ['K'] + [str(k) for k in Dist_k_values]
        table.add_row(['Dist@K'] + [f'{v:.2f}' for v in dis_list])
        print(table.get_string(title=f"Dist@K Performances on GTA_UAV (Meter)"))
        print()
        table = PrettyTable()
        table.field_names = ['Metric', 'AP']
        table.add_row(['Value', f'{100 * np.average(aps):.2f}'])
        print(table.get_string(title=f"AP Performance on GTA_UAV (%)"))

    return d, predictions, sdm_list, dis_list



def Get_Recalls_AP_SDM_Dist_DenseUAV(r_list: np.array,
                                    q_list: np.array,
                                    Recall_k_values: list,
                                    SDM_k_values: list,
                                    Dist_k_values: list,
                                    gt,
                                    loc_xy_list,
                                    q_img_path,
                                    db_img_path,
                                    type='Euclidean',
                                    print_results=True,
                                    faiss_gpu=False):

    embed_size = r_list.shape[1]
    if 'eu' in type.lower():
        # 是否使用faiss_gpu
        if faiss_gpu:
            res = faiss.StandardGpuResources()  # 收集标准GPU资源
            flat_config = faiss.GpuIndexFlatConfig()
            flat_config.useFloat16 = True
            flat_config.device = 0
            faiss_index = faiss.GpuIndexFlatL2(res, embed_size, flat_config)  # 初始化示例
        else:
            faiss_index = faiss.IndexFlatL2(embed_size)  # 初始化cpu实例

        # Add references
        faiss_index.add(r_list)  # 将索引加入faiss_index

        # Search for queries in the index
        _, predictions = faiss_index.search(q_list, max(max(Recall_k_values),100))  # 在建立的索引中寻找每一个检索对应的前k个匹配数据库索引
        # _, predictions = faiss_index.search(q_list, len(r_list))#在建立的索引中寻找每一个检索对应的前k个匹配数据库索引
    elif 'cos' in type.lower():
        # Normalize the vectors to unit length for cosine similarity
        def normalize(vectors):
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            return vectors / norms

        # Normalize reference and query lists
        r_list_normalized = normalize(r_list)
        q_list_normalized = normalize(q_list)

        # Check if GPU is enabled
        if faiss_gpu:
            res = faiss.StandardGpuResources()  # Collect standard GPU resources
            flat_config = faiss.GpuIndexFlatConfig()
            flat_config.useFloat16 = True
            flat_config.device = 0
            # Use GpuIndexFlatIP for inner product (cosine similarity)
            faiss_index = faiss.GpuIndexFlatIP(res, embed_size, flat_config)  # Initialize GPU instance
        else:
            # Use IndexFlatIP for inner product (cosine similarity)
            faiss_index = faiss.IndexFlatIP(embed_size)  # Initialize CPU instance

        # Add normalized references to the index
        faiss_index.add(r_list_normalized)  # Add the index to faiss_index

        # Search for queries in the index
        # For cosine similarity, larger values indicate more similarity
        _, predictions = faiss_index.search(q_list_normalized, max(Recall_k_values))
        # _, predictions = faiss_index.search(q_list_normalized, len(r_list))
    else:
        raise KeyError(f"{type} not implemented!")
    #TODO:弄完指标计算
    correct_at_k = np.zeros(len(Recall_k_values))
    sdm_list = []
    dis_list = []
    aps = []
    for q_idx, pred in enumerate(predictions):
        for i, n in enumerate(Recall_k_values):
            if np.any(np.in1d(pred[:n], gt[q_idx])):
                correct_at_k[i:] += 1
                break
        query_loc_xy_list = loc_xy_list[q_img_path[q_idx].split("/")[-2]]
        gallery_path_list = [db_img_path[i].split("/")[-2] for i in pred]
        gallery_loc_xy_list = [loc_xy_list[i] for i in gallery_path_list]

        sdm_list.append(sdm(query_loc_xy_list, SDM_k_values, gallery_loc_xy_list))
        dis_list.append(get_dis(query_loc_xy_list, Dist_k_values, gallery_loc_xy_list))

        good_index = torch.tensor(gt[q_idx]).reshape(len(gt[q_idx]), 1)
        ap = compute_AP(pred, good_index, np.array([]))
        aps.append(ap)


    correct_at_k = correct_at_k / len(predictions)
    d = {k: v for (k, v) in zip(Recall_k_values, correct_at_k)}
    sdm_list = np.mean(np.array(sdm_list), axis=0)
    dis_list = np.mean(np.array(dis_list), axis=0)

    if print_results:
        print()  # print a new line
        table = PrettyTable()
        table.field_names = ['K'] + [str(k) for k in Recall_k_values]
        table.add_row(['Recall@K'] + [f'{100 * v:.2f}' for v in correct_at_k])
        print(table.get_string(title=f"Recall@K Performances on DenseUAV (%)"))
        print()
        table = PrettyTable()
        table.field_names = ['K'] + [str(k) for k in SDM_k_values]
        table.add_row(['SDM@K'] + [f'{100 * v:.2f}' for v in sdm_list])
        print(table.get_string(title=f"SDM@K Performances on DenseUAV (%)"))
        print()
        table = PrettyTable()
        table.field_names = ['K'] + [str(k) for k in Dist_k_values]
        table.add_row(['Dist@K'] + [f'{v:.2f}' for v in dis_list])
        print(table.get_string(title=f"Dist@K Performances on DenseUAV (Meter)"))
        print()
        table = PrettyTable()
        table.field_names = ['Metric', 'AP']
        table.add_row(['Value', f'{100 * np.average(aps):.2f}'])
        print(table.get_string(title=f"AP Performance on DenseUAV (%)"))

    return d, predictions, sdm_list, dis_list