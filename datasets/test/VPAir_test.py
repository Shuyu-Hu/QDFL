from pathlib import Path
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

DATASET_ROOT = '/media/whu/Largedisk/datasets/VPAir/'
# GT_ROOT = './datasets_cache/' # BECAREFUL, this is the ground truth that comes with GSV-Cities
GT_ROOT = '/media/whu/Largedisk/datasets/VPAir/VPAir_q_db_gt/'   # test

path_obj = Path(DATASET_ROOT)
if not path_obj.exists():
    raise Exception(f'Please make sure the path {DATASET_ROOT} to VPAir dataset is correct')

if not path_obj.joinpath('ref') or not path_obj.joinpath('query'):
    raise Exception(f'Please make sure the directories query and ref are situated in the directory {DATASET_ROOT}')


class VPAir_test(Dataset):
    def __init__(self, input_transform=None):
        self.input_transform = input_transform

        # reference images names
        self.dbImages = np.load(GT_ROOT + '/VPAir_dbImages.npy')

        # query images names
        self.qImages = np.load(GT_ROOT + '/VPAir_qImages.npy')

        # ground truth
        self.ground_truth = np.load(GT_ROOT + '/VPAir_gt.npy', allow_pickle=True)

        # reference images then query images
        self.images = np.concatenate((self.dbImages, self.qImages))

        self.num_references = len(self.dbImages)
        self.num_queries = len(self.qImages)

    def __getitem__(self, index):
        img = Image.open(DATASET_ROOT + self.images[index])

        if self.input_transform:
            img = self.input_transform(img)

        return img

    def __len__(self):
        return len(self.images)


def check_object_array_columns(array, expected_columns):
    num = 0
    for sub_array in array:
        if isinstance(sub_array, np.ndarray) and sub_array.ndim == expected_columns:
            num += 1
        else:
            continue
    return num


if __name__ == "__main__":
    import numpy as np

    # parse_dbStruct('/media/whu/Largedisk/datasets/mat/netvlad_v100_datasets/datasets_cache/tokyo247.mat')
    x = VPAir_test(input_transform=None)
    print(x.__getitem__(1).size)
    # test = np.load('/media/whu/Largedisk/datasets/VPAir/vpair_gt.npy',allow_pickle=True)
    # test2 = test[:,1]
    # test3 = []
    # for i,it in enumerate(test2):
    #     test3.append(np.array(it))
    # test3 = np.array(test3,dtype=object)
    # np.save('/home/whu/Documents/codespace/mixvpr/MixVPR/datasets_cache/VPAir/VPAir_gt.npy',test3)

    # num = check_object_array_columns(x.ground_truth, 2)
    # print(r"子数组{}列都是2".format(num))

    # which = 'query'
    #
    # file_names = sorted(os.listdir(fr'/media/whu/Largedisk/datasets/VPAir/{which}/'))
    #
    # # 为每个文件名加上前缀
    # file_names_with_prefix = [os.path.join(fr'{which}/', file_name) for file_name in file_names]
    #
    # # 将文件名列表转换为 NumPy 数组
    # file_names_array = np.array(file_names_with_prefix)
    #
    # # 保存到 .npy 文件中
    # output_file = '/home/whu/Documents/codespace/mixvpr/MixVPR/datasets_cache/VPAir/VPAir_qImages.npy'
    # np.save(output_file, file_names_array)
    # print(1)
