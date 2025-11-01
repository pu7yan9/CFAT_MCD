import os
import cv2
import glob
import random
import numpy
import torch.utils.data



dataset_1 = '/home/yan/Documents/Dataset/Work2/#1-Italy'
dataset_2 = '/home/yan/Documents/Dataset/Work2/#2-Yellow'
dataset_3 = '/home/yan/Documents/Dataset/Work2/#3-Shuguag'
dataset_4 = '/home/yan/Documents/Dataset/Work2/#4-Glources2'
dataset_5 = '/home/yan/Documents/Dataset/Work2/#5-Glources1'
dataset_6 = '/home/yan/Documents/Dataset/Work2/#6-California'
dataset_7 = '/home/yan/Documents/Dataset/Work2/#7-France'
datasets = [dataset_1, dataset_2, dataset_3, dataset_4, dataset_5, dataset_6, dataset_7]


def read_image_list(dataset, p_tra, patch_size, core=6, num=None):
    array_of_train_img_patch_T1_true = []
    array_of_train_img_patch_T1_false = []
    array_of_train_img_patch_T2_true = []
    array_of_train_img_patch_T2_false = []
    array_of_test_img_patch_T1 = []
    array_of_test_img_patch_T2 = []
    array_of_train_label_true = []
    array_of_train_label_false = []
    array_of_test_label = []
    SIZE = patch_size

    GT = cv2.imread(glob.glob(dataset + '/GT.*')[0], cv2.IMREAD_GRAYSCALE)
    T1 = cv2.imread(glob.glob(dataset + '/T1.*')[0])
    T2 = cv2.imread(glob.glob(dataset + '/T2.*')[0])
    T1 = cv2.copyMakeBorder(T1, SIZE // 2, SIZE // 2, SIZE // 2, SIZE // 2, cv2.BORDER_REFLECT)
    T2 = cv2.copyMakeBorder(T2, SIZE // 2, SIZE // 2, SIZE // 2, SIZE // 2, cv2.BORDER_REFLECT)

    pseudo_label = cv2.imread(dataset + '/Pseudo/CM_{}.png'.format(p_tra), cv2.IMREAD_GRAYSCALE)
    erosion_pseudo_label = cv2.imread(dataset + '/Pseudo_ero/{}/CM_{}_eroded_d{}.png'.format(p_tra, p_tra, core), cv2.IMREAD_GRAYSCALE)

    # 测试样本
    for i in range(T1.shape[0] - SIZE + 1):
        for j in range(T1.shape[1] - SIZE + 1):
            T1_patch = T1[i:i + SIZE, j:j + SIZE, ]
            T2_patch = T2[i:i + SIZE, j:j + SIZE, ]
            array_of_test_img_patch_T1.append(T1_patch)
            array_of_test_img_patch_T2.append(T2_patch)
            array_of_test_label.append(numpy.array(GT[i, j] // 255).reshape(1))

    # 负样本
    for i in range(T1.shape[0] - SIZE + 1):
        for j in range(T1.shape[1] - SIZE + 1):
            T1_patch = T1[i:i + SIZE, j:j + SIZE, ]
            T2_patch = T2[i:i + SIZE, j:j + SIZE, ]
            if (pseudo_label[i, j] == 0 and erosion_pseudo_label[i, j] == 0):
                array_of_train_img_patch_T1_false.append(T1_patch)
                array_of_train_img_patch_T2_false.append(T2_patch)
                array_of_train_label_false.append(numpy.array(0).reshape(1))

    # 正样本
    for i in range(T1.shape[0] - SIZE + 1):
        for j in range(T1.shape[1] - SIZE + 1):
            T1_patch = T1[i:i + SIZE, j:j + SIZE, ]
            T2_patch = T2[i:i + SIZE, j:j + SIZE, ]
            if (pseudo_label[i, j] != 0 and erosion_pseudo_label[i, j] != 0):
                array_of_train_img_patch_T1_true.append(T1_patch)
                array_of_train_img_patch_T2_true.append(T2_patch)
                array_of_train_label_true.append(numpy.array(1).reshape(1))

    def shuffle_together(a, b, c):
        packed = list(zip(a, b, c))
        random.shuffle(packed)
        if not packed:
            return [], [], []
        a2, b2, c2 = zip(*packed)
        return list(a2), list(b2), list(c2)

    (array_of_train_img_patch_T1_true,
     array_of_train_img_patch_T2_true,
     array_of_train_label_true) = shuffle_together(
        array_of_train_img_patch_T1_true,
        array_of_train_img_patch_T2_true,
        array_of_train_label_true
    )

    (array_of_train_img_patch_T1_false,
     array_of_train_img_patch_T2_false,
     array_of_train_label_false) = shuffle_together(
        array_of_train_img_patch_T1_false,
        array_of_train_img_patch_T2_false,
        array_of_train_label_false
    )

    total_samples = min(len(array_of_train_label_true), len(array_of_train_label_false)) * 2

    if num is not None and total_samples > num:
        half = num // 2
        array_of_train_img_patch_T1 = array_of_train_img_patch_T1_true[:half] + array_of_train_img_patch_T1_false[
                                                                                :half]
        array_of_train_img_patch_T2 = array_of_train_img_patch_T2_true[:half] + array_of_train_img_patch_T2_false[
                                                                                :half]
        array_of_train_label = array_of_train_label_true[:half] + array_of_train_label_false[:half]
    else:
        array_of_train_img_patch_T1 = array_of_train_img_patch_T1_true + array_of_train_img_patch_T1_false
        array_of_train_img_patch_T2 = array_of_train_img_patch_T2_true + array_of_train_img_patch_T2_false
        array_of_train_label = array_of_train_label_true + array_of_train_label_false

    pack = list(zip(array_of_train_img_patch_T1, array_of_train_img_patch_T2, array_of_train_label))
    random.shuffle(pack)
    if pack:
        array_of_train_img_patch_T1, array_of_train_img_patch_T2, array_of_train_label = map(list, zip(*pack))
    else:
        array_of_train_img_patch_T1, array_of_train_img_patch_T2, array_of_train_label = [], [], []

    return (array_of_train_img_patch_T1, array_of_train_img_patch_T2, array_of_train_label,
            array_of_test_img_patch_T1, array_of_test_img_patch_T2, array_of_test_label)


class HCD(torch.utils.data.Dataset):
    def __init__(self, train=True, transform=None, dataset_id=1, p_tra='IRGMcS',  patch_size=13, num=None):
        super(HCD, self).__init__()
        dataset = datasets[dataset_id - 1]
        if train:
            seg_img_1, seg_img_2, seg_label, _, _, _ = read_image_list(dataset, p_tra, patch_size, num=num)
        else:
            _, _, _, seg_img_1, seg_img_2, seg_label = read_image_list(dataset, p_tra, patch_size, num=num)
        self.seq_img_1 = seg_img_1
        self.seg_img_2 = seg_img_2
        self.seq_label = seg_label
        self.transform = transform

    def __getitem__(self, index):
        img_1 = self.seq_img_1[index]
        img_2 = self.seg_img_2[index]
        label = self.seq_label[index]
        if self.transform is not None:
            img_1 = self.transform(img_1)
            img_2 = self.transform(img_2)
            label = torch.Tensor(label)
        return img_1, img_2, label

    def __len__(self):
        return len(self.seq_label)
