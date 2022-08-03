# -*- coding: utf-8 -*-

"""
@date: 2022/4/28 下午9:18
@file: extract_cub200.py
@author: zj
@description: Collate data and generate
1. classes.txt: class list
2. train.txt: train image path list
3. test.txt: test image path list
"""

import os

import numpy as np
from zcls2.config.key_word import KEY_SEP


def load_data(data_root):
    assert os.path.isdir(data_root), data_root

    cls_path = os.path.join(data_root, "classes.txt")
    classes = np.loadtxt(cls_path, dtype=str, delimiter=' ')[:, 1]

    images_path = os.path.join(data_root, 'images.txt')
    images_list = np.loadtxt(images_path, dtype=str, delimiter=' ')[:, 1]

    labels_path = os.path.join(data_root, 'image_class_labels.txt')
    labels_list = np.loadtxt(labels_path, dtype=int, delimiter=' ')[:, 1]

    train_test_split_path = os.path.join(data_root, 'train_test_split.txt')
    train_test_split_list = np.loadtxt(train_test_split_path, dtype=int, delimiter=' ')[:, 1]

    train_data_list = list()
    test_data_list = list()
    for img_path, target, is_train in zip(images_list, labels_list, train_test_split_list):
        if is_train:
            train_data_list.append([os.path.join("CUB_200_2011", "images", img_path), target-1])
        else:
            test_data_list.append([os.path.join("CUB_200_2011", "images", img_path), target-1])

    return classes, train_data_list, test_data_list


def save_classes(classes, class_path):
    assert not os.path.exists(class_path), class_path
    np.savetxt(class_path, classes, fmt='%s', delimiter=' ', newline='\n', header='', )


def save_img_paths(img_path_list, data_path):
    assert not os.path.exists(data_path), data_path

    length = len(img_path_list)
    with open(data_path, 'w') as f:
        for idx, (img_path, target) in enumerate(img_path_list):
            if idx < (length - 1):
                f.write(f"{img_path}{KEY_SEP}{target}\n")
            else:
                f.write(f"{img_path}{KEY_SEP}{target}")


def process(data_root, classes, train_data_list, test_data_list):
    if not os.path.exists(data_root):
        os.makedirs(data_root)

    # classes
    cls_path = os.path.join(data_root, 'classes.txt')
    save_classes(classes, cls_path)

    # train
    print('process train')
    train_data_path = os.path.join(data_root, 'train.txt')
    save_img_paths(train_data_list, train_data_path)

    # test
    print("process test")
    test_data_path = os.path.join(data_root, 'test.txt')
    save_img_paths(test_data_list, test_data_path)


def main():
    data_root = 'CUB_200_2011/CUB_200_2011'
    if not os.path.exists(data_root):
        os.makedirs(data_root)
    classes, train_data_list, test_data_list = load_data(data_root)

    process('CUB_200_2011/', classes, train_data_list, test_data_list)


if __name__ == '__main__':
    main()
