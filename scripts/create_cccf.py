# -*- coding: utf-8 -*-

"""
@date: 2022/4/28 下午9:17
@file: create_cccf.py
@author: zj
@description: 
"""
from typing import List
import os

import numpy as np
from tqdm import tqdm
from zcls2.config.key_word import KEY_SEP


def load_data(data_root):
    assert os.path.isdir(data_root), data_root

    class_path = os.path.join(data_root, 'classes.txt')
    classes = np.loadtxt(class_path, dtype=str, delimiter=' ')

    train_list = list()
    train_path = os.path.join(data_root, 'train.txt')
    with open(train_path, 'r') as f:
        for line in f:
            tmp_list = line.strip().split(KEY_SEP)
            train_list.append(tmp_list)

    test_list = list()
    test_path = os.path.join(data_root, 'test.txt')
    with open(test_path, 'r') as f:
        for line in f:
            tmp_list = line.strip().split(KEY_SEP)
            test_list.append(tmp_list)

    return classes, train_list, test_list


def process(data_root, dst_classes: List, dst_train_list: List, dst_test_list: List):
    classes, train_list, test_list = load_data(data_root)

    current_class_num = len(dst_classes)
    dst_classes.extend(classes)

    for item in tqdm(train_list):
        img_path, target = item
        dst_train_list.append([os.path.join(data_root, img_path), int(target) + current_class_num])

    for item in tqdm(test_list):
        img_path, target = item
        dst_test_list.append([os.path.join(data_root, img_path), int(target) + current_class_num])


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


def main():
    cifar100_dir = 'cifar100'
    caltech101_dir = 'caltech-101'
    cub200_dir = 'CUB_200_2011'
    food101_dir = 'food-101'

    print('process ...')
    classes = list()
    train_list = list()
    test_list = list()
    process(cifar100_dir, classes, train_list, test_list)
    process(caltech101_dir, classes, train_list, test_list)
    process(cub200_dir, classes, train_list, test_list)
    process(food101_dir, classes, train_list, test_list)

    print('save ...')
    dst_classes_path = './classes.txt'
    save_classes(classes, dst_classes_path)
    dst_train_path = './train.txt'
    save_img_paths(train_list, dst_train_path)
    dst_test_path = './test.txt'
    save_img_paths(test_list, dst_test_path)


if __name__ == '__main__':
    main()
