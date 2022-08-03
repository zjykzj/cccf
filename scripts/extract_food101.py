# -*- coding: utf-8 -*-

"""
@date: 2022/4/28 下午9:34
@file: extract_fod101.py
@author: zj
@description: Collate data and generate
1. classes.txt: class list
2. train.txt: train image path list
3. test.txt: test image path list
"""

import os

import numpy as np
from tqdm import tqdm
from zcls2.config.key_word import KEY_SEP


def load_data(data_root):
    assert os.path.isdir(data_root), data_root

    cls_path = os.path.join(data_root, "meta/classes.txt")
    classes = np.loadtxt(cls_path, dtype=str, delimiter=' ')

    train_path = os.path.join(data_root, 'meta/train.txt')
    tmp_train_list = np.loadtxt(train_path, dtype=str, delimiter=' ')
    train_list = [os.path.join("images", img_path + ".jpg") for img_path in tmp_train_list]

    test_path = os.path.join(data_root, 'meta/test.txt')
    tmp_test_list = np.loadtxt(test_path, dtype=str, delimiter=' ')
    test_list = [os.path.join("images", img_path + ".jpg") for img_path in tmp_test_list]

    return list(classes), train_list, test_list


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


def process(data_root, classes, train_list, test_list):
    if not os.path.exists(data_root):
        os.makedirs(data_root)

    # classes
    cls_path = os.path.join(data_root, 'classes.txt')
    save_classes(classes, cls_path)

    # train
    print('process train')
    train_data_list = list()
    for idx, img_path in tqdm(enumerate(train_list)):
        img_dir = os.path.split(img_path)[0]
        class_name = os.path.split(img_dir)[1]
        target = classes.index(class_name)

        train_data_list.append([img_path, target])
    train_data_path = os.path.join(data_root, 'train.txt')
    save_img_paths(train_data_list, train_data_path)

    # test
    print("process test")
    test_data_list = list()
    for idx, img_path in tqdm(enumerate(test_list)):
        img_dir = os.path.split(img_path)[0]
        class_name = os.path.split(img_dir)[1]
        target = classes.index(class_name)

        test_data_list.append([img_path, target])
    test_data_path = os.path.join(data_root, 'test.txt')
    save_img_paths(test_data_list, test_data_path)


def main():
    data_root = 'food-101/'
    if not os.path.exists(data_root):
        os.makedirs(data_root)
    classes, train_list, test_list = load_data(data_root)

    process('food-101/', classes, train_list, test_list)


if __name__ == '__main__':
    main()
