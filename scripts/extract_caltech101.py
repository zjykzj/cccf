# -*- coding: utf-8 -*-

"""
@date: 2022/4/28 下午10:08
@file: extract_caltech101.py
@author: zj
@description: Collate data and generate
1. classes.txt: class list
2. train.txt: train image path list
3. test.txt: test image path list
"""

import os
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from zcls2.config.key_word import KEY_SEP


def load_data(data_root):
    assert os.path.isdir(data_root), data_root

    classes = os.listdir(data_root)
    class_list = list()
    for class_name in classes:
        if 'BACKGROUND_Google' == class_name:
            continue
        else:
            class_list.append(class_name)

    data_list = list()
    p = Path(data_root)
    for path in tqdm(p.rglob('*.jpg')):
        img_path = str(path).strip()[12:]
        if 'BACKGROUND_Google' in img_path:
            continue
        data_list.append(img_path)
    return data_list, class_list


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


def process(data_root, classes, data_list):
    if not os.path.exists(data_root):
        os.makedirs(data_root)

    # classes
    cls_path = os.path.join(data_root, 'classes.txt')
    save_classes(classes, cls_path)

    # train/test split
    train_dataset, test_dataset = train_test_split(data_list, test_size=0.2, random_state=1, shuffle=True)

    # train
    print('process train')
    train_data_list = list()
    for idx, img_path in tqdm(enumerate(train_dataset)):
        img_dir = os.path.split(img_path)[0]
        class_name = os.path.split(img_dir)[1]
        target = classes.index(class_name)

        train_data_list.append([img_path, target])

    train_data_path = os.path.join(data_root, 'train.txt')
    save_img_paths(train_data_list, train_data_path)

    # test
    print("process test")
    test_data_list = list()
    for idx, img_path in tqdm(enumerate(test_dataset)):
        img_dir = os.path.split(img_path)[0]
        class_name = os.path.split(img_dir)[1]
        target = classes.index(class_name)

        test_data_list.append([img_path, target])

    test_data_path = os.path.join(data_root, 'test.txt')
    save_img_paths(test_data_list, test_data_path)


def main():
    data_root = 'caltech-101/caltech-101/101_ObjectCategories'
    if not os.path.exists(data_root):
        os.makedirs(data_root)
    data_list, classes = load_data(data_root)

    process('caltech-101/', classes, data_list)


if __name__ == '__main__':
    main()
