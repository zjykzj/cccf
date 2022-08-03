# -*- coding: utf-8 -*-

"""
@date: 2022/4/28 下午10:08
@file: extract_cifar100.py
@author: zj
@description: Collate data and generate
1. classes.txt: class list
2. train.txt: train image path list
3. test.txt: test image path list
"""

import os

import numpy as np
from PIL.Image import Image
from tqdm import tqdm
from torchvision.datasets import cifar

from zcls2.config.key_word import KEY_SEP


def load_data(data_root):
    assert os.path.isdir(data_root), data_root

    train_dataset = cifar.CIFAR100(data_root, train=True, download=True)
    test_dataset = cifar.CIFAR100(data_root, train=False, download=True)

    return train_dataset, test_dataset


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


def process(data_root, train_dataset, test_dataset):
    assert isinstance(train_dataset, cifar.CIFAR100) and isinstance(test_dataset, cifar.CIFAR100)
    if not os.path.exists(data_root):
        os.makedirs(data_root)

    # classes
    cls_path = os.path.join(data_root, 'classes.txt')
    classes = train_dataset.classes
    save_classes(classes, cls_path)

    # train
    print('process train')
    train_data_root = os.path.join(data_root, 'train')
    if not os.path.exists(train_data_root):
        os.makedirs(train_data_root)

    train_data_dict = dict()
    train_data_list = list()
    for idx, (pil_img, target) in tqdm(enumerate(iter(train_dataset))):
        assert isinstance(pil_img, Image)
        class_name = classes[target]
        if class_name not in train_data_dict.keys():
            train_data_dict[class_name] = 0

        cls_dir = os.path.join(train_data_root, class_name)
        if not os.path.exists(cls_dir):
            os.makedirs(cls_dir)

        img_path = os.path.join(cls_dir, f'{idx}.jpg')
        pil_img.save(img_path)

        img_path = os.path.join("train", class_name, f'{idx}.jpg')
        train_data_list.append([img_path, target])

    train_data_path = os.path.join(data_root, 'train.txt')
    save_img_paths(train_data_list, train_data_path)

    # test
    print("process test")
    test_data_root = os.path.join(data_root, 'test')
    if not os.path.exists(test_data_root):
        os.makedirs(test_data_root)

    test_data_dict = dict()
    test_data_list = list()
    for idx, (pil_img, target) in tqdm(enumerate(iter(test_dataset))):
        assert isinstance(pil_img, Image)
        class_name = classes[target]
        if class_name not in test_data_dict.keys():
            test_data_dict[class_name] = 0

        cls_dir = os.path.join(test_data_root, class_name)
        if not os.path.exists(cls_dir):
            os.makedirs(cls_dir)

        img_path = os.path.join(cls_dir, f'{idx}.jpg')
        pil_img.save(img_path)

        img_path = os.path.join("test", class_name, f'{idx}.jpg')
        test_data_list.append([img_path, target])

    test_data_path = os.path.join(data_root, 'test.txt')
    save_img_paths(test_data_list, test_data_path)


def main():
    data_root = 'cifar100'
    if not os.path.exists(data_root):
        os.makedirs(data_root)
    train_dataset, test_dataset = load_data(data_root)

    process(data_root, train_dataset, test_dataset)


if __name__ == '__main__':
    main()
