# -*- coding: utf-8 -*-

"""
@date: 2022/8/3 下午9:16
@file: create_gallery_and_query_data.py
@author: zj
@description: Create feature set and query set images

× For feature sets, 20 pieces of each type are extracted from the training file
× For the query set, extract 5 pieces of each type from the test file
"""

import os
import random

import numpy as np
from zcls2.config.key_word import KEY_SEP


def load_data(data_path):
    assert os.path.isfile(data_path), data_path

    data_dict = dict()
    with open(data_path, 'r') as f:
        for line in f:
            if line.strip() == '':
                continue

            img_path, label = line.strip().split(KEY_SEP)
            if label not in data_dict.keys():
                data_dict[label] = list()
            data_dict[label].append(img_path)

    return data_dict


def save_data(data_list, data_path):
    assert not os.path.exists(data_path), data_path

    length = len(data_list)
    with open(data_path, 'w') as f:
        for idx, (img_path, key) in enumerate(data_list):
            if idx < (length - 1):
                f.write(f"{img_path}{KEY_SEP}{key}\n")
            else:
                f.write(f"{img_path}{KEY_SEP}{key}")


def main():
    random.seed(10)
    np.random.seed(10)

    train_path = 'train.txt'
    train_dict = load_data(train_path)

    sample_num = 20
    gallery_list = list()
    for key, value in train_dict.items():
        if len(value) <= sample_num:
            img_path_list = value
        else:
            img_path_list = np.random.choice(value, sample_num, replace=False)
        gallery_list.extend([[img_path, key] for img_path in img_path_list])

    gallery_path = 'gallery.txt'
    save_data(gallery_list, gallery_path)

    test_path = 'test.txt'
    test_dict = load_data(test_path)

    sample_num = 5
    query_list = list()
    for key, value in test_dict.items():
        if len(value) < sample_num:
            img_path_list = value
        else:
            img_path_list = np.random.choice(value, sample_num, replace=False)
        query_list.extend([[img_path, key] for img_path in img_path_list])

    # shuffle
    random.shuffle(query_list)

    query_path = 'query.txt'
    save_data(query_list, query_path)


if __name__ == '__main__':
    main()
