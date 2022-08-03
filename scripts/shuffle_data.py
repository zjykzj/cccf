# -*- coding: utf-8 -*-

"""
@date: 2022/5/30 下午8:29
@file: shuffle_data.py
@author: zj
@description: Given the data file, disturb the data path
"""

import os
import random


def load_txt(txt_path):
    assert os.path.isfile(txt_path), txt_path

    line_list = list()
    with open(txt_path, 'r') as f:
        for line in f:
            if line.strip() != '':
                line_list.append(line.strip())

    return line_list


def save_txt(line_list, dst_path):
    assert not os.path.isfile(dst_path), dst_path

    length = len(line_list)
    with open(dst_path, 'w') as f:
        for idx, line in enumerate(line_list):
            if idx < (length - 1):
                f.write(f'{line}\n')
            else:
                f.write(line)


def main():
    src_path = 'test.txt'
    line_list = load_txt(src_path)

    # shuffle
    random.shuffle(line_list)

    rename_path = 'test_unsorted.txt'
    print(f'rename {src_path} to {rename_path}')
    os.rename(src_path, rename_path)

    dst_path = 'test.txt'
    print(f'save to {dst_path}')
    save_txt(line_list, dst_path)


if __name__ == '__main__':
    main()
