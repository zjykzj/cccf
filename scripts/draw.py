# -*- coding: utf-8 -*-

"""
@date: 2022/8/3 下午9:16
@file: draw.py
@author: zj
@description: 
"""
import random
from typing import List

import os
import cv2

import numpy as np
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt


def load_data(data_root: str) -> List[str]:
    assert os.path.isdir(data_root), data_root

    data_list = list()
    p = Path(data_root)
    for path in tqdm(p.rglob('*.jpg')):
        data_list.append(str(path))
    return data_list


def draw(img_list: List[str], draw_path='./img.png'):
    plt.figure(figsize=(20, 20))  # 设置窗口大小

    col_num = 20
    row_num = 20
    for i in range(row_num):
        for j in range(col_num):
            img_idx = i * row_num + j

            img_path = img_list[img_idx]
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            plt.subplot(row_num, col_num, img_idx + 1)
            plt.imshow(img)
            plt.axis('off')

    plt.savefig(draw_path, bbox_inches='tight', pad_inches=0.05)
    plt.show()


def main():
    random.seed(20)
    np.random.seed(20)

    data_root = './'
    data_list = load_data(data_root)

    img_list = np.random.choice(data_list, 400, replace=False)
    draw(img_list)


if __name__ == '__main__':
    main()
