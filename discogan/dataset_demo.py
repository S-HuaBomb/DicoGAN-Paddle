# encoding=utf8
# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from paddle.io import DataLoader

import numpy as np
import cv2
import pandas as pd
import sys
sys.path.append(r'D:\code_sources\from_github\paddlepaddle\14s\DiscoGAN-Paddle\discogan')
from dataset import CelebaDataset


def dataset_demo():
    img_dir = r'D:\code_sources\from_github\paddlepaddle\14s\DiscoGAN-Paddle\datasets\celeba\celeba_demo'
    attr_file = r'D:\code_sources\from_github\paddlepaddle\14s\DiscoGAN-Paddle\datasets\celeba\list_attr_celeba_demo.txt'
    test_set = CelebaDataset(img_dir=img_dir, attr_file=attr_file, test=True, n_test=4)
    train_set = CelebaDataset(img_dir=img_dir, attr_file=attr_file, test=False, n_test=4)

    print(f"testset len: {len(test_set)}; trainset len: {len(train_set)}")
    test_loader = DataLoader(test_set, batch_size=2)
    train_loader = DataLoader(train_set, batch_size=2)
    print(f"testloader len: {len(test_loader)}; trainloader len: {len(train_loader)}")

    for (A, B) in test_loader:
        print(A.shape, B.shape)

    for (A, B) in train_loader:
        print(A.shape, B.shape)


def read_attr_file(attr_path, image_dir):
    """
    pandas 读取 attr_list.txt
    """
    f = open(attr_path)
    lines = f.readlines()
    lines = list(map(lambda line: line.strip(), lines))
    columns = ['image_path'] + lines[1].split()
    lines = lines[2:]

    items = map(lambda line: line.split(), lines)
    df = pd.DataFrame(items, columns=columns)
    df['image_path'] = df['image_path'].map(lambda x: os.path.join(image_dir, x))

    return df

def get_celebA_files(self):
    """
    返回每个图片的绝对路径
    """
    image_data = read_attr_file(self.attr_file, self.img_dir)

    if self.constraint:
        image_data = image_data[image_data[self.constraint] == self.constraint_type]

    style_A_data = image_data[image_data[self.style_A] == '1']['image_path'].values
    if self.style_B:
        style_B_data = image_data[image_data[self.style_B] == '1']['image_path'].values
    else:
        style_B_data = image_data[image_data[self.style_A] == '-1']['image_path'].values

    if self.test == False:
        return style_A_data[:-self.n_test], style_B_data[:-self.n_test]
    if self.test == True:
        return style_A_data[-self.n_test:], style_B_data[-self.n_test:]

def read_images(self, filenames, image_size=64):
    """
    遍历上面函数返回的data_path，读取图片
    """
    images = []
    for fn in filenames:
        image = cv2.imread(fn)
        if image is None:
            continue

        image = cv2.resize(image, (image_size, image_size))
        image = image.astype(np.float32) / 255.
        image = image.transpose(2, 0, 1)
        images.append(image)

    images = np.stack(images)
    return images


if __name__ == '__main__':
    dataset_demo()
