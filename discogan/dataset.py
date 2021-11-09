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

import os
import cv2
import numpy as np
import pandas as pd
from paddle.io import Dataset

dataset_path = './datasets/'
celebA_path = os.path.join(dataset_path, 'celebA')
handbag_path = os.path.join(dataset_path, 'edges2handbags')
shoe_path = os.path.join(dataset_path, 'edges2shoes')
facescrub_path = os.path.join(dataset_path, 'facescrub')
chair_path = os.path.join(dataset_path, 'rendered_chairs')
face_3d_path = os.path.join(dataset_path, 'PublicMM1', '05_renderings')
face_real_path = os.path.join(dataset_path, 'real_face')
car_path = os.path.join(dataset_path, 'data', 'cars')


def read_attr_file(attr_path, image_dir):
    f = open(attr_path)
    lines = f.readlines()
    lines = list(map(lambda line: line.strip(), lines))
    columns = ['image_path'] + lines[1].split()
    lines = lines[2:]

    items = map(lambda line: line.split(), lines)
    df = pd.DataFrame(items, columns=columns)
    df['image_path'] = df['image_path'].map(lambda x: os.path.join(image_dir, x))

    return df


class CelebaDataset(Dataset):
    def __init__(self, img_dir, attr_file, image_size=64, style_A='Male', style_B=None,
                 constraint=None, constraint_type=None, test=False, n_test=200):
        super(CelebaDataset, self).__init__()
        self.img_dir = img_dir  # img_align_celeba 文件夹路径
        self.attr_file = attr_file  # list_attr_celeba.txt 路径
        self.image_size = image_size
        self.style_A = style_A
        self.style_B = style_B
        self.constraint = constraint
        self.constraint_type = constraint_type
        self.test = test
        self.n_test = n_test  # 测试集数量

        self.style_A_paths, self.style_B_paths = self.__get_celebA_files()

        self.style_A_imgs = self.__read_images(filenames=self.style_A_paths, image_size=self.image_size)
        self.style_B_imgs = self.__read_images(filenames=self.style_B_paths, image_size=self.image_size)

    def __len__(self):
        return min(len(self.style_A_paths), len(self.style_B_paths))

    def __getitem__(self, index):
        return self.style_A_imgs[index], self.style_B_imgs[index]

    def __read_images(self, filenames, image_size=64):
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

    def __get_celebA_files(self):
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
