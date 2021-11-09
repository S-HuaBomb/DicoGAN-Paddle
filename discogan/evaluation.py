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
import argparse

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from paddle.io import DataLoader, BatchSampler

from dataset import CelebaDataset
from model import *
from PIL import Image

parser = argparse.ArgumentParser(description='Paddle implementation of DiscoGAN')
parser.add_argument('--task_name', type=str, default='facescrub', help='Set data name')
parser.add_argument('--model_arch', type=str, default='discogan',
                    help='choose among gan/recongan/discogan. gan - standard GAN, recongan - GAN with reconstruction, discogan - DiscoGAN.')

parser.add_argument('--batch_size', type=int, default=64, help='Set batch size')
parser.add_argument('--num_workers', type=int, default=4, help='dataloader num_workers')
parser.add_argument('--result_path', type=str, default='./results/',
                    help='Set the path the result images will be saved.')

parser.add_argument('--image_dir', type=str, default=None,
                    help='Path to img_align_celeba dir')
parser.add_argument('--attr_file', type=str, default=None,
                    help='Path to list_attr_celeba.txt')
parser.add_argument('--ckpt_path', type=str, default=None,
                    help='Path to generator model path')

parser.add_argument('--style_A', type=str, default='Male',
                    help='Style for CelebA dataset. Could be any attributes in celebA (Young, Male, Blond_Hair, Wearing_Hat ...)')
parser.add_argument('--style_B', type=str, default=None,
                    help='Style for CelebA dataset. Could be any attributes in celebA (Young, Male, Blond_Hair, Wearing_Hat ...)')
parser.add_argument('--constraint', type=str, default=None,
                    help='Constraint for celebA dataset. Only images satisfying this constraint is used. For example, if --constraint=Male, and --constraint_type=1, only male images are used for both style/domain.')
parser.add_argument('--constraint_type', type=str, default=None,
                    help='Used along with --constraint. If --constraint_type=1, only images satisfying the constraint are used. If --constraint_type=-1, only images not satisfying the constraint are used.')
parser.add_argument('--n_test', type=int, default=200, help='Number of test data.')

args = parser.parse_args()

paddle.set_device('gpu') if paddle.is_compiled_with_cuda() else paddle.set_device('cpu')


def get_data():
    celeba_test_set = CelebaDataset(img_dir=args.image_dir, attr_file=args.attr_file,
                                    style_A=args.style_A, style_B=args.style_B,
                                    constraint=args.constraint,
                                    constraint_type=args.constraint_type,
                                    test=True, n_test=args.n_test)

    test_batch_sampler = BatchSampler(dataset=celeba_test_set,
                                      batch_size=args.batch_size,
                                      shuffle=False, drop_last=False)
    test_loader = DataLoader(dataset=celeba_test_set,  # return test_A, test_B
                             batch_sampler=test_batch_sampler,
                             num_workers=args.num_workers)
    print(f"test set: {len(celeba_test_set)}, test loader: {len(test_loader)}")

    return test_loader


test_loader = get_data()


def main():
    global args

    result_path = os.path.join(args.result_path, args.task_name)
    if args.style_A:
        result_path = os.path.join(result_path, args.style_A)
    result_path = os.path.join(result_path, args.model_arch)

    test_loader = get_data()

    if not os.path.exists(result_path):
        os.makedirs(result_path, exist_ok=True)

    generator_A = Generator()
    generator_B = Generator()

    if args.ckpt_path:
        # 加载预训练模型
        discoGAN_ckpt = paddle.load(args.ckpt_path)
        generator_A.set_state_dict(discoGAN_ckpt['generator_A'])
        generator_B.set_state_dict(discoGAN_ckpt['generator_B'])

        print(f"load pretrained ckpt from {args.ckpt_path}")
    else:
        raise ValueError(f"need pretrained checkpoint to generate fake images")

    generator_A.eval()
    generator_B.eval()

    with paddle.no_grad():
        for test_A, test_B in test_loader:
            test_A = paddle.to_tensor(test_A, dtype=paddle.float32)
            test_B = paddle.to_tensor(test_B, dtype=paddle.float32)

            AB = generator_B(test_A)
            BA = generator_A(test_B)
            ABA = generator_A(AB)
            BAB = generator_B(BA)

            n_testset = min(10, test_A.shape[0], test_B.shape[0])

            subdir_path = result_path

            if os.path.exists(subdir_path):
                pass
            else:
                os.makedirs(subdir_path, exist_ok=True)

            for im_idx in range(n_testset):
                A_val = test_A[im_idx].numpy().transpose(1, 2, 0) * 255.
                B_val = test_B[im_idx].numpy().transpose(1, 2, 0) * 255.
                BA_val = BA[im_idx].numpy().transpose(1, 2, 0) * 255.
                ABA_val = ABA[im_idx].numpy().transpose(1, 2, 0) * 255.
                AB_val = AB[im_idx].numpy().transpose(1, 2, 0) * 255.
                BAB_val = BAB[im_idx].numpy().transpose(1, 2, 0) * 255.

                filename_prefix = os.path.join(subdir_path, str(im_idx))
                Image.fromarray(A_val.astype(np.uint8)[:, :, ::-1]).save(filename_prefix + '.A.jpg')
                Image.fromarray(B_val.astype(np.uint8)[:, :, ::-1]).save(filename_prefix + '.B.jpg')
                Image.fromarray(BA_val.astype(np.uint8)[:, :, ::-1]).save(filename_prefix + '.BA.jpg')
                Image.fromarray(AB_val.astype(np.uint8)[:, :, ::-1]).save(filename_prefix + '.AB.jpg')
                Image.fromarray(ABA_val.astype(np.uint8)[:, :, ::-1]).save(filename_prefix + '.ABA.jpg')
                Image.fromarray(BAB_val.astype(np.uint8)[:, :, ::-1]).save(filename_prefix + '.BAB.jpg')
        print(f"Test images saved to {subdir_path}")


if __name__ == '__main__':
    main()
