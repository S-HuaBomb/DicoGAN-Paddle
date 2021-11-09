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
from itertools import chain
import logging

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import paddle.optimizer as optim
from paddle import distributed as dist
from paddle.io import DataLoader, DistributedBatchSampler, BatchSampler

from loss_fn import HingeEmbeddingLoss
from dataset import CelebaDataset
from model import *
from PIL import Image


parser = argparse.ArgumentParser(description='Paddle implementation of DiscoGAN')
parser.add_argument('--task_name', type=str, default='facescrub', help='Set data name')
parser.add_argument('--epoch_size', type=int, default=5000, help='Set epoch size')
parser.add_argument('--batch_size', type=int, default=200, help='Set batch size')
parser.add_argument('--num_workers', type=int, default=4, help='dataloader num_workers')
parser.add_argument('--learning_rate', type=float, default=0.0002, help='Set learning rate for optimizer')
parser.add_argument('--result_path', type=str, default='./results/',
                    help='Set the path the result images will be saved.')
parser.add_argument('--model_path', type=str, default='./models/', help='Set the path for trained models')
parser.add_argument('--model_arch', type=str, default='discogan',
                    help='choose among gan/recongan/discogan. gan - standard GAN, recongan - GAN with reconstruction, discogan - DiscoGAN.')
parser.add_argument('--image_size', type=int, default=64, help='Image size. 64 for every experiment in the paper')

parser.add_argument('--gan_curriculum', type=int, default=10000,
                    help='Strong GAN loss for certain period at the beginning')
parser.add_argument('--iters', type=int, default=0,
                    help='iters you have run')
parser.add_argument('--starting_rate', type=float, default=0.01,
                    help='Set the lambda weight between GAN loss and Recon loss during curriculum period at the beginning. We used the 0.01 weight.')
parser.add_argument('--default_rate', type=float, default=0.5,
                    help='Set the lambda weight between GAN loss and Recon loss after curriculum period. We used the 0.5 weight.')

parser.add_argument('--image_dir', type=str, default=None,
                    help='Path to img_align_celeba dir')
parser.add_argument('--attr_file', type=str, default=None,
                    help='Path to list_attr_celeba.txt')
parser.add_argument('--ckpt_path', type=str, default=None,
                    help='Path to generator model path')
parser.add_argument('--data_path', type=str, default=None,
                    help='Path to CelebA dataset parent root')
parser.add_argument('--style_A', type=str, default='Male',
                    help='Style for CelebA dataset. Could be any attributes in celebA (Young, Male, Blond_Hair, Wearing_Hat ...)')
parser.add_argument('--style_B', type=str, default=None,
                    help='Style for CelebA dataset. Could be any attributes in celebA (Young, Male, Blond_Hair, Wearing_Hat ...)')
parser.add_argument('--constraint', type=str, default=None,
                    help='Constraint for celebA dataset. Only images satisfying this constraint is used. For example, if --constraint=Male, and --constraint_type=1, only male images are used for both style/domain.')
parser.add_argument('--constraint_type', type=str, default=None,
                    help='Used along with --constraint. If --constraint_type=1, only images satisfying the constraint are used. If --constraint_type=-1, only images not satisfying the constraint are used.')
parser.add_argument('--n_test', type=int, default=200, help='Number of test data.')

parser.add_argument('--update_interval', type=int, default=3, help='')
parser.add_argument('--log_interval', type=int, default=50, help='Print loss values every log_interval iterations.')
parser.add_argument('--image_save_interval', type=int, default=1000,
                    help='Save test results every image_save_interval iterations.')
parser.add_argument('--model_save_interval', type=int, default=10000,
                    help='Save models every model_save_interval iterations.')

parser.add_argument('--local_rank', type=int, default=-1,
                    help='setup the master gpu, -1 means use single gpu or cpu')
parser.add_argument('--log_out', type=str, default='./logs',
                    help='setup the master gpu, -1 means use single gpu or cpu')

paddle.set_device('gpu') if paddle.is_compiled_with_cuda() else paddle.set_device('cpu')


def get_logger(args, name=__name__, verbosity=2):
    log_levels = {
        0: logging.WARNING,
        1: logging.INFO,
        2: logging.DEBUG
    }
    logging.basicConfig(format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                        datefmt="%m/%d/%Y %H:%M:%S",
                        level=log_levels[verbosity] if args.local_rank in [-1, 0] else logging.INFO,
                        filename=f'{args.log_out}/train_{args.style_A}.log',
                        filemode='a')

    msg_verbosity = 'verbosity option {} is invalid. Valid options are {}.'.format(verbosity,
                                                                                   log_levels.keys())
    assert verbosity in log_levels, msg_verbosity
    logger = logging.getLogger(name)
    chlr = logging.StreamHandler()  # 输出到控制台的handler
    logger.addHandler(chlr)
    return logger


def as_np(data):
    return data.numpy()


def get_data():
    if args.local_rank == -1:
        batch_sampler = BatchSampler
    else:
        batch_sampler = DistributedBatchSampler

    celeba_train_set = CelebaDataset(img_dir=args.image_dir, attr_file=args.attr_file,
                                     style_A=args.style_A, style_B=args.style_B,
                                     constraint=args.constraint,
                                     constraint_type=args.constraint_type,
                                     test=False, n_test=args.n_test)
    celeba_test_set = CelebaDataset(img_dir=args.image_dir, attr_file=args.attr_file,
                                    style_A=args.style_A, style_B=args.style_B,
                                    constraint=args.constraint,
                                    constraint_type=args.constraint_type,
                                    test=True, n_test=args.n_test)
    print(f"celeba_train_set len: {len(celeba_train_set)}")

    train_batch_sampler = batch_sampler(dataset=celeba_train_set,
                                        batch_size=args.batch_size,
                                        shuffle=True, drop_last=False)
    train_loader = DataLoader(dataset=celeba_train_set,  # return data_A, data_B
                              batch_sampler=train_batch_sampler,
                              num_workers=args.num_workers)

    test_batch_sampler = batch_sampler(dataset=celeba_test_set,
                                       batch_size=args.batch_size,
                                       shuffle=True, drop_last=False)
    test_loader = DataLoader(dataset=celeba_test_set,  # return test_A, test_B
                             batch_sampler=test_batch_sampler,
                             num_workers=args.num_workers)

    return train_loader, test_loader


def get_fm_loss(real_feats, fake_feats, criterion):
    losses = 0
    for real_feat, fake_feat in zip(real_feats, fake_feats):
        l2 = (real_feat.mean(0) - fake_feat.mean(0)) * (real_feat.mean(0) - fake_feat.mean(0))
        loss = criterion(l2, paddle.ones(l2.shape))
        losses += loss

    return losses


def get_gan_loss(dis_real, dis_fake, criterion):
    labels_dis_real = paddle.ones([dis_real.shape[0], 1])
    labels_dis_fake = paddle.zeros([dis_fake.shape[0], 1])
    labels_gen = paddle.ones([dis_fake.shape[0], 1])

    dis_real = dis_real.reshape([dis_real.shape[0], 1])
    dis_fake = dis_fake.reshape([dis_fake.shape[0], 1])
    dis_loss = criterion(dis_real, labels_dis_real) * 0.5 + criterion(dis_fake, labels_dis_fake) * 0.5
    gen_loss = criterion(dis_fake, labels_gen)

    return dis_loss, gen_loss


def main():
    global args, local_master, logger
    args = parser.parse_args()
    local_master = (args.local_rank == -1 or dist.get_rank() == 0)

    if args.local_rank != -1:
        dist.init_parallel_env()

    epoch_size = args.epoch_size
    batch_size = args.batch_size

    result_path = os.path.join(args.result_path, args.task_name)
    if args.style_A:
        result_path = os.path.join(result_path, args.style_A)
    result_path = os.path.join(result_path, args.model_arch)

    model_path = os.path.join(args.model_path, args.task_name)
    if args.style_A:
        model_path = os.path.join(model_path, args.style_A)
    model_path = os.path.join(model_path, args.model_arch)

    train_loader, test_loader = get_data()

    if local_master:
        # local master create dir
        if not os.path.exists(result_path):
            os.makedirs(result_path, exist_ok=True)
        if not os.path.exists(model_path):
            os.makedirs(model_path, exist_ok=True)
        if not os.path.exists(args.log_out):
            os.makedirs(args.log_out, exist_ok=True)

    logger = get_logger(args) if local_master else None

    if local_master:
        logger.info(f"num of iters: {len(train_loader)}; len of test_loader: {len(test_loader)}")

    logger.info(
        f'[Process {os.getpid()}] world_size = {dist.get_world_size()}, '
        + f'rank = {dist.get_rank()}'
    ) if local_master else None
    print(
        f'[Process {os.getpid()}] world_size = {dist.get_world_size()}, '
        + f'rank = {dist.get_rank()}'
    )

    generator_A = Generator()
    generator_B = Generator()
    discriminator_A = Discriminator()
    discriminator_B = Discriminator()

    if args.local_rank != -1:
        # 第3处改动，增加paddle.DataParallel封装
        generator_A = paddle.DataParallel(generator_A, find_unused_parameters=True)
        generator_B = paddle.DataParallel(generator_B, find_unused_parameters=True)
        discriminator_A = paddle.DataParallel(discriminator_A, find_unused_parameters=True)
        discriminator_B = paddle.DataParallel(discriminator_B, find_unused_parameters=True)

    if args.ckpt_path:
        # 加载预训练模型
        discoGAN_ckpt = paddle.load(args.ckpt_path)
        generator_A.set_state_dict(discoGAN_ckpt['generator_A'])
        generator_B.set_state_dict(discoGAN_ckpt['generator_B'])
        discriminator_A.set_state_dict(discoGAN_ckpt['discriminator_A'])
        discriminator_B.set_state_dict(discoGAN_ckpt['discriminator_B'])
        logger.info(f"resume ckpt from {args.ckpt_path}") if local_master else None

    generator_A.train()
    generator_B.train()
    discriminator_A.train()
    discriminator_B.train()

    data_size = len(train_loader)
    n_batches = (data_size // batch_size)

    recon_criterion = nn.MSELoss()
    gan_criterion = nn.BCELoss()
    feat_criterion = HingeEmbeddingLoss()

    gen_params = chain(generator_A.parameters(), generator_B.parameters())
    dis_params = chain(discriminator_A.parameters(), discriminator_B.parameters())

    optim_gen = optim.Adam(parameters=gen_params,
                           learning_rate=args.learning_rate,
                           beta1=0.5, beta2=0.999,
                           weight_decay=0.00001)
    optim_dis = optim.Adam(parameters=dis_params,
                           learning_rate=args.learning_rate,
                           beta1=0.5, beta2=0.999,
                           weight_decay=0.00001)

    iters = args.iters

    for epoch in range(epoch_size):

        for data_style_A, data_style_B in train_loader:
            A = paddle.to_tensor(data_style_A, dtype=paddle.float32)
            B = paddle.to_tensor(data_style_B, dtype=paddle.float32)

            AB = generator_B(A)
            BA = generator_A(B)

            ABA = generator_A(AB)
            BAB = generator_B(BA)

            # Reconstruction Loss
            recon_loss_A = recon_criterion(ABA, A)
            recon_loss_B = recon_criterion(BAB, B)

            # Real/Fake GAN Loss (A)
            A_dis_real, A_feats_real = discriminator_A(A)
            A_dis_fake, A_feats_fake = discriminator_A(BA)

            dis_loss_A, gen_loss_A = get_gan_loss(A_dis_real, A_dis_fake, gan_criterion)
            fm_loss_A = get_fm_loss(A_feats_real, A_feats_fake, feat_criterion)

            # Real/Fake GAN Loss (B)
            B_dis_real, B_feats_real = discriminator_B(B)
            B_dis_fake, B_feats_fake = discriminator_B(AB)

            dis_loss_B, gen_loss_B = get_gan_loss(B_dis_real, B_dis_fake, gan_criterion)
            fm_loss_B = get_fm_loss(B_feats_real, B_feats_fake, feat_criterion)

            # Total Loss

            if iters < args.gan_curriculum:
                rate = args.starting_rate
            else:
                rate = args.default_rate

            gen_loss_A_total = (gen_loss_B * 0.1 + fm_loss_B * 0.9) * (1. - rate) + recon_loss_A * rate
            gen_loss_B_total = (gen_loss_A * 0.1 + fm_loss_A * 0.9) * (1. - rate) + recon_loss_B * rate

            if args.model_arch == 'discogan':
                gen_loss = gen_loss_A_total + gen_loss_B_total
                dis_loss = dis_loss_A + dis_loss_B
            elif args.model_arch == 'recongan':
                gen_loss = gen_loss_A_total
                dis_loss = dis_loss_B
            elif args.model_arch == 'gan':
                gen_loss = (gen_loss_B * 0.1 + fm_loss_B * 0.9)
                dis_loss = dis_loss_B

            if iters % args.update_interval == 0:
                dis_loss.backward()
                optim_dis.step()
            else:
                gen_loss.backward()
                optim_gen.step()

            optim_dis.clear_grad()
            optim_gen.clear_grad()

            if iters % args.log_interval == 0 and local_master:
                logger.info(f"Epoch: {epoch} - Iter: {iters} - Total GEN Loss: {round(gen_loss.item(), 7)} - "
                            f"Total DIS Loss: {round(dis_loss.item(), 7)} - "
                            f"GEN Loss: {as_np(gen_loss_A.mean())}, {as_np(gen_loss_B.mean())} - "
                            f"Feature Matching Loss: {as_np(fm_loss_A.mean())}, {as_np(fm_loss_B.mean())} - "
                            f"RECON Loss: {as_np(recon_loss_A.mean())}, {as_np(recon_loss_B.mean())} - "
                            f"DIS Loss: {as_np(dis_loss_A.mean())}, {as_np(dis_loss_B.mean())} \n")

            if (iters + 1) % args.image_save_interval == 0 and local_master:
                with paddle.no_grad():
                    for test_A, test_B in test_loader:
                        test_A = paddle.to_tensor(test_A, dtype=paddle.float32)
                        test_B = paddle.to_tensor(test_B, dtype=paddle.float32)

                        AB = generator_B(test_A)
                        BA = generator_A(test_B)
                        ABA = generator_A(AB)
                        BAB = generator_B(BA)

                        n_testset = min(10, test_A.shape[0], test_B.shape[0])

                        subdir_path = os.path.join(result_path, str(iters / args.image_save_interval))

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
                    logger.info(f"Test images saved to {subdir_path}")

            if (iters + 1) % args.model_save_interval == 0 and local_master:
                total_model_state_dict = {
                    'generator_A': generator_A.state_dict(),
                    'generator_B': generator_B.state_dict(),
                    'discriminator_A': discriminator_A.state_dict(),
                    'discriminator_B': discriminator_B.state_dict()
                }
                paddle.save(total_model_state_dict, os.path.join(model_path,
                                                                 'discoGAN' + str(
                                                                     iters / args.model_save_interval) + '.pdparams'))
                logger.info(f"discoGAN model saved to {model_path} \n")

            iters += 1


if __name__ == "__main__":
    main()

