# -*- coding: utf-8 -*-
"""
DeepJSCC Training.

Features:
1. CBR can be specified from the command line, e.g., 1/192, 1/96, 1/64, 1/48, 1/32.
2. SNR is randomly selected from snr_list for each training batch.
3. Validation evaluates every SNR in snr_list.
4. Compatible with the DeepJSCC encoder whose output is [B, 2c, H/8, W/8].
"""

import os
import time
import random
import argparse
import numpy as np
import torch
import torch.optim as optim

from fractions import Fraction
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

from DeepJSCC_model import DeepJSCC, ratio2filtersize
from Datasets.dataset_H5 import H5Dataset


def image_normalization(norm_type):
    def _inner(tensor: torch.Tensor):
        if norm_type == 'normalization':
            return tensor / 255.0
        elif norm_type == 'denormalization':
            return tensor * 255.0
        else:
            raise Exception('Unknown type of normalization')
    return _inner


def parse_ratio(value):
    try:
        return float(Fraction(value))
    except Exception:
        raise argparse.ArgumentTypeError(f'Invalid CBR ratio: {value}')


def ratio_to_string(ratio):
    fraction = Fraction(ratio).limit_denominator(10000)
    return f'{fraction.numerator}_{fraction.denominator}'


def config_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--project_root_dir',
        type=str,
        default='/home/czx/code_JSCC/Semantic_diffusion'
    )

    train_dataset = '/home/czx/data_sdb/ImgData/codec_datasets/dataset_train.hdf5'
    test_dataset = '/home/czx/code_JSCC/Semantic_diffusion/Datasets/Kodak.hdf5'

    parser.add_argument('--train_dataset', default=train_dataset, type=str, help='training dataset')
    parser.add_argument('--test_dataset', default=test_dataset, type=str, help='validation dataset')
    parser.add_argument('--dataloader_num_workers', default=8, type=int)
    parser.add_argument('--train_patch_size', default=512, type=int)

    parser.add_argument('--channel_type', default='AWGN', type=str, choices=['AWGN', 'Rayleigh'], help='channel type')

    parser.add_argument('--snr_list', nargs='+', type=float, default=[0, 2, 4, 6, 8, 10], help='candidate SNR values in dB')

    parser.add_argument('--CBR_ratio', default=parse_ratio('1/192'), type=parse_ratio, help='channel bandwidth ratio, e.g. 1/192, 1/96, 1/64, 1/48, 1/32')

    parser.add_argument('--batch_size', default=128, type=int, help='training batch size')
    parser.add_argument('--train_epochs', default=1000, type=int, help='number of training epochs')
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--weight_decay', default=5e-4, type=float, help='weight decay')
    parser.add_argument('--step_size', default=100, type=int, help='StepLR step size')
    parser.add_argument('--save_freq', default=100, type=int, help='checkpoint saving frequency')

    parser.add_argument('--gpu', default=0, type=int, help='GPU ID, -1 for CPU')
    parser.add_argument('--seed', default=42, type=int, help='random seed')
    parser.add_argument('--disable_tqdm', action='store_true')

    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main_train_run():
    args = config_args()
    set_seed(args.seed)

    if args.train_patch_size % 8 != 0:
        raise ValueError(
            f'train_patch_size must be divisible by 8 for the current DeepJSCC architecture, '
            f'but got {args.train_patch_size}.'
        )

    device = torch.device(
        f'cuda:{args.gpu}'
        if torch.cuda.is_available() and args.gpu != -1
        else 'cpu'
    )

    cbr_fraction = Fraction(args.CBR_ratio).limit_denominator(10000)
    cbr_string = f'{cbr_fraction.numerator}/{cbr_fraction.denominator}'
    cbr_path_string = ratio_to_string(args.CBR_ratio)

    print(f'Device: {device}')
    print(f'Channel: {args.channel_type}')
    print(f'Requested CBR: {cbr_string} = {args.CBR_ratio:.8f}')
    print(f'SNR list: {args.snr_list}')

    log_path = os.path.join(
        args.project_root_dir,
        'checkpoints',
        'DeepJSCC',
        f'DeepJSCC_{args.channel_type}_CBR_{cbr_path_string}'
    )

    os.makedirs(log_path, exist_ok=True)

    print(f'Checkpoint path: {log_path}')

    train_dataset = H5Dataset(
        args.train_dataset,
        transform=transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomCrop((args.train_patch_size, args.train_patch_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor()
        ])
    )

    test_dataset = H5Dataset(
        args.test_dataset,
        transform=transforms.Compose([
            transforms.ToTensor()
        ])
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.dataloader_num_workers,
        shuffle=True,
        pin_memory=True,
        drop_last=True
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=1,
        num_workers=args.dataloader_num_workers,
        shuffle=False,
        pin_memory=True
    )

    image_first = train_dataset[0]

    c = ratio2filtersize(image_first, args.CBR_ratio)

    with torch.no_grad():
        temp_model = DeepJSCC(c=c, channel_type=args.channel_type)
        temp_model.eval()
        temp_input = image_first.unsqueeze(0)
        temp_latent = temp_model.encoder(temp_input)

    source_symbols = image_first.numel()
    real_latent_symbols = temp_latent[0].numel()
    complex_channel_symbols = real_latent_symbols / 2
    actual_cbr = complex_channel_symbols / source_symbols

    actual_cbr_fraction = Fraction(actual_cbr).limit_denominator(10000)

    print(f'Input image size: {tuple(image_first.shape)}')
    print(f'Latent parameter c: {c}')
    print(f'Encoder output channels: {2 * c}')
    print(f'Encoder output size: {tuple(temp_latent.shape)}')
    print(f'Number of real source samples: {source_symbols}')
    print(f'Number of complex channel symbols: {int(complex_channel_symbols)}')
    print(
        f'Actual CBR: {actual_cbr_fraction.numerator}/{actual_cbr_fraction.denominator} '
        f'= {actual_cbr:.8f}'
    )

    del temp_model
    del temp_latent

    if abs(actual_cbr - args.CBR_ratio) > 1e-8:
        print(
            f'Warning: requested CBR={args.CBR_ratio:.8f}, '
            f'but the implemented CBR is {actual_cbr:.8f}.'
        )

    model = DeepJSCC(c=c, channel_type=args.channel_type)
    model = model.to(device)

    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=args.step_size,
        gamma=0.5
    )

    epoch_train_losses = []
    epoch_val_losses = []
    epoch_val_losses_per_snr = []
    per_epoch_time = []

    t0 = time.time()

    with tqdm(range(args.train_epochs), disable=args.disable_tqdm) as progress_bar:
        for epoch in progress_bar:
            epoch_start = time.time()
            progress_bar.set_description(f'Epoch {epoch + 1}')

            model.train()
            train_loss_sum = 0.0
            snr_counter = {snr: 0 for snr in args.snr_list}

            for image_batch in train_dataloader:
                image_batch = image_batch.to(device, non_blocking=True)

                train_snr = random.choice(args.snr_list)
                snr_counter[train_snr] += 1

                optimizer.zero_grad()

                outputs = model(image_batch, snr=train_snr)

                outputs_denorm = image_normalization('denormalization')(outputs)
                images_denorm = image_normalization('denormalization')(image_batch)

                loss = model.loss(outputs_denorm, images_denorm)

                loss.backward()
                optimizer.step()

                train_loss_sum += loss.detach().item()

            epoch_train_loss = train_loss_sum / len(train_dataloader)
            epoch_train_losses.append(epoch_train_loss)

            model.eval()
            val_loss_per_snr = {}

            with torch.no_grad():
                for val_snr in args.snr_list:
                    snr_loss_sum = 0.0

                    for images in test_dataloader:
                        images = images.to(device, non_blocking=True)

                        outputs = model(images, snr=val_snr)

                        outputs_denorm = image_normalization('denormalization')(outputs)
                        images_denorm = image_normalization('denormalization')(images)

                        loss = model.loss(outputs_denorm, images_denorm)
                        snr_loss_sum += loss.detach().item()

                    val_loss_per_snr[val_snr] = snr_loss_sum / len(test_dataloader)

            epoch_val_loss = np.mean(list(val_loss_per_snr.values()))

            epoch_val_losses.append(epoch_val_loss)
            epoch_val_losses_per_snr.append(val_loss_per_snr)

            scheduler.step()

            epoch_time = time.time() - epoch_start
            per_epoch_time.append(epoch_time)

            progress_bar.set_postfix(
                train_loss=f'{epoch_train_loss:.6f}',
                val_loss=f'{epoch_val_loss:.6f}',
                lr=optimizer.param_groups[0]['lr'],
                time=f'{epoch_time:.1f}s'
            )

            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f'\nEpoch {epoch + 1}')
                print(f'Training SNR usage: {snr_counter}')

                for snr in args.snr_list:
                    print(
                        f'SNR = {snr:g} dB, '
                        f'validation loss = {val_loss_per_snr[snr]:.6f}'
                    )

            if (epoch + 1) % args.save_freq == 0 or epoch + 1 == args.train_epochs:
                model_save_name = f'DeepJSCC_ep{epoch + 1}.pt'

                checkpoint = {
                    'epoch': epoch + 1,
                    'model_state': model.state_dict(),
                    'optimizer_state': optimizer.state_dict(),
                    'scheduler_state': scheduler.state_dict(),
                    'config': vars(args),
                    'c': c,
                    'CBR_ratio': args.CBR_ratio,
                    'actual_CBR_ratio': actual_cbr,
                    'CBR_fraction': cbr_string,
                    'snr_list': args.snr_list,
                    'train_loss': epoch_train_losses,
                    'val_loss': epoch_val_losses,
                    'val_loss_per_snr': epoch_val_losses_per_snr
                }

                save_path = os.path.join(log_path, model_save_name)
                torch.save(checkpoint, save_path)

                print(f'\nCheckpoint saved to: {save_path}')

    print('\nTraining finished.')
    print(f'Total epochs: {args.train_epochs}')
    print(f'TOTAL TIME TAKEN: {time.time() - t0:.4f}s')
    print(f'AVG TIME PER EPOCH: {np.mean(per_epoch_time):.4f}s')


if __name__ == '__main__':
    main_train_run()