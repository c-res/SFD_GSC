# -*- coding: utf-8 -*-
"""
DeepJSCC Evaluation.

Features:
1. Evaluates multiple CBRs and SNRs.
2. Loads checkpoints produced by train_DeepJSCC.py.
3. Uses the same image preprocessing and metrics as the proposed-method evaluation.
4. Supports AWGN and Rayleigh channels.
5. Saves reconstructed images, NPZ results, and CSV files.
"""

import os
import re
import argparse
import numpy as np
import torch
from fractions import Fraction
from pathlib import Path
from PIL import Image
from torchvision import transforms
from accelerate.utils import set_seed
import pyiqa
from torchmetrics.image import FrechetInceptionDistance, KernelInceptionDistance, LearnedPerceptualImagePatchSimilarity
from neuralcompression.metrics import update_patch_fid
from DeepJSCC_model import DeepJSCC


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
    parser.add_argument('--test_dataset', type=str, default='Kodak', choices=['Kodak', 'DIV2K_valid_HR', 'CLIC2020Professional_test'])
    parser.add_argument('--eval_SNR_list', nargs='+', type=float, default=[0, 2, 4, 6, 8, 10])
    parser.add_argument('--CBR_ratio_list', nargs='+', type=parse_ratio, default=[parse_ratio('1/192'), parse_ratio('1/96'), parse_ratio('1/64'), parse_ratio('1/48'), parse_ratio('1/32')])
    # parser.add_argument('--CBR_ratio_list', nargs='+', type=parse_ratio, default=[parse_ratio('1/96')])
    parser.add_argument('--channel_type', type=str, default='AWGN', choices=['AWGN', 'Rayleigh'])
    parser.add_argument('--checkpoint_epoch', type=int, default=None, help='Checkpoint epoch. If omitted, the latest checkpoint is used.')
    parser.add_argument('--crop_base', type=int, default=128)
    parser.add_argument('--store_rec_imgs', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID, -1 for CPU')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--project_root_dir', type=str, default='/home/czx/code_JSCC/Semantic_diffusion')
    return parser.parse_args()


def preprocess_image(image_path: Path, transform) -> torch.Tensor:
    with Image.open(image_path) as image:
        image = image.convert('RGB')
        return transform(image)


def center_crop_to_multiple(img: torch.Tensor, base: int = 128) -> torch.Tensor:
    _, h, w = img.shape
    new_h = h - h % base
    new_w = w - w % base
    if new_h == h and new_w == w:
        return img
    if new_h <= 0 or new_w <= 0:
        raise ValueError(f'Image size too small for base={base}: got ({h}, {w})')
    return transforms.CenterCrop((new_h, new_w))(img)


def find_checkpoint(checkpoint_dir: Path, checkpoint_epoch=None) -> Path:
    if checkpoint_epoch is not None:
        checkpoint_path = checkpoint_dir / f'DeepJSCC_ep{checkpoint_epoch}.pt'
        if not checkpoint_path.exists():
            raise FileNotFoundError(f'Checkpoint not found: {checkpoint_path}')
        return checkpoint_path
    candidates = list(checkpoint_dir.glob('DeepJSCC_ep*.pt'))
    if len(candidates) == 0:
        raise FileNotFoundError(f'No DeepJSCC checkpoints found in {checkpoint_dir}')
    def get_epoch(path):
        match = re.search(r'DeepJSCC_ep(\d+)\.pt$', path.name)
        return int(match.group(1)) if match else -1
    return max(candidates, key=get_epoch)


def load_model(checkpoint_path: Path, channel_type: str, requested_cbr: float, device):
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if 'c' not in checkpoint:
        raise KeyError(f'Checkpoint does not contain latent parameter c: {checkpoint_path}')
    c = int(checkpoint['c'])
    checkpoint_cbr = checkpoint.get('actual_CBR_ratio', checkpoint.get('CBR_ratio', None))
    if checkpoint_cbr is not None and abs(float(checkpoint_cbr) - requested_cbr) > 1e-8:
        raise ValueError(f'CBR mismatch: requested CBR={requested_cbr:.8f}, checkpoint CBR={float(checkpoint_cbr):.8f}')
    checkpoint_config = checkpoint.get('config', {})
    checkpoint_channel = checkpoint_config.get('channel_type', channel_type) if isinstance(checkpoint_config, dict) else channel_type
    if str(checkpoint_channel).lower() != channel_type.lower():
        raise ValueError(f'Channel mismatch: checkpoint channel={checkpoint_channel}, requested channel={channel_type}')
    model = DeepJSCC(c=c, channel_type=channel_type).to(device)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    return model, c, checkpoint_cbr


def main():
    args = config_args()
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    if args.seed is not None:
        set_seed(args.seed)
    test_image_path = Path(args.project_root_dir) / 'Datasets' / args.test_dataset
    if args.test_dataset == 'CLIC2020Professional_test':
        test_image_path = test_image_path / 'professional'
    test_img_path_list = sorted([x for x in test_image_path.glob('*.[jpJP][pnPN]*[gG]')])
    print(f'\nFind {len(test_img_path_list)} images in {test_image_path}\n')
    if len(test_img_path_list) == 0:
        raise FileNotFoundError(f'No images found in {test_image_path}')
    metric_paired_dict = {}
    metric_paired_dict['psnr'] = pyiqa.create_metric('psnr').to(device)
    metric_paired_dict['dists'] = pyiqa.create_metric('dists').to(device)
    metric_paired_dict['ms_ssim'] = pyiqa.create_metric('ms_ssim').to(device)
    metric_paired_dict['lpips'] = LearnedPerceptualImagePatchSimilarity(normalize=True).to(device)
    fid_metric = FrechetInceptionDistance(normalize=True).to(device)
    kid_metric = KernelInceptionDistance(normalize=True, subset_size=min(24, len(test_img_path_list))).to(device)
    SNR_list = args.eval_SNR_list
    CBR_list = args.CBR_ratio_list
    result_snr = np.zeros((len(SNR_list), len(CBR_list)))
    result_cbr = np.zeros((len(SNR_list), len(CBR_list)))
    result_psnr = np.zeros((len(SNR_list), len(CBR_list)))
    result_dists = np.zeros((len(SNR_list), len(CBR_list)))
    result_ms_ssim = np.zeros((len(SNR_list), len(CBR_list)))
    result_lpips = np.zeros((len(SNR_list), len(CBR_list)))
    result_fid = np.zeros((len(SNR_list), len(CBR_list)))
    result_kid_mean = np.zeros((len(SNR_list), len(CBR_list)))
    result_kid_std = np.zeros((len(SNR_list), len(CBR_list)))
    result_root = Path(args.project_root_dir) / 'c_results' / f'DeepJSCC_{args.channel_type}' / args.test_dataset
    result_root.mkdir(parents=True, exist_ok=True)
    for cbr_ind, requested_cbr in enumerate(CBR_list):
        cbr_string = ratio_to_string(requested_cbr)
        checkpoint_dir = Path(args.project_root_dir) / 'checkpoints' / 'DeepJSCC' / f'DeepJSCC_{args.channel_type}_CBR_{cbr_string}'
        checkpoint_path = find_checkpoint(checkpoint_dir, args.checkpoint_epoch)
        model, c, checkpoint_cbr = load_model(checkpoint_path, args.channel_type, requested_cbr, device)
        print(f'CBR={Fraction(requested_cbr).limit_denominator(10000)}, c={c}, checkpoint={checkpoint_path}')
        for snr_ind, SNR in enumerate(SNR_list):
            fid_metric.reset()
            kid_metric.reset()
            result_one = {}
            for img_path in test_img_path_list:
                print('[Processing]', 'CBR=', Fraction(requested_cbr).limit_denominator(10000), 'SNR=', SNR, 'img_path=', img_path)
                ori_img = preprocess_image(img_path, transforms.ToTensor())
                ori_img = center_crop_to_multiple(ori_img, base=args.crop_base)
                ori_img = ori_img.unsqueeze(0).to(device)
                with torch.no_grad():
                    latent = model.encoder(ori_img)
                    z_hat = model.channel(latent, snr=SNR)
                    recon_image = model.decoder(z_hat)
                recon_image = recon_image.clamp(0.0, 1.0)
                real_latent_symbols = latent[0].numel()
                complex_channel_symbols = real_latent_symbols / 2.0
                actual_cbr = complex_channel_symbols / ori_img[0].numel()
                if args.store_rec_imgs:
                    rec_img_path = result_root / 'rec_imgs' / f'CBR_{cbr_string}' / f'SNR_{SNR:g}dB'
                    rec_img_path.mkdir(parents=True, exist_ok=True)
                    outf = rec_img_path / f'{Path(img_path).stem}.png'
                    transforms.ToPILImage()(recon_image[0].cpu()).save(outf)
                result_one['cbr'] = result_one.get('cbr', 0.0) + float(actual_cbr)
                for key, metric in metric_paired_dict.items():
                    value = metric(recon_image, ori_img).item()
                    result_one[key] = result_one.get(key, 0.0) + value
                update_patch_fid(ori_img, recon_image, fid_metric=fid_metric, kid_metric=kid_metric)
            result_one['fid'] = float(fid_metric.compute())
            kid_tuple = kid_metric.compute()
            result_one['kid_mean'], result_one['kid_std'] = float(kid_tuple[0]), float(kid_tuple[1])
            result_snr[snr_ind][cbr_ind] = SNR
            result_cbr[snr_ind][cbr_ind] = result_one['cbr'] / len(test_img_path_list)
            result_psnr[snr_ind][cbr_ind] = result_one['psnr'] / len(test_img_path_list)
            result_dists[snr_ind][cbr_ind] = result_one['dists'] / len(test_img_path_list)
            result_ms_ssim[snr_ind][cbr_ind] = result_one['ms_ssim'] / len(test_img_path_list)
            result_lpips[snr_ind][cbr_ind] = result_one['lpips'] / len(test_img_path_list)
            result_fid[snr_ind][cbr_ind] = result_one['fid']
            result_kid_mean[snr_ind][cbr_ind] = result_one['kid_mean']
            result_kid_std[snr_ind][cbr_ind] = result_one['kid_std']
            print(f'[Result] CBR={Fraction(requested_cbr).limit_denominator(10000)}, SNR={SNR:g} dB, PSNR={result_psnr[snr_ind][cbr_ind]:.6f}, DISTS={result_dists[snr_ind][cbr_ind]:.6f}, MS-SSIM={result_ms_ssim[snr_ind][cbr_ind]:.6f}, LPIPS={result_lpips[snr_ind][cbr_ind]:.6f}, FID={result_fid[snr_ind][cbr_ind]:.6f}, KID={result_kid_mean[snr_ind][cbr_ind]:.6f}')
    save_npz_path = result_root / 'eval_results.npz'
    np.savez_compressed(save_npz_path, result_snr=result_snr, result_cbr=result_cbr, result_psnr=result_psnr, result_dists=result_dists, result_ms_ssim=result_ms_ssim, result_lpips=result_lpips, result_fid=result_fid, result_kid_mean=result_kid_mean, result_kid_std=result_kid_std, SNR_list=np.array(SNR_list), CBR_list=np.array(CBR_list))
    np.savetxt(result_root / 'result_psnr.csv', result_psnr, delimiter=',', fmt='%.6f')
    np.savetxt(result_root / 'result_dists.csv', result_dists, delimiter=',', fmt='%.6f')
    np.savetxt(result_root / 'result_ms_ssim.csv', result_ms_ssim, delimiter=',', fmt='%.6f')
    np.savetxt(result_root / 'result_lpips.csv', result_lpips, delimiter=',', fmt='%.6f')
    np.savetxt(result_root / 'result_fid.csv', result_fid, delimiter=',', fmt='%.6f')
    np.savetxt(result_root / 'result_kid_mean.csv', result_kid_mean, delimiter=',', fmt='%.6f')
    np.savetxt(result_root / 'result_kid_std.csv', result_kid_std, delimiter=',', fmt='%.6f')
    print(f'Saved all results to {save_npz_path}')


if __name__ == '__main__':
    main()
