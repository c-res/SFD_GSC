import os
import sys
import json

import argparse
import torch
import torch.utils.checkpoint
from pathlib import Path
import numpy as np

from torchvision import transforms
from accelerate.utils import set_seed
from diffusers.utils.import_utils import is_xformers_available

from Models.DM_JSCC_sepED import DM_JSCC
from Models.Denoiser import VE_DiTDenoiser


from Models.wireless_channel import Wireless_Channel

from PIL import Image

import pyiqa
from torchmetrics.image import (
    FrechetInceptionDistance,  # FID
    KernelInceptionDistance,  # KID
    LearnedPerceptualImagePatchSimilarity,  # LPIPS
)
from neuralcompression.metrics import update_patch_fid


def fine_tune_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_root_dir", type=str, default="")
    parser.add_argument("--test_dataset", type=str, default="Kodak", choices=["Kodak", "DIV2K_valid_HR", "CLIC2020Professional_test"],)
    parser.add_argument("--eval_SNR_list", nargs="+", type=int, default=[0, 2, 4, 6, 8, 10])  # the CBR = [B, 256, H/32, W/32] / 2* [B, 3, H, W], = c/6144 if ch_num=256, cbr=0.0417, 64->1/96
    parser.add_argument("--channel_num_list", nargs="+", type=int, default=[64])  # [32, 64, 96, 128, 192]-> 1/192, 1/96, 1/64, 1/48, 1/32
    parser.add_argument("--data_channels", default=64, type=int, help="the dimension of the latent feature",)  # TODO:
    parser.add_argument("--channel_type", type=str, default="awgn", choices=["none", "awgn", "rayleigh", "rayleigh_complex"],)
    parser.add_argument("--store_rec_imgs", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gpu", type=int, default=1, help="GPU ID, -1 for CPU")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers.",)
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Optional exact Stage 3 checkpoint directory.",)
    parser.add_argument("--checkpoint_step", type=int, default=10000)
    return parser.parse_args()


def preprocess_image(image_path: Path, transform) -> torch.Tensor:
    with Image.open(image_path) as image:  # PIL image
        image = image.convert("RGB")
        return transform(image)


def center_crop_to_multiple(img: torch.Tensor, base: int = 128) -> torch.Tensor:
    """
    img: [C, H, W]
    Crop image so H and W are multiples of `base`.
    """
    _, h, w = img.shape
    new_h = h - (h % base)
    new_w = w - (w % base)

    if new_h == h and new_w == w:
        return img
    if new_h <= 0 or new_w <= 0:
        raise ValueError(f"Image size too small for base={base}: got ({h}, {w})")

    crop = transforms.CenterCrop((new_h, new_w))  # (H, W)
    return crop(img)


CHECK_CONTEXT = "initialization"

def rebuild_scheduler_from_cpu(model, device):
    """Keep the loaded config; avoid default-CUDA to evaluation-GPU transfers."""
    with torch.device("cpu"):
        scheduler = type(model.sched).from_config(model.sched.config)
    reference = scheduler.alphas_cumprod.detach().cpu().clone()
    if not torch.isfinite(reference).all() or (reference <= 0).any():
        raise ValueError(
            "Rebuilt scheduler has invalid/zero alpha; refusing to change its noise schedule."
        )
    scheduler.set_timesteps(1, device=device)
    transferred = reference.to(device=device, dtype=reference.dtype, copy=True)
    if not torch.equal(transferred.cpu(), reference):
        raise RuntimeError(
            "CPU-to-GPU scheduler alpha transfer changed values; investigate the CUDA runtime/device."
        )
    scheduler.alphas_cumprod = transferred
    model.sched = scheduler
    return reference


def trace_jscc_decoding(model, z_hat):
    """Exact operations from the supplied DM_JSCC.decoding, with boundary checks."""
    batch = z_hat.shape[0]
    pos_caption_enc = model.pos_caption_enc.to(z_hat.device).expand(batch, -1, -1)
    res_aux = model.aux_decoder(z_hat)
    x_hat = model.latent_JSCC_decoder(z_hat)
    timesteps = model.timesteps.to(z_hat.device)
    model_pred = model.unet(
        x_hat, timesteps, encoder_hidden_states=pos_caption_enc
    ).sample
    model.sched.set_timesteps(1, device=z_hat.device)
    model.sched.alphas_cumprod = model.sched.alphas_cumprod.to(device=z_hat.device)
    reference = getattr(model, "_eval_alpha_reference_cpu", None)
    if reference is not None and not torch.equal(
        model.sched.alphas_cumprod.detach().cpu(), reference
    ):
        raise RuntimeError(
            "Scheduler alphas changed after verified initialization. The evaluation stopped before division; do not clamp alpha."
        )

    scheduler_out = model.sched.step(
        model_pred, timesteps, x_hat[:, :4], return_dict=True
    ).prev_sample
    x_denoised = scheduler_out + res_aux
    scaling_factor = float(model.vae.config.scaling_factor)
    if not np.isfinite(scaling_factor) or scaling_factor == 0:
        raise ValueError(f"Invalid VAE scaling_factor={scaling_factor}")
    vae_input = x_denoised / scaling_factor
    raw_image = model.vae.decode(vae_input).sample
    return raw_image.clamp(-1, 1)


def rayleigh_ls_observation(z, snr, complex_fading=True):
    z = z.float()
    if z.ndim != 4 or z.shape[1] % 2:
        raise ValueError("Expected [B,C,H,W] latent with even C.")
    if not torch.isfinite(z).all():
        raise FloatingPointError("Nonfinite encoder latent.")
    half = z.shape[1] // 2
    scale = (2.0 * z.square().mean(dim=(1, 2, 3), keepdim=True)).clamp_min(1e-12).sqrt()
    tx = torch.complex(z[:, :half] / scale, z[:, half:] / scale)
    h_complex = torch.complex(
        torch.randn_like(tx.real), torch.randn_like(tx.real)
    ) / np.sqrt(2.0)
    h = h_complex if complex_fading else h_complex.abs()
    amplitude = h.abs()
    if (amplitude == 0).any() or not torch.isfinite(amplitude).all():
        raise FloatingPointError(
            "Strict LS cannot invert zero/nonfinite fading coefficients."
        )
    std = float(10.0 ** (-float(snr) / 20.0) / np.sqrt(2.0))
    noise = torch.complex(torch.randn_like(tx.real), torch.randn_like(tx.real)) * std
    received = h * tx + noise
    equalized = (received / h) * scale
    sigma = std * scale / amplitude
    z_ls = torch.cat((equalized.real, equalized.imag), dim=1)
    sigma_map = torch.cat((sigma, sigma), dim=1)
    return z_ls, sigma_map


def denoise_awgn_weights_on_ls(denoiser, z_ls, sigma_map):
    """Inference-only heterogeneous-noise adaptation of the original X_predict net.

    Same as the previous denoise_ls path with an exactly zero noise adapter,
    but creates no new parameters. Scalar time uses mean(log sigma); local
    EDM scaling uses the full map. This is not an exact uniform-AWGN equivalence.
    """
    z_ls = z_ls.float()
    sigma = torch.broadcast_to(
        sigma_map.to(device=z_ls.device, dtype=z_ls.dtype), z_ls.shape
    )
    if denoiser.sigma_data <= 0 or (sigma < 0).any() or not torch.isfinite(sigma).all():
        raise ValueError("Invalid noise map or sigma_data.")
    sd = torch.full_like(sigma, float(denoiser.sigma_data))
    norm = torch.hypot(sigma, sd)
    c_skip = (sd / norm).square()
    c_out = sd * (sigma / norm)
    c_noise = 0.25 * sigma.clamp_min(1e-12).log().mean(dim=(1, 2, 3))
    net_input = z_ls / norm
    residual = denoiser.net(net_input, c_noise)
    result = c_skip * z_ls + c_out * residual
    return result


@torch.no_grad()
def main():
    global CHECK_CONTEXT
    args = fine_tune_args()
    device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else "cpu")

    if args.seed is not None:
        set_seed(args.seed)

    checkpoint_path = args.checkpoint_path
    if checkpoint_path is None:
        checkpoint_path = os.path.join(args.project_root_dir, "checkpoints", f"transceiver_finetune_{args.data_channels}", f"iter_{args.checkpoint_step}",)
    checkpoint_path = os.path.abspath(checkpoint_path)

    configs_path = checkpoint_path + "/configs.pt"
    configs = torch.load(configs_path, map_location="cpu", weights_only=False)
    DMJSCC_args = configs["DMJSCC_config"]
    denoiser_args = configs["Denoiser_config"]
    DMJSCC_encoder_path = checkpoint_path + "/encoder.pt"
    DMJSCC_decoder_path = checkpoint_path + "/decoder.pt"
    if DMJSCC_args.sd_path is None:
        from huggingface_hub import snapshot_download
        sd_path = snapshot_download(repo_id="stabilityai/sd-turbo")
    else:
        sd_path = DMJSCC_args.sd_path

    DMJSCC_net = DM_JSCC(sd_path=sd_path, args=DMJSCC_args)
    # load model parameters:
    DMJSCC_net.load_model(DMJSCC_encoder_path, DMJSCC_decoder_path)
    DMJSCC_net.to(device)
    DMJSCC_net.set_eval()
    DMJSCC_net._eval_alpha_reference_cpu = rebuild_scheduler_from_cpu(
        DMJSCC_net, device
    )


    if args.enable_xformers_memory_efficient_attention or getattr(DMJSCC_args, "enable_xformers_memory_efficient_attention", False):
        if is_xformers_available():
            DMJSCC_net.unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available, please install it by running `pip install xformers`")

    denoiser_checkpoint_path = checkpoint_path + "/denoiser.pt"
    denoiser_contents = torch.load(denoiser_checkpoint_path, map_location="cpu", weights_only=False)
    denoiser_state = denoiser_contents["model_state"]
    denoiser_model = VE_DiTDenoiser(denoiser_args)
    denoiser_model.load_state_dict(denoiser_state, strict=True)
    denoiser_model.to(device)
    denoiser_model.eval()
    denoiser_model.requires_grad_(False)


    wireless_channel = Wireless_Channel(channel_type=args.channel_type)
    # the test image path:
    test_image_path = args.project_root_dir + "/Datasets/" + args.test_dataset
    if args.test_dataset == "CLIC2020Professional_test":
        test_image_path = test_image_path + "/professional"
    test_image_path = Path(test_image_path) if not isinstance(test_image_path, Path) else test_image_path
    test_img_path_list = sorted(x for x in test_image_path.iterdir() if x.is_file() and x.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"})

    # metrics:
    metric_paired_dict = {}
    metric_paired_dict["psnr"] = pyiqa.create_metric("psnr", device=device)
    metric_paired_dict["dists"] = pyiqa.create_metric("dists", device=device)
    metric_paired_dict["ms_ssim"] = pyiqa.create_metric("ms_ssim", device=device)
    metric_paired_dict["lpips"] = LearnedPerceptualImagePatchSimilarity(normalize=True).to(device)
    fid_metric = FrechetInceptionDistance(normalize=True).to(device)
    kid_metric = KernelInceptionDistance(
        normalize=True, subset_size=min(24, len(test_img_path_list))
    ).to(device)

    for metric in metric_paired_dict.values():
        metric.eval()

    # results list:
    for snr_ind, SNR in enumerate(args.eval_SNR_list):
        for cbr_ind, cbr_channel_num in enumerate(args.channel_num_list):
            fid_metric.reset()
            kid_metric.reset()
            for metric in metric_paired_dict.values():
                if hasattr(metric, "reset"):
                    metric.reset()

            result_one = {}
            # test each image:
            for img_path in test_img_path_list:
                CHECK_CONTEXT = f"SNR={SNR}, image={img_path.name}"
                ori_img = preprocess_image(
                    img_path, transforms.ToTensor()
                )  # [C, H, W] in [0, 1]
                ori_img = center_crop_to_multiple(ori_img, base=128)  # crop on CPU
                ori_img = ori_img.unsqueeze(0).to(device)
                transform_range = transforms.Normalize(
                    [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
                )  # [-1, 1]
                ori_img_input = transform_range(ori_img)
                ori_img_input = ori_img_input.to(device)  # [1, C, H, W] in [-1, 1]

                with torch.no_grad():
                    z, cbr_value = DMJSCC_net.encoding(ori_img_input, given_rate=None)
                    if args.channel_type in ("rayleigh", "rayleigh_complex"):
                        z_ls, sigma_map = rayleigh_ls_observation(
                            z,
                            SNR,
                            complex_fading=args.channel_type == "rayleigh_complex",
                        )
                        z_pred = denoise_awgn_weights_on_ls(denoiser_model, z_ls, sigma_map)
                    elif args.channel_type == "none":
                        z_pred = z
                    else:
                        z_hat, sigma = wireless_channel(z.float(), snr=SNR)
                        sigma_y = torch.as_tensor(sigma, device=z_hat.device, dtype=z_hat.dtype)
                        z_pred, _ = denoiser_model.edm_sampling(z_hat, sigma_y)

                    # pass the denoised latent to the JSCC decoder:
                    recon_image = trace_jscc_decoding(DMJSCC_net, z_pred)
                    out_img = ((recon_image * 0.5 + 0.5).float().cpu().detach())  # map to [0, 1]

                out_img = out_img[0].clamp(0.0, 1.0).cpu()
                # compute the metrics:
                recon_tensor = out_img.unsqueeze(0).to(device)

                if recon_tensor.shape != ori_img.shape:
                    raise ValueError(
                        f"Reconstruction/reference shape mismatch: {recon_tensor.shape} versus {ori_img.shape}"
                    )
                result_one["cbr"] = result_one.get("cbr", 0.0) + float(
                    cbr_value.item() if torch.is_tensor(cbr_value) else cbr_value
                )

                for key, metric in metric_paired_dict.items():
                    value = metric(recon_tensor, ori_img).item()
                    result_one[key] = result_one.get(key, 0) + value

                update_patch_fid(ori_img, recon_tensor, fid_metric=fid_metric, kid_metric=kid_metric)

            # compute the fig and kid:
            result_one["fid"] = float(fid_metric.compute())
            kid_tuple = kid_metric.compute()
            result_one["kid_mean"], result_one["kid_std"] = (
                float(kid_tuple[0]),
                float(kid_tuple[1]),
            )
            image_count = len(test_img_path_list)
            print(
                f"[Result] SNR={SNR} dB, CBR={result_one['cbr'] / image_count:.6f}, PSNR={result_one['psnr'] / image_count:.4f}, LPIPS={result_one['lpips'] / image_count:.4f}, DISTS={result_one['dists'] / image_count:.4f}, MS-SSIM={result_one['ms_ssim'] / image_count:.4f}, patch-FID={result_one['fid']:.4f}, patch-KID={result_one['kid_mean']:.6f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
