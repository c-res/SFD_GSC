import os
import lpips
import torch
import torch.utils.checkpoint
import argparse
os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'INFO'
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
from torchmetrics.image import LearnedPerceptualImagePatchSimilarity
from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers.utils.import_utils import is_xformers_available
from Models.loss_funcs import CLIPLoss
from Datasets.dataset_H5 import H5Dataset
from Models.DM_JSCC_sepED import DM_JSCC


def parse_args_training(input_args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_root_dir", type=str, default="")
    # pretrained weights
    parser.add_argument("--sd_path", default='', help="Path to SD-Turbo")
    parser.add_argument("--elic_path", default='', help="Path to pretrained ELIC model")
    # dataset
    parser.add_argument("--train_dataset", default='', help="Path to training dataset (hdf5)")

    # Wireless channel settings:
    parser.add_argument('--channel_type', default='none', type=str, choices=["none", 'awgn', 'rayleigh', 'rayleigh_complex'], help='channel')
    parser.add_argument("--snr_list", nargs="+", type=int, default=[0, 2, 4, 6, 8, 10])
    # the CBR = [B, 256, H/32, W/32] / 2* [B, 3, H, W], = c/6144 if ch_num=256, cbr=0.0417, 64->1/96
    parser.add_argument("--tx_channel_num_list", nargs="+", type=int, default=[32, 64, 96, 128, 192])
    parser.add_argument("--fix_tx_channel_num", type=int, default=128, help='whether use the fixed CBR, if not, set as None')


    # loss function weights:
    parser.add_argument("--gan_loss_type", default="multilevel_sigmoid_s") # GAN loss, only used for finetuning
    parser.add_argument("--lambda_gan", default=0.1, type=float) # GAN loss weight, only used for finetuning
    parser.add_argument("--lambda_clip", default=0.1, type=float)
    parser.add_argument("--lambda_lpips", default=1.0, type=float)
    parser.add_argument("--lambda_l2", default=2.0, type=float) # weight of MSE loss
    parser.add_argument("--lambda_rate", default=0.5, type=float) # The weight of rate

    # model details
    parser.add_argument("--lora_rank_unet", default=32, type=int) # the LoRA rank in UNet, SD-Turbo
    parser.add_argument("--lora_rank_vae", default=16, type=int) # the LoRA rank in VAE, SD-Turbo
    parser.add_argument("--pos_prompt", type=str, default="A high-resolution and ultra-realistic image with sharp focus, vibrant colors, and natural lighting.")

    # training details
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument("--train_patch_size", type=int, default=256) #
    parser.add_argument("--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader.")
    parser.add_argument("--max_train_steps", type=int, default=120000)
    parser.add_argument("--checkpointing_steps", type=int, default=10000)

    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--lr_scheduler", type=str, default="constant")
    parser.add_argument("--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument("--dataloader_num_workers", type=int, default=8)
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--allow_tf32", action="store_true")
    parser.add_argument("--report_to", type=str, default="wandb")
    parser.add_argument("--mixed_precision", type=str, default=None, choices=["no", "fp16", "bf16"], )
    parser.add_argument("--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers.")
    parser.add_argument("--set_grads_to_none", action="store_true")

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    return args


def to_log_tensor(value, device):
    if torch.is_tensor(value):
        return value.detach().float().mean().reshape(1).to(device)
    return torch.tensor([float(value)], device=device, dtype=torch.float32)

def main():
    args = parse_args_training()
    # check whether SD_Turbo is downloaded.
    if args.sd_path is None:
        from huggingface_hub import snapshot_download
        sd_path = snapshot_download(repo_id="stabilityai/sd-turbo")
    else:
        sd_path = args.sd_path

    # Initialize Accelerate,
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        # log_with=args.report_to,
    )

    log_path = args.project_root_dir + '/checkpoints/' + 'transceiver_basic_' + args.channel_type + '_' + str(args.fix_tx_channel_num)

    if accelerator.is_main_process:
        os.makedirs(log_path, exist_ok=True)
        print('checkpoint_path=', log_path)

    accelerator.wait_for_everyone()

    if args.seed is not None:
        set_seed(args.seed)

    # Dataset / Dataloader
    train_dataset = H5Dataset(
        args.train_dataset,
        transform=transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomCrop((args.train_patch_size, args.train_patch_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
    )

    net = DM_JSCC(sd_path=sd_path, args=args)
    net.set_train()
    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            net.unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available, please install it by running `pip install xformers`")
    # save memory by recomputing:
    if args.gradient_checkpointing:
        net.unet.enable_gradient_checkpointing()
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    # select the trainable parameters;
    layers_to_opt = list(net.latent_JSCC_encoder.parameters())
    layers_to_opt += list(net.latent_JSCC_decoder.parameters())
    layers_to_opt += list(net.aux_decoder.parameters())
    layers_to_opt += list(net.encoder_adapter.parameters())
    for n, _p in net.unet.named_parameters():
        if "lora" in n:
            assert _p.requires_grad
            layers_to_opt.append(_p)
    layers_to_opt += list(net.unet.conv_in.parameters())
    for n, _p in net.vae.named_parameters():
        if "lora" in n:
            assert _p.requires_grad
            layers_to_opt.append(_p)
    optimizer = torch.optim.AdamW(layers_to_opt, lr=1e-4, betas=(args.adam_beta1, args.adam_beta2),
                                  weight_decay=args.adam_weight_decay, eps=args.adam_epsilon)

    # =========================
    # Accelerate prepare
    # =========================
    net, optimizer, train_dataloader = accelerator.prepare(net, optimizer, train_dataloader)

    params_to_clip = []
    for group in optimizer.param_groups:
        params_to_clip.extend(group["params"])


    effective_batch_size = (args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps)
    accelerator.print(f"num_processes = {accelerator.num_processes}")
    accelerator.print(f"per-device batch size = {args.train_batch_size}")
    accelerator.print(f"gradient accumulation = {args.gradient_accumulation_steps}")
    accelerator.print(f"effective batch size = {effective_batch_size}")

    # the loss computing networks:
    net_lpips = lpips.LPIPS(net='vgg').to(accelerator.device)  # LPIPS for training use
    net_lpips.requires_grad_(False)
    net_lpips.eval()

    alex_lpips = LearnedPerceptualImagePatchSimilarity(normalize=True).to(accelerator.device) # LPIPS for validation use
    alex_lpips.requires_grad_(False)
    alex_lpips.eval()

    # net_clip = CLIPLoss(clip_model_name='/your_local_dir/clip-vit-base-patch32').cuda()
    net_clip = CLIPLoss().to(accelerator.device)
    net_clip.requires_grad_(False)
    net_clip.eval()

    mse_loss = torch.nn.MSELoss()

    progress_bar = tqdm(range(0, args.max_train_steps), initial=0, desc="Steps", disable=not accelerator.is_local_main_process)

    global_step = 0
    while global_step < args.max_train_steps:
        for batch in train_dataloader:
            batch = batch.to(accelerator.device, non_blocking=True)  # [B, 3, 512, 512]
            # accumulate to the defined times, accelerator.sync_gradients=True, then update
            with accelerator.accumulate(net):
                # B, C, H, W = batch.shape
                x_hat, cbr_value, snr = net(batch)
                x = batch.detach().float()  #
                x_hat = x_hat.float()

                loss_l2 = mse_loss(x_hat, x)
                loss_lpips = net_lpips(x_hat, x).mean()
                loss_clip = net_clip(x_hat, x)

                loss = loss_l2 * args.lambda_l2 + loss_lpips * args.lambda_lpips + loss_clip * args.lambda_clip
                accelerator.backward(loss)
                # only clip after synchronize:
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)

                optimizer.step()
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)

            if accelerator.sync_gradients:
                global_step += 1
                progress_bar.update(1)

                with torch.no_grad():
                    ori_img = (batch + 1) / 2 # [0, 1]
                    recon_img = (x_hat.detach() + 1 ) / 2
                    recon_img = recon_img.clamp(0., 1.)
                    mse1 = torch.mean((ori_img - recon_img) ** 2)
                    psnr = torch.where(
                        mse1 > 0,
                        10.0 * torch.log10(1.0 / mse1),
                        torch.tensor(float("inf"), device=accelerator.device),
                    )

                loss_log = accelerator.gather_for_metrics(loss.detach().float().reshape(1)).mean().item()
                l2_log = accelerator.gather_for_metrics(loss_l2.detach().float().reshape(1)).mean().item()
                lpips_log = accelerator.gather_for_metrics(loss_lpips.detach().float().reshape(1)).mean().item()
                clip_log = accelerator.gather_for_metrics(loss_clip.detach().float().reshape(1)).mean().item()
                psnr_log = accelerator.gather_for_metrics(psnr.detach().float().reshape(1)).mean().item()
                cbr_log = accelerator.gather_for_metrics(to_log_tensor(cbr_value, accelerator.device)).mean().item()
                snr_log = accelerator.gather_for_metrics(to_log_tensor(snr, accelerator.device)).mean().item()

                if accelerator.is_main_process:
                    print(
                        "global_step=", global_step,
                        "loss=", loss_log,
                        "loss_l2=", l2_log,
                        "loss_lpips=", lpips_log,
                        "loss_clip=", clip_log,
                        "CBR=", cbr_log,
                        "SNR=", snr_log,
                        "psnr=", psnr_log,
                    )

                if global_step % args.checkpointing_steps == 0 or global_step == args.max_train_steps:
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        model_save_file_name = log_path + '/transceiver_iter' + str(global_step)
                        os.makedirs(model_save_file_name, exist_ok=True)

                        encoder_out_f_name = os.path.join(model_save_file_name, "encoder.pt")
                        decoder_out_f_name = os.path.join(model_save_file_name, "decoder.pt")

                        unwrapped_net = accelerator.unwrap_model(net)
                        unwrapped_net.save_model(encoder_out_f_name, decoder_out_f_name)

                        # save the training configs:
                        config_save_name = f"config.pt"
                        config_out_f_name = os.path.join(model_save_file_name, config_save_name)
                        torch.save({'args': args}, config_out_f_name)

                        accelerator.print(f"[Checkpoint] Saved to {log_path}")

                    accelerator.wait_for_everyone()


                if global_step >= args.max_train_steps:
                    break


if __name__ == "__main__":
    main()
