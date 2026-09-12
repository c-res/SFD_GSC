import os
import random
import argparse
import torch
import torch.nn as nn
import torch.utils.checkpoint
import gc
from torch.utils.data import DataLoader
from torchvision import transforms
from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers.utils.import_utils import is_xformers_available
from Datasets.dataset_H5 import H5Dataset
from Models.DM_JSCC_sepED import DM_JSCC
from Models.Denoiser import VE_DiTDenoiser
from Models.DiT_utils import add_weight_decay
from Models.wireless_channel import Wireless_Channel
from Models.vision_aided_loss.cv_discriminator import Discriminator
import lpips
from torchmetrics.image import LearnedPerceptualImagePatchSimilarity
from Models.loss_funcs import CLIPLoss

os.environ["TORCH_DISTRIBUTED_DEBUG"] = "INFO"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def fine_tune_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_root_dir", type=str, default="")

    parser.add_argument("--train_patch_size", type=int, default=256)  #
    parser.add_argument("--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader.",)

    parser.add_argument("--channel_type", default="awgn", type=str, choices=["awgn", "rayleigh", "rayleigh_complex"], help="channel",)
    parser.add_argument("--data_channels", default=64, type=int, help="the dimension of the latent feature",)  # TODO:
    parser.add_argument("--DMJSCC_lr", default=1e-5, type=float, help="lr for DMJSCC in finetuning")
    parser.add_argument("--denoiser_lr", default=1e-5, type=float, help="lr for DMJSCC in finetuning")

    parser.add_argument("--num_workers", default=8, type=int, help="for dataloader")
    parser.add_argument("--max_train_steps", type=int, default=10000)
    parser.add_argument("--checkpointing_steps", type=int, default=1000)

    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="Number of updates steps to accumulate before performing a backward/update pass.",)
    parser.add_argument("--mixed_precision", type=str, default="no", choices=["no", "fp16", "bf16"],)
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument("--gpu", type=int, default=1, help="GPU ID, -1 for CPU")

    return parser.parse_args()


class JSCC_denoiser_Wrapper(nn.Module):
    def __init__(self, JSCC_net, denoiser_model, wireless_channel):
        super().__init__()
        self.JSCC_net = JSCC_net
        self.denoiser_model = denoiser_model
        self.wireless_channel = wireless_channel

    def forward(self, x, snr_value):
        # input to the encoder to get the latent feature for transmission:
        z, cbr_value = self.JSCC_net.encoding(x, given_rate=None)

        if self.wireless_channel.chan_type in ("rayleigh", "rayleigh_complex"):
            received, sigma, h = self.wireless_channel(z, snr=snr_value, return_h=True)
            z_ls, sigma_map = self.wireless_channel.ls_equalize(received, sigma, h)
            z_pred = self.denoiser_model.denoise_ls(z_ls, sigma_map)
        else:
            z_hat, sigma = self.wireless_channel(z.float(), snr=snr_value)
            sigma_y = torch.as_tensor(sigma, device=z_hat.device, dtype=z_hat.dtype)
            z_pred, _ = self.denoiser_model.edm_sampling(z_hat, sigma_y)

        # pass the denoised latent to the JSCC decoder:
        x_hat = self.JSCC_net.decoding(z_pred)
        x_hat = x_hat.float()

        return x_hat, z_pred, z, cbr_value


def main():
    args = fine_tune_args()
    torch.autograd.set_detect_anomaly(True)

    # Initialize Accelerate,
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        # log_with=args.report_to,
    )

    device = accelerator.device

    log_path = args.project_root_dir + "/checkpoints/" + "transceiver_finetune_" + args.channel_type + "_" + str(args.data_channels)
    if accelerator.is_main_process:
        os.makedirs(log_path, exist_ok=True)
        print("checkpoint_path=", log_path)

    accelerator.wait_for_everyone()
    if args.seed is not None:
        set_seed(args.seed)

    # =========================
    # 1. load the JSCC model and training configs
    # =============================
    DMJSCC_checkpoint_path = args.project_root_dir + "/checkpoints/" + "transceiver_basic_none_" + str(args.data_channels) + "/transceiver_iter50000"

    # the DMJSCC training configs:
    DMJSCC_config_path = DMJSCC_checkpoint_path + "/config.pt"
    DMJSCC_config = torch.load(DMJSCC_config_path, map_location="cpu", weights_only=False)["args"]

    DMJSCC_encoder_path = DMJSCC_checkpoint_path + "/encoder.pt"
    DMJSCC_decoder_path = DMJSCC_checkpoint_path + "/decoder.pt"

    # create the DMJSCC model:
    if DMJSCC_config.sd_path is None:
        from huggingface_hub import snapshot_download

        with accelerator.main_process_first():
            sd_path = snapshot_download(repo_id="stabilityai/sd-turbo")
    else:
        sd_path = DMJSCC_config.sd_path

    net = DM_JSCC(sd_path=sd_path, args=DMJSCC_config)
    net.load_model(DMJSCC_encoder_path, DMJSCC_decoder_path)
    net.set_train()

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
    optimizer_JSCC = torch.optim.AdamW(
        layers_to_opt,
        lr=args.DMJSCC_lr,
        betas=(DMJSCC_config.adam_beta1, DMJSCC_config.adam_beta2),
        weight_decay=DMJSCC_config.adam_weight_decay,
        eps=DMJSCC_config.adam_epsilon,
    )

    if DMJSCC_config.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            net.unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError(
                "xformers is not available, please install it by running `pip install xformers`"
            )

    if DMJSCC_config.seed is not None:
        set_seed(DMJSCC_config.seed)

    # load the denoiser model:
    denoiser_checkpoint_path = args.project_root_dir + "/checkpoints/" + "Denoiser_" + str(args.data_channels) + "/VE_DiTiter20000.pt"
    denoiser_contents = torch.load(denoiser_checkpoint_path, map_location="cpu", weights_only=False)
    denoiser_args = denoiser_contents["config"]
    denoiser_state = denoiser_contents["model_state"]

    denoiser_model = VE_DiTDenoiser(denoiser_args)
    denoiser_model.load_state_dict(denoiser_state)
    if denoiser_args.data_channels != args.data_channels:
        raise ValueError("Stage 2 checkpoint data_channels does not match training configuration.")
    if args.channel_type in ("rayleigh", "rayleigh_complex"):
        if getattr(denoiser_args, "predict_obj", "X_predict") != "X_predict":
            raise ValueError("Rayleigh LS path requires Stage 2 X_predict weights.")
        denoiser_model.enable_noise_conditioning()

    param_groups = add_weight_decay(denoiser_model, denoiser_args.weight_decay)
    optimizer_denoiser = torch.optim.AdamW(
        param_groups, lr=args.denoiser_lr, betas=(0.9, 0.95)
    )

    wireless_channel = Wireless_Channel(channel_type=args.channel_type)

    # Wrap the JSCC model, denoiser model, and the wireless channel:
    overall_model = JSCC_denoiser_Wrapper(JSCC_net=net, denoiser_model=denoiser_model, wireless_channel=wireless_channel)

    # GAN:
    net_disc = Discriminator(
        cv_type="dinov2_reg",
        output_type="conv_multi_level",
        loss_type=DMJSCC_config.gan_loss_type,
        device=device,
    )

    net_disc.requires_grad_(True)
    net_disc.cv_ensemble.requires_grad_(False)
    net_disc.train()

    optimizer_disc = torch.optim.AdamW(
        net_disc.parameters(),
        lr=2e-5,
        betas=(DMJSCC_config.adam_beta1, DMJSCC_config.adam_beta2),
        weight_decay=DMJSCC_config.adam_weight_decay,
        eps=DMJSCC_config.adam_epsilon,
    )

    for name, module in net_disc.named_modules():
        if "attn" in name:
            module.fused_attn = False

    # the loss computing networks:
    net_lpips = lpips.LPIPS(net="vgg").to(device)  # LPIPS for training use
    net_lpips.requires_grad_(False)
    net_lpips.eval()

    alex_lpips = LearnedPerceptualImagePatchSimilarity(normalize=True).to(
        device
    )  # LPIPS for validation use
    alex_lpips.requires_grad_(False)
    alex_lpips.eval()

    # net_clip = CLIPLoss(clip_model_name='/your_local_dir/clip-vit-base-patch32').cuda()
    net_clip = CLIPLoss().to(device)
    net_clip.requires_grad_(False)
    net_clip.eval()

    mse_loss = torch.nn.MSELoss()

    # =========================
    # Dataset / Dataloader
    # =========================
    train_dataset = H5Dataset(
        DMJSCC_config.train_dataset,
        transform=transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.RandomCrop((args.train_patch_size, args.train_patch_size)),
                transforms.RandomHorizontalFlip(),
                # transforms.RandomVerticalFlip(),
                transforms.ToTensor(),  # [0, 1]
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),  # [-1, 1]
            ]
        ),
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,  # per GPU batch size
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
    )

    (
        overall_model,
        net_disc,
        optimizer_JSCC,
        optimizer_denoiser,
        optimizer_disc,
        train_dataloader,
    ) = accelerator.prepare(
        overall_model,
        net_disc,
        optimizer_JSCC,
        optimizer_denoiser,
        optimizer_disc,
        train_dataloader,
    )

    if len(train_dataloader) == 0:
        raise ValueError("Training dataloader is empty.")
    global_step = 0
    while global_step < args.max_train_steps:
        for batch in train_dataloader:
            batch = batch.to(accelerator.device, non_blocking=True)  # [B, 3, 256, 256]
            x = batch.detach().float()

            with accelerator.accumulate(overall_model, net_disc):
                net_disc.requires_grad_(False)

                snr_value = random.choice(DMJSCC_config.snr_list)
                # TODO: here we select the denoising method:, equalization:
                x_hat, z_pred, z, cbr_value = overall_model(batch, snr_value)

                # compute loss:
                loss_l2 = mse_loss(x_hat, x)
                loss_lpips = net_lpips(x_hat, x).mean()
                loss_clip = net_clip(x_hat, x)
                loss_adv = accelerator.unwrap_model(net_disc)(x_hat, for_G=True).mean()
                loss_D = (
                    loss_l2 * DMJSCC_config.lambda_l2
                    + loss_lpips * DMJSCC_config.lambda_lpips
                    + loss_clip * DMJSCC_config.lambda_clip
                    + loss_adv * DMJSCC_config.lambda_gan
                )

                # here should concide with the training settings: TODO
                loss_latent = torch.nn.functional.mse_loss(z_pred, z.detach())

                loss = loss_D + 0.1 * loss_latent
                accelerator.backward(loss)
                # loss.backward()
                optimizer_JSCC.step()
                optimizer_denoiser.step()

                optimizer_JSCC.zero_grad(set_to_none=True)
                optimizer_denoiser.zero_grad(set_to_none=True)

                # -------------------------
                # Train discriminator
                # -------------------------
                net_disc.requires_grad_(True)
                accelerator.unwrap_model(net_disc).cv_ensemble.requires_grad_(False)
                loss_real = (
                    net_disc(x.detach(), for_real=True).mean()
                    * DMJSCC_config.lambda_gan
                )
                accelerator.backward(loss_real)
                loss_fake = (
                    net_disc(x_hat.detach(), for_real=False).mean()
                    * DMJSCC_config.lambda_gan
                )
                accelerator.backward(loss_fake)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(net_disc.parameters(), 1.0)
                optimizer_disc.step()
                optimizer_disc.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                global_step = global_step + 1
                accelerator.print(
                    f"global_step={global_step}, "
                    f"loss={loss.detach().item():.6f}, "
                    f"loss_D={loss_D.detach().item():.6f}, "
                    f"loss_latent={loss_latent.detach().item():.6f}, "
                    f"loss_real={loss_real.detach().item():.6f}, "
                    f"loss_fake={loss_fake.detach().item():.6f}"
                )

                if global_step % args.checkpointing_steps == 0 or global_step == args.max_train_steps:
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        save_path_name = log_path + "/iter_" + str(global_step)
                        os.makedirs(save_path_name, exist_ok=True)

                        unwrapped_overall_model = accelerator.unwrap_model(overall_model)
                        unwrapped_disc = accelerator.unwrap_model(net_disc)

                        encoder_out_f_name = os.path.join(save_path_name, "DMJSCC_encoder.pt")
                        decoder_out_f_name = os.path.join(save_path_name, "DMJSCC_decoder.pt")
                        unwrapped_overall_model.JSCC_net.save_model(encoder_out_f_name, decoder_out_f_name)


                        # save the denoiser:
                        accelerator.save(
                            {
                                "model_state": unwrapped_overall_model.denoiser_model.state_dict(),
                                "config": denoiser_args,
                                "noise_conditioning": args.channel_type
                                in ("rayleigh", "rayleigh_complex"),
                                "receiver": "perfect_csi_ls"
                                if args.channel_type in ("rayleigh", "rayleigh_complex")
                                else "awgn",
                            },
                            os.path.join(save_path_name, "VE_DiT.pt"),
                        )

                        # save the GAN:
                        accelerator.save(
                            {"model_state": unwrapped_disc.state_dict()},
                            os.path.join(save_path_name, "GAN.pt"),
                        )

                        accelerator.save(
                            {
                                "DMJSCC_config": DMJSCC_config,
                                "Denoiser_config": denoiser_args,
                                "finetune_args": args,
                            },
                            os.path.join(save_path_name, "configs.pt"),
                        )

                    accelerator.wait_for_everyone()

                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                if global_step >= args.max_train_steps:
                    break


if __name__ == "__main__":
    main()
