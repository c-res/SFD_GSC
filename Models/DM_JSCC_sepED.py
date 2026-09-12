import random

import torch
import torch.nn as nn
from pyiqa.matlab_utils.resize import padding
from transformers import AutoTokenizer, CLIPTextModel
from diffusers import AutoencoderKL, UNet2DConditionModel
from peft import LoraConfig
from Models.model import make_1step_sched, my_lora_fwd
# from my_utils.vaehook import VAEHook

import sys

sys.path.append("..")
from ELIC.model.elic_official import ELIC

# from Models.latent_codec import LatentCodec
from Models.latent_JSCC import Downsample, AuxDecoder
from Models.latent_JSCC import Latent_JSCC_Encoder, Latent_JSCC_Decoder
from Models.wireless_channel import Wireless_Channel



class Encoder_Adapter(nn.Module):
    def __init__(self, main_in_ch, main_out_ch, aux_in_ch, aux_out_ch):
        super().__init__()
        self.main_encoder_adapter = Downsample(main_in_ch, main_out_ch)
        self.aux_encoder_adapter = nn.Conv2d(aux_in_ch, aux_out_ch, kernel_size=3, padding=1)

    def forward(self, main_latent, aux_latent):
        main_latent = self.main_encoder_adapter(main_latent)
        aux_latent = self.aux_encoder_adapter(aux_latent)
        latent = torch.cat((main_latent, aux_latent), dim=1)  # concatenates the main and auxiliary latents

        return latent



class DM_JSCC(torch.nn.Module):
    def __init__(self, sd_path=None, args=None):
        super().__init__()
        # ------------------------------------------------------------
        # 1. load the SD-Turbo model and config the LoRA:
        # SD-Turbo is a fast generative text-to-image model that can synthesize images from a text prompt in a single network evaluation.
        # ------------------------------------------------------------
        print("[SD-Turbo]: Building SD-Turbo ......")
        self.tokenizer = AutoTokenizer.from_pretrained(sd_path, subfolder="tokenizer")  # transform prompt to tokens
        self.text_encoder = CLIPTextModel.from_pretrained(sd_path, subfolder="text_encoder")  # trans tokens to textual embeddings
        self.sched = make_1step_sched(sd_path)  # contract one-step DDPM scheduler
        self.guidance_scale = 1.07  # This is the default value in the SD-Turbo

        self.vae = AutoencoderKL.from_pretrained(sd_path, subfolder="vae")
        self.unet = UNet2DConditionModel.from_pretrained(sd_path, subfolder="unet")

        self.register_buffer("timesteps", torch.tensor([999], dtype=torch.long), persistent=False)
        # self.timesteps = torch.tensor([999], device="cuda").long()
        self.text_encoder.requires_grad_(False)
        print("[SD-Turbo]: Done!")

        # ------------------------------------------------------------
        # 1.1 LoRA
        # ------------------------------------------------------------
        print("[LoRA]: Initializing LoRA ......")
        target_modules_vae = r"^encoder\..*(conv1|conv2|conv_in|conv_shortcut|conv|conv_out|to_k|to_q|to_v|to_out\.0)$"
        target_modules_unet = [
            "to_k", "to_q", "to_v", "to_out.0", "conv", "conv1", "conv2", "conv_shortcut", "conv_out",
            "proj_in", "proj_out", "ff.net.2", "ff.net.0.proj"
        ]
        lora_rank_vae = args.lora_rank_vae
        lora_rank_unet = args.lora_rank_unet

        vae_lora_config = LoraConfig(r=lora_rank_vae, init_lora_weights="gaussian", target_modules=target_modules_vae)
        self.vae.add_adapter(vae_lora_config, adapter_name="vae_skip")
        unet_lora_config = LoraConfig(r=lora_rank_unet, init_lora_weights="gaussian", target_modules=target_modules_unet)
        self.unet.add_adapter(unet_lora_config)
        # change the VAE forward method to my_forward
        self.vae_lora_layers = []
        for name, module in self.vae.named_modules():
            if 'base_layer' in name:
                self.vae_lora_layers.append(name[:-len(".base_layer")])
        for name, module in self.vae.named_modules():
            if name in self.vae_lora_layers:
                module.forward = my_lora_fwd.__get__(module, module.__class__)

        self.unet_lora_layers = []
        for name, module in self.unet.named_modules():
            if 'base_layer' in name:
                self.unet_lora_layers.append(name[:-len(".base_layer")])
        for name, module in self.unet.named_modules():
            if name in self.unet_lora_layers:
                module.forward = my_lora_fwd.__get__(module, module.__class__)
        print("[LoRA]: Done!")

        # ------------------------------------------------------------
        # 1.2 The input layer of the U-Net in SD-Turbo:
        # ------------------------------------------------------------
        temp_layer = nn.Conv2d(320, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.unet.conv_in = temp_layer  # to adapt the UNet:
        print("[Latent Codec]: Done!")

        # ------------------------------------------------------------
        # 1.3. generate the fixed text prompt embedding of the U-Net:
        # ------------------------------------------------------------
        # use fix text prompt for the SD-Turbo, store in self.pos_caption_enc
        print("[Prompt]: Setting Prompt ......")
        self.set_prompt(args.pos_prompt)
        del self.tokenizer
        del self.text_encoder
        print("[Prompt]: Done!")

        # ------------------------------------------------------------
        # 2.1. Auxiliary Encoder:
        # ------------------------------------------------------------
        print("[Auxiliary Encoder]: Loading Pretrained Weights ......")
        model = ELIC()
        checkpoint = torch.load(args.elic_path, map_location="cpu")
        model.load_state_dict(checkpoint)
        self.aux_encoder = model.g_a
        self.aux_encoder.eval()
        self.aux_encoder.requires_grad_(False)
        print("[Auxiliary Encoder]: Done!")

        # ------------------------------------------------------------
        # 2.2. The adapter of the the encoders of SD-Turbo's encoder and the ELIC encoder:
        # ------------------------------------------------------------
        self.encoder_adapter = Encoder_Adapter(main_in_ch=4, main_out_ch=128, aux_in_ch=320, aux_out_ch=64)

        # ------------------------------------------------------------
        # 2.3. Auxiliary Decoder:
        # ------------------------------------------------------------
        self.aux_decoder = AuxDecoder(in_ch_num=args.fix_tx_channel_num)

        # ------------------------------------------------------------
        # 3. Latent JSCC:
        # ------------------------------------------------------------
        print("[Latent JSCC]: Initializing Latent JSCC ......")
        # self.codec = LatentCodec(args.lambda_rate) # entropy encoding module
        self.snr_list = args.snr_list
        self.tx_channel_num_list = args.tx_channel_num_list
        self.fix_tx_channel_num = args.fix_tx_channel_num


        self.latent_JSCC_encoder = Latent_JSCC_Encoder(out_ch_num=args.fix_tx_channel_num)
        self.latent_JSCC_decoder = Latent_JSCC_Decoder(in_ch_num=args.fix_tx_channel_num)

        self.wireless_channel = Wireless_Channel(channel_type=args.channel_type)

        # self.latent_JSCC = Latent_JSCC(channel_type=args.channel_type,
        #                                fixed_tx_ch_num=args.fix_tx_channel_num)


    def save_encoder_model(self, out_f):
        e_params = {}
        e_params["state_dict_vae"] = {k: v for k, v in self.vae.state_dict().items() if "lora" in k}
        e_params["state_dict_encoder_adapter"] = {k: v for k, v in self.encoder_adapter.state_dict().items()}
        e_params["state_dict_latentJSCC_encoder"] = {k: v for k, v in self.latent_JSCC_encoder.state_dict().items()}

        torch.save(e_params, out_f)

    def save_decoder_model(self, out_f):
        d_params = {}
        d_params["state_dict_latentJSCC_decoder"] = {k: v for k, v in self.latent_JSCC_decoder.state_dict().items()}
        d_params["state_dict_aux_decoder"] = {k: v for k, v in self.aux_decoder.state_dict().items()}
        d_params["state_dict_unet"] = {k: v for k, v in self.unet.state_dict().items() if "lora" in k or "conv_in" in k}
        torch.save(d_params, out_f)

    def save_model(self, e_path, d_path):
        self.save_encoder_model(e_path)
        self.save_decoder_model(d_path)


    def load_encoder_model(self, e_path):
        contents = torch.load(e_path, map_location="cpu")
        # 1. the vae of SD-Turbo:
        _sd_vae = self.vae.state_dict()
        for k in contents["state_dict_vae"]:
            _sd_vae[k] = contents["state_dict_vae"][k]
        self.vae.load_state_dict(_sd_vae)

        # 2. the adapters:
        _encoder_adapter = self.encoder_adapter.state_dict()
        for k in contents["state_dict_encoder_adapter"]:
            _encoder_adapter[k] = contents["state_dict_encoder_adapter"][k]
        self.encoder_adapter.load_state_dict(_encoder_adapter)

        # 3. the latent JSCC encoder:
        _latent_jscc_encoder = self.latent_JSCC_encoder.state_dict()
        for k in contents["state_dict_latentJSCC_encoder"]:
            _latent_jscc_encoder[k] = contents["state_dict_latentJSCC_encoder"][k]
        self.latent_JSCC_encoder.load_state_dict(_latent_jscc_encoder)

    def load_decoder_model(self, d_path):
        contents = torch.load(d_path, map_location="cpu")
        # 1. the latent JSCC decoder:
        _latent_jscc_decoder = self.latent_JSCC_decoder.state_dict()
        for k in contents["state_dict_latentJSCC_decoder"]:
            _latent_jscc_decoder[k] = contents["state_dict_latentJSCC_decoder"][k]
        self.latent_JSCC_decoder.load_state_dict(_latent_jscc_decoder)

        # 2. the auxiliary decoder:
        _aux_decoder = self.aux_decoder.state_dict()
        for k in contents["state_dict_aux_decoder"]:
            _aux_decoder[k] = contents["state_dict_aux_decoder"][k]
        self.aux_decoder.load_state_dict(_aux_decoder)

        # 3. the unet of SD-Turbo:
        _sd_unet = self.unet.state_dict()
        for k in contents["state_dict_unet"]:
            _sd_unet[k] = contents["state_dict_unet"][k]
        self.unet.load_state_dict(_sd_unet)

    def load_model(self, e_path, d_path):
        self.load_encoder_model(e_path)
        self.load_decoder_model(d_path)

    def set_encoder_eval(self):
        self.vae.eval()
        self.aux_encoder.eval()
        self.encoder_adapter.eval()
        self.latent_JSCC_encoder.eval()

        self.vae.requires_grad_(False)
        self.aux_encoder.requires_grad_(False)
        self.encoder_adapter.requires_grad_(False)
        self.latent_JSCC_encoder.requires_grad_(False)

    def set_decoder_eval(self):
        self.latent_JSCC_decoder.eval()
        self.aux_decoder.eval()
        self.unet.eval()

        self.latent_JSCC_decoder.requires_grad_(False)
        self.aux_decoder.requires_grad_(False)
        self.unet.requires_grad_(False)

    def set_eval(self):
        self.set_encoder_eval()
        self.set_decoder_eval()

    def set_encoder_train(self):
        self.vae.train()
        self.aux_encoder.eval()
        self.encoder_adapter.train()
        self.latent_JSCC_encoder.train()

        self.vae.requires_grad_(False)
        self.aux_encoder.requires_grad_(False)
        self.encoder_adapter.requires_grad_(True)
        self.latent_JSCC_encoder.requires_grad_(True)

        for n, _p in self.vae.named_parameters():
            if "lora" in n:
                _p.requires_grad = True


    def set_decoder_train(self):
        self.latent_JSCC_decoder.train()
        self.aux_decoder.train()
        self.unet.train()

        self.latent_JSCC_decoder.requires_grad_(True)
        self.aux_decoder.requires_grad_(True)
        self.unet.requires_grad_(False)

        for n, _p in self.unet.named_parameters():
            if "lora" in n:
                _p.requires_grad = True

        self.unet.conv_in.requires_grad_(True)

    def set_train(self):
        self.set_encoder_train()
        self.set_decoder_train()


    def encoding(self, x, given_rate=None):
        # B = x.shape[0]
        # Encoder
        with torch.no_grad():
            # because ELIC require the image in [0, 1],
            aux_latent = self.aux_encoder((x + 1) / 2).detach() # [B, C, H, W]-> [B, 320, H/16, W/16]: [B, 320, 32, 32]
            # pos_caption_enc = self.pos_caption_enc.to(x.device).expand(B, -1, -1)
        # .latent_dist.mode(): take the mean, not sampling.
        main_latent = self.vae.encode(x).latent_dist.mode() * self.vae.config.scaling_factor  # [B, C, H, W]-> [B, 4, H/8, W/8]=[1, 4, 64, 64]


        # aux encoder: [B, C, H, W]-> [B, 320, H/16, W/16], ==> adapter: [B, 64, H/16, W/16]
        # main encoder: [B, C, H, W]-> [B, 4, H/8, W/8], ==> adapter: [B, 128, H/16, W/16]
        # concanate: [B, 192, H/16, W/16], thus: out_cbr = 192/ (256 * 3) = 0.25
        # encoder adapter: main_latent, aux_latent
        latent = self.encoder_adapter(main_latent, aux_latent)
        out_cbr = latent.numel() / x.numel()  # this part is real -> real

        # latent JSCC encoder:
        z, latent_cbr = self.latent_JSCC_encoder(latent)

        cbr_value = out_cbr * latent_cbr

        return z, cbr_value


    def decoding(self, z_hat):
        B = z_hat.shape[0]
        with torch.no_grad():
            pos_caption_enc = self.pos_caption_enc.to(z_hat.device).expand(B, -1, -1)

        res_aux = self.aux_decoder(z_hat)
        x_hat = self.latent_JSCC_decoder(z_hat)

        # One-Step Denoiser
        timesteps = self.timesteps.to(z_hat.device)
        model_pred = self.unet(x_hat, timesteps, encoder_hidden_states=pos_caption_enc).sample  # [1, 4, 64, 64]

        self.sched.set_timesteps(1, device=z_hat.device)
        self.sched.alphas_cumprod = self.sched.alphas_cumprod.to(device=z_hat.device)

        x_denoised = self.sched.step(model_pred, timesteps, x_hat[:, :4], return_dict=True).prev_sample + res_aux

        # Decoder
        output_image = (self.vae.decode(x_denoised / self.vae.config.scaling_factor).sample).clamp(-1, 1)

        return output_image


    def forward(self, x, given_snr=None, given_rate=None):

        if given_snr is not None:
            snr = given_snr
        else:
            snr = random.choice(self.snr_list)

        # 1. encoding:
        z, cbr_value = self.encoding(x, given_rate=given_rate)

        # 2. pass the wireless channel:
        z_hat, _ = self.wireless_channel(z, snr=snr)

        # 3. decoding:
        output_image = self.decoding(z_hat)

        return output_image, cbr_value, snr


    def set_prompt(self, pos_prompt):
        with torch.no_grad():
            caption_tokens = self.tokenizer(pos_prompt,
                                            max_length=self.tokenizer.model_max_length,
                                            padding="max_length",
                                            truncation=True,
                                            return_tensors="pt").input_ids.to(next(self.text_encoder.parameters()).device)

            pos_caption_enc = self.text_encoder(caption_tokens)[0].detach()

        self.register_buffer("pos_caption_enc", pos_caption_enc, persistent=False)


    # def set_prompt(self, pos_prompt):
    #     caption_tokens = self.tokenizer(pos_prompt, max_length=self.tokenizer.model_max_length, padding="max_length",
    #                                     truncation=True, return_tensors="pt").input_ids.cuda()
    #     self.pos_caption_enc = self.text_encoder(caption_tokens)[0]







