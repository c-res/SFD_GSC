# import math
# from torch import Tensor
# from typing import NamedTuple

import torch
import torch.nn as nn
# import sys
# sys.path.append("..")


from Models.wireless_channel import Wireless_Channel


class InceptionDWConv2d(nn.Module):
    def __init__(self, split_indexes, square_kernel_size=3, band_kernel_size=11):
        super().__init__()
        # [B, C1, H, W] -> [B, C1, H, W]
        self.dwconv_hw = nn.Conv2d(split_indexes[1], split_indexes[1], square_kernel_size, padding=square_kernel_size//2, groups=split_indexes[1])
        # [B, C2, H, W] -> [B, C2, H, W]
        self.dwconv_w = nn.Conv2d(split_indexes[2], split_indexes[2], kernel_size=(1, band_kernel_size), padding=(0, band_kernel_size//2), groups=split_indexes[2])
        # [B, C3, H, W] -> [B, C3, H, W]
        self.dwconv_h = nn.Conv2d(split_indexes[3], split_indexes[3], kernel_size=(band_kernel_size, 1), padding=(band_kernel_size//2, 0), groups=split_indexes[3])
        self.split_indexes = split_indexes

    def forward(self, x):
        # x: [B, C, H, W] --> out: [B, C, H, W]
        id, x_hw, x_w, x_h = torch.split(x, self.split_indexes, dim=1)
        return torch.cat((id, self.dwconv_hw(x_hw), self.dwconv_w(x_w), self.dwconv_h(x_h)), dim=1)

class InceptionNeXt(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.depthconv = InceptionDWConv2d((in_ch - (in_ch // 8) * 3, in_ch // 8, in_ch // 8, in_ch // 8)) # [B, C, H, W] -> [B, C, H, W]
        self.conv1 = nn.Conv2d(in_ch, in_ch * 2, 1) # [B, C, H, W] -> [B, 2C, H, W]
        self.conv2 = nn.Conv2d(in_ch * 2, in_ch, 1) # [B, 2C, H, W] -> [B, C, H, W]
        self.act = nn.GELU()

    def forward(self, x):
        # x: [B, C, H, W] --> out: [B, C, H, W]
        shortcut = x
        x = self.depthconv(x)
        x = self.conv1(x)
        x = self.act(x)
        x = self.conv2(x)
        return x + shortcut

class GatedCNNBlock(nn.Module):
    def __init__(self, in_ch, expansion_ratio=2):
        super().__init__()
        self.norm = nn.LayerNorm(in_ch, eps=1e-6)
        hidden = int(expansion_ratio * in_ch)
        self.fc1 = nn.Conv2d(in_ch, hidden * 2, 1)
        self.act = nn.GELU()
        self.conv = nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, groups=hidden)
        self.fc2 = nn.Conv2d(hidden, in_ch, 1)

    def forward(self, x):
        # x: [B, C, H, W] --> out: [B, C, H, W]
        shortcut = x
        x = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2) # [B, C, H, W] -> [B, C, H, W]
        x1, x2 = self.fc1(x).chunk(2, 1) # [B, C, H, W] -> x1, x2: [B, hidden, H, W]
        x = self.fc2(self.act(x1) * self.conv(x2)) # -> [B, C, H, W]
        return x + shortcut

class BasicBlock(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.blocks = nn.Sequential(
            InceptionNeXt(in_ch),
            GatedCNNBlock(in_ch),
        )

    def forward(self, x):
        # x: [B, C, H, W] --> out: [B, C, H, W]
        x = self.blocks(x)
        return x

class Downsample(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1), # [B, C, H, W] -> [B, C, H, W]
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=2, padding=1), # [B, C, H, W] -> [B, C, H/2, W/2]
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1), # [B, C, H, W] -> [B, C, H, W]
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=5, stride=2, padding=2, groups=out_ch), # [B, C, H, W] -> [B, C, H/2, W/2]
        )

    def forward(self, x):
        # x: [B, C, H, W] --> out: [B, C, H/2, W/2]
        return self.branch1(x) + self.branch2(x)

class Upsample(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1), # [B, C, H, W] -> [B, C, H, W]
            nn.GELU(),
            nn.Conv2d(in_ch, out_ch * 4, kernel_size=1, padding=0), # [B, C, H, W] -> [B, 4* out_C, H, W]
            nn.PixelShuffle(2), # [B, 4* out_C, H, W]-> [B, out_C, 2H, 2W]
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=5, padding=2, groups=in_ch),
            nn.GELU(),
            nn.Conv2d(in_ch, out_ch * 4, kernel_size=1, padding=0),
            nn.PixelShuffle(2),
        )

    def forward(self, x):
        return self.branch1(x) + self.branch2(x)





class Latent_JSCC_Encoder(nn.Module):
    def __init__(self, out_ch_num=None):
        super().__init__()
        # self.pre1 and self.pre2 belong to the adapter.
        # self.pre1 = Downsample(4, 128) # [B, 4, H/8, W/8] -> [B, 128, H/16, W/16]
        # self.pre2 = nn.Conv2d(320, 64, kernel_size=3, padding=1) # [B, 320, H/16, W/16] -> [B, 64, H/16, W/16]
        self.encoder = nn.Sequential(
            BasicBlock(192), # [B, 192, H/16, W/16] -> [B, 192, H/16, W/16]
            Downsample(192, 256), # [B, 192, H/16, W/16] -> [B, 256, H/32, W/32]
            BasicBlock(256),  # [B, 256, H/32, W/32]
            # Downsample(256, 320),  # [B, 256, H/32, W/32] -> [B, 320, H/64, W/64]
            # BasicBlock(320), #  [B, 320, H/64, W/64]
        )

        self.out_ch_num = out_ch_num

        if out_ch_num is not None:
            self.out_head = nn.Conv2d(256, out_ch_num, 1)

    def forward(self, latent):
        # latent: [B, 192, H/16, W/16]
        # output: -> [B, 320, H/64, W/64]
        # x = torch.cat((self.pre1(latent), self.pre2(latent2)), dim=1) # [B, 192, H/16, W/16]
        x = self.encoder(latent)

        if self.out_ch_num is not None:
            x = self.out_head(x)

        # compute the CBR: [B, 192, H/16, W/16] -- [B, 320, H/64, W/64] -- [B, c, H/64, W/64]
        latent_cbr = x.numel() / 2 / latent.numel()

        return x, latent_cbr

class Latent_JSCC_Decoder(nn.Module):
    def __init__(self, in_ch_num=None) -> None:
        super().__init__()
        self.decoder = nn.Sequential(
            # BasicBlock(320), # [B, 320, H/64, W/64]
            # Upsample(320, 320), # [B, 320, H/64, W/64] -> [B, 320, H/32, W/32]
            BasicBlock(256), # [B, 320, H/32, W/32]
            Upsample(256, 320), # [B, 320, H/32, W/32] -> [B, 320, H/16, W/16]
            BasicBlock(320), # [B, 320, H/16, W/16]
            Upsample(320, 320), # [B, 320, H/16, W/16] -> # [B, 320, H/8, W/8]
        )

        self.in_ch_num = in_ch_num
        if in_ch_num is not None:
            self.in_head = nn.Conv2d(in_ch_num, 256, 1)

    def forward(self, x):
        if self.in_ch_num is not None:
            x = self.in_head(x)

        x = self.decoder(x)

        return x


class AuxDecoder(nn.Module):
    def __init__(self, in_ch_num=None) -> None:
        super().__init__()
        self.block = nn.Sequential(
            # BasicBlock(320),  # [B, 320, H/64, W/64]
            # Upsample(320, 256),  # [B, 320, H/64, W/64] -> [B, 256, H/32, W/32]
            BasicBlock(256),  # [B, 256, H/32, W/32] ->
            Upsample(256, 128),  # [B, 192, H/16, W/16] ->
            BasicBlock(128),  # [B, 192, H/16, W/16]
            Upsample(128, 4),  # [B, 4, H/8, W/8]
        )

        self.in_ch_num = in_ch_num
        if in_ch_num is not None:
            self.in_head = nn.Conv2d(in_ch_num, 256, 1)

    def forward(self, x):
        if self.in_ch_num is not None:
            x = self.in_head(x)

        x = self.block(x)
        return x


class Latent_JSCC(nn.Module):
    def __init__(self, channel_type, fixed_tx_ch_num=None):
        super().__init__()

        self.encoder = Latent_JSCC_Encoder(out_ch_num=fixed_tx_ch_num) # g_a and the adapter after the encoders
        self.decoder = Latent_JSCC_Decoder(in_ch_num=fixed_tx_ch_num) # g_s
        self.wireless_channel = Wireless_Channel(channel_type=channel_type)

    def forward(self, latent, given_snr=10., given_rate=None):
        # latent: the output of SD-Turbo's VAE encoder: [B, C, H, W]-> [B, 4, H/8, W/8]
        # latent2: the output of the auxiliary VAE encoder: [B, C, H, W]-> [B, 320, H/16, W/16]
        z, latent_cbr = self.encoder(latent)

        z_hat = self.wireless_channel(z, snr=given_snr)

        x_hat = self.decoder(z_hat)
        # res = self.aux(y_hat)

        return x_hat, z_hat, latent_cbr

