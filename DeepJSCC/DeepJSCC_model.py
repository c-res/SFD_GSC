# -*- coding: utf-8 -*-
"""
Deep Joint Source-Channel Coding for Wireless Image Transmission, TCCN, 2019.
1. SNR is dynamically passed to forward().
2. CBR is used to determine the latent channel number c.
3. Encoder output resolution is H/8 x W/8.
4. Supports CBR = 1/192, 1/96, 1/64, 1/48, 1/32, etc.
5. Supports AWGN and Rayleigh channels.

For an RGB image [B, 3, H, W]:
    Encoder output: [B, 2c, H/8, W/8]

The number of complex channel symbols is:
    k = c * H/8 * W/8

Therefore:
    CBR = k / (3HW) = c / 192
"""

import torch
import torch.nn as nn


class Wireless_Channel(nn.Module):
    def __init__(self, channel_type='AWGN'):
        super(Wireless_Channel, self).__init__()
        if channel_type not in ['AWGN', 'Rayleigh']:
            raise ValueError("channel_type must be 'AWGN' or 'Rayleigh'")
        self.channel_type = channel_type

    def forward(self, z_hat, snr):
        if z_hat.dim() not in {3, 4}:
            raise ValueError('Input tensor must be 3D or 4D')

        squeeze_batch = False
        if z_hat.dim() == 3:
            z_hat = z_hat.unsqueeze(0)
            squeeze_batch = True

        batch_size = z_hat.size(0)
        snr = torch.as_tensor(snr, dtype=z_hat.dtype, device=z_hat.device)

        if snr.numel() == 1:
            snr = snr.reshape(1, 1, 1, 1)
        elif snr.numel() == batch_size:
            snr = snr.reshape(batch_size, 1, 1, 1)
        else:
            raise ValueError(f'SNR must contain either 1 value or {batch_size} values.')

        sig_pwr = torch.mean(torch.abs(z_hat).square(), dim=(1, 2, 3), keepdim=True)
        snr_linear = 10 ** (snr / 10.0)
        noi_pwr = sig_pwr / snr_linear
        noise = torch.randn_like(z_hat) * torch.sqrt(noi_pwr / 2.0)

        if self.channel_type == 'Rayleigh':
            hc = torch.randn(2, dtype=z_hat.dtype, device=z_hat.device)
            z_hat = z_hat.clone()
            half_channel = z_hat.size(1) // 2
            z_hat[:, :half_channel] = hc[0] * z_hat[:, :half_channel]
            z_hat[:, half_channel:] = hc[1] * z_hat[:, half_channel:]

        received = z_hat + noise

        if squeeze_batch:
            received = received.squeeze(0)

        return received

    def get_channel(self):
        return self.channel_type


def ratio2filtersize(x: torch.Tensor, ratio):
    """
    Compute c according to the requested CBR.

    For the current architecture:
        input:  [3, H, W]
        latent: [2c, H/8, W/8]

    Number of complex channel symbols:
        k = c * (H/8) * (W/8)

    CBR:
        rho = k / (3HW)

    For dimensions divisible by 8:
        rho = c / 192
    """

    if ratio <= 0:
        raise ValueError('CBR ratio must be positive.')

    if x.dim() == 4:
        before_size = x[0].numel()
    elif x.dim() == 3:
        before_size = x.numel()
    else:
        raise ValueError('Input image must be 3D or 4D.')

    encoder_temp = Encoder(is_temp=True)

    with torch.no_grad():
        x_temp = x.unsqueeze(0) if x.dim() == 3 else x
        z_temp = encoder_temp(x_temp)

    spatial_size = z_temp.size(-2) * z_temp.size(-1)

    c_float = before_size * ratio / spatial_size
    c = int(round(c_float))

    if c < 1:
        min_ratio = spatial_size / before_size
        raise ValueError(
            f'Requested CBR={ratio:.8f} is too small for the current architecture. '
            f'Minimum achievable CBR is approximately {min_ratio:.8f}.'
        )

    actual_ratio = c * spatial_size / before_size

    if abs(actual_ratio - ratio) > 1e-8:
        print(
            f'Warning: requested CBR={ratio:.8f} cannot be represented exactly. '
            f'Using c={c}, actual CBR={actual_ratio:.8f}.'
        )

    return c


def calculate_cbr(x: torch.Tensor, c):
    """
    Calculate the actual CBR for a given input size and c.
    """

    if x.dim() == 4:
        source_size = x[0].numel()
        h = x.size(-2)
        w = x.size(-1)
    elif x.dim() == 3:
        source_size = x.numel()
        h = x.size(-2)
        w = x.size(-1)
    else:
        raise ValueError('Input image must be 3D or 4D.')

    latent_h = h // 8
    latent_w = w // 8
    complex_channel_symbols = c * latent_h * latent_w

    return complex_channel_symbols / source_size


class ConvWithPReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(ConvWithPReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.prelu = nn.PReLU()
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.conv(x)
        x = self.prelu(x)
        return x


class TransConvWithPReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, activate='prelu'):
        super(TransConvWithPReLU, self).__init__()
        self.transconv = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            output_padding
        )

        if activate == 'prelu':
            self.activate = nn.PReLU()
            nn.init.kaiming_normal_(self.transconv.weight, mode='fan_out', nonlinearity='leaky_relu')
        elif activate == 'sigmoid':
            self.activate = nn.Sigmoid()
            nn.init.xavier_normal_(self.transconv.weight)
        else:
            raise ValueError('Unknown activation type.')

    def forward(self, x):
        x = self.transconv(x)
        x = self.activate(x)
        return x


class Encoder(nn.Module):
    def __init__(self, c=1, is_temp=False, P=1.0):
        super(Encoder, self).__init__()

        self.is_temp = is_temp
        self.P = P

        # [B, 3, H, W] -> [B, 16, H/2, W/2]
        self.conv1 = ConvWithPReLU(
            in_channels=3,
            out_channels=16,
            kernel_size=5,
            stride=2,
            padding=2
        )

        # [B, 16, H/2, W/2] -> [B, 32, H/4, W/4]
        self.conv2 = ConvWithPReLU(
            in_channels=16,
            out_channels=32,
            kernel_size=5,
            stride=2,
            padding=2
        )

        # [B, 32, H/4, W/4] -> [B, 32, H/4, W/4]
        self.conv3 = ConvWithPReLU(
            in_channels=32,
            out_channels=32,
            kernel_size=5,
            stride=1,
            padding=2
        )

        # [B, 32, H/4, W/4] -> [B, 32, H/4, W/4]
        self.conv4 = ConvWithPReLU(
            in_channels=32,
            out_channels=32,
            kernel_size=5,
            stride=1,
            padding=2
        )

        # Additional downsampling:
        # [B, 32, H/4, W/4] -> [B, 32, H/8, W/8]
        self.conv5 = ConvWithPReLU(
            in_channels=32,
            out_channels=32,
            kernel_size=5,
            stride=2,
            padding=2
        )

        # Channel projection:
        # [B, 32, H/8, W/8] -> [B, 2c, H/8, W/8]
        self.conv6 = ConvWithPReLU(
            in_channels=32,
            out_channels=2 * c,
            kernel_size=5,
            stride=1,
            padding=2
        )

    def power_normalization(self, z):
        squeeze_batch = False

        if z.dim() == 3:
            z = z.unsqueeze(0)
            squeeze_batch = True
        elif z.dim() != 4:
            raise ValueError('Input tensor must be 3D or 4D.')

        k = z[0].numel()
        power = torch.sum(z.square(), dim=(1, 2, 3), keepdim=True)
        scale = torch.sqrt(torch.tensor(self.P * k, dtype=z.dtype, device=z.device))
        z = scale * z / torch.sqrt(power + 1e-8)

        if squeeze_batch:
            z = z.squeeze(0)

        return z

    def forward(self, x):
        # Input:
        # [B, 3, H, W]
        #
        # Output:
        # [B, 2c, H/8, W/8]
        #
        # CBR:
        # rho = c / 192

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        if self.is_temp:
            return x

        x = self.conv6(x)
        x = self.power_normalization(x)

        return x


class Decoder(nn.Module):
    def __init__(self, c=1):
        super(Decoder, self).__init__()

        # [B, 2c, H/8, W/8] -> [B, 32, H/8, W/8]
        self.tconv1 = TransConvWithPReLU(
            in_channels=2 * c,
            out_channels=32,
            kernel_size=5,
            stride=1,
            padding=2
        )

        # [B, 32, H/8, W/8] -> [B, 32, H/8, W/8]
        self.tconv2 = TransConvWithPReLU(
            in_channels=32,
            out_channels=32,
            kernel_size=5,
            stride=1,
            padding=2
        )

        # [B, 32, H/8, W/8] -> [B, 32, H/8, W/8]
        self.tconv3 = TransConvWithPReLU(
            in_channels=32,
            out_channels=32,
            kernel_size=5,
            stride=1,
            padding=2
        )

        # [B, 32, H/8, W/8] -> [B, 32, H/4, W/4]
        self.tconv4 = TransConvWithPReLU(
            in_channels=32,
            out_channels=32,
            kernel_size=5,
            stride=2,
            padding=2,
            output_padding=1
        )

        # [B, 32, H/4, W/4] -> [B, 16, H/2, W/2]
        self.tconv5 = TransConvWithPReLU(
            in_channels=32,
            out_channels=16,
            kernel_size=5,
            stride=2,
            padding=2,
            output_padding=1
        )

        # [B, 16, H/2, W/2] -> [B, 3, H, W]
        self.tconv6 = TransConvWithPReLU(
            in_channels=16,
            out_channels=3,
            kernel_size=5,
            stride=2,
            padding=2,
            output_padding=1,
            activate='sigmoid'
        )

    def forward(self, x):
        x = self.tconv1(x)
        x = self.tconv2(x)
        x = self.tconv3(x)
        x = self.tconv4(x)
        x = self.tconv5(x)
        x = self.tconv6(x)
        return x


class DeepJSCC(nn.Module):
    def __init__(self, c, channel_type='AWGN'):
        super(DeepJSCC, self).__init__()

        self.c = c
        self.encoder = Encoder(c=c)
        self.channel = Wireless_Channel(channel_type=channel_type)
        self.decoder = Decoder(c=c)

    def forward(self, x, snr):
        z = self.encoder(x)
        z_hat = self.channel(z, snr=snr)
        x_hat = self.decoder(z_hat)
        return x_hat

    def encode(self, x):
        return self.encoder(x)

    def channel_forward(self, z, snr):
        return self.channel(z, snr=snr)

    def decode(self, z_hat):
        return self.decoder(z_hat)

    def change_channel(self, channel_type='AWGN'):
        self.channel = Wireless_Channel(channel_type=channel_type)

    def get_channel(self):
        return self.channel.get_channel()

    @staticmethod
    def loss(prd, gt):
        criterion = nn.MSELoss(reduction='mean')
        return criterion(prd, gt)