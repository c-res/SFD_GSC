import torch.nn as nn
import numpy as np
import torch


class Wireless_Channel(nn.Module):
    """
    Wireless channel layer for JSCC.
    Supported channel types:
        0 or 'none'     : noiseless channel
        1 or 'awgn'     : AWGN channel
        2 or 'rayleigh' : Rayleigh fading channel, in which channel is real only has amplitude, as in many existing works
        3 or 'rayleigh_complex': Rayleigh fading channel, in which channel is complex
    """

    def __init__(self, channel_type='awgn'):
        super(Wireless_Channel, self).__init__()
        if isinstance(channel_type, str):
            channel_type = channel_type.lower()

        self.chan_type = channel_type

    def gaussian_noise_layer(self, input_layer, std):
        """
        AWGN channel: y = x + n
        where:
            n = n_real + j n_imag
            n_real, n_imag ~ N(0, std^2)
        """
        device = input_layer.device
        dtype = input_layer.real.dtype
        noise_real = torch.normal(mean=0.0, std=std, size=input_layer.shape, device=device, dtype=dtype)
        noise_imag = torch.normal(mean=0.0, std=std, size=input_layer.shape, device=device, dtype=dtype)
        noise = torch.complex(noise_real, noise_imag)
        return input_layer + noise

    def rayleigh_noise_layer(self, input_layer, std, complex_rayleigh=False, return_h=False):
        """
        Rayleigh fading channel: y = h * x + n

        If complex_rayleigh=True:
            h = h_real + j h_imag
            h_real, h_imag ~ N(0, 1/2)

        If self.complex_rayleigh=False: h is real-valued Rayleigh amplitude.
        """
        device = input_layer.device
        dtype = input_layer.real.dtype
        noise_real = torch.normal(mean=0.0, std=std, size=input_layer.shape, device=device, dtype=dtype)
        noise_imag = torch.normal(mean=0.0, std=std, size=input_layer.shape, device=device, dtype=dtype)
        noise = torch.complex(noise_real, noise_imag)
        if complex_rayleigh:
            h_real = torch.normal(mean=0.0, std=1.0 / np.sqrt(2), size=input_layer.shape, device=device, dtype=dtype)
            h_imag = torch.normal(mean=0.0, std=1.0 / np.sqrt(2), size=input_layer.shape, device=device, dtype=dtype)
            h = torch.complex(h_real, h_imag)
        else:
            # Note: here h is real number, only consider the amplitude of the channel, this is widely used in existing works
            h1 = torch.normal(mean=0.0, std=1.0, size=input_layer.shape, device=device, dtype=dtype)
            h2 = torch.normal(mean=0.0, std=1.0, size=input_layer.shape, device=device, dtype=dtype)
            h = torch.sqrt(h1 ** 2 + h2 ** 2) / np.sqrt(2)

        output = input_layer * h + noise
        return (output, h) if return_h else output

    def complex_normalize(self, x, power=1.0):
        """
        Normalize real-valued latent tensor before converting to complex symbols.
        Since two real numbers form one complex symbol:
            E[|z|^2] = E[x_real^2 + x_imag^2]
                     ~= 2 * E[x^2]
        """
        # normalize for each image:
        pwr = torch.mean(x ** 2, dim=(1, 2, 3), keepdim=True) * 2.0
        pwr = pwr.clamp_min(1e-12)
        out = np.sqrt(power) * x / torch.sqrt(pwr)

        return out, pwr

    def real_to_complex(self, x):
        """
        Convert [B, C, H, W] real tensor to complex tensor [B, C//2, H, W].
        The first half of channels are real parts, and the second half are imaginary parts.
        """
        if x.dim() != 4:
            raise ValueError(f"Input must be 4D [B, C, H, W], but got shape {x.shape}")

        B, C, H, W = x.shape
        if C % 2 != 0:
            raise ValueError(f"Channel dimension C must be even to form complex symbols, but got C={C}.")

        real = x[:, : C // 2, :, :]
        imag = x[:, C // 2:, :, :]

        return torch.complex(real, imag)

    def complex_to_real(self, x_complex):
        """
        Convert complex tensor [B, C//2, H, W] back to real tensor [B, C, H, W].
        """
        return torch.cat([x_complex.real, x_complex.imag], dim=1)

    def forward(self, input, snr=10, avg_pwr=None, return_h=False):
        """
        input: real-valued latent feature, shape [B, C, H, W]
        snr: SNR in dB
        avg_pwr: optional average power. If provided, it should represent average real-valued feature power.
        """
        if return_h:
            if avg_pwr is not None:
                raise ValueError('return_h=True currently uses per-image power; omit avg_pwr.')
            return self.forward_with_csi(input, snr)

        if input.dim() != 4:
            raise ValueError(f"Input must be 4D [B, C, H, W], but got shape {input.shape}")

        if avg_pwr is not None:
            power = 1
            pwr = avg_pwr * 2
            channel_tx = power * input / torch.sqrt(pwr)
        else:
            channel_tx, pwr = self.complex_normalize(input, power=1.0)

        channel_in = self.real_to_complex(channel_tx)
        channel_output = self.complex_forward(channel_in, snr)
        channel_output = self.complex_to_real(channel_output)

        #
        snr_linear = 10 ** (snr / 10)
        sigma = np.sqrt(1.0 / (2 * snr_linear))  # Here because the real and imag part multiply sigma

        if self.chan_type in (0, 'none'):
            sigma = 0.0
        if avg_pwr is not None:
            return channel_output * torch.sqrt(avg_pwr * 2), sigma * torch.sqrt(avg_pwr * 2)
        else:
            return channel_output * torch.sqrt(pwr), sigma * torch.sqrt(pwr)


    def complex_forward(self, channel_in, snr):
        """
        Apply channel to complex-valued tensor.
        channel_in: [B, C//2, H, W], complex tensor
        snr: SNR in dB
        """
        if self.chan_type == 0 or self.chan_type == "none":
            return channel_in

        snr_linear = 10 ** (snr / 10)
        sigma = np.sqrt(1.0 / (2 * snr_linear))  # Here because the real and imag part multiply sigma
        if self.chan_type == 1 or self.chan_type == "awgn":
            return self.gaussian_noise_layer(channel_in, std=sigma)
        elif self.chan_type == 2 or self.chan_type == "rayleigh":
            return self.rayleigh_noise_layer(channel_in, std=sigma, complex_rayleigh=False)
        elif self.chan_type == 3 or self.chan_type == "rayleigh_complex":
            return self.rayleigh_noise_layer(channel_in, std=sigma, complex_rayleigh=True)
        else:
            raise ValueError(f"Unsupported channel type: {self.chan_type}")




    def forward_with_csi(self, input, snr=10):
        """Return raw received latent, real-component noise std, and complex CSI.

        Perfect receiver CSI; independent fading per complex symbol.
        The original forward() API remains unchanged.
        """
        if input.ndim != 4 or input.shape[1] % 2:
            raise ValueError('Input must be [B, C, H, W] with even C.')
        input = input.float()  # complex arithmetic outside half precision
        tx, pwr = self.complex_normalize(input)
        tx = self.real_to_complex(tx)
        std = float(10 ** (-float(snr) / 20) / np.sqrt(2.0))
        if self.chan_type in (2, 'rayleigh', 3, 'rayleigh_complex'):
            received, h = self.rayleigh_noise_layer(
                tx, std, complex_rayleigh=self.chan_type in (3, 'rayleigh_complex'),
                return_h=True)
        elif self.chan_type in (1, 'awgn'):
            received, h = self.gaussian_noise_layer(tx, std), torch.ones_like(tx)
        elif self.chan_type in (0, 'none'):
            received, h, std = tx, torch.ones_like(tx), 0.0
        else:
            raise ValueError(f'Unsupported channel type: {self.chan_type}')
        scale = pwr.sqrt()
        return self.complex_to_real(received) * scale, std * scale, h

    def ls_equalize(self, received, sigma, h):
        """Exact LS (zero forcing) for known, nonzero, per-symbol complex h.

        Returns real-packed y/h and its local per-real-component noise std.
        No denominator clipping, regularization, or prior power is used.
        Exactly zero/nonfinite h is rejected instead of silently changing LS.
        """
        y = self.real_to_complex(received.float())
        h = torch.as_tensor(h, device=y.device, dtype=y.dtype)
        h = torch.broadcast_to(h, y.shape)
        amplitude = h.abs()
        if not torch.isfinite(h).all() or (amplitude == 0).any():
            raise ValueError('LS requires finite, nonzero h; an erased symbol cannot be inverted.')
        sigma = torch.as_tensor(sigma, device=y.device, dtype=y.real.dtype)
        if sigma.ndim == 1:
            sigma = sigma.reshape(-1, 1, 1, 1)
        if not torch.isfinite(sigma).all() or (sigma < 0).any():
            raise ValueError('sigma must be finite and nonnegative.')
        equalized = y / h
        noise_std = torch.broadcast_to(sigma / amplitude, y.shape)
        if not torch.isfinite(equalized).all() or not torch.isfinite(noise_std).all():
            raise FloatingPointError('LS overflow in a deep fade; no clipping was applied.')
        return self.complex_to_real(equalized), torch.cat([noise_std, noise_std], dim=1)
