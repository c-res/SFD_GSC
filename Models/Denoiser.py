import torch
import torch.nn as nn
import math

from Models.DiT_model import DiT_NN
# from DiT_model import DiT_NN

'''
This is to implement the VE version of SF-DiT-CE
'''

class VE_DiTDenoiser(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.predict_obj = getattr(args, 'predict_obj', 'X_predict')
        self.noise_adapter = None
        self.data_channels = args.data_channels
        self.net = DiT_NN(
            # img_size=args.img_size,
                           in_channels=args.data_channels,
                           patch_size=args.patch_size,
                           hidden_size=args.hidden_size,
                           num_heads=args.num_heads,
                           depth=args.depth,
                           mlp_ratio=args.mlp_ratio,
                           attn_drop=args.attn_drop,
                           proj_drop=args.proj_drop)

        # self.img_size = args.img_size
        self.sigma_data = args.sigma_data

        self.P_mean = args.P_mean
        self.P_std = args.P_std

        self.sigma_min = args.sigma_min
        self.sigma_max = args.sigma_max
        # ema
        self.ema_decay1 = args.ema_decay1
        self.ema_decay2 = args.ema_decay2
        self.ema_params1 = {}
        self.ema_params2 = {}

    def get_scaling_for_boundary_condition(self, sigma, sigma_data=1.0, sigma_min=0.0):
        c_skip = (sigma_data ** 2) / ((sigma - sigma_min) ** 2 + sigma_data ** 2)
        c_out = (sigma - sigma_min) * sigma_data / torch.sqrt(sigma ** 2 + sigma_data ** 2)
        c_in = 1 / torch.sqrt(sigma ** 2 + sigma_data ** 2)

        return c_skip, c_out, c_in

    def sample_sigma(self, n: int, device=None):
        rnd = torch.randn(n, device=device)
        log_sigma = (rnd * self.P_std + self.P_mean).clamp(math.log(self.sigma_min), math.log(self.sigma_max))
        sigma = log_sigma.exp()
        return log_sigma, sigma


    def forward(self, x):
        log_sigma, sigma = self.sample_sigma(x.size(0), device=x.device) # [B]
        sigma = sigma.view(-1, *([1] * (x.ndim - 1)))  # [B, 1, 1, 1]
        eps = torch.randn_like(x)
        z = x + sigma * eps
        c_skip, c_out, c_in = self.get_scaling_for_boundary_condition(sigma=sigma, sigma_data=self.sigma_data, sigma_min=0.0)
        z_in = z * c_in # multiply c_in before feeding into the net so the net's input is roughly unit scale.
        c_noise = 0.25 * log_sigma # [B]
        x_pred = self.net(z_in, c_noise)
        x_pred = c_skip * z + c_out * x_pred
        v_pred = (x_pred - z) / sigma
        loss = (v_pred + eps) ** 2  # the velocity = -eps
        loss = loss.mean(dim=(1, 2, 3)).mean()
        return loss

    def forward_score_loss(self, x):
        log_sigma, sigma = self.sample_sigma(x.size(0), device=x.device)  # [B]
        sigma = sigma.view(-1, *([1] * (x.ndim - 1)))  # [B, 1, 1, 1]
        eps = torch.randn_like(x)
        z = x + sigma * eps  #
        c_skip, c_out, c_in = self.get_scaling_for_boundary_condition(sigma=sigma, sigma_data=self.sigma_data, sigma_min=0.0)
        z_in = z * c_in
        c_noise = 0.25 * log_sigma  # [B]
        x_pred = self.net(z_in, c_noise)
        x_pred = c_skip * z + c_out * x_pred
        # --------------- score-loss: ----------------
        epsilon_true = (z - x) / sigma
        score_true = -epsilon_true / sigma
        epsilon_pred = (z - x_pred) / sigma
        score_pred = -epsilon_pred / sigma
        loss = (score_true - score_pred) ** 2
        loss = loss.mean(dim=(1, 2, 3)).mean()

        return loss


    def forward_x_loss(self, x):
        log_sigma, sigma = self.sample_sigma(x.size(0), device=x.device)  # [B]
        sigma = sigma.view(-1, *([1] * (x.ndim - 1)))  # [B, 1, 1, 1]
        eps = torch.randn_like(x)
        z = x + sigma * eps  #
        c_skip, c_out, c_in = self.get_scaling_for_boundary_condition(sigma=sigma, sigma_data=self.sigma_data,
                                                                      sigma_min=0.0)
        z_in = z * c_in
        c_noise = 0.25 * log_sigma  # [B]
        x_pred = self.net(z_in, c_noise)
        x_pred = c_skip * z + c_out * x_pred
        # --------- x-loss: --------------
        loss = (x_pred - x) ** 2
        loss = loss.mean(dim=(1, 2, 3)).mean()
        return loss

    def forward_score(self, x):
        log_sigma, sigma = self.sample_sigma(x.size(0), device=x.device)  # [B]
        sigma = sigma.view(-1, *([1] * (x.ndim - 1)))  # [B, 1, 1, 1]
        eps = torch.randn_like(x)
        z = x + sigma * eps  #
        c_skip, c_out, c_in = self.get_scaling_for_boundary_condition(sigma=sigma, sigma_data=self.sigma_data,
                                                                      sigma_min=0.0)
        z_in = z * c_in
        c_noise = 0.25 * log_sigma  # [B]
        # ---- ===== net predicts SCORE directly  ======== ----
        score_pred = self.net(z_in, c_noise)
        score_target = (x - z) / (sigma ** 2)
        loss = (score_pred - score_target) ** 2
        loss = loss.mean(dim=(1, 2, 3)).mean()
        return loss


    def forward_velocity(self, x):
        log_sigma, sigma = self.sample_sigma(x.size(0), device=x.device)  # [B]
        sigma = sigma.view(-1, *([1] * (x.ndim - 1)))  # [B, 1, 1, 1]
        eps = torch.randn_like(x)
        z = x + sigma * eps
        c_skip, c_out, c_in = self.get_scaling_for_boundary_condition(sigma=sigma, sigma_data=self.sigma_data,
                                                                      sigma_min=0.0)
        z_in = z * c_in
        c_noise = 0.25 * log_sigma  # [B]
        # ---- net predicts VELOCITY directly ----
        v_pred = self.net(z_in, c_noise)
        v_target = (z - x) / sigma  # == eps
        loss = (v_pred - v_target) ** 2
        loss = loss.mean(dim=(1, 2, 3)).mean()
        return loss


    def x0_from_score(self, z, sigma):
        """
        z:     [B,C,H,W] (or [B,2,Nr,Nt]) noisy sample
        sigma: [B,1,1,1]
        return: x0_hat same shape as z
        """
        _, _, c_in = self.get_scaling_for_boundary_condition(
            sigma=sigma, sigma_data=self.sigma_data, sigma_min=0.0
        )
        c_noise = 0.25 * torch.log(sigma[:, 0, 0, 0])  # [B]
        score_pred = self.net(z * c_in, c_noise)  # net outputs score in z-space
        x0_hat = z + (sigma ** 2) * score_pred # Tweedie's formula
        return x0_hat, score_pred


    def x0_from_velocity(self, z, sigma):
        """
        z:     [B,C,H,W] noisy sample
        sigma: [B,1,1,1]
        return: x0_hat same shape as z
        """
        _, _, c_in = self.get_scaling_for_boundary_condition(
            sigma=sigma, sigma_data=self.sigma_data, sigma_min=0.0
        )
        c_noise = 0.25 * torch.log(sigma[:, 0, 0, 0])  # [B]
        v_pred = self.net(z * c_in, c_noise)  # net outputs v = (z-x0)/sigma
        x0_hat = z - sigma * v_pred
        return x0_hat, v_pred


    def edm_sampling(self, H_noisy, sigma):
        """
        Use the model to predict the clean channel image based on noisy image
            H_noisy: [B,2,Nr,Nt] noisy channel image at noise level sigma
            sigma: [B, 1, 1, 1]
            return: the predicted image H0_hat, and the velocity:
        """
        c_skip, c_out, c_in = self.get_scaling_for_boundary_condition(sigma=sigma, sigma_data=self.sigma_data, sigma_min=0.0)
        c_noise = 0.25 * torch.log(sigma[:, 0, 0, 0]) # [B]
        H_pred = self.net(H_noisy * c_in, c_noise)
        H0_hat = c_skip * H_noisy + c_out * H_pred
        v = (H_noisy - H0_hat) / sigma # the velocity
        return H0_hat, v

    def enable_noise_conditioning(self):
        """Call AFTER loading Stage 2 weights and BEFORE building the optimizer.

        For Stage 3 reloads call BEFORE loading its state_dict instead.
        """
        if self.noise_adapter is None:
            self.noise_adapter = nn.Conv2d(self.data_channels,
                                             self.data_channels, 1)
            nn.init.zeros_(self.noise_adapter.weight)
            nn.init.zeros_(self.noise_adapter.bias)
            self.noise_adapter.to(next(self.net.parameters()))

    def denoise_awgn(self, z, sigma):
        sigma = torch.as_tensor(sigma, device=z.device, dtype=z.dtype)
        if sigma.ndim == 0:
            sigma = sigma.expand(z.shape[0]).reshape(-1, 1, 1, 1)
        elif sigma.ndim == 1:
            sigma = sigma.reshape(-1, 1, 1, 1)
        sigma = sigma.clamp_min(1e-8)
        if self.predict_obj == 'X_predict':
            return self.edm_sampling(z, sigma)
        if self.predict_obj == 'V_predict':
            return self.x0_from_velocity(z, sigma)
        if self.predict_obj == 'epsilon_predict':
            x0, _ = self.x0_from_score(z, sigma)
            return x0, (z - x0) / sigma
        raise ValueError(f'Unknown prediction objective: {self.predict_obj}')

    def denoise_ls(self, z_ls, sigma_map):
        """Denoise LS output z_ls = z + n_ls using its full noise std map.

        Keep the pretrained scalar DiT embedding (mean log noise), and inject
        local log-noise deviations using a zero-initialized 1x1 adapter.
        Local EDM coefficients use the actual sigma map without upper clipping.
        This new heterogeneous-noise path needs Stage 3 fine-tuning.
        """
        if self.predict_obj != 'X_predict':
            raise ValueError('Rayleigh LS fine-tuning requires Stage 2 X_predict weights.')
        if self.noise_adapter is None:
            raise RuntimeError('Call enable_noise_conditioning() before building the optimizer.')
        z_ls = z_ls.float()
        sigma = torch.as_tensor(sigma_map, device=z_ls.device, dtype=z_ls.dtype)
        sigma = torch.broadcast_to(sigma, z_ls.shape)
        if not torch.isfinite(sigma).all() or (sigma < 0).any():
            raise ValueError('sigma_map must be finite and nonnegative.')
        if not torch.isfinite(z_ls).all():
            raise FloatingPointError('Nonfinite LS observation.')
        sd = torch.full_like(sigma, float(self.sigma_data))
        if self.sigma_data <= 0:
            raise ValueError('sigma_data must be positive.')
        # hypot avoids overflow from explicitly squaring large LS noise stds.
        norm = torch.hypot(sigma, sd)
        c_skip = (sd / norm).square()
        c_out = (sigma / norm) * sd
        log_sigma = sigma.clamp_min(1e-12).log()
        global_log_sigma = log_sigma.mean(dim=(1, 2, 3), keepdim=True)
        local_noise = 0.25 * (log_sigma - global_log_sigma)
        c_noise = 0.25 * global_log_sigma.flatten()
        net_input = z_ls / norm + self.noise_adapter(local_noise)
        residual = self.net(net_input, c_noise)
        return c_skip * z_ls + c_out * residual

    @torch.no_grad()
    def update_ema(self):
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            # if the first time or has new parameters
            if name not in self.ema_params1:
                self.ema_params1[name] = p.detach().clone()
            else:
                self.ema_params1[name].mul_(self.ema_decay1).add_(p.data, alpha=1 - self.ema_decay1)
            if name not in self.ema_params2:
                self.ema_params2[name] = p.detach().clone()
            else:
                self.ema_params2[name].mul_(self.ema_decay2).add_(p.data, alpha=1 - self.ema_decay2)

    @torch.no_grad()
    def init_ema(self):
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            self.ema_params1[name] = p.detach().clone()
            self.ema_params2[name] = p.detach().clone()


