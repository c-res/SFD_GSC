import os
import sys
import math
import torch
import torch.utils.checkpoint
import argparse
os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'INFO'
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from torch.utils.data import DataLoader
from torchvision import transforms
from accelerate.utils import set_seed
from diffusers.utils.import_utils import is_xformers_available
from Datasets.dataset_H5 import H5Dataset
from Models.DM_JSCC_sepED import DM_JSCC
from Models.Denoiser import VE_DiTDenoiser
from Models.DiT_utils import add_weight_decay


def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--patch_size', default=[4, 4], nargs='+', type=int, help='patch size')
    parser.add_argument('--hidden_size', default=128, type=int, help='The hidden dimension of the model')
    parser.add_argument('--num_heads', default=8, type=int, help='The number of attention heads')
    parser.add_argument('--depth', default=2, type=int, help='The number of Transformer blocks')
    parser.add_argument('--mlp_ratio', default=4, type=int, help='The ratio of hidden dim of the MLP')
    parser.add_argument('--attn_drop', type=float, default=0.0, help='Attention dropout rate')
    parser.add_argument('--proj_drop', type=float, default=0.0, help='Projection dropout rate')

    parser.add_argument('--data_channels', default=128, type=int, help="the dimension of the latent feature") # TODO:
    parser.add_argument('--sigma_data', default=1.0, type=float, help="std of channel image")
    parser.add_argument('--num_workers', default=8, type=int, help="for dataloader")
    parser.add_argument('--train_batch_size', default=64, type=int, help="the training batch size")

    parser.add_argument("--max_train_steps", type=int, default=20000)
    parser.add_argument('--weight_decay', default=0.000, type=float, help="weight decay of optimizer")
    parser.add_argument('--lr', default=0.0001, type=float, help="learning rate")
    parser.add_argument('--P_mean',  default=-1.2, type=float, help='Hyperparameter for noise schedule')
    parser.add_argument('--P_std', default=1.2, type=float, help='Hyperparameter for noise schedule')
    parser.add_argument('--sigma_min', default=0.002, type=float, help='Hyperparameter for noise schedule')
    parser.add_argument('--sigma_max', default=10, type=float, help='Hyperparameter for noise schedule')
    parser.add_argument('--ema_decay1', type=float, default=0.9999, help='The first ema to track. Use the first ema for sampling by default.')
    parser.add_argument('--ema_decay2', type=float, default=0.9996, help='The second ema to track')
    parser.add_argument('--gpu', type=int, default=1, help="GPU ID, -1 for CPU")

    parser.add_argument('--predict_obj', default='X_predict', choices=['X_predict', 'V_predict', 'epsilon_predict'],
                        help="The network prediction objective")
    parser.add_argument('--loss_type', default='V_loss', choices=['V_loss', 'X_loss', 'epsilon_loss'],
                        help="The loss type")

    parser.add_argument('--channel_type', default='awgn', type=str, choices=['awgn', 'rayleigh', 'rayleigh_complex'],
                        help='channel')
    return parser.parse_args()


def main():

    args = args_parser()
    device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    log_path = '../checkpoints/' + 'Denoiser_' + str(args.data_channels)
    os.makedirs(log_path, exist_ok=True)
    # =========================
    # load the JSCC model and training configs
    # =============================
    parent_dir = os.path.dirname(os.getcwd())
    DMJSCC_checkpoint_path = parent_dir + '/checkpoints/' + 'transceiver_basic_none_' + str(args.data_channels) + '/OneDM_basic_iter50000'

    # the DMJSCC training configs:
    DMJSCC_train_config_path = DMJSCC_checkpoint_path + '/config.pt'
    DMJSCC_train_config = torch.load(DMJSCC_train_config_path, map_location="cpu", weights_only=False)['args']

    # the encoder params:
    DMJSCC_encoder_path = DMJSCC_checkpoint_path + '/encoder.pt'
    DMJSCC_decoder_path = DMJSCC_checkpoint_path + '/decoder.pt'
    if DMJSCC_train_config.sd_path is None:
        from huggingface_hub import snapshot_download
        sd_path = snapshot_download(repo_id="stabilityai/sd-turbo")
    else:
        sd_path = DMJSCC_train_config.sd_path

    net = DM_JSCC(sd_path=sd_path, args=DMJSCC_train_config)

    # load model parameters:
    net.load_model(DMJSCC_encoder_path, DMJSCC_decoder_path)
    net = net.to(device)
    net.set_eval()

    if DMJSCC_train_config.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            net.unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available, please install it by running `pip install xformers`")


    if DMJSCC_train_config.seed is not None:
        set_seed(DMJSCC_train_config.seed)

    # =========================
    # Dataset / Dataloader
    # =========================
    train_dataset = H5Dataset(
        DMJSCC_train_config.train_dataset,
        transform=transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomCrop((DMJSCC_train_config.train_patch_size, DMJSCC_train_config.train_patch_size)),
            transforms.RandomHorizontalFlip(),
            # transforms.RandomVerticalFlip(),
            transforms.ToTensor(), # [0, 1]
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]), # [-1, 1]
        ])
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size, # per GPU batch size
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
    )

    denoiser_model = VE_DiTDenoiser(args)

    print("Model =", denoiser_model)
    n_params = sum(p.numel() for p in denoiser_model.parameters() if p.requires_grad)
    print("Number of trainable parameters: {:.6f}M".format(n_params / 1e6))
    denoiser_model.to(device)
    # set up optimizer:
    param_groups = add_weight_decay(denoiser_model, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)


    global_step = 0
    while global_step < args.max_train_steps:
        for batch in train_dataloader:
            batch = batch.to(device)  # [B, 3, 256, 256]

            # input to the encoder to get the latent feature for transmission:
            z, cbr_value = net.encoding(batch, given_rate=None) # [4, 192, 8, 8]

            # use the latent z to train the denoiser:
            if args.predict_obj == 'X_predict':
                if args.loss_type == 'V_loss':
                    loss = denoiser_model(z)
                elif args.loss_type == 'X_loss':
                    loss = denoiser_model.forward_x_loss(z)
                elif args.loss_type == 'epsilon_loss':
                    loss = denoiser_model.forward_score_loss(z)
                else:
                    raise ValueError(f"Unknown loss type: {args.loss_type}")
            elif args.predict_obj == 'V_predict':
                loss = denoiser_model.forward_velocity(z)
            elif args.predict_obj == 'epsilon_predict':
                loss = denoiser_model.forward_score(z)
            else:
                raise ValueError(f"Unknown Network Prediction Objective: {args.predict_obj}")

            #
            loss_value = loss.item()

            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value))
                sys.exit(1)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            global_step = global_step + 1

            # save checkpoints periodically:
            print('global_step=', global_step, 'loss=', loss.item())
            save_freq = 2000
            if global_step % save_freq == 0 or global_step == args.max_train_steps:
                model_save_name = 'VE_DiT' + 'iter' + str(global_step) + '.pt'
                torch.save({'model_state': denoiser_model.state_dict(),
                            'config': args},
                           os.path.join(log_path, model_save_name))





if __name__ == "__main__":
    main()
