import torch
from src.datasets.SoftGrasp_dataset import ImitationEpisode
from torch.utils.data import SequentialSampler, BatchSampler
from src.models.encoders import (
    make_image_encoder,
    make_torque_Proprioceptionencoder,
    make_angle_Proprioceptionencoder,
)
from src.models.SoftGrasp_models import Actor
from src.engines.engine import ImiEngine
from torch.utils.data import DataLoader,RandomSampler
from src.train_utils import save_config, start_training
import pandas as pd
import numpy as np
from torch.utils.data import Sampler
from torch.utils.tensorboard import SummaryWriter
torch.multiprocessing.set_sharing_strategy("file_system")


def strip_sd(state_dict, prefix):
    """
    strip prefix from state dictionary
    """
    return {k.lstrip(prefix): v for k, v in state_dict.items() if k.startswith(prefix)}


def main(args):
    train_csv = pd.read_csv(args.train_csv)
    val_csv = pd.read_csv(args.val_csv)
    test_csv = pd.read_csv(args.test_csv)
    
    if args.num_episode is None:
        train_num_episode = len(train_csv)
        val_num_episode = len(val_csv)
        test_num_episode = len(test_csv)
    else:
        train_num_episode = args.num_episode
        val_num_episode = args.num_episode
        test_num_episode = args.num_episode
    train_set = torch.utils.data.ConcatDataset(
        [
            ImitationEpisode(args.train_csv, args, i, args.data_folder)
            for i in range(train_num_episode)
        ]
    )
    val_set = torch.utils.data.ConcatDataset(
        [
            ImitationEpisode(args.val_csv, args, i, args.data_folder, False)
            for i in range(val_num_episode)
        ]
    )
    test_set = torch.utils.data.ConcatDataset(
        [
            ImitationEpisode(args.test_csv, args, i, args.data_folder, False)
            for i in range(test_num_episode)
        ]
    )

    train_loader = DataLoader(train_set, args.batch_size, num_workers=8)
    val_loader = DataLoader(val_set, args.batch_size, num_workers=8, shuffle=False)
    test_loader = DataLoader(test_set, 1, num_workers=8, shuffle=False)
    I_encoder = make_image_encoder(args.encoder_dim)

    if args.use_one_hot:
        T_encoder = make_torque_Proprioceptionencoder(args.one_hot_torque_dim, args.encoder_dim )
        A_encoder = make_angle_Proprioceptionencoder(args.one_hot_angle_dim, args.encoder_dim )
    else:
        T_encoder = make_torque_Proprioceptionencoder(args.torque_dim, args.encoder_dim )
        A_encoder = make_angle_Proprioceptionencoder(args.angle_dim, args.encoder_dim )
    
    imi_model = Actor(I_encoder,T_encoder,A_encoder, args).cuda()
    optimizer = torch.optim.Adam(imi_model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=args.period, gamma=args.gamma
    )

    exp_dir = save_config(args)
    pl_module = ImiEngine(
        imi_model, optimizer, train_loader, val_loader, test_loader,scheduler, args
    )

    start_training(args, exp_dir, pl_module)


if __name__ == "__main__":
    import configargparse

    p = configargparse.ArgParser()
    p.add("-c", "--config", is_config_file=True, default="conf/imi/SoftGrasp_imi_learn.yaml")
    p.add("--batch_size", default=32, type=int)
    p.add("--lr", default=1e-4, type=float)
    p.add("--gamma", default=0.9, type=float)
    p.add("--gradient_clip_val", default=0.5, type=float)
    p.add("--period", default=3)
    p.add("--epochs", default=2000, type=int)
    p.add("--num_trials", default=2000, type=int)
    p.add("--resume", default=None)
    p.add("--num_workers", default=8, type=int)
    # imi_stuff
    p.add("--conv_bottleneck", required=True, type=int)
    p.add("--encoder_dim", default=8, type=int)
    p.add("--observation_dim", default=8, type=int)
    p.add("--torque_dim", required=True, type=int)
    p.add("--angle_dim", default=3, type=int)
    p.add("--action_dim", default=3, type=int)
    p.add("--one_hot_torque_dim", default=3, type=int)
    p.add("--one_hot_angle_dim", default=3, type=int)
    p.add("--picture_dim", default=3, type=int)
    p.add("--share_dim", default=3, type=int)
    p.add("--num_stack", default=3, type=int)
    p.add("--frameskip", default=3, type=int)
    p.add("--use_pos", required=True, type=int)
    p.add("--use_one_hot", required=True, type=int)
    p.add("--use_way", required=True)

    # data
    p.add("--train_csv", default="data/data_csv/train.csv")
    p.add("--val_csv", default="data/data_csv/val.csv")
    p.add("--test_csv", default="data/data_csv/test.csv")
    p.add("--data_folder", default="data/test_recordings")
    p.add("--resized_height_v", default=3, type=int)
    p.add("--resized_width_v", default=3, type=int)
    p.add("--resized_height_t", default=3, type=int)
    p.add("--resized_width_t", default=3, type=int)
    p.add("--num_episode", default=None, type=int)
    p.add("--crop_percent", required=True, type=float)
    p.add("--ablation", required=True)
    p.add("--encoder", required=True)
    p.add("--num_heads", default=3, type=int)
    p.add("--use_flow", default=True, action="store_true")
    p.add("--use_holebase", default=True, action="store_true")
    p.add("--task", type=str)
    p.add("--norm_audio", default=True, action="store_true")
    p.add("--aux_multiplier", type=float)
    p.add("--nocrop",required=True, type=int)
    p.add("--ROI",required=True, type=int)
    p.add("--ckpt", default="04-29-11:12:31-jobid=0-epoch=0-step=325.ckpt")
    p.add("--train", default=True, action="store_true")
    p.add("--mask_height_t", required=True, type=int)
    p.add("--mask_height_v", required=True, type=int)
    p.add("--mask_width_t", required=True, type=int)
    p.add("--mask_width_v", required=True, type=int)
    ### loss
    p.add("--a_loss", required=True, type=int)
    p.add("--f_loss", required=True, type=int)
    p.add("--loss", default="l1")
    p.add("--delta", default=0.5, type=float)
    p.add("--fix_loss_delta", default=0.5, type=float)

    ## action type
    p.add("--dis_actions", required=True, type=int)
    ## data
    p.add("--train_data", type=str)
    p.add("--val_data", type=str)

    
    args = p.parse_args()
    args.batch_size *= torch.cuda.device_count()
    main(args)
