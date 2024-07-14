import mj_envs
import click 
import os
import gym
import torch
import numpy as np
import pickle
import time
import matplotlib.pyplot as plt
from einops import rearrange
from PIL import Image
from mjrl.utils.gym_env import GymEnv
from src.models.imi_models_real import Actor
from src.models.encoders import (
    make_vision_encoder,
    make_force_encoder,
    make_joint_encoder
)
from torch.utils.data import DataLoader,RandomSampler,IterableDataset
from hand_relocate_v1_mujoco import make_sim_env
from PIL import Image


def main(args):
    camera_names = args.camera_names
    v_encoder = make_vision_encoder(args.encoder_dim)
    f_encoder = make_force_encoder(args.force_dim,args.encoder_dim )
    j_encoder = make_joint_encoder(args.observation_dim,args.encoder_dim * args.num_stack )
    actor = Actor(v_encoder, f_encoder, j_encoder, args).cuda()
    e = make_sim_env()
    # 加载ckpt文件中的模型参数
    checkpoint = torch.load(args.ckpt)
    # txt_file_path = "checkpoint_content.txt"
    # with open(txt_file_path, 'w') as f:
    #     # 逐一将键和对应的值写入文本文件
    #     for key, value in checkpoint.items():
    #         f.write(f"Key: {key}\n")
    #         f.write(f"Value: {value}\n\n")

    # print(f"Checkpoint 内容已保存到 {txt_file_path}")
    # 打印模型结构
    # print(actor)

    # 打印检查点文件中保存的模型结构
    # print(checkpoint['state_dict']) 
    state_dict = checkpoint['state_dict']
    new_state_dict = {key.replace("actor.", "", 1): value for key, value in state_dict.items()}
    loading_status = actor.load_state_dict(new_state_dict) 
    print(loading_status)
    actor.eval()
    for _ in range(100):
        ts = e.reset()
        obs = ts.observation
        step = 0
        reward_sum = 0
        fig, ax = plt.subplots()  
        plt_img = ax.imshow(ts.observation['images']['top'])
        while step < 100:  
                step += 1
                qpos_numpy = np.array(ts.observation['observation'])
                curr_images = []
                curr_image = rearrange(ts.observation['images']['top'], 'h w c -> c h w')
                curr_images.append(curr_image)
                batch_images = np.stack(curr_images, axis=0) 
                num_stack = 1 
                Hv, Wv = batch_images.shape[-2:]  
                
                batch_images = np.repeat(batch_images, 3, axis=0)
                batch_images = np.repeat(batch_images, num_stack, axis=1) 
                batch_size = 1 
                batch_images = np.expand_dims(batch_images, axis=0)

                qpos_numpy = np.expand_dims(qpos_numpy, axis=0)
     
                qpos_tensor = torch.from_numpy(qpos_numpy).float().cuda()
                images_tensor = torch.from_numpy(batch_images / 255.0).float().cuda()
                raw_action = actor(images_tensor,qpos_tensor) 
                raw_action_tensor = raw_action[2]
                raw_action_cpu = raw_action_tensor.cpu()  
                raw_action_numpy = raw_action_cpu.detach().numpy() 
                ts = e.step(raw_action_numpy)
                obs = ts.observation
                image_data = obs['images']['top']  # 获取更新后的图像数据
                plt_img.set_data(image_data)
                plt.pause(0.02)
                plt.draw()  # 强制绘制
        
        # 在每次循环结束时打印总奖励
        print(f'Total reward: {reward_sum}')
    
    # 循环结束后重置环境
    e.reset()


if __name__ == '__main__':
    import configargparse
    p = configargparse.ArgParser()
    p.add("-c", "--config", is_config_file=True, default="conf/imi/imi_visualize.yaml")
    p.add("--batch_size", default=32)
    p.add("--lr", default=1e-4, type=float)
    p.add("--gamma", default=0.9, type=float)
    p.add("--period", default=3)
    p.add("--epochs", default=65, type=int)
    p.add("--resume", default=None)
    p.add("--num_workers", default=8, type=int)
    p.add("--conv_bottleneck", required=True, type=int)
    p.add("--exp_name", required=True, type=str)
    p.add("--encoder_dim", required=True, type=int)
    p.add("--observation_dim", required=True, type=int)
    p.add("--force_dim", required=True, type=int)
    p.add("--action_dim", default=3, type=int)
    p.add("--joint_dim", default=3, type=int)
    p.add("--num_stack", required=True, type=int)
    p.add("--frameskip", required=True, type=int)
    p.add("--use_mha", default=True, action="store_true")
    p.add("--train_csv", default="data/train.csv")
    p.add("--val_csv", default="data/val.csv")
    p.add("--data_folder", default="data/test_recordings")
    p.add("--env_name", default="hand-v1")
    p.add("--resized_height_v", required=True, type=int)
    p.add("--resized_width_v", required=True, type=int)
    p.add("--resized_height_t", required=True, type=int)
    p.add("--resized_width_t", required=True, type=int)
    p.add("--num_episode", default=None, type=int)
    p.add("--crop_percent", required=True, type=float)
    p.add("--ablation", required=True)
    p.add("--num_heads", required=True, type=int)
    p.add("--use_flow", default=False, action="store_true")
    p.add("--use_holebase", default=False, action="store_true")
    p.add("--task", type=str)
    p.add("--norm_audio", default=False, action="store_true")
    p.add("--aux_multiplier", type=float)
    p.add("--nocrop", default=False, action="store_true")
    ### visualize
    p.add("--camera_names", default="top")
    p.add("--ckpt", default="03-26-09:13:04-jobid=0-epoch=3-step=32.ckpt")
    p.add("--arm", default="top")
    p.add("--real_robot", default="top")

    args = p.parse_args()
    args.batch_size *= torch.cuda.device_count()
    main(args)




