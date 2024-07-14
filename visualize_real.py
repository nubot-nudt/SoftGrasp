import mj_envs
import click 
import os
import gym
import json
import torch
import numpy as np
import pickle
import time
import matplotlib.pyplot as plt
import torchvision.transforms as T
from einops import rearrange
from PIL import Image
# from mjrl.utils.gym_env import GymEnv
# from src.models.imi_models_real import Actor_share,Actor,Actor_share_score,Actor_share_new
from src.models.SoftGrasp_models_real import Actor
from src.models.encoders import (
    make_image_encoder,
    make_torque_encoder,
    make_angle_encoder,
    make_share_encoder,
    make_share_POS_encoder,
    make_torque_Proprioceptionencoder,
    make_angle_Proprioceptionencoder,
)
from torch.utils.data import DataLoader,RandomSampler,IterableDataset
from hand_relocate_v1_mujoco import make_sim_env
from PIL import Image
# from robot.real_env import make_real_env_hand_arm,make_real_env_hand
from robot.real_env import make_real_env_hand
import cv2
# from matplotlib import rcParams
# config = {
#     "font.family":'serif',
#     "font.serif":['Times'],
#     "font.size": 14,
#     "mathtext.fontset":'stix',
#     "font.serif": ['SimSun'],
#     'text.usetex':True
# }
# rcParams.update(config)

def main(args):
    camera_names = args.camera_names
    resized_height_v = args.resized_height_v
    resized_width_v = args.resized_width_v
    resized_height_t = args.resized_height_t
    resized_width_t = args.resized_width_t
    I_encoder = make_image_encoder(args.encoder_dim)
    if "mlp" in args.encoder:
        if args.use_one_hot:
            T_encoder = make_torque_Proprioceptionencoder(args.one_hot_torque_dim, args.encoder_dim )
            A_encoder = make_angle_Proprioceptionencoder(args.one_hot_angle_dim, args.encoder_dim )
            share_encoder = make_share_POS_encoder(args.share_dim, args.encoder_dim )
        else:
            T_encoder = make_torque_Proprioceptionencoder(args.torque_dim, args.encoder_dim )
            A_encoder = make_angle_Proprioceptionencoder(args.angle_dim, args.encoder_dim )
            share_encoder = make_share_POS_encoder(args.share_dim, args.encoder_dim )
    else:
        T_encoder = make_torque_encoder(args.torque_dim, args.encoder_dim * args.num_stack )
        A_encoder = make_angle_encoder(args.angle_dim, args.encoder_dim * args.num_stack )
        share_encoder = make_share_encoder(args.torque_dim , args.encoder_dim * args.num_stack)
    modalities = args.ablation.split("_")
    
    actor = Actor(I_encoder,T_encoder,A_encoder,share_encoder, args).cuda()
    
    if args.real_robot:
        e = make_real_env_hand(init_node=True, setup_robots=True, setup_base=True)
    else:
        e = make_sim_env()
    # 加载ckpt文件中的模型参数
    
    modalities = args.ablation.split("_")
    print('modalities',modalities)
    print('len(modalities) ',len(modalities) )
    if len(modalities) == 3 :
        if "a_mha" in args.use_way:
            if args.use_pos:
                checkpoint = torch.load(args.j_f_vf_m_pos_ckpt)
                print(1111)
            else :
                checkpoint = torch.load(args.j_f_vf_m_ckpt)
        else:
            checkpoint = torch.load(args.j_f_vf_ckpt)
    
    if len(modalities) == 2 :
        if "vf" in modalities and "j" in modalities:
            checkpoint = torch.load(args.v_j_ckpt)

        if "vf" in modalities and "f" in modalities:
            checkpoint = torch.load(args.v_f_ckpt)
            print(1111)

        if "f" in modalities and "j" in modalities:
            checkpoint = torch.load(args.j_f_ckpt)
    
    if len(modalities) == 1 :

        if "vf" in modalities :
            checkpoint = torch.load(args.v_ckpt)
            

        if "f" in modalities :
            checkpoint = torch.load(args.f_ckpt)

        if  "j" in modalities:
            checkpoint = torch.load(args.j_ckpt)

    state_dict = checkpoint['state_dict']
    new_state_dict = {key.replace("actor.", "", 1): value for key, value in state_dict.items()}
    loading_status = actor.load_state_dict(new_state_dict) 
    actor.eval()
    plt.ion()  # 启用Matplotlib的交互模式
    try:
        for _ in range(1):
            ts = e.reset()
            obs = ts.observation
            # print("obs:",obs)
            step = 0
            reward_sum = 0
            # if "vf" in modalities: 
            fig, ax1 = plt.subplots()  
            fig2, ax2 = plt.subplots()  
            image_brg = ts.observation['image']

            stacked_images_rgb = []
            for img_brg in image_brg:
                img_rgb = cv2.cvtColor(img_brg, cv2.COLOR_BGR2RGB)
                img_rgb = rearrange(img_rgb, 'h w c -> c h w')
                resized_image = cv2.resize(img_rgb.transpose(1, 2, 0), (args.resized_height_v, args.resized_height_t)).transpose(2, 0, 1)
                stacked_images_rgb.append(resized_image)
            stacked_images_rgb = np.array(stacked_images_rgb)

            while step < 500:  
                    step += 1
                    print('step:',step)
                    # if "j" in modalities:
                    joint_numpy = np.array(ts.observation['joint'] / args.inspire)
                    joint_tensor = torch.from_numpy(joint_numpy).float().cuda()
                    joint_tensor = joint_tensor.unsqueeze(0)
                    force_numpy = np.array((ts.observation['force'] +1000)/2000 )
                    force_tensor = torch.from_numpy(force_numpy).float().cuda()
                    force_tensor = force_tensor.unsqueeze(0)
                    print('force_tensor',force_tensor)

                    
                    images_tensor = torch.from_numpy(stacked_images_rgb / 255.0).float().cuda()
                    images_tensor = torch.unsqueeze(images_tensor, 0)

                    raw_action = actor(joint_tensor,force_tensor,images_tensor) 
                    scores = raw_action[1]
                    print('scores_tensor',scores)
                    raw_action_tensor = raw_action[2]
                    force_tensor = raw_action[3]
                    force = force_tensor.cpu().detach().numpy()
                    force = force.squeeze()
                    raw_action_cpu = raw_action_tensor.cpu()  
                    raw_action_numpy = raw_action_cpu.detach().numpy() 
                    # ts = e.step(args,scores,raw_action_numpy,weight,force)
                    ts = e.step(args,scores,raw_action_numpy,force)
                    # ts = e.step(raw_action_numpy,weight)
                    obs = ts.observation
                    if "vf" in modalities: 
                        image_data = obs['image']  # 获取更新后的图像数据
                        # image_data = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)
                        # mask = np.zeros(image_data.shape[:2], dtype=np.uint8)
                        # r1 = (resized_height_t,resized_height_v, resized_width_t,resized_width_v)
                        # mask[r1[0]:r1[1], r1[2]:r1[3]] = 255
                        # ROI_image = cv2.bitwise_and(image_data, image_data, mask=mask)
                        # plt_img.set_data(ROI_image)
                        # plt.title("更新后的图像数据")  
                        plt.pause(0.02)
                        # plt.draw()  # 强制绘制

            print(f'Total reward: {reward_sum}')
    except KeyboardInterrupt:
        print("强制退出循环")
    finally:
        # 清理部分
        if 'actor' in locals():
            del actor
        torch.cuda.empty_cache()
        plt.ioff()
        plt.close('all')
        e.reset()


if __name__ == '__main__':
    import configargparse
    p = configargparse.ArgParser()
    p.add("-c", "--config", is_config_file=True, default="conf/imi/SoftGrasp_imi_visualize.yaml")
    p.add("--batch_size", default=16)
    p.add("--lr", default=1e-4, type=float)
    p.add("--gamma", default=0.9, type=float)
    p.add("--period", default=3)
    p.add("--epochs", default=65, type=int)
    p.add("--resume", default=None)
    p.add("--num_workers", default=8, type=int)
    p.add("--conv_bottleneck", required=True, type=int)
    # p.add("--exp_name", required=True, type=str)

    p.add("--encoder_dim", required=True, type=int)
    p.add("--observation_dim", required=True, type=int)
    p.add("--torque_dim", required=True, type=int)
    p.add("--angle_dim", default=3, type=int)
    p.add("--action_dim", default=3, type=int)
    p.add("--share_dim", default=3, type=int)
    p.add("--one_hot_torque_dim", default=3, type=int)
    p.add("--one_hot_angle_dim", default=3, type=int)
    p.add("--picture_dim", default=3, type=int)
    p.add("--num_stack", required=True, type=int)
    p.add("--frameskip", required=True, type=int)

    p.add("--use_mha", default=True, action="store_true")
    p.add("--use_amha", default=True, action="store_true")
    p.add("--use_pos", required=True, type=int)
    p.add("--use_one_hot", required=True, type=int)
    p.add("--use_way", required=True)

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
    p.add("--encoder", required=True)
    p.add("--use_flow", default=False, action="store_true")
    p.add("--use_holebase", default=False, action="store_true")
    p.add("--task", type=str)
    p.add("--norm_audio", default=False, action="store_true")
    p.add("--aux_multiplier", type=float)
    p.add("--nocrop", default=False, type=int)
    p.add("--inspire", default=1000, type=int)

    ## visualize 
    p.add("--weights_visualize_interval", required=True, type=int)
    p.add("--weights_visualize", required=True, type=int)
    p.add("--scores_visualize", required=True, type=int)
    p.add("--save_scores", required=True, type=int)
    p.add("--save_weights", required=True, type=int)

    p.add("--camera_names", default="top")
    p.add("--real_robot", default="top")
    p.add("--arm", default="top")
    p.add("--ckpt", default="04-29-11:12:31-jobid=0-epoch=0-step=325.ckpt")
    p.add("--f_ckpt", default="04-29-11:12:31-jobid=0-epoch=0-step=325.ckpt")
    p.add("--v_ckpt", default="04-29-11:12:31-jobid=0-epoch=0-step=325.ckpt")
    p.add("--j_ckpt", default="04-29-11:12:31-jobid=0-epoch=0-step=325.ckpt")
    p.add("--v_j_ckpt", default="04-29-11:12:31-jobid=0-epoch=0-step=325.ckpt")
    p.add("--v_f_ckpt", default="04-29-11:12:31-jobid=0-epoch=0-step=325.ckpt")
    p.add("--j_f_ckpt", default="04-29-11:12:31-jobid=0-epoch=0-step=325.ckpt")
    p.add("--j_f_vf_m_pos_ckpt", default="04-29-11:12:31-jobid=0-epoch=0-step=325.ckpt")
    p.add("--j_f_vf_m_ckpt", default="04-29-11:12:31-jobid=0-epoch=0-step=325.ckpt")
    p.add("--j_f_vf_ckpt", default="04-29-11:12:31-jobid=0-epoch=0-step=325.ckpt")

    # transformer
    p.add("--hidden_dim", required=True, type=int)
    p.add("--dropout", required=True, type=float)
    p.add("--nheads", required=True, type=int)
    p.add("--dim_feedforward", required=True, type=int)
    p.add("--enc_layers", required=True, type=int)
    p.add("--dec_layers", required=True, type=int)
    p.add("--pre_norm", default=True, action="store_true")

    args = p.parse_args()
    args.batch_size *= torch.cuda.device_count()
    main(args)




