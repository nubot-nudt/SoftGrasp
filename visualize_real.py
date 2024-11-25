import click 
import os
import json
import torch
import numpy as np
import pickle
import time
import matplotlib.pyplot as plt
import torchvision.transforms as T
from einops import rearrange
from PIL import Image
from src.models.SoftGrasp_models_real import Actor
from src.models.encoders import (
    make_image_encoder,
    make_torque_Proprioceptionencoder,
    make_angle_Proprioceptionencoder,
)
from torch.utils.data import DataLoader,RandomSampler,IterableDataset
from PIL import Image
from robot.real_env import make_real_env_hand
import cv2


def stack_images(image_brg, num_stack, resized_height_v, resized_height_t):
    stacked_images_rgb = []
    print('len(image_brg):',len(image_brg))
    for i in range(0, len(image_brg) - num_stack + 1):
        current_images = image_brg[i:i +  num_stack]
        stack = []
        for img_brg in current_images:
            img_rgb = cv2.cvtColor(img_brg, cv2.COLOR_BGR2RGB)
            img_rgb = rearrange(img_rgb, 'h w c -> c h w')
            
            try:
                resized_image = cv2.resize(img_rgb.transpose(1, 2, 0), (resized_height_v, resized_height_t)).transpose(2, 0, 1)
                stack.append(resized_image)
            except Exception as e:
                print(f"Error processing image: {e}")
        stacked_images_rgb.append(np.stack(stack, axis=0))
    
    stacked_images_rgb = np.array(stacked_images_rgb)
    
    return stacked_images_rgb

def main(args):
    camera_names = args.camera_names
    resized_height_v = args.resized_height_v
    resized_width_v = args.resized_width_v
    resized_height_t = args.resized_height_t
    resized_width_t = args.resized_width_t
    num_stack = args.num_stack
    I_encoder = make_image_encoder(args.encoder_dim)
    if args.use_one_hot:
        T_encoder = make_torque_Proprioceptionencoder(args.one_hot_torque_dim, args.encoder_dim )
        A_encoder = make_angle_Proprioceptionencoder(args.one_hot_angle_dim, args.encoder_dim )
    else:
        T_encoder = make_torque_Proprioceptionencoder(args.torque_dim, args.encoder_dim )
        A_encoder = make_angle_Proprioceptionencoder(args.angle_dim, args.encoder_dim )

    modalities = args.ablation.split("_")
    
    actor = Actor(I_encoder,T_encoder,A_encoder, args).cuda()
    

    e = make_real_env_hand(init_node=True, setup_robots=True, setup_base=True,num_stack=args.num_stack)

    
    modalities = args.ablation.split("_")
    print('modalities',modalities)
    print('len(modalities) ',len(modalities) )
    if len(modalities) == 3 :
        if "a_mha" in args.use_way:
            if args.use_pos:
                checkpoint = torch.load(args.j_f_vf_m_pos_ckpt)
            else :
                checkpoint = torch.load(args.j_f_vf_m_ckpt)
        else:
            checkpoint = torch.load(args.j_f_vf_ckpt)
    
    if len(modalities) == 2 :
        if "I" in modalities and "A" in modalities:
            checkpoint = torch.load(args.v_j_ckpt)

        if "I" in modalities and "T" in modalities:
            checkpoint = torch.load(args.v_f_ckpt)

        if "T" in modalities and "A" in modalities:
            checkpoint = torch.load(args.j_f_ckpt)
    
    if len(modalities) == 1 :

        if "I" in modalities :
            checkpoint = torch.load(args.v_ckpt)
            

        if "T" in modalities :
            checkpoint = torch.load(args.f_ckpt)

        if "A" in modalities:
            checkpoint = torch.load(args.j_ckpt)

    state_dict = checkpoint['state_dict']
    new_state_dict = {key.replace("actor.", "", 1): value for key, value in state_dict.items()}
    loading_status = actor.load_state_dict(new_state_dict) 
    actor.eval()
    plt.ion() 
    try:
        for _ in range(1):
            ts = e.reset()
            obs = ts.observation
            step = 0
            reward_sum = 0
            fig, ax1 = plt.subplots()  
            fig2, ax2 = plt.subplots()  
            image_brg = ts.observation['image']

            stacked_images_rgb = stack_images(image_brg, num_stack, resized_height_v, resized_height_t)
        
            while step < 500:  
                    step += 1
                    joint_numpy = np.array(ts.observation['joint'] / args.inspire)
                    joint_tensor = torch.from_numpy(joint_numpy).float().cuda()
                    joint_tensor = joint_tensor.unsqueeze(0)

                    torque_numpy = np.array((ts.observation['torque'] +1000)/2000 )
                    torque_tensor = torch.from_numpy(torque_numpy).float().cuda()
                    torque_tensor = torque_tensor.unsqueeze(0)
                    print('force_tensor',torque_tensor)
              
                    images_tensor = torch.from_numpy(stacked_images_rgb / 255.0).float().cuda()
                    start_time = time.time() 
                    raw_action = actor(joint_tensor,force_tensor,images_tensor) 
                    raw_action_tensor = raw_action[1]
                    force_tensor = raw_action[2]
                    force = force_tensor.cpu().detach().numpy()
                    force = force.squeeze()
                    raw_action_cpu = raw_action_tensor.cpu()  
                    raw_action_numpy = raw_action_cpu.detach().numpy() 
                    ts = e.step(args,raw_action_numpy,force)
                    obs = ts.observation
                    end_time = time.time()  
                    step_time = end_time - start_time 
                    print(f'Step {step} took {step_time:.4f} seconds') 
                    if "vf" in modalities: 
                        image_data = obs['image']   
                        plt.pause(0.02)

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


    args = p.parse_args()
    args.batch_size *= torch.cuda.device_count()
    main(args)




