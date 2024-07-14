import os
import torch
import torchvision.transforms as T

from src.datasets.base import EpisodeDataset
import numpy as np
from PIL import Image
import pickle



class ImitationEpisode(EpisodeDataset):
    def __init__(self, log_file, args, dataset_idx, data_folder, train=True):
        super().__init__(log_file, data_folder)
        self.train = train
        self.num_stack = args.num_stack
        self.frameskip = args.frameskip
        self.ROI = args.ROI
        self.max_len = (self.num_stack - 1) * self.frameskip

        self.resized_height_v = args.resized_height_v
        self.resized_width_v = args.resized_width_v
        self.resized_height_t = args.resized_height_t
        self.resized_width_t = args.resized_width_t

        self.mask_height_t = args.mask_height_t
        self.mask_height_v = args.mask_height_v
        self.mask_width_t = args.mask_width_t
        self.mask_width_v = args.mask_width_v

        self._crop_height_v = int(self.resized_height_v * (1.0 - args.crop_percent))
        self._crop_width_v = int(self.resized_width_v * (1.0 - args.crop_percent))
        self._crop_height_t = int(self.resized_height_t * (1.0 - args.crop_percent))
        self._crop_width_t = int(self.resized_width_t * (1.0 - args.crop_percent))
        (
            self.trial,
            self.format_time,
            self.hand_demos,
            self.num_frames,
        ) = self.get_episode(dataset_idx, ablation=args.ablation)

        self.action_dim = args.action_dim
        self.task = args.task
        self.use_flow = args.use_flow
        self.modalities = args.ablation.split("_")
        self.nocrop = args.nocrop

        # dis_action
        self.is_dis_actions = args.dis_actions

        # if self.train:
        self.transform_cam = [
            T.Resize((self.resized_height_v, self.resized_width_v)),
            T.ColorJitter(brightness=0.2, contrast=0.02, saturation=0.02),
        ]
        self.transform_cam = T.Compose(self.transform_cam)
    def __len__(self):
        return self.num_frames

    def __getitem__(self, idx):
        start = idx - self.max_len
        if start < 0:
            frame_idx = np.arange(0, self.max_len + 1, self.frameskip)
        elif idx >= self.num_frames - 1:
            frame_idx = np.arange(self.num_frames-self.max_len-2, self.num_frames-1, self.frameskip)
        else:
            frame_idx = np.arange(start, idx + 1, self.frameskip)
            frame_idx = sorted(frame_idx)
            frame_idx = np.array(frame_idx)

        frame_idx[frame_idx < 0] = -1

        a_framestack = 0
        t_framestack = 0
        
        cam_framestack = torch.stack(
            [
                self.transform_cam(
                    self.load_image(self.trial, self.format_time, timestep, self.mask_height_t, self.mask_height_v, self.mask_width_t, self.mask_width_v, self.ROI)
                )
                for timestep in frame_idx
            ],
            dim=0,
        )

        stacked_force = []
        stacked_joint = []
        for timestep in frame_idx:
            obs_force = (self.hand_demos["force"][timestep]+1000)/2000
            obs_force_tensor = torch.tensor(obs_force, dtype=torch.float32)
            stacked_force.append(obs_force_tensor)

            obs_joint = (self.hand_demos["joint"][timestep])/1000
            obs_joint_tensor = torch.tensor(obs_joint, dtype=torch.float32)
            stacked_joint.append(obs_joint_tensor)

        t_framestack = torch.stack(stacked_force, dim=0)
        a_framestack = torch.stack(stacked_joint, dim=0)

 
        if not self.is_dis_actions:
            actions = torch.Tensor((self.hand_demos["joint"][frame_idx[-1]+1])/1000)
        else:
            actions = torch.Tensor(self.hand_demos["dis_actions"][frame_idx[-1]])

        torques = torch.Tensor((self.hand_demos["force"][frame_idx[-1]+1]+1000)/2000)

        init_state_dict = [self.hand_demos["init_state_dict"]]
        for state_dict in init_state_dict:
            target_pos = state_dict['target_pos']
    
        optical_flow = 0
  
        return (
            (
                cam_framestack,
                a_framestack,
                t_framestack,
            ),
            optical_flow,
            start,
            actions,
            torques,
            target_pos,
        )
        
        

class TransformerEpisode(ImitationEpisode):
    @staticmethod
    def load_image(trial, stream, timestep):
        """
        Do not duplicate first frame for padding, instead return all zeros
        """
        return_null = timestep == -1
        if timestep == -1:
            timestep = 0
        img_path = os.path.join(trial, stream, str(timestep) + ".jpg")
        image = (
            torch.as_tensor(np.array(Image.open(img_path))).float().permute(2, 0, 1)
            / 255
        )
        if return_null:
            image = torch.zeros_like(image)
        return image
