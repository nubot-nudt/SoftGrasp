import json
import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
import pandas as pd
import torchaudio
import soundfile as sf
import pickle
import cv2
import matplotlib.pyplot as plt


class EpisodeDataset(Dataset):
    def __init__(self, log_file, data_folder="data/test_recordings_0214"):
        """
        neg_ratio: ratio of silence audio clips to sample
        """
        super().__init__()
        self.logs = pd.read_csv(log_file)
        self.data_folder = data_folder
        self.sr = 44100
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sr,
            n_fft=int(self.sr * 0.025),
            hop_length=int(self.sr * 0.01),
            n_mels=64,
            center=False,
        )
        self.streams = [
            "cam_gripper_color",
            "cam_fixed_color",
            "left_gelsight_flow",
            "left_gelsight_frame",
        ]
        # self.gelsight_offset = torch.as_tensor(np.array(Image.open("gelsight_offset.png"))).float().permute(2, 0, 1) / 255
        pass

    def get_episode(self, idx, ablation=""):
     
        modes = ablation.split("_")

        def load(file):
            fullpath = os.path.join(trial, file)
            if os.path.exists(fullpath):
                return sf.read(fullpath)[0]
            else:
                return None

        format_time = self.logs.iloc[idx].Time.replace(":", "_")
        # print("override" + '#' * 50)
        trial = os.path.join(self.data_folder, format_time)
        # with open(os.path.join(trial, "timestamps.json")) as ts:
        #     timestamps = json.load(ts)
        demos = pickle.load(open(os.path.join(trial, format_time + '.pickle'), 'rb'))
        # num_frames = 0
 
        num_frames = len(demos["joint"] ) # 计算动作数组的行数作为长度
        num_frames = num_frames
        # print('trial',trial)
        # print('num_frames',num_frames)
        
        return (
            trial,
            # timestamps,
            format_time,
            demos,
            num_frames,
        )

    def __getitem__(self, idx):
        raise NotImplementedError

    @staticmethod
    # def load_image(trial, stream, timestep,mask_height_t ,mask_height_v ,mask_width_t,mask_width_v):
      
    #     if timestep == -1:
    #         timestep = 0
    #     # print('timestep',timestep)
    #     img_path = os.path.join(trial, stream, str(timestep) + ".jpg")
    #     # print('img_path:',img_path)
    #     image = (
    #         torch.as_tensor(np.array(Image.open(img_path))).float().permute(2, 0, 1)
    #         / 255
    #     )
    #     mask = np.zeros(image.shape[:2], dtype=np.uint8)
    #     r1 = (mask_height_t,mask_height_v, mask_width_t,mask_width_v)
    #     mask[r1[0]:r1[1], r1[2]:r1[3]] = 255
    #     ROI_image = cv2.bitwise_and(image, image, mask=mask)
    #     # return image
    #     return ROI_image
    def load_image(trial, stream, timestep, mask_height_t, mask_height_v, mask_width_t, mask_width_v,ROI):
      
        if timestep == -1:
            timestep = 0
        if ROI:
            img_path = os.path.join(trial, stream, f"{timestep}.jpg")
            # print('img_path:',img_path)
            image = torch.as_tensor(np.array(Image.open(img_path))).float().permute(2, 0, 1) / 255
            image_np = image.numpy().transpose(1, 2, 0)*255
            mask = np.zeros(image_np.shape[:2], dtype=np.uint8)
            r1 = (mask_height_t, mask_height_v, mask_width_t, mask_width_v)
            mask[r1[0]:r1[1], r1[2]:r1[3]] = 255  
            masked_image = cv2.bitwise_and(image_np, image_np, mask=mask)
            save_path = os.path.join(trial, stream, f"masked_{timestep}.jpg")
            cv2.imwrite(save_path, cv2.cvtColor(masked_image, cv2.COLOR_RGB2BGR))
            # ROI_image = torch.as_tensor(masked_image.transpose(2, 0, 1)) 
            ROI_image = torch.as_tensor(masked_image) .float().permute(2, 0, 1) / 255
            # print('ROI_image.shape',ROI_image.shape)
        else:
            img_path = os.path.join(trial, stream, f"{timestep}.jpg")
            ROI_image = torch.as_tensor(np.array(Image.open(img_path))).float().permute(2, 0, 1) / 255
            # print('ROI_image.shape',ROI_image.shape)
        return ROI_image

    @staticmethod
    def load_flow(trial, stream, timestep):
        """
        Args:
            trial: the folder of the current episode
            stream: ["cam_gripper_color", "cam_fixed_color", "left_gelsight_frame"]
                for "left_gelsight_flow", please add another method to this class using torch.load("xxx.pt")
            timestep: the timestep of frame you want to extract
        """
        img_path = os.path.join(trial, stream, str(timestep) + ".pt")
        image = torch.as_tensor(torch.load(img_path))
        return image
    

    @staticmethod
    def clip_resample(audio, audio_start, audio_end):
        left_pad, right_pad = torch.Tensor([]), torch.Tensor([])
        if audio_start < 0:
            left_pad = torch.zeros((audio.shape[0], -audio_start))
            audio_start = 0
        if audio_end >= audio.size(-1):
            right_pad = torch.zeros((audio.shape[0], audio_end - audio.size(-1)))
            audio_end = audio.size(-1)
        audio_clip = torch.cat(
            [left_pad, audio[:, audio_start:audio_end], right_pad], dim=1
        )
        audio_clip = torchaudio.functional.resample(audio_clip, 44100, 16000)
        return audio_clip

    def __len__(self):
        return len(self.logs)

    @staticmethod
    def resize_image(image, size):
        assert len(image.size()) == 3  # [3, H, W]
        return torch.nn.functional.interpolate(
            image.unsqueeze(0), size=size, mode="bilinear"
        ).squeeze(0)
