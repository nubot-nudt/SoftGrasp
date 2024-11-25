from torch.nn.modules.activation import MultiheadAttention
from torchvision.models import resnet18
from torchvision.models.feature_extraction import create_feature_extractor
import torch
from torch import nn
import os

# from engines.imi_engine import Future_Prediction
import cv2
import numpy as np
import time
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class PositionalEncoding(torch.nn.Module):
    def __init__(self, num_features, max_len=500):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, num_features)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, num_features, 2).float() * (-torch.log(torch.tensor(10000.0)) / num_features))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x
    
    
class Actor(torch.nn.Module):
    def __init__(self, I_encoder,T_encoder,A_encoder,args):
        super().__init__()
        self.I_encoder = I_encoder
        self.A_encoder = A_encoder
        self.T_encoder = T_encoder
        self.mlp = None
        self.layernorm_embed_shape = args.encoder_dim * args.num_stack
        self.encoder_dim = args.encoder_dim
        self.ablation = args.ablation
        self.action_dim = args.action_dim
        self.num_heads = args.num_heads
        self.use_way = args.use_way
        self.use_pos = args.use_pos
        self.use_one_hot = args.use_one_hot
        self.query = nn.Parameter(torch.randn(1, 1, self.layernorm_embed_shape))

        ## load models
        self.modalities = self.ablation.split("_")
        print(f"Using modalities: {self.modalities}")
        self.embed_dim = self.layernorm_embed_shape * len(self.modalities)
        self.layernorm = nn.LayerNorm(self.layernorm_embed_shape)
        
        self.bottleneck = nn.Linear(
            self.embed_dim, self.layernorm_embed_shape
        ) 

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(self.layernorm_embed_shape, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, args.action_dim),
        )
        self.a_mlp = torch.nn.Sequential(
            torch.nn.Linear(self.layernorm_embed_shape, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, args.action_dim),
        )
        self.aux_mlp = torch.nn.Linear(self.layernorm_embed_shape, self.action_dim)

        if self.use_pos:
            self.I_pos_encoder = PositionalEncoding(num_features=args.picture_dim).to(device)
            self.T_pos_encoder = PositionalEncoding(num_features=args.torque_dim).to(device)
            self.A_pos_encoder = PositionalEncoding(num_features=args.angle_dim).to(device)

        if "m_mha" in self.use_way:
            self.mha = MultiheadAttention(self.layernorm_embed_shape,  self.num_heads)

        
    def forward(self, inputs,_):
        """
        Args:
            cam_framestack
            image_inp: [batch, num_stack, 3, H, W]
            angle    :[batch, num_stack, 6]
            torque   :[batch, num_stack, 6]
        
        """
        image_inp, angle,torque= inputs
        
        embeds = []

        if self.use_pos:
            if "I" in self.modalities:
                batch, num_stack, _, Hv, Wv = image_inp.shape
                image_inp_flat = image_inp.view(batch, num_stack, -1)
                image_inp_flat_pos = self.I_pos_encoder(image_inp_flat)
                image_inp_with_pos = torch.add(image_inp_flat, image_inp_flat_pos)
                image_inp_pos = image_inp_with_pos.view(batch * num_stack, 3, Hv, Wv)
                I_embeds = self.I_encoder(image_inp_pos)  
                I_embeds = I_embeds.view(
                    -1, self.layernorm_embed_shape
                )  
                embeds.append(I_embeds)
            if "A" in self.modalities:
                batch, num_stack, state = angle.shape
                angle_flat = angle.view(batch, num_stack, -1)
                angle_flat_pos = self.A_pos_encoder(angle_flat)
                angle_with_pos = torch.add(angle_flat, angle_flat_pos)
                if  self.use_one_hot:
                    expanded_angle_list = []
                    for i in range(batch):
                        one_hot = torch.eye(state).to(angle_with_pos.device)  
                        angle_i = angle_with_pos[i].unsqueeze(1)
                        combined_one_hot = one_hot.unsqueeze(0).expand(num_stack, -1, -1) 
                        expanded_with_angles = torch.cat((combined_one_hot, angle_i), dim=1) 
                        expanded_angle_list.append(expanded_with_angles)
                    angle_final_result = torch.stack(expanded_angle_list, dim=0) 
                    angle_state = angle_final_result.shape[2]*angle_final_result.shape[3]
                    angle_final_result = angle_final_result.view(batch, num_stack, -1)
                    angle_final_result = angle_final_result.view(batch * num_stack, angle_state)
                else:
                    angle_final_result = angle_with_pos.view(batch * num_stack, state)
                A_embeds = self.A_encoder(angle_final_result)
                A_embeds = A_embeds.view(-1, self.layernorm_embed_shape)
                embeds.append(A_embeds)

            if "T" in self.modalities:
                batch, num_stack, state = torque.shape
                torque_flat = torque.view(batch, num_stack, -1)
                torque_flat_pos = self.T_pos_encoder(torque_flat)
                torque_with_pos = torch.add(torque_flat, torque_flat_pos)
                if  self.use_one_hot:
                    expanded_torque_list = []
                    for i in range(batch):
                        one_hot = torch.eye(state).to(torque_with_pos.device) 
                        torque_i = torque_with_pos[i].unsqueeze(1)
                        combined_one_hot = one_hot.unsqueeze(0).expand(num_stack, -1, -1) # (num_stack, 6, 6)
                        expanded_with_torques = torch.cat((combined_one_hot, torque_i), dim=1) #  (num_stack, 7, 6)
                        expanded_torque_list.append(expanded_with_torques)
                    torque_final_result = torch.stack(expanded_torque_list, dim=0) # [batch, num_stack, 7, 6]
                    torque_state = torque_final_result.shape[2]*torque_final_result.shape[3]
                    torque_final_result = torque_final_result.view(batch, num_stack, -1)
                    torque_final_result = torque_final_result.view(batch * num_stack, torque_state)
                else:
                    torque_final_result = torque_with_pos.view(batch * num_stack, state)
                T_embeds = self.T_encoder(torque_final_result)
                T_embeds = T_embeds.view(-1, self.layernorm_embed_shape)
                embeds.append(T_embeds)

        else:
            if "I" in self.modalities:
                batch, num_stack, _, Hv, Wv = image_inp.shape
                image_inp = image_inp.view(batch * num_stack, 3, Hv, Wv)
                I_embeds = self.I_encoder(image_inp)  
                I_embeds = I_embeds.view(
                    -1, self.layernorm_embed_shape
                )  
                embeds.append(I_embeds)
            if "A" in self.modalities:
                batch,num_stack,state = angle.shape
                A_angle = angle.view(batch * num_stack,state)
                A_angle = A_angle
                A_embeds = self.A_encoder(A_angle) 
                A_embeds = A_embeds.view(
                    -1, self.layernorm_embed_shape
                ) 
                embeds.append(A_embeds)
            if "T" in self.modalities:
                batch,num_stack,state = torque.shape
                T_torque = torque.view(batch * num_stack,state)
                T_torque = T_torque
                T_embeds = self.T_encoder(T_torque) 
                T_embeds = T_embeds.view(
                    -1, self.layernorm_embed_shape
                ) 
                embeds.append(T_embeds)
            
        
        if "m_mha" in self.use_way:
            mlp_inp_start = torch.stack(embeds, dim=0) 
            mlp_inp_start = mlp_inp_start.float()
            mha_out, weights = self.mha(mlp_inp_start, mlp_inp_start, mlp_inp_start) 
            mha_out_o = mha_out
            mha_out += mlp_inp_start
            mlp_inp = torch.cat([mha_out[i] for i in range(mha_out.shape[0])], 1)
            mlp_inp = self.bottleneck(mlp_inp)
        if "no_mha" in self.use_way:
            mlp_inp = torch.cat(embeds, dim=-1)
            mlp_inp = self.bottleneck(mlp_inp)
            mha_out_o = mlp_inp
            weights = None
        torque = self.mlp(mlp_inp)
        action = self.a_mlp(mlp_inp)

        return mha_out_o, action, torque, weights     
    

if __name__ == "__main__":
    pass

