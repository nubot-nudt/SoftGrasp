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

task2actiondim = {"pouring": 2, "insertion": 3}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CrossMultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(CrossMultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.attention_heads = nn.ModuleList([
            nn.MultiheadAttention(embed_dim, 1) for _ in range(num_heads)
        ])
        self.output_linear = nn.Linear(embed_dim * num_heads, embed_dim)

    def forward(self, query, key, value):
        attn_outputs = []
        attn_weights = []

        for head in self.attention_heads:
            attn_output, attn_weight = head(query, key, value)
            attn_outputs.append(attn_output)
            attn_weights.append(attn_weight)
        
        # Concatenate the outputs of each head
        attn_output = torch.cat(attn_outputs, dim=-1)
        attn_output = self.output_linear(attn_output)

        # Stack attention weights along the new head dimension
        attn_weights = torch.stack(attn_weights, dim=0)

        return attn_output, attn_weights

class AdaptiveAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super(AdaptiveAttention, self).__init__()
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.scale_factor = nn.Parameter(torch.ones(1, embed_dim))
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.residual = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.LayerNorm(embed_dim)
        )

    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None):
        attn_output, attn_weights = self.mha(query, key, value, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        # Apply adaptive scaling
        attn_output = attn_output * self.scale_factor
        # Apply dropout and add residual connection
        attn_output = self.norm(attn_output + self.dropout(self.residual(attn_output)))
        return attn_output, attn_weights
    
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

class CustomMultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(CustomMultiHeadAttention, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.proj_query = nn.Linear(embed_dim, embed_dim)
        self.proj_key = nn.Linear(embed_dim, embed_dim)
        self.proj_value = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, query, key, value, key_padding_mask=None, attn_mask=None):
        query = self.proj_query(query)
        key = self.proj_key(key)
        value = self.proj_value(value)
        
        attn_output, attn_weights = self.multihead_attn(query, key, value, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
       
        attn_output = self.out_proj(attn_output)
        
        return attn_output, attn_weights

# 定义高斯分布的专家乘积层
class ExpertProduct(nn.Module):
    def __init__(self, num_modalities, in_features, out_features):
        super(ExpertProduct, self).__init__()
        self.num_modalities = num_modalities
        self.in_features = in_features
        self.out_features = out_features
        self.expert1 = nn.Linear(in_features, out_features)
        # 添加更多专家层...

    def forward(self, modalities):
        # 将每个模态的特征输入专家层中
        expert_outputs = []
        for i in range(self.num_modalities):
            expert_outputs.append(self.expert1(modalities[i]))
            # 添加更多模态的处理...
        
        # 多个专家层的乘积
        product = torch.prod(torch.stack(expert_outputs, dim=1), dim=1)
        
        return product

class Actor(torch.nn.Module):
    def __init__(self, I_encoder,T_encoder,A_encoder,share_encoder,args):
        super().__init__()
        self.I_encoder = I_encoder
        self.A_encoder = A_encoder
        self.T_encoder = T_encoder
        self.share_encoder = share_encoder
        self.mlp = None
        self.layernorm_embed_shape = args.encoder_dim * args.num_stack
        self.encoder_dim = args.encoder_dim
        self.ablation = args.ablation
        self.action_dim = args.action_dim
        self.use_mha = args.use_mha
        self.use_amha = args.use_amha
        self.num_heads = args.num_heads
        self.use_way = args.use_way
        self.use_pos = args.use_pos
        self.use_one_hot = args.use_one_hot
        self.scores_visualize = args.scores_visualize
        self.save_scores  = args.save_scores
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
            if "share" in self.modalities:
                self.sf_pos_encoder = PositionalEncoding(num_features=args.torque_dim).to(device)
                self.sj_pos_encoder = PositionalEncoding(num_features=args.angle_dim).to(device)

        if "a_mha" in self.use_way:
            self.amha = AdaptiveAttention(self.layernorm_embed_shape, self.num_heads)
        if "c_mha" in self.use_way:
            self.cmha = CrossMultiHeadAttention(self.layernorm_embed_shape, self.num_heads)
        if "w_mha" in self.use_way:
            self.wmha = CustomMultiHeadAttention(self.layernorm_embed_shape, self.num_heads)
        if "m_mha" in self.use_way:
            self.mha = MultiheadAttention(self.layernorm_embed_shape,  self.num_heads)
        if "expert" in self.use_way:
            self.expert_product = ExpertProduct(len(self.modalities), self.layernorm_embed_shape, args.encoder_dim )

        
    def forward(self, angle,torque,image_inp):
        """
        Args:
            cam_framestack
            image_inp: [batch, num_stack, 3, H, W]
            angle    :[batch, num_stack, 6]
            torque   :[batch, num_stack, 6]
        
        """
        # vf_inp,angle,torque= inputs
        
        embeds = []

        if self.use_pos:
            if "I" in self.modalities:
                batch, num_stack, _, Hv, Wv = image_inp.shape
                image_inp_flat = image_inp.view(batch, num_stack, -1)
                # Add positional encoding
                image_inp_flat_pos = self.I_pos_encoder(image_inp_flat)
                # 将位置编码与输入数据相加
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
                # 将位置编码与输入数据相加
                angle_with_pos = torch.add(angle_flat, angle_flat_pos)
                if  self.use_one_hot:
                    expanded_angle_list = []
                    for i in range(batch):
                        one_hot = torch.eye(state).to(angle_with_pos.device)  # 生成 (6, 6) 的 one-hot 矩阵
                        angle_i = angle_with_pos[i].unsqueeze(1)
                        combined_one_hot = one_hot.unsqueeze(0).expand(num_stack, -1, -1) # (num_stack, 6, 6)
                        expanded_with_angles = torch.cat((combined_one_hot, angle_i), dim=1) #  (num_stack, 7, 6)
                        expanded_angle_list.append(expanded_with_angles)
                    angle_final_result = torch.stack(expanded_angle_list, dim=0) # [batch, num_stack, 7, 6]
                    angle_state = angle_final_result.shape[2]*angle_final_result.shape[3]
                    angle_final_result = angle_final_result.view(batch, num_stack, -1)
                    angle_final_result = angle_final_result.view(batch * num_stack, angle_state)
                else:
                    angle_final_result = angle_with_pos.view(batch * num_stack, state)
                # angle_pos = angle_pos
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
                        one_hot = torch.eye(state).to(torque_with_pos.device)  # 生成 (6, 6) 的 one-hot 矩阵
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
                # torque_pos = torque_pos
                T_embeds = self.T_encoder(torque_final_result)
                T_embeds = T_embeds.view(-1, self.layernorm_embed_shape)
                embeds.append(T_embeds)

            if "share" in self.modalities:
                batch, num_stack, state = torque.shape
                torque_flat = torque.view(batch, num_stack, -1)
                torque_flat_pos = self.sf_pos_encoder(torque_flat)
                angle_flat = angle.view(batch, num_stack, -1)
                angle_flat_pos = self.sj_pos_encoder(angle_flat)
                
                expanded_share_list = []
                for i in range(batch):
                    one_hot = torch.eye(state).to(torque_flat_pos.device)  # 生成 (6, 6) 的 one-hot 矩阵
                    angle_i = angle_flat_pos[i].unsqueeze(1)
                    torque_i = torque_flat_pos[i].unsqueeze(1)

                    combined_one_hot = one_hot.unsqueeze(0).expand(num_stack, -1, -1) # (num_stack, 6, 6)
                    # 将 angle 信息添加到 one_hot 矩阵
                    expanded_with_angles = torch.cat((combined_one_hot, angle_i), dim=1) #  (num_stack, 7, 6)
                    # 将 torque 信息添加到 added angles 
                    expanded_share = torch.cat((expanded_with_angles, torque_i), dim=1) #  (num_stack, 8, 6)
                    expanded_share_list.append(expanded_share)

                # 在 batch 维度连接所有批次的结果
                final_result = torch.stack(expanded_share_list, dim=0) # [batch, num_stack, 8, 6]
                final_result = final_result.view(batch, num_stack, -1)
                final_result = final_result.view(batch * num_stack, 48)
                final_result = final_result
                share_embeds = self.share_encoder(final_result)
                share_embeds = share_embeds.view(-1, self.layernorm_embed_shape)
                embeds.append(share_embeds)
        ############     有位置编码     #############
        
        ############    没有位置编码    #############
        else:
            if "I" in self.modalities:
                batch, num_stack, _, Hv, Wv = image_inp.shape
                image_inp = image_inp.view(batch * num_stack, 3, Hv, Wv)
                I_embeds = self.I_encoder(image_inp)  
                # print('vf_embeds.shape:',vf_embeds.shape)
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
            if "share" in self.modalities:
                batch, num_stack,state = torque.shape
                expanded_share_list = []
                for i in range(batch):
                    one_hot = torch.eye(state).to(torque.device)  # 生成 (6, 6) 的 one-hot 矩阵
                    angle_i = angle[i].unsqueeze(1)
                    torque_i = torque[i].unsqueeze(1)
                    combined_one_hot = one_hot.unsqueeze(0).expand(num_stack, -1, -1) # (num_stack, 6, 6)
                    # 将 angle 信息添加到 one_hot 矩阵
                    expanded_with_angles = torch.cat((combined_one_hot, angle_i), dim=1) #  (num_stack, 7, 6)
                    # 将 torque 信息添加到 added angles 
                    expanded_share = torch.cat((expanded_with_angles, torque_i), dim=1) #  (num_stack, 8, 6)
                    expanded_share_list.append(expanded_share)

                # 在 batch 维度连接所有批次的结果
                final_result = torch.stack(expanded_share_list, dim=0) # [batch, num_stack, 8, 6]
                final_result = final_result.view(batch, num_stack, -1)
                final_result = final_result.view(batch * num_stack, 48)
                final_result = final_result
                share_embeds = self.share_encoder(final_result)
                share_embeds = share_embeds.view(-1, self.layernorm_embed_shape)
                embeds.append(share_embeds)
        ############    没有位置编码    #############
        if "m_mha" in self.use_way:
            mlp_inp_start = torch.stack(embeds, dim=0) 
            mlp_inp_start = mlp_inp_start.float()
            mha_out, weights = self.mha(mlp_inp_start, mlp_inp_start, mlp_inp_start) 
            mha_out_o = mha_out
            mha_out += mlp_inp_start
            mlp_inp = torch.cat([mha_out[i] for i in range(mha_out.shape[0])], 1)
            mlp_inp = self.bottleneck(mlp_inp)
        if "a_mha" in self.use_way:
            mlp_inp_start = torch.stack(embeds, dim=0) 
            mlp_inp_start = mlp_inp_start.float()
            mha_out, weights = self.amha(mlp_inp_start, mlp_inp_start, mlp_inp_start) 
            mha_out_o = mha_out
            mha_out += mlp_inp_start
            mlp_inp = torch.cat([mha_out[i] for i in range(mha_out.shape[0])], 1)
            mlp_inp = self.bottleneck(mlp_inp)
        if "c_mha" in self.use_way:
            mlp_inp_start = torch.stack(embeds, dim=0) 
            mlp_inp_start = mlp_inp_start.float()
            mha_out, weights = self.cmha(mlp_inp_start, mlp_inp_start, mlp_inp_start) 
            mha_out_o = mha_out
            mha_out += mlp_inp_start
            mlp_inp = torch.cat([mha_out[i] for i in range(mha_out.shape[0])], 1)
            mlp_inp = self.bottleneck(mlp_inp)
        if "no_mha" in self.use_way:
            mlp_inp = torch.cat(embeds, dim=-1)
            mlp_inp = self.bottleneck(mlp_inp)
            mha_out_o = mlp_inp
            weights = None
            scores = 0
        # action = self.aux_mlp(mlp_inp)
        torque = self.mlp(mlp_inp)
        action = self.a_mlp(mlp_inp)
        ############    可视化     #############
        if self.scores_visualize :
            if "c_mha" in self.use_way:
                mlp_inp_start_reshaped = mlp_inp_start.permute(1, 0, 2) 
                batch_size = mlp_inp_start_reshaped.shape[0]

                attention_scores = torch.zeros(batch_size,  len(self.modalities),len(self.modalities)).to(device)

                linear_layer = torch.nn.Linear( self.layernorm_embed_shape, len(self.modalities)).to(device)
                softmax = torch.nn.Softmax(dim=-1)

                for head in range(self.num_heads):
                    weighted_sum = torch.einsum('bmd,bmm->bmd', mlp_inp_start_reshaped, weights[head])
                    attention_scores += linear_layer(weighted_sum)  
                final_attention_scores = softmax(attention_scores)
                attention_scores /= self.num_heads
                summarized_scores, _ = torch.max(final_attention_scores, dim=0)
                scores = summarized_scores[0].cpu().detach().numpy()
                modalities_labels = [f'Modality {i + 1}' for i in range(len(self.modalities))]

                plt.plot(modalities_labels, scores, marker='o')
                plt.xlabel('Modalities')
                plt.ylabel('Attention Score')
                plt.title('Attention Scores for Each Modality')

                output_dir = "visualizations"
                os.makedirs(output_dir, exist_ok=True)
                output_path = os.path.join(output_dir, "attention_scores.png")
                if self.save_scores:
                    plt.savefig(output_path)
                plt.close()  
            if "a_mha" in self.use_way or "m_mha" in self.use_way:
                mlp_inp_start_reshaped = mlp_inp_start.permute(1, 0, 2)  # (batch, modalities, 768)
                weighted_sum = torch.einsum('bmd,bmm->bmd', mlp_inp_start_reshaped, weights)
                linear_layer = torch.nn.Linear(self.layernorm_embed_shape,  len(self.modalities)).to(device)
                attention_scores = linear_layer(weighted_sum)
                softmax = torch.nn.Softmax(dim=-1)
                final_attention_scores = softmax(attention_scores)
                summarized_scores, _ = torch.max(final_attention_scores, dim=-1)
                scores = summarized_scores[0].cpu().detach().numpy()
                modalities_labels = [f'Modality {i + 1}' for i in range(len(self.modalities))]
                # 清空之前的图表
                plt.clf()
                plt.cla()
                plt.bar(modalities_labels, scores)
                plt.xlabel('Modalities')
                plt.ylabel('Attention Score')
                plt.title('Attention Scores for Each Modality')
                output_dir = "visualizations"
                os.makedirs(output_dir, exist_ok=True) 
                output_path = os.path.join(output_dir, "attention_scores.png")
                if self.save_scores:
                    plt.savefig(output_path)
        else:
            scores = 0

        return mha_out_o, scores, action, torque, weights    
    


if __name__ == "__main__":
    pass
  
