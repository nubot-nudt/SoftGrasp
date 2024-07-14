import time
from pytorch_lightning import LightningModule
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import torchvision
import matplotlib.pyplot as plt
import pandas as pd
import os
import seaborn as sns
import io
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import logging

plt.switch_backend('agg')  # 用于非交互式环境

class ImiEngine(LightningModule):
    def __init__(self, actor, optimizer, train_loader, val_loader,test_loader, scheduler, config):
        super(ImiEngine, self).__init__()
        self.actor = actor
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.scheduler = scheduler
        self.config = config

        self.loss_cce = torch.nn.CrossEntropyLoss()
        self.all_weights = []  
        self.a_loss = 0
        self.f_loss = 0
        self.loss = 0

        self.validation_step_outputs = []
        self.writer = SummaryWriter()
        self.save_hyperparameters(config)
        self.lambda_param = None

    def set_lambda_param(self, lambda_param):
        self.lambda_param = lambda_param

    def training_step(self, batch, batch_idx):
        inputs, optical_flow, start ,act ,torques,target_pos = batch
        #############   (imitation)   ################3
        mha_out_o, mlp_inp ,act_pred,torque_pred, weights = self.actor(inputs, start) 
        if torch.isnan(act_pred).any():
            print("模型输出act_pred包含 NaN，检测到异常数据！")
        if torch.isnan(torque_pred).any():
            print("模型输出torque_pred包含 NaN，检测到异常数据！")
        # self.all_weights.append(weights.detach().cpu().numpy())  
        if self.config.loss == 'mse':
            a_mse_loss = F.mse_loss(act, act_pred)
            f_mse_loss = F.mse_loss(torques, torque_pred)
            loss = self.config.a_loss * a_mse_loss + self.config.f_loss * f_mse_loss* self.lambda_param
            self.log_dict(
                {"train/a_mse_loss": a_mse_loss,"train/f_mse_loss": f_mse_loss,"train/loss": loss}, on_step=True,on_epoch=True,prog_bar=True
            )
        if self.config.loss == 'l1':
            a_l1_loss = F.l1_loss(act, act_pred)
            f_l1_loss = F.l1_loss(torques, torque_pred)
            loss = self.config.a_loss * a_l1_loss + self.config.f_loss * f_l1_loss* self.lambda_param
            self.log_dict(
                {"train/a_l1_loss": a_l1_loss,"train/f_l1_loss": f_l1_loss,"train/loss": loss}, on_step=True,on_epoch=True,prog_bar=True
            )
        if self.config.loss == 'huber':
            a_huber_loss = F.huber_loss(act, act_pred,delta=self.config.delta)
            f_huber_loss = F.huber_loss(torques, torque_pred,delta=self.config.delta)
            # loss = self.config.a_loss * a_huber_loss + self.config.f_loss * f_huber_loss* self.lambda_param
            loss = self.config.a_loss * a_huber_loss + self.config.f_loss * f_huber_loss* 0.1
            self.log_dict(
                {"train/a_huber_loss": a_huber_loss,"train/f_huber_loss": f_huber_loss,"train/loss": loss}, on_step=True,on_epoch=True,prog_bar=True
            )

        # -------------------- 可视化注意力权重 --------------------
        if self.config.weights_visualize :
            # print("self.config.weights_visualize value:", self.config.weights_visualize)
            if batch_idx % self.config.weights_visualize_interval == 0:
                if "c_mha" in self.config.use_way:
                    batch_id = 0 
                    attn_weights = weights 
                    num_heads = attn_weights.shape[0]
                    for head_idx in range(num_heads):
                        attn_weights_sample = attn_weights[head_idx, batch_id].cpu().detach().numpy() 
                        fig, ax = plt.subplots(figsize=(8, 6))
                        sns.heatmap(attn_weights_sample, annot=True, cmap="viridis", ax=ax)
                        ax.set_title(f'Attention Scores for Batch {batch_idx}, Head {head_idx}, and Batch Id {batch_id}')
                        ax.set_xlabel('Query Token Index')
                        ax.set_ylabel('Key Token Index')

                        buf = io.BytesIO()
                        plt.savefig(buf, format='png')
                        buf.seek(0)
                        img = Image.open(buf)
                       
                        file_name = f'imitation_2_attention_heatmap_batch_{batch_idx}_sample_{batch_id}_head_{head_idx}.png'
                        directory = 'train/imitation_2/heatmap'
                        if not os.path.exists(directory):
                            os.makedirs(directory)
                        file_path = os.path.join(directory, file_name)
                        if self.config.save_weights:
                            img.save(file_path)
                        plt.close(fig)
                else:
                    batch_id = 0 
                    attn_weights = weights  # [(batch_size, query_len, key_len)]
                    attn_weights_sample = attn_weights[batch_id] 
                    # 可视化注意力权重矩阵
                    fig, ax = plt.subplots(figsize=(8, 6))
                    sns.heatmap(attn_weights_sample.detach().cpu().numpy(), annot=True, cmap="viridis", ax=ax)
                    ax.set_title(f'Attention Scores for Batch {batch_idx} and Batch Id {batch_id}')
                    ax.set_xlabel('Query Token Index')
                    ax.set_ylabel('Key Token Index')

                    buf = io.BytesIO()
                    plt.savefig(buf, format='png')
                    buf.seek(0)
                    img = Image.open(buf)
                    # 保存图像到本地
                    file_name = f'imitation_2_attention_heatmap_batch_{batch_idx}_sample_{batch_id}.png'
                    directory = 'train/imitation_2/heatmap'
                    if not os.path.exists(directory):
                        os.makedirs(directory)
                    file_path = os.path.join(directory, file_name)
                    if self.config.save_weights:
                        img.save(file_path)
                    # 将图像添加到 TensorBoard 中
                    self.writer.add_image(f'Attention Scores Batch {batch_idx} Sample {batch_id}', np.array(img).transpose(2, 0, 1), self.current_epoch, dataformats='CHW')
                    self.writer.close()
                    plt.close(fig)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs,optical_flow, start, act ,torques ,target_pos = batch
        image_inp,angle,torque= inputs
        if torch.isnan(act).any():
            print("模型输入act包含 NaN，检测到异常数据！") 
        if torch.isnan(torques).any():
            print("模型输入torques包含 NaN，检测到异常数据！") 
        if torch.isnan(angle).any():
            print("模型输入angle包含 NaN，检测到异常数据！") 
        if torch.isnan(torque).any():
            print("模型输入torque包含 NaN，检测到异常数据！") 
        mha_out_o, mlp_inp,  act_pred, torque_pred,weights = self.actor(inputs, start) 
        if torch.isnan(act_pred).any():
            print("模型输出act_pred包含 NaN，检测到异常数据！")
        if torch.isnan(torque_pred).any():
            print("模型输出torque_pred包含 NaN，检测到异常数据！")
        if self.config.loss == 'mse':
            a_loss = F.mse_loss(act, act_pred)
            f_loss = F.mse_loss(torques, torque_pred)
        if self.config.loss == 'l1':
            a_loss = F.l1_loss(act, act_pred)
            f_loss = F.l1_loss(torques, torque_pred)
        if self.config.loss == 'huber':
            a_loss = F.huber_loss(act, act_pred,delta=self.config.delta)
            f_loss = F.huber_loss(torques, torque_pred,delta=self.config.delta)
        # loss = self.config.a_loss * a_loss + self.config.f_loss * f_loss* self.lambda_param
        # loss = self.config.a_loss  + self.config.f_loss *0.1
        loss = a_loss  + f_loss *self.config.fix_loss_delta

        # tqdm.write("a_loss:%.8f"% a_loss.item()+", f_loss:%.8f"% f_loss.item()+", loss:%.8f"% loss.item())
        # self.a_loss = a_loss.item()
        # self.f_loss = f_loss.item()
        # self.loss = loss.item()
        tqdm.write("a_loss:%.8f" % a_loss + ", f_loss:%.8f" % f_loss + ", loss:%.8f" % loss)
        self.a_loss = a_loss
        self.f_loss = f_loss
        self.loss = loss
        return loss
    
    # def on_validation_epoch_end(self):
    #     torque_pred_values = self.validation_step_outputs 
    #     for epoch, torque_pred_value in enumerate(torque_pred_values):
    #         self.writer.add_scalar('Validation/torque_Pred', torque_pred_value, global_step=epoch)
  
    def on_validation_epoch_end(self):
    # Log val/loss
        self.log_dict({"val/loss": self.loss}, prog_bar=True, on_epoch=True)

    def test_step(self, batch, batch_idx):  
        inputs,optical_flow, start, act ,torques ,target_pos = batch
        mha_out_o, mlp_inp,  act_pred, torque_pred,weights = self.actor(inputs, start) 
        if self.config.loss == 'mse':
            a_mse_loss = F.mse_loss(act, act_pred)
            f_mse_loss = F.mse_loss(torques, torque_pred)
            loss = self.config.a_loss * a_mse_loss + self.config.f_loss * f_mse_loss
            print("a_mse_loss:", a_mse_loss.item())
            print("f_mse_loss:", f_mse_loss.item())
            print("loss:", loss.item())
            self.log_dict(
            {"test/loss":loss}, prog_bar=True,on_step=True)
        if self.config.loss == 'l1':
            a_l1_loss = F.l1_loss(act, act_pred)
            f_l1_loss = F.l1_loss(torques, torque_pred)
            loss = self.config.a_loss * a_l1_loss + self.config.f_loss * f_l1_loss
            print("a_l1_loss:", a_l1_loss.item())
            print("f_l1_loss:", f_l1_loss.item())
            print("loss:", loss.item())
            self.log_dict(
            {"test/loss":loss}, prog_bar=True,on_step=True)
        if self.config.loss == 'huber':
            a_huber_loss = F.huber_loss(act, act_pred)
            f_huber_loss = F.huber_loss(torques, torque_pred)
            loss = self.config.a_loss * a_huber_loss + self.config.f_loss * f_huber_loss
            print("a_huber_loss:", a_huber_loss.item())
            print("f_huber_loss:", f_huber_loss.item())
            print("loss:", loss.item())
            self.log_dict(
            {"test/loss":loss}, prog_bar=True,on_step=True)


    # def on_train_epoch_end(self):
    #     weights_np = np.array(self.all_weights)
    #     np.save(os.path.join(self.trainer.log_dir, f"weights_epoch{self.current_epoch}.txt"), weights_np)
    #     self.all_weights = []  # 清空列表以准备下一个 epoch

    def train_dataloader(self):
        """Training dataloader"""
        return self.train_loader

    def val_dataloader(self):
        """Validation dataloader"""
        return self.val_loader
    
    def test_dataloader(self):
        """test dataloader"""
        return self.test_loader

    def configure_optimizers(self):
        return [self.optimizer], [self.scheduler]
    
    def forward(self, x):
    # 定义模型的前向传播过程
    # 在这里使用 self.actor 对输入进行处理并返回结果
        return self.actor(x)
