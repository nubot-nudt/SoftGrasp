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
        mha_out_o,act_pred,torque_pred, weights = self.actor(inputs, start) 
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
        return loss

    def validation_step(self, batch, batch_idx):
        inputs,optical_flow, start, act ,torques ,target_pos = batch
        image_inp,angle,torque= inputs
        mha_out_o, act_pred, torque_pred,weights = self.actor(inputs, start) 
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

        tqdm.write("a_loss:%.8f" % a_loss + ", f_loss:%.8f" % f_loss + ", loss:%.8f" % loss)
        self.a_loss = a_loss
        self.f_loss = f_loss
        self.loss = loss
        return loss
    
  
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
        return self.actor(x)
