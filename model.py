import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
from .utils import *
# from torch.utils.data import DataLoader
from torch import optim
# from torch.autograd import Variable
# from data_generator import AudioSampleGenerator
# from scipy.io import wavfile


class UNET(nn.Module):

    def __init__(self):
        super().__init__()
        # encoder
        self.down1 = down(1, 16)
        self.down2 = down(16, 32)
        self.down3 = down(32, 64)
        self.down4 = down(64, 128)
        self.down5 = down(128, 256)
        self.down6 = down(256, 512)

        #decoder
        self.up1 = up(512, 256, dropout_flag=True)
        self.up2 = up(512, 128, dropout_flag=True)
        self.up3 = up(256, 64, dropout_flag=True)
        self.up4 = up(128, 32)
        self.up5 = up(64, 16)
        self.fconv = final_conv(32, 1)


    def forward(self, X):
        x1 = self.down(X)
        x2 = self.down(x1)
        x3 = self.down(x2)
        x4 = self.down(x3)
        x5 = self.down(x4)
        x6 = self.down(x5)
        x = self.up1(x6)
        x = self.up2(x, x5)
        x = self.up3(x, x4)
        x = self.up4(x, x3)
        x = self.up5(x, x2)
        x = self.fconv(x, x1)
        return x

def TrainModel(Xlist, Ylist, epoch=40, savefile="unet.model"):
    def __init__(self):
        super().__init__()
        with self.init_scope():
            self.unet = unet

    def __call__(self, X, Y):
        theta = self.unet(X)
        self.loss = F.mean_absolute_error(X * theta, Y)
        return self.loss
