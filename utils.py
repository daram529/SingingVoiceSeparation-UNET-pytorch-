import torch
import torch.nn as nn
import torch.nn.functional as F



class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=5, stride=2, padding=1),
       		nn.BatchNorm2d(out_ch),
        	nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch, dropout_flag=False):
        super().__init__()
        self.convTrans = nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_ch, out_channels=out_ch, kernel_size=5, stride=2, padding=1),
       		nn.BatchNorm2d(out_ch),
        	nn.ReLU(inplace=True)
        )
        self.dropout_flag = dropout_flag
        if self.dropout_flag:
        	self.dropout = nn.Dropout(0.5)

    def forward(self, x1, x2=None):
    	if x2:
    		x1 = torch.cat([x1, x2], dim=1)
        x = self.convTrans(x1)
        if self.dropout_flag:
        	x = self.dropout(x)
        return x

class final_conv(nn.Module):
	def __init__(self, in_ch, out_ch):
       	super().__init__()
        self.convTrans = nn.ConvTranspose2d(in_channels=in_ch, out_channels=out_ch, kernel_size=5, stride=2, padding=1)

    def forward(self, x1, x2):
    	x = torch.cat([x1, x2], dim=1)
        x = self.convTrans(x1)
        x = nn.sigmoid(x)
        return x