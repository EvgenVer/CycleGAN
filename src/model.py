import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlockGen(nn.Module):
    
    def __init__(self, input_ch,
                 output_ch, downsampling=True,
                 activation=True, **kwargs):
        super(ConvBlockGen, self).__init__()
        
        if downsampling:
            self.conv = nn.Sequential(nn.Conv2d(input_ch, output_ch, **kwargs), 
                                      nn.InstanceNorm2d(output_ch),
                                      nn.ReLU(inplace=True) if activation else nn.Identity())
        else:
            self.conv = nn.Sequential(nn.ConvTranspose2d(input_ch, output_ch, **kwargs),
                                      nn.InstanceNorm2d(output_ch),
                                      nn.ReLU(inplace=True) if activation else nn.Identity())
        
    def forward(self, x):
        return self.conv(x)
    
    
class ResBlock(nn.Module):
    def __init__(self, num_channels):
        super(ResBlock, self).__init__()
        self.res = nn.Sequential(
            ConvBlockGen(num_channels, num_channels, activation=True, kernel_size=3, padding=1), 
            ConvBlockGen(num_channels, num_channels, activation=False, kernel_size=3, padding=1))
        

    def forward(self, x):
        return x + self.res(x)
    

class Generator(nn.Module):
    def __init__(self, img_ch=3, num_hid_channels=64, num_residuals=9):
        super(Generator, self).__init__()
        self.encoder = nn.Sequential()
        