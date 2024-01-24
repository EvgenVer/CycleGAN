import torch
import torch.nn as nn


class ConvBlockGen(nn.Module):
    
    def __init__(self, input_ch,
                 output_ch, downsampling=True,
                 activation=True, **kwargs):
        super(ConvBlockGen, self).__init__()
        
        if downsampling:
            self.conv = nn.Sequential(
                nn.Conv2d(input_ch, output_ch, padding_mode="reflect", **kwargs), 
                nn.InstanceNorm2d(output_ch),
                nn.ReLU(inplace=True) if activation else nn.Identity()
            )
        else:
            self.conv = nn.Sequential(
                nn.ConvTranspose2d(input_ch, output_ch, **kwargs),
                nn.InstanceNorm2d(output_ch),
                nn.ReLU(inplace=True) if activation else nn.Identity()
            )
        
    def forward(self, x):
        return self.conv(x)
    
    
class ResBlock(nn.Module):
    def __init__(self, num_channels):
        super(ResBlock, self).__init__()
        self.res = nn.Sequential(
            ConvBlockGen(num_channels, num_channels, activation=True, kernel_size=3, padding=1), 
            ConvBlockGen(num_channels, num_channels, activation=False, kernel_size=3, padding=1)
        )
        

    def forward(self, x):
        return x + self.res(x)
    

class Generator(nn.Module):
    def __init__(self, img_ch=3, num_hid_channels=64, num_residuals=9):
        super(Generator, self).__init__()
        
        self.encoder = nn.Sequential(
            ConvBlockGen(img_ch, num_hid_channels, kernel_size=7, stride=1, padding=3),
            ConvBlockGen(num_hid_channels, num_hid_channels*2, kernel_size=3, stride=2, padding=1),
            ConvBlockGen(num_hid_channels*2, num_hid_channels*4, kernel_size=3, stride=2, padding=1)
        )
        
        self.transform = nn.Sequential(
            *[ResBlock(num_hid_channels*4) for _ in range(num_residuals)]
        )
        
        self.decoder = nn.Sequential(
            ConvBlockGen(num_hid_channels*4, num_hid_channels*2, downsampling=False,
                         kernel_size=3, stride=2, padding=1, output_padding=1),
            ConvBlockGen(num_hid_channels*2, num_hid_channels, downsampling=False,
                         kernel_size=3, stride=2, padding=1, output_padding=1),
            ConvBlockGen(num_hid_channels, img_ch, kernel_size=7, stride=1, padding=3)
        )
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.transform(x)
        x = self.decoder(x)
        return torch.tanh(x)
    
class ConvBlockDis(nn.Module):
    
    def __init__(self, input_ch, output_ch, **kwargs):
        super(ConvBlockDis, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(input_ch, output_ch, padding_mode="reflect", **kwargs), 
            nn.InstanceNorm2d(output_ch), 
            nn.LeakyReLU(0.2, inplace=True)
        )
        
    def forward(self, x):
        return self.conv(x)
    
class Discriminator(nn.Module):
    def __init__(self, img_ch=3, num_hid_channels=64):
        super(Generator, self).__init__()
        
        self.result = nn.Sequential(
            ConvBlockDis(img_ch, num_hid_channels, kernel_size=4, stride=2, padding=1),
            ConvBlockDis(num_hid_channels, num_hid_channels*2, kernel_size=4, stride=2, padding=1),
            ConvBlockDis(num_hid_channels*2, num_hid_channels*4, kernel_size=4, stride=2, padding=1),
            ConvBlockDis(num_hid_channels*4, num_hid_channels*8, kernel_size=4, stride=2, padding=1),
            nn.Conv2d(num_hid_channels*8, 1, kernel_size=3, stride=1, padding=1, padding_mode="reflect"),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.result(x)
        
        
        
        