import torch
import torch.nn as nn
import torch.nn.functional as F
from .block import Block

class UNet(nn.Module):
    def __init__(self, n_blocks=[1,2,6,10,4], n_channels=[24,32,48,96,192,384]):
        super(UNet, self).__init__()

        backbone = []
        in_channel = n_channels[0]
        for i in range(len(n_blocks)):
            channel = n_channels[i+1]
            layers = [Block(in_channel,channel)]
            for _ in range(n_blocks[i]-1):
                layers.append(Block(channel,channel))
            layers.append(Block(channel,channel,2))
            in_channel = channel
            backbone.append(nn.Sequential(*layers))
        self.backbone = nn.ModuleList(backbone)

        upstep = [Block(n_channels[1], n_channels[0])]
        for i in range(len(n_blocks)-1):
            channel = n_channels[i+1]
            upstep.append(Block(channel*2, channel))
        self.upstep = nn.ModuleList(upstep)

        downstep = []
        out_channel = n_channels[0]
        for i in range(len(n_blocks)):
            channel = n_channels[i+1]
            downstep.append(Block(channel, out_channel))
            out_channel = channel
        self.downstep = nn.ModuleList(downstep)

    def forward(self, x):
        back_out = []
        for i in range(len(self.backbone)):
            x = self.backbone[i](x)
            back_out.append(x)

        out = back_out[len(back_out)-1]
        for i in range(len(back_out)-1):
            low = self.downstep[len(self.downstep)-i-1](out)
            up1 = F.interpolate(low, scale_factor=2)
            up2 = back_out[len(back_out)-i-2]
            up = torch.cat([up1,up2], dim=1)
            out = self.upstep[len(self.upstep)-i-1](up)
        return self.upstep[0](out)


class DetectionNet(nn.Module):
    def __init__(self):
        super(DetectionNet, self).__init__()

        self.conv1 = nn.Conv2d(1, 24, 3, 1, 1, bias=True)
        self.block1 = UNet()

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        self.out1 = nn.Conv2d(24, 1, 1, bias=True)
        self.out2 = nn.Conv2d(24, 1, 1, bias=True)
        self.out3 = nn.Conv2d(24, 2, 1, bias=True)

    def forward(self, input):
        x = self.conv1(input)
        x = self.block1(x)
        out = self.relu(x)
        result = {'hm_wd':self.sigmoid(self.out1(out)),
                'hm_sent':self.sigmoid(self.out2(out)),
                'of_size':self.sigmoid(self.out3(out))}
        return result

def get_detectionnet():
    model = DetectionNet()
    return model
