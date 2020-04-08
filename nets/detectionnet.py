import torch
import torch.nn as nn
import torch.nn.functional as F

from .block import Mish, SeparableConv2d, Block

class HourglassNet(nn.Module):
    def __init__(self, depth, channel):
        super(HourglassNet, self).__init__()
        self.depth = depth
        hg = []
        for _ in range(self.depth):
            hg.append([
                Block(channel,channel,3,1,activation=Mish()),
                Block(channel,channel,2,2,activation=Mish()),
                Block(channel,channel,3,1,activation=Mish())
            ])
        hg[0].append(Block(channel,channel,3,1,activation=Mish()))
        hg = [nn.ModuleList(h) for h in hg]
        self.hg = nn.ModuleList(hg)

    def _hour_glass_forward(self, n, x):
        up1 = self.hg[n-1][0](x)
        low1 = self.hg[n-1][1](up1)

        if n > 1:
            low2 = self._hour_glass_forward(n-1, low1)
        else:
            low2 = self.hg[n-1][3](low1)

        low3 = self.hg[n-1][2](low2)
        up2 = F.interpolate(low3, scale_factor=2)
        out = up1 + up2
        return out

    def forward(self, x):
        return self._hour_glass_forward(self.depth, x)

class XceptionHourglass(nn.Module):
    def __init__(self, use_grid_offset=True, use_offset_pooling=False):
        super(XceptionHourglass, self).__init__()
        self.use_grid_offset = use_grid_offset
        self.use_offset_pooling = use_offset_pooling

        self.conv1 = nn.Conv2d(1, 64, 3, 2, 1, bias=True)
        self.bn1 = nn.BatchNorm2d(64)
        self.mish = Mish()

        self.conv2 = nn.Conv2d(64, 96, 3, 1, 1, bias=True)
        self.bn2 = nn.BatchNorm2d(96)

        self.head = Block(96,96,2,2)

        self.block1 = HourglassNet(3, 96)
        self.bn3 = nn.BatchNorm2d(96)
        self.block2 = HourglassNet(3, 96)

        self.sigmoid = nn.Sigmoid()

        self.conv3_1 = nn.Conv2d(96, 1, 1, bias=True)
        self.conv3_2 = nn.Conv2d(96, 1, 1, bias=True)
        self.conv3_3 = nn.Conv2d(96, 2, 1, bias=True)
        if self.use_grid_offset:
            self.conv3_4 = nn.Conv2d(96, 2, 1, bias=True)
        self.conv4_1 = nn.Conv2d(96, 1, 1, bias=True)
        self.conv4_2 = nn.Conv2d(96, 1, 1, bias=True)
        self.conv4_3 = nn.Conv2d(96, 2, 1, bias=True)
        if self.use_grid_offset:
            self.conv4_4 = nn.Conv2d(96, 2, 1, bias=True)

    def forward(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.mish(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.mish(x)

        x = self.head(x)
        out1 = self.block1(x)
        x = self.bn3(out1)
        x = self.mish(x)
        out2 = self.block2(x)

        out1 = self.mish(out1)
        out2 = self.mish(out2)

        out1_off = out1
        out2_off = out2
        if self.use_offset_pooling:
            out1_off = F.avg_pool2d(out1, 2, 2)
            out2_off = F.avg_pool2d(out2, 2, 2)

        result = [{'hm_wd':self.sigmoid(self.conv3_1(out1)),
                'hm_sent':self.sigmoid(self.conv3_2(out1)),
                'of_size':self.sigmoid(self.conv3_3(out1_off))},
                {'hm_wd':self.sigmoid(self.conv4_1(out2)),
                'hm_sent':self.sigmoid(self.conv4_2(out2)),
                'of_size':self.sigmoid(self.conv4_3(out2_off))}]
        if self.use_grid_offset:
            result[0]['of_grid'] = self.sigmoid(self.conv3_4(out1_off))
            result[1]['of_grid'] = self.sigmoid(self.conv4_4(out2_off))
        return result

def get_detectionnet(use_grid_offset, use_offset_pooling):
    model = XceptionHourglass(use_grid_offset, use_offset_pooling)
    return model
