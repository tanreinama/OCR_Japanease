import torch
import torch.nn as nn
import torch.nn.functional as F

class Block(nn.Module):
    def __init__(self,in_channel,out_channel,stride=1):
        super(Block, self).__init__()

        if in_channel != out_channel or stride != 1:
            self.skip = nn.Conv2d(in_channel,out_channel,1,stride=stride, bias=False)
            self.skipbn = nn.BatchNorm2d(out_channel)
        else:
            self.skip=None

        self.act = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channel,out_channel,3,stride=stride,padding=1,bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.conv2 = nn.Conv2d(out_channel,out_channel,3,stride=1,padding=1,bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)

    def forward(self,inp):
        x = self.act(inp)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.bn2(x)

        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp

        x += skip
        return x
