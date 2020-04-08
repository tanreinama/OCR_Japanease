import torch
import torch.nn as nn
import torch.nn.functional as F

from .block import Mish, SeparableConv2d, Block

class WideTipXception(nn.Module):
    def __init__(self, num_class):
        super(WideTipXception, self).__init__()

        self.conv1 = nn.Conv2d(1, 192, 3, 2, 1, bias=True)
        self.bn1 = nn.BatchNorm2d(192)
        self.mish = Mish()

        self.conv2 = nn.Conv2d(192, 512, 3, 1, 1, bias=True)
        self.bn2 = nn.BatchNorm2d(512)

        self.block1 = Block(512,1024,3,1)
        self.block2 = Block(1024,1024,3,1)
        self.block3 = Block(1024,1024,3,1)
        self.block4 = Block(1024,1024,3,1)
        self.block5 = Block(1024,1024,3,1)
        self.block6 = Block(1024,2048,2,2)
        self.block7 = Block(2048,3072,2,2)

        self.conv3 = SeparableConv2d(3072,4096,3,stride=1,padding=0,bias=True)
        self.fc = nn.Linear(4096, num_class)

    def forward(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.mish(x)

        x = self.conv2(x)
        x = self.bn2(x)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)

        x = self.mish(x)
        x = self.conv3(x)

        x = self.mish(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        result = self.fc(x)

        return result

def get_classifiernet(num_class):
    model = WideTipXception(num_class)
    return model
