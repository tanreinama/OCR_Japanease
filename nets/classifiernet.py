import torch
import torch.nn as nn
import torch.nn.functional as F


class CharactorNet(nn.Module):
    def __init__(self, n_class, n_convs=[1,4,9,1], n_channels=[128,384,1152,3456]):
        super(CharactorNet, self).__init__()
        channel = 42
        layers = [nn.Conv2d(1, channel, 3, 1, 1),
                        nn.ReLU(),
                        nn.Conv2d(channel,channel,3,1,1,groups=channel),
                        nn.ReLU(),
                        nn.BatchNorm2d(channel)]
        for i in range(len(n_convs)):
            in_channel = channel
            channel = n_channels[i]
            layers.extend([
                nn.Conv2d(in_channel,channel,1,1,bias=False),
                nn.ReLU(),
                nn.Conv2d(channel,channel,3,1,1,groups=channel),
                nn.ReLU(),
                nn.BatchNorm2d(channel)
            ])
            for _ in range(n_convs[i]-1):
                layers.extend([
                    nn.Conv2d(channel,channel,1,1,bias=False),
                    nn.ReLU(),
                    nn.Conv2d(channel,channel,3,1,1,groups=channel),
                    nn.ReLU(),
                    nn.BatchNorm2d(channel)
                ])
            if i < len(n_convs)-1:
                layers.extend([
                    nn.MaxPool2d(3,2,1),
                ])

        self.layer = nn.Sequential(*layers)
        self.fc = nn.Linear(channel,n_class)

    def forward(self, input):
        x = self.layer(input)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        result = self.fc(x)
        return result

def get_classifiernet(num_class):
    model = CharactorNet(num_class)
    return model
