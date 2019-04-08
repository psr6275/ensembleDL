import torch
import torch.nn as nn
from .module import Flatten
import math
cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
}

__all__ = ['cifar10','cifar100']
DIM=128

class CIFAR10(nn.Module):
    def __init__(self):
        super(CIFAR10, self).__init__()
        self.classifier = nn.Sequential(
                nn.Conv2d(3,96,3),
                nn.GroupNorm(32,96),
                nn.ELU(),

                nn.Conv2d(96,96,3),
                nn.GroupNorm(32,96),
                nn.ELU(),

                nn.Conv2d(96,96,3, stride=2),
                nn.GroupNorm(32,96),
                nn.ELU(),

                nn.Dropout2d(0.5),

                nn.Conv2d(96,192,3),
                nn.GroupNorm(32,192),
                nn.ELU(),

                nn.Conv2d(192,192,3),
                nn.GroupNorm(32,192),
                nn.ELU(),

                nn.Conv2d(192,192,3,stride=2),
                nn.GroupNorm(32,192),
                nn.ELU(),

                nn.Dropout2d(0.5),

                nn.Conv2d(192,192,3),
                nn.GroupNorm(32,192),
                nn.ELU(),

                nn.Conv2d(192,192,1),
                nn.GroupNorm(32,192),
                nn.ELU(),

                nn.Conv2d(192,10,1),

                nn.AvgPool2d(2),
                Flatten()
            )
    def forward(self,x):
        x = self.classifier(x)
        return x

class CIFAR10_VGG(nn.Module):
    def __init__(self, vgg_name):
        super(CIFAR10_VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, 10)
        self._initialize_weights()

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()



def vgg11():
    model = CIFAR10_VGG('VGG11')
    return model

def vgg13():
    model = CIFAR10_VGG('VGG13')
    return model

def vgg16():
    model = CIFAR10_VGG('VGG16')
    return model

def vgg19():
    model = CIFAR10_VGG('VGG19')
    return model

def simpleCNN():
    model = CIFAR10()
    return model

