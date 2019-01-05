import torch
import torch.nn as nn
from collections import OrderedDict
from quant_uni_type import QtUni

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, vgg_name, cifar):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, cifar)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        in_channels = 3
        x = cfg[0]
        layers = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(in_channels, x, kernel_size=3, padding=1)),
            ('norm0', nn.BatchNorm2d(x)),
            ('relu0', nn.ReLU(inplace=True))
        ]))
        in_channels = x
        
        index_pool = 0; index_block = 1
        for x in cfg[1:]:
            if x == 'M':
                layers.add_module('pool%d' % index_pool, 
                                  nn.MaxPool2d(kernel_size=2, stride=2))
                index_pool += 1
            else:
                layers.add_module('conv%d' % index_block, 
                                  nn.Conv2d(in_channels, x, kernel_size=3, padding=1)),
                layers.add_module('norm%d' % index_block, 
                                  nn.BatchNorm2d(x)),
                layers.add_module('relu%d' % index_block, 
                                  nn.ReLU(inplace=True))
                in_channels = x
                index_block += 1
#         layers.add_module('avg_pool%d' % index_pool, 
#                           nn.AvgPool2d(kernel_size=1, stride=1))
        return layers


class QtUniVGG(nn.Module):
    def __init__(self, vgg_name, cifar, bit, ffun = 'relu', bfun = 'relu', rate_factor = 0.1, gd_alpha = False,
                gd_type = 'mean'):
        super(QtUniVGG, self).__init__()
        self.bit = bit
        self.ffun = ffun
        self.bfun = bfun
        self.r = rate_factor
        self.g = gd_alpha
        self.g_t = gd_type
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, cifar)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        in_channels = 3
        x = cfg[0]
        layers = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(in_channels, x, kernel_size=3, padding=1)),
            ('norm0', nn.BatchNorm2d(x)),
            ('relu0', QtUni(self.bit,ffun=self.ffun,bfun=self.bfun,rate_factor=self.r,gd_alpha=self.g,gd_type=self.g_t))
        ]))
        in_channels = x
        
        index_pool = 0; index_block = 1
        for x in cfg[1:]:
            if x == 'M':
                layers.add_module('pool%d' % index_pool, 
                                  nn.MaxPool2d(kernel_size=2, stride=2))
                index_pool += 1
            else:
                layers.add_module('conv%d' % index_block, 
                                  nn.Conv2d(in_channels, x, kernel_size=3, padding=1)),
                layers.add_module('norm%d' % index_block, 
                                  nn.BatchNorm2d(x)),
                layers.add_module('relu%d' % index_block,
                                  QtUni(self.bit,ffun=self.ffun,bfun=self.bfun,rate_factor=self.r,gd_alpha=self.g,
                                        gd_type=self.g_t))
                in_channels = x
                index_block += 1
#         layers.add_module('avg_pool%d' % index_pool, 
#                           nn.AvgPool2d(kernel_size=1, stride=1))
        return layers