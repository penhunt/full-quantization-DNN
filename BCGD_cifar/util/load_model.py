import os
import sys
sys.path.append('../')
import torch
import torch.nn as nn
import torch.optim as optim
import config as cgs
import numpy as np
import matplotlib.pyplot as plt
import torchvision.models as models

def load_model(net, name):
    assert os.path.isdir('models'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('models/' + name)
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['best_acc']
    start_epoch = checkpoint['start_epoch']
    return best_acc, start_epoch

def load_model_quant(net, name):
    assert os.path.isdir('models'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('models/' + name)
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['best_acc']
    start_epoch = checkpoint['start_epoch']
    all_G_kernels = checkpoint['G_kernels']
    return best_acc, start_epoch, all_G_kernels


def get_model2(model, lr = cgs.lr, momentum = cgs.momentum, weight_decay = cgs.weight_decay):
    # set the first layer not trainable
    # model.features.conv0.weight.requires_grad = False

    # all fc layers
    weights = [
        p for n, p in model.named_parameters()
        if 'weight' in n and 'conv' not in n
    ]

    # all conv layers
    weights_to_be_quantized = [
        p for n, p in model.named_parameters()
        # if 'conv' in n and 'conv0' not in n
        if 'conv' in n and 'weight' in n
    ]

    biases = [
        p for n, p in model.named_parameters()
        if 'bias' in n
    ]    

    params = [
        {'params': weights, 'weight_decay': cgs.weight_decay},
        {'params': weights_to_be_quantized, 'weight_decay': cgs.weight_decay},
        {'params': biases,  'weight_decay': cgs.weight_decay}
    ]
    optimizer = optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)

    loss = nn.CrossEntropyLoss().to(cgs.device)
    model = model.to(cgs.device)  # move the model to gpu
    return model, loss, optimizer

def get_model2_a(model, lr = cgs.lr, momentum = cgs.momentum, weight_decay = cgs.weight_decay, decay_factor=1):

    # set the first layer not trainable
    # model.features.conv0.weight.requires_grad = False

    # all fc layers
    weights = [
        p for n, p in model.named_parameters()
        if 'weight' in n and 'conv' not in n
    ]

    # all conv layers
    weights_to_be_quantized = [
        p for n, p in model.named_parameters()
        # if 'conv' in n and 'conv0' not in n
        if 'conv' in n and 'weight' in n
    ]

    biases = [
        p for n, p in model.named_parameters()
        if 'bias' in n
    ]
    
    alphas = [
        p for n, p in model.named_parameters()
        if 'alpha' in n and 'gd_alpha' not in n
    ]

    params = [
        {'params': weights, 'weight_decay': weight_decay},
        {'params': weights_to_be_quantized, 'weight_decay': weight_decay},
        {'params': biases,  'weight_decay': weight_decay}
    ]
    
    params_a = [
        {'params': weights, 'weight_decay': weight_decay},
        {'params': weights_to_be_quantized, 'weight_decay': weight_decay},
        {'params': biases,  'weight_decay': weight_decay},
        {'params': alphas,  'weight_decay': weight_decay*decay_factor}
    ]
    
    optimizer = optim.SGD(params, lr=lr, momentum=momentum)
    
    optimizer_a = optim.SGD(params_a, lr=lr, momentum=momentum)

    loss = nn.CrossEntropyLoss().cuda()
    model = model.cuda()  # move the model to gpu
    return model, loss, optimizer, optimizer_a



def init_alpha(model, a = 0.67):
    for n, p in model.named_parameters():
        if 'alpha' in n and 'gd_alpha' not in n:
            p.data[0] = a
            
def init_factor(model, a = 0.01):
    for n, p in model.named_parameters():
        if 'rate_factor' in n:
            p.data[0] = a