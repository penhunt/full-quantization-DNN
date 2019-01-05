import argparse
import sys
sys.path.append('./util/')
sys.path.append('../nets/')
import math
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision.datasets as datasets_th

import config as cgs
from vgg_type import QtUniVGG
from resnet_type_cifar import QtUniRes
from train_net_alpha import train, train_alpha, train_alm1_alpha, train_alm1_eta_alpha, validate, val_quant
from tests import reptest_float, reptest_ft_alpha, reptest_bc_alpha, reptest_br_alpha


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained float model')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--method', default='float', type=str, metavar='M',
                    help='training method')
parser.add_argument('--rho', default=0.0, type=float, metavar='R',
                    help='blended parameter')

cifar = 10

def weight_init(m):
    if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()


def main():
    global args
    args = parser.parse_args()


    # Data
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

    trainset = datasets_th.CIFAR10(root='../../data', train=True, download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=cgs.batch_size, shuffle=True, num_workers=args.workers)

    testset = datasets_th.CIFAR10(root='../../data', train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=args.workers)
    
    n_batches = len(train_loader)
    n_validation_batches = len(test_loader)
    print("There are ", n_batches, " batches in the train set.")
    print("There are ", n_validation_batches, " batches in the val set.")


    # Model
    if args.method == 'float':
        ffun, bfun = 'relu', 'relu'
    else:
        ffun, bfun = cgs.ffun, cgs.bfun
    if 'vgg' in cgs.arch:
        net = QtUniVGG(cgs.arch.upper(), cifar, cgs.bit, ffun = ffun, bfun = bfun,
            rate_factor = cgs.rate_factor, gd_type = cgs.gd_type)
    elif 'res' in cgs.arch:
        net = QtUniRes(cgs.arch, num_classes = cifar, bit = cgs.bit, ffun = ffun, bfun = bfun,
            rate_factor = cgs.rate_factor, gd_type = cgs.gd_type)
    net.apply(weight_init)
    net = net.to(cgs.device)
    if cgs.device == 'cuda':
        net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

    
    if args.method == 'ft':
        reptest_ft_alpha(train_loader, test_loader, net, args, train_alpha, validate)
    elif args.method == 'bc':
        reptest_bc_alpha(train_loader, test_loader, net, args, train_alm1_alpha, val_quant)
    elif args.method == 'br':
        reptest_br_alpha(train_loader, test_loader, net, args, train_alm1_eta_alpha, val_quant)
    else:
        reptest_float(train_loader, test_loader, net, args, train, validate, load_float_model=0)
        
    return
    

if __name__ == '__main__':
    main()
