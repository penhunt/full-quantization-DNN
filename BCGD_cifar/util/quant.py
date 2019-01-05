import torch
from torch.autograd import Variable
import torch.nn.functional as F
import pdb
import sys
sys.path.append('../')
import config as cgs


def check_value(kernel):
    print(kernel[1].data.abs().max())

def quantize_bw(kernel):
    """
    binary quantization
    Return quantized weights of a layer.
    """
    if cgs.binary == 'median':
        delta = kernel.abs().median()
    else:
        delta = kernel.abs().mean()
    sign = kernel.sign().float()
    return sign*delta

def quantize_tnn(kernel):
    """
    ternary quantization
    Return quantized weights of a layer.
    """
    data = kernel.abs()
    delta = 0.7*data.mean()
    delta = min(delta, 100.0)
    index = data.ge(delta).float()
    sign = kernel.sign().float()
    scale = (data*index).mean()
    return scale*index*sign

def quantize_fbit(kernel):
    """
    four bit quantization
    """
    data = kernel.abs()
    delta = data.max()/15
    delta = min(delta, 10.0)
    sign = kernel.sign().float()
    q = 0.0*data

    for i in range(3,17,2):
        if i<15:
            index = data.gt((i-2)*delta).float()*data.le(i*delta).float()
        else:
            index = data.gt(13*delta).float()
        q += (i-1)/2*index
    
    scale = (data*q).sum()/(q*q).sum()
    return scale*q*sign


def _accuracy(target, output, topk=(1,5)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


quant_dict = {1: quantize_bw, 2: quantize_tnn, 4: quantize_fbit}


def optimization_step(model, loss, x_batch, y_batch, optimizer_list, param_list, eta=0):
    """
    steps:
        1. quantized W to get G
        2. use G in forward
        3. update W with W - grad(G)

    methods: ALM, no lambda

    optimizer_list:  
        optimizer: optimizer tool for NN
        optimizer_qunat: lr=0, used to recoder G
    """
    # switch to train mode
    model.train()

    optimizer, optimizer_quant = optimizer_list
    rho, lip = param_list

    x_batch, y_batch = x_batch.to(cgs.device), y_batch.to(cgs.device)

    # get all kernels
    all_W_kernels = optimizer.param_groups[1]['params']
    all_G_kernels = optimizer_quant.param_groups[0]['params']

    for i in range(len(all_W_kernels)):
        k_W = all_W_kernels[i]
        k_G = all_G_kernels[i]
        V = k_W.data
        if eta > 0:
            k_G.data = (eta*quant_dict[cgs.quant](V)+V)/(1+eta)
        else:
            k_G.data = quant_dict[cgs.quant](V)
        k_W.data, k_G.data = k_G.data, k_W.data

    # forward pass using quantized model
    logits = model(x_batch)
    
    # compute logloss
    batch_loss = loss(logits, y_batch)
    
    # compute accuracies
    pred = logits.data
    batch_accuracy = _accuracy(y_batch, pred, topk=(1,5))
    
    # compute grads
    optimizer.zero_grad()
    batch_loss.backward()

    for i in range(len(all_W_kernels)):
        k_W = all_W_kernels[i]
        k_G = all_G_kernels[i]
        k_W.data, k_G.data = k_G.data, k_W.data
        k_W.data = lip*k_W.data + rho*k_G.data

    # update parameters
    optimizer.step()

    return batch_loss.item(), batch_accuracy
