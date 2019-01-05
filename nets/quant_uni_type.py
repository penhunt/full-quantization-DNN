import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


bit_alpha = {1:1.2221, 2:0.6511, 4:0.1946}


dtype = torch.cuda.FloatTensor

class QtUniFunction(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """

    def __init__(self, bit, ffun = 'quant', bfun = 'inplt', rate_factor = 0.001,
                 gd_alpha = False, gd_type = 'mean'):
        global fw, bw, level, r, g, g_t
        fw = ffun
        bw = bfun
        level = 2**bit-1
        r = rate_factor
        g = gd_alpha
        g_t = gd_type

    @staticmethod
    def forward(ctx, input, alpha):
        global fw, level
        m = 1e-3
        M = 5
        alpha.data = alpha.data.clamp(min=m,max=M)
        a = alpha.item()
        ctx.save_for_backward(input,alpha)
        output = input.clone()
        output[input <= 0] = 0
        if fw == 'quant':
            ind = input.gt(0) * input.le(a*(level-1))
            output[ind] = (input[ind]/a).ceil() * a
            output[input > a*(level-1)] = a * level
        return output

    @staticmethod
    def backward(ctx, grad_output):
        global fw, bw, level, r, g, g_t
        input, alpha = ctx.saved_tensors
        a = alpha.item()
        grad_input = grad_alpha = None
        grad_input = grad_output.clone()
        if bw != 'idnty':
            grad_input[input <= 0] = 0
        if bw == 'inplt':
            grad_input[input > a*level] = 0
        # grad_alpha = (grad_output * 0).sum().expand(1)
        if not g:
            a = input.abs().max() / level
            alpha.data.fill_(a)
        if fw == 'quant' and g:
            grad_0 = grad_output.clone()
            grad_0.data.zero_()
            if g_t == 'ae':
                ind = input.gt(0) * input.le(a*(level-1))
                grad_0[ind] = (input[ind]/a).ceil()
                grad_0[input > a*(level-1)] = level
            elif g_t == 'pact':
                grad_0[input > a*level] = level
            else:
                ind = input.gt(0) * input.le(a*level)
                if g_t == 'min':
                    grad_0[ind] = 1
                else:
                    grad_0[ind] = (level+1)/2
            # grad_0[ind] = 1
            grad_0[input > a*level] = level
            grad_alpha = (grad_output * grad_0).sum() * r
            grad_alpha = torch.clamp(grad_alpha,-1,1)
            grad_alpha = grad_alpha.expand(1)
            # print(grad_alpha)
        return grad_input, grad_alpha
    
class QtUni(nn.Module):
    def __init__(self, bit, ffun = 'relu', bfun = 'relu', rate_factor = 0.01,
                 gd_alpha = False, gd_type = 'mean'):
        super(QtUni, self).__init__()
        self.fw = ffun
        self.bw = bfun
        self.bit = bit
        self.a = bit_alpha[bit]
        self.r = rate_factor
        self.g = int(gd_alpha==True)
        self.alpha = nn.Parameter(torch.Tensor(1))
        self.rate_factor = nn.Parameter(torch.Tensor(1), requires_grad=False)
        self.gd_alpha = nn.Parameter(torch.Tensor(1), requires_grad=False)
        self.g_t = gd_type
        self.reset_parameters()
        
    def reset_parameters(self):
        self.alpha.data = torch.ones(1) * self.a
        self.rate_factor.data = torch.ones(1) * self.r
        self.gd_alpha.data = torch.ones(1) * self.g
        
    def forward(self, input):
        r = self.rate_factor.item()
        g = bool(self.gd_alpha.item())
        qtuni = QtUniFunction(self.bit,ffun=self.fw,bfun=self.bw,rate_factor=r,gd_alpha=g,gd_type=self.g_t)
        return qtuni.apply(input, self.alpha)