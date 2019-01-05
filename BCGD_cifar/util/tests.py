import sys
sys.path.append('../')

import torch.optim as optim
from torch.autograd import Variable

import config as cgs
from load_model import load_model, load_model_quant, get_model2, get_model2_a, init_alpha, init_factor

        
def reptest_float(trainloader, testloader, net, args, train, validate, load_float_model = 0):

    n_epochs = cgs.n_epochs
    best_prec1 = 0
    start_epoch = 0

    model, loss, optimizer = get_model2(net)
    name = cgs.arch+'_float.t7'
    params_float = [best_prec1, start_epoch, name]
    
    if load_float_model:
        print("=> using pre-trained model '{}'".format(cgs.arch.upper()))
        best_acc, start_epoch = load_model(net, name=cgs.arch+'_float.t7')
        # load_pretrained(net, cgs.arch)
        best_acc = 0

    if args.evaluate:
        best_acc, start_epoch = load_model(net, name=name)
        validate(testloader, model, loss)
        return
 
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=cgs.steps, gamma=0.1) 
    train(model, loss, optimizer, 
          trainloader, testloader, params_float, 
          n_epochs=n_epochs, lr_scheduler=lr_scheduler)
    
    
def reptest_ft_alpha(trainloader, testloader, net, args, train, validate):

    n_epochs = cgs.n_epochs
    best_prec1 = 0
    start_epoch = 0
    
    model, loss, optimizer, optimizer_a = get_model2_a(net)
    test = str(cgs.bit)+'aft'
    name = cgs.arch+'_'+test+'.t7'
    params_float = [best_prec1, start_epoch, name]

    if cgs.load_float_model:
        print("=> using pre-trained model '{}'".format(cgs.arch))
        best_acc, start_epoch = load_model(net, name=cgs.arch+'_float.t7')
        # load_pretrained(net, cgs.arch)
        best_acc = 0    

    if args.evaluate:
        best_acc, start_epoch = load_model(net, name=name)
        validate(testloader, model, loss)
        return

    if cgs.initial_alpha > 0:
        init_alpha(model, cgs.initial_alpha)

    if cgs.rate_factor > 0:
        init_factor(model, cgs.rate_factor)

    #lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9) 
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=cgs.steps, gamma=0.1) 

    train(model, loss, optimizer, optimizer_a,
        trainloader, testloader,
        params_float, n_epochs=n_epochs,
        lr_scheduler=lr_scheduler,
        stage=cgs.stage, gamma=cgs.gamma)

    
def reptest_bc_alpha(trainloader, testloader, net, args, train, validate, load_model_type = 'float'):
    
    lip = 1-args.rho
    n_epochs = cgs.n_epochs
    all_G_kernels = []
    best_prec1 = 0
    start_epoch = 0
    
    model, loss, optimizer, optimizer_a = get_model2_a(net)
    test = str(cgs.bit)+'a'+str(cgs.quant)+'w'
    if cgs.quant == 1:
        name = cgs.arch+'_BC_'+test+'_'+cgs.binary+'.t7'
    else:
        name = cgs.arch+'_BC_'+test+'.t7'
    params_quant = [args.rho, lip, best_prec1, start_epoch, name]
    
    if load_model_type == 'float':
        print("=> using pre-trained model '{}'".format(cgs.arch))
        best_acc, start_epoch = load_model(net, name=cgs.arch+'_float.t7')
        # load_pretrained(net, cgs.arch)
        best_acc = 0
        all_G_kernels = [
            Variable(kernel.data.clone(), requires_grad=True) 
            for kernel in optimizer.param_groups[1]['params']
        ]
    elif load_model_type == 'quant':
        print("=> resuming pre-trained quant model '{}'".format(cgs.arch))
        best_acc, start_epoch, all_G_kernels = load_model_quant(net, name=name)
     
    all_W_kernels = [kernel for kernel in optimizer.param_groups[1]['params']]
    kernels = [{'params': all_G_kernels}]
    optimizer_quant = optim.SGD(kernels, lr=0)
    
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=cgs.steps, gamma=0.1)
    
    if args.evaluate:
        best_acc, start_epoch, all_G_kernels = load_model_quant(net, name=name)
        validate(testloader, model, loss, [all_W_kernels, all_G_kernels])
        return
    
    if cgs.initial_alpha > 0:
        init_alpha(model, cgs.initial_alpha)

    if cgs.rate_factor > 0:
        init_factor(model, cgs.rate_factor)
    
    train(model, loss, [optimizer, optimizer_a, optimizer_quant],
        [all_W_kernels, all_G_kernels],
        trainloader, testloader, params_quant,
        n_epochs=n_epochs, lr_scheduler=lr_scheduler,
        rho_rate=cgs.rho_rate,
        stage=cgs.stage, gamma=cgs.gamma)
    
    
def reptest_br_alpha(trainloader, testloader, net, args, train, validate, load_model_type = 'float'):
    
    lip = 1-args.rho
    n_epochs = cgs.n_epochs
    all_G_kernels = []
    best_prec1 = 0
    start_epoch = 0
    
    model, loss, optimizer, optimizer_a = get_model2_a(net)
    test = str(cgs.bit)+'a'+str(cgs.quant)+'w'
    if cgs.quant == 1:
        name = cgs.arch+'_BR_'+test+'_'+cgs.binary+'.t7'
    else:
        name = cgs.arch+'_BR_'+test+'.t7'
    params_quant = [args.rho, lip, best_prec1, start_epoch, name]
    
    if load_model_type == 'float':
        print("=> using pre-trained model '{}'".format(cgs.arch))
        best_acc, start_epoch = load_model(net, name=cgs.arch+'_float.t7')
        # load_pretrained(net, cgs.arch)
        best_acc = 0
        all_G_kernels = [
            Variable(kernel.data.clone(), requires_grad=True) 
            for kernel in optimizer.param_groups[1]['params']
        ]
    elif load_model_type == 'quant':
        print("=> resuming pre-trained quant model '{}'".format(cgs.arch))
        best_acc, start_epoch, all_G_kernels = load_model_quant(net, name=name)
        
    all_W_kernels = [kernel for kernel in optimizer.param_groups[1]['params']]
    kernels = [{'params': all_G_kernels}]
    optimizer_quant = optim.SGD(kernels, lr=0)
    
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=cgs.steps, gamma=0.1)
    
    if args.evaluate:
        best_acc, start_epoch, all_G_kernels = load_model_quant(net, name=name)
        validate(testloader, model, loss, [all_W_kernels, all_G_kernels])
        return
    
    train(model, loss, [optimizer, optimizer_a, optimizer_quant],
        [all_W_kernels, all_G_kernels],
        trainloader, testloader, params_quant,
        n_epochs=n_epochs, lr_scheduler=lr_scheduler,
        rho_rate=cgs.rho_rate, eta_rate=cgs.eta_rate,
        m_epochs=cgs.m_epochs,
        stage=cgs.stage, gamma=cgs.gamma)
