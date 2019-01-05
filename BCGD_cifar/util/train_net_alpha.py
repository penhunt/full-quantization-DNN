from torch.autograd import Variable
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
import time
import pdb
import os
import torch
import sys
sys.path.append('../')
import config as cgs

import datetime
now = datetime.datetime.now()
import numpy as np

from util.quant import optimization_step

def save_model(model, name, best_acc, epoch):
    print('Saving..')
    state = {
        'net': model.state_dict(),
        'best_acc': best_acc,
        'start_epoch': epoch
    }
    if not os.path.isdir('models'):
            os.mkdir('models')
    torch.save(state, 'models/'+name)

def save_model_quant(model, name, best_acc, epoch, all_G_kernels):
    print('Saving..')
    state = {
        'net': model.state_dict(),
        'best_acc': best_acc,
        'start_epoch': epoch,
        'G_kernels': all_G_kernels
    }
    if not os.path.isdir('models'):
            os.mkdir('models')
    torch.save(state, 'models/'+name)


def check_value(kernel):
    return kernel[1].data.abs().mean()

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

## evaluate float model
def _evaluate(model, loss, val_iterator):

    loss_value = AverageMeter()
    accuracy = AverageMeter()

    n_validation_batches = len(val_iterator)

    for j, (x_batch, y_batch) in enumerate(val_iterator):
        x_batch, y_batch = x_batch.to('cuda'), y_batch.to('cuda')

        logits = model(x_batch)

        # compute logloss
        batch_loss = loss(logits, y_batch)

        # compute accuracies
        # pred = F.softmax(logits)
        pred = logits.data
        batch_accuracy = _accuracy(y_batch, pred, topk=(1,5))

        loss_value.update(batch_loss.item(), x_batch.size(0))
        accuracy.update(batch_accuracy[0], x_batch.size(0))

        if j >= n_validation_batches:
            break
    return loss_value.avg, accuracy.avg.item()

def _evaluate_quant(model, loss, val_iterator, kernels_list):
    W, G = kernels_list
    for i in range(len(W)):
        k_W = W[i]
        k_quant = G[i]    
        k_W.data, k_quant.data = k_quant.data, k_W.data

    test_loss, test_accuracy = _evaluate(model, loss, val_iterator)
        
    for i in range(len(W)):
        k_W = W[i]
        k_quant = G[i]    
        k_W.data, k_quant.data = k_quant.data, k_W.data

    return test_loss, test_accuracy


def validate(val_iterator, model, loss):
    # evaluation
    start_time = time.time()
    model.eval()
    # n_validation_batches = len(val_iterator)

    test_loss, test_accuracy = _evaluate(model, loss, val_iterator)
    
    time_epoch = time.time() - start_time
    print('{0}  {1:.3f}'.format((test_loss, test_accuracy), time_epoch))


def val_quant(val_iterator, model, loss, kernels_list):
    # evaluation
    start_time = time.time()
    model.eval()
    
    test_loss, test_accuracy = _evaluate_quant(model, loss, val_iterator, kernels_list)

    time_epoch = time.time() - start_time
    print('{0}  {1:.3f}'.format((test_loss, test_accuracy), time_epoch))


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def optimization_step_float(model, loss, x_batch, y_batch, optimizer):
    """Make forward pass and update model parameters with gradients."""
    # switch to train mode
    model.train()

    x_batch, y_batch = x_batch.to(cgs.device), y_batch.to(cgs.device)

    logits = model(x_batch)

    # compute logloss
    batch_loss = loss(logits, y_batch)

    # compute accuracies
    # pred = F.softmax(logits)
    pred = logits.data
    batch_accuracy = _accuracy(y_batch, pred, topk=(1,5))

    # compute gradients
    optimizer.zero_grad()
    batch_loss.backward()

    # update params
    optimizer.step()

    return batch_loss.item(), batch_accuracy


def train(model, loss, optimizer,
          train_iterator, val_iterator, 
          params,
          n_epochs=200,
          lr_scheduler=None):

    best_prec1, start_epoch, name = params
    
    if not os.path.isdir('logs'):
            os.mkdir('logs')
    fname = './logs/'+name[:-3]+'_'+now.strftime("%Y-%m-%d-%H-%M")
    f_train = open(fname+'_train.txt', 'w')
    f_val = open(fname+'_val.txt', 'w')
    f_train.write('epoch,loss,top1,time\n')
    f_val.write('epoch,loss,top1,best_top1\n')

    for epoch in range(start_epoch, n_epochs):
        
        model.train()  # set train mode
        start_time = time.time()
        losses = AverageMeter()
        top1 = AverageMeter()
        
        if lr_scheduler is not None:
            lr_scheduler.step()

        # main training loop
        for step, (x_batch, y_batch) in enumerate(train_iterator):
            batch_loss, batch_accuracy = optimization_step_float(model, loss, x_batch, y_batch, optimizer)
            losses.update(batch_loss, x_batch.size(0))
            top1.update(batch_accuracy[0], x_batch.size(0))
            
        loss_train, prec1_train = losses.avg, top1.avg.item()
         
        # evaluation
        model.eval()
        loss_val, prec1 = _evaluate(model, loss, val_iterator)

        time_epoch = time.time() - start_time
        print('{0}  {1:.3f}'.format((epoch, loss_train, loss_val, prec1_train, prec1), time_epoch))

        if prec1 > best_prec1:
            best_prec1 = prec1
            print('update best test accuracy as', best_prec1)
            save_model(model, name=name,
                       best_acc=best_prec1, 
                       epoch=epoch + start_epoch)

        # record results
        for tem in [epoch, loss_train, prec1_train]:
            f_train.write(str(tem)+',')
        f_train.write(str(time_epoch)+'\n')

        for tem in [epoch, loss_val, prec1]:
            f_val.write(str(tem)+',')
        f_val.write(str(best_prec1)+'\n')
        
    f_train.close()
    f_val.close()


def train_alpha(model, loss, optimizer, optimizer_a,
                train_iterator, val_iterator, 
                params, n_epochs=200,
                lr_scheduler=None, lr_scheduler_a=None,
                stage=150, gamma=0, steps=None):
    """
    Train 'model' by minimizing 'loss' using 'optimization_step_fn'
    for parameter updates.
    """

    best_prec1, start_epoch, name = params

    if not os.path.isdir('logs'):
            os.mkdir('logs')
    fname = './logs/'+name[:-3]+'_'+now.strftime("%Y-%m-%d-%H-%M")
    f_train = open(fname+'_train.txt', 'w')
    f_val = open(fname+'_val.txt', 'w')
    f_train.write('epoch,loss,top1,time\n')
    f_val.write('epoch,loss,top1,best_top1\n')

    for epoch in range(start_epoch, n_epochs):
        
        model.train()  # set train mode
        start_time = time.time()
        losses = AverageMeter()
        top1 = AverageMeter()
        
        if lr_scheduler is not None:
            lr_scheduler.step()
        
        if epoch >= stage and lr_scheduler_a is not None:
            lr_scheduler_a.step()

        if steps is not None:
            if epoch in steps and gamma > 0:
                for n, p in model.named_parameters():
                    if 'rate_factor' in n:
                        p.data[0] /= gamma

        # main training loop
        if epoch == max(stage, start_epoch):
            for n, p in model.named_parameters():
                if 'gd_alpha' in n:
                    p.fill_(1)
        if epoch >= stage:
            for step, (x_batch, y_batch) in enumerate(train_iterator):
                batch_loss, batch_accuracy = optimization_step_float(model, loss, x_batch, y_batch, optimizer_a)
                losses.update(batch_loss, x_batch.size(0))
                top1.update(batch_accuracy[0], x_batch.size(0))
        else:
            for step, (x_batch, y_batch) in enumerate(train_iterator):
                batch_loss, batch_accuracy = optimization_step_float(model, loss, x_batch, y_batch, optimizer)
                losses.update(batch_loss, x_batch.size(0))
                top1.update(batch_accuracy[0], x_batch.size(0))


        loss_train, prec1_train = losses.avg, top1.avg.item()
         
        # evaluation
        model.eval()
        loss_val, prec1 = _evaluate(model, loss, val_iterator)

        time_epoch = time.time() - start_time
        print('{0}  {1:.3f}'.format((epoch, loss_train, loss_val, prec1_train, prec1), time_epoch))

        if prec1 > best_prec1:
            best_prec1 = prec1
            print('update best test accuracy as', best_prec1)
            save_model(model, name=name,
                       best_acc=best_prec1, 
                       epoch=epoch + start_epoch)
        
        # record results
        for tem in [epoch, loss_train, prec1_train]:
            f_train.write(str(tem)+',')
        f_train.write(str(time_epoch)+'\n')

        for tem in [epoch, loss_val, prec1]:
            f_val.write(str(tem)+',')
        f_val.write(str(best_prec1)+'\n')
        
    f_train.close()
    f_val.close()


def train_alm1_alpha(model, loss, optimizer_list,
                     kernels_list,
                     train_iterator, val_iterator,
                     params, n_epochs=200,
                     lr_scheduler=None, lr_scheduler_a=None,
                     rho_rate=1.01,
                     stage=150, gamma=0, steps=None):
    """
    Train 'model' by minimizing 'loss' using 'optimization_step_fn'
    for parameter updates.
    """

    rho, lip, best_prec1, start_epoch, name = params
    optimizer, optimizer_a, optimizer_quant = optimizer_list

    if not os.path.isdir('logs'):
            os.mkdir('logs')
    fname = './logs/'+name[:-3]+'_'+now.strftime("%Y-%m-%d-%H-%M")
    f_train = open(fname+'_train.txt', 'w')
    f_val = open(fname+'_val.txt', 'w')
    f_train.write('epoch,loss,top1,time\n')
    f_val.write('epoch,loss,top1,best_top1\n')

    for epoch in range(start_epoch, n_epochs):

        # adaptive change rho and lip values
        rho = min(rho_rate*rho, .01)
        lip = 1 - rho

        model.train()  # set train mode
        start_time = time.time()
        losses = AverageMeter()
        top1 = AverageMeter()
        
        if lr_scheduler is not None:
            lr_scheduler.step()
        
        if epoch >= stage and lr_scheduler_a is not None:
            lr_scheduler_a.step()

        if steps is not None:
            if epoch in steps and gamma > 0:
                for n, p in model.named_parameters():
                    if 'rate_factor' in n:
                        p.data[0] /= gamma

        # main training loop
        if epoch == max(stage, start_epoch):
            for n, p in model.named_parameters():
                if 'gd_alpha' in n:
                    p.fill_(1)
        if epoch >= stage:
            for step, (x_batch, y_batch) in enumerate(train_iterator):
                batch_loss, batch_accuracy = optimization_step(
                    model, loss, x_batch, y_batch, [optimizer_a, optimizer_quant], [rho, lip])
                losses.update(batch_loss, x_batch.size(0))
                top1.update(batch_accuracy[0], x_batch.size(0))
        else:
            for step, (x_batch, y_batch) in enumerate(train_iterator):
                batch_loss, batch_accuracy = optimization_step(
                    model, loss, x_batch, y_batch, [optimizer, optimizer_quant], [rho, lip])
                losses.update(batch_loss, x_batch.size(0))
                top1.update(batch_accuracy[0], x_batch.size(0))

        loss_train, prec1_train = losses.avg, top1.avg.item()
        
        if rho != 0:
            print('check value of rho ', rho)

        # evaluation
        model.eval()
        loss_val, prec1 = _evaluate_quant(model, loss, val_iterator, kernels_list)

        time_epoch = time.time() - start_time
        print('{0}  {1:.3f}'.format((epoch, loss_train, loss_val, prec1_train, prec1), time_epoch))

        if prec1 > best_prec1:
            best_prec1 = prec1
            print('update best test accuracy as', best_prec1)
            save_model(model, name=name,
                       best_acc=best_prec1, 
                       epoch=epoch + start_epoch)
        
        # record results
        for tem in [epoch, loss_train, prec1_train]:
            f_train.write(str(tem)+',')
        f_train.write(str(time_epoch)+'\n')

        for tem in [epoch, loss_val, prec1]:
            f_val.write(str(tem)+',')
        f_val.write(str(best_prec1)+'\n')
        
    f_train.close()
    f_val.close()

def train_alm1_eta_alpha(model, loss, optimizer_list,
                         kernels_list,
                         train_iterator, val_iterator,
                         params, n_epochs=200,
                         lr_scheduler=None, lr_scheduler_a=None,
                         rho_rate=1.01,
                         eta=1, eta_rate=1.05, m_epochs=80, divs=2,
                         stage=150, gamma=0, steps=None):
    """
    Train 'model' by minimizing 'loss' using 'optimization_step_fn'
    for parameter updates.
    """

    rho, lip, best_prec1, start_epoch, name = params
    optimizer, optimizer_a, optimizer_quant = optimizer_list

    if not os.path.isdir('logs'):
            os.mkdir('logs')
    fname = './logs/'+name[:-3]+'_'+now.strftime("%Y-%m-%d-%H-%M")
    f_train = open(fname+'_train.txt', 'w')
    f_val = open(fname+'_val.txt', 'w')
    f_train.write('epoch,loss,top1,time\n')
    f_val.write('epoch,loss,top1,best_top1\n')
    
    eta_step = np.ceil(len(train_iterator)/divs)

    for epoch in range(start_epoch, n_epochs):

        # adaptive change rho and lip values
        rho = min(rho_rate*rho, .01)
        lip = 1 - rho

        model.train()  # set train mode
        start_time = time.time()
        losses = AverageMeter()
        top1 = AverageMeter()
        
        if lr_scheduler is not None:
            lr_scheduler.step()
        
        if epoch >= stage and lr_scheduler_a is not None:
            lr_scheduler_a.step()

        if steps is not None:
            if epoch in steps and gamma > 0:
                for n, p in model.named_parameters():
                    if 'rate_factor' in n:
                        p.data[0] /= gamma
     
        # main training loop
        if epoch == max(stage, start_epoch):
            for n, p in model.named_parameters():
                if 'gd_alpha' in n:
                    p.fill_(1)

        if epoch == m_epochs:
            best_prec1 = 0
            eta = 0
        
        if epoch >= stage:
            for step, (x_batch, y_batch) in enumerate(train_iterator):
                if epoch < m_epochs and step % eta_step == 0:
                    eta = eta_rate*eta
                batch_loss, batch_accuracy = optimization_step(
                    model, loss, x_batch, y_batch, [optimizer_a, optimizer_quant], [rho, lip], eta=eta)
                losses.update(batch_loss, x_batch.size(0))
                top1.update(batch_accuracy[0], x_batch.size(0))
        else:
            for step, (x_batch, y_batch) in enumerate(train_iterator):
                if epoch < m_epochs and step % eta_step == 0:
                    eta = eta_rate*eta
                batch_loss, batch_accuracy = optimization_step(
                    model, loss, x_batch, y_batch, [optimizer, optimizer_quant], [rho, lip], eta=eta)
                losses.update(batch_loss, x_batch.size(0))
                top1.update(batch_accuracy[0], x_batch.size(0))

        loss_train, prec1_train = losses.avg, top1.avg.item()
        
        if rho != 0:
            print('check value of rho and eta', rho, eta)
        elif eta != 0:
            print('check value of eta', eta)

        # evaluation
        model.eval()
        loss_val, prec1 = _evaluate_quant(model, loss, val_iterator, kernels_list)

        time_epoch = time.time() - start_time
        print('{0}  {1:.3f}'.format((epoch, loss_train, loss_val, prec1_train, prec1), time_epoch))

        if prec1 > best_prec1:
            best_prec1 = prec1
            print('update best test accuracy as', best_prec1)
            save_model(model, name=name,
                       best_acc=best_prec1, 
                       epoch=epoch + start_epoch)
        
        # record results
        for tem in [epoch, loss_train, prec1_train]:
            f_train.write(str(tem)+',')
        f_train.write(str(time_epoch)+'\n')

        for tem in [epoch, loss_val, prec1]:
            f_val.write(str(tem)+',')
        f_val.write(str(best_prec1)+'\n')
        
    f_train.close()
    f_val.close()
