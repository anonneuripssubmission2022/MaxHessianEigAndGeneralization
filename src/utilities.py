'''
Modified from: https://github.com/locuslab/edge-of-stability
'''

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch import Tensor
from torch.nn.utils import parameters_to_vector, vector_to_parameters
import numpy as np
from typing import List
from scipy.sparse.linalg import LinearOperator, eigsh, svds
from math import sqrt, isclose
from time import time
import os
from torch.optim import SGD, Adam
from os import makedirs
import torch.nn.functional as F
from torch import log
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import sys
from sam import SAM

DEFAULT_BATCH_SIZE = 100
DEFAULT_PHYS_BATCH_SIZE = 1000

def get_path(dataset, lr, batch_size, arch_id, opt, loss, rho=0.05):
    """ Return directory where results should be saved """
    if opt=='sam':
        opt=f'{opt}_{rho}'
    directory = f"{dataset}/arch_{arch_id}/{loss}/{opt}/no_change_lr/lr_{lr}_bs_{batch_size}"
    return directory

def save_every(epoch: int):
    """ Save specified checkpoints """
    return(epoch <= 100 and epoch % 10 == 0) or \
    (epoch > 100 and epoch <= 1000 and epoch % 100 == 0) or \
    (epoch > 1000 and epoch % 1000 == 0)

def save_final(directory,network,train_loss,test_loss,train_acc,test_acc):
    """ Save final checkpoint, and test and train losses + accuracies """
    torch.save(network.state_dict(), f"{directory}/snapshot_final")          
    torch.save(train_loss, f"{directory}/train_loss")
    torch.save(test_loss, f"{directory}/test_loss")
    torch.save(train_acc, f"{directory}/train_acc")
    torch.save(test_acc, f"{directory}/test_acc")
        
def save_most_recent(directory,network,train_loss,test_loss,train_acc,test_acc):
    """ Save most recent checkpoint, and test and train losses + accuracies """
    torch.save(network.state_dict(), f"{directory}/snapshot_most_recent")          
    torch.save(train_loss, f"{directory}/train_loss_most_recent")
    torch.save(test_loss, f"{directory}/test_loss_most_recent")
    torch.save(train_acc, f"{directory}/train_acc_most_recent")
    torch.save(test_acc, f"{directory}/test_acc_most_recent") 

class AccuracyCE(nn.Module):
    def __init__(self):
        super(AccuracyCE, self).__init__()

    def forward(self, input, target):
        return (input.argmax(1) == target).float().sum()
    
def get_loss_and_acc(loss: str):
    """ Return the loss and accuracy functions (currently, only cross-entropy loss is supported)"""
    loss_fns = {
            'ce': [nn.CrossEntropyLoss(reduction='sum'), AccuracyCE()]
    }
    loss_fn = loss_fns.get(loss)
    if not loss_fn:
        raise NotImplementedError(f"no such loss function: {loss}")
    return loss_fn 

def get_optimizer(opt: str, parameters, lr: float, rho=0.05):
    """ Return the optimizer (currently, only SGD and SAM are supported)"""
    if opt == "sgd":
        return SGD(parameters, lr=lr)
    elif opt == 'sam':
        base_optimizer = SGD
        return SAM(parameters, base_optimizer, rho=rho, lr=lr, momentum=0)
    raise NotImplementedError(f"no such optimizer: {opt}")

def compute_losses(network: nn.Module, loss_functions: List[nn.Module],
                   dataset: Dataset, batch_size: int = DEFAULT_BATCH_SIZE):
    """ Compute loss and accuracy over a dataset """
    L = len(loss_functions)
    losses = [0. for l in range(L)]
    with torch.no_grad():
        for (X, y) in iterate_dataset(dataset, batch_size):
            preds = network(X)
            for l, loss_fn in enumerate(loss_functions):
                losses[l] += loss_fn(preds, y) / len(dataset)
    return losses
        
def iterate_dataset(dataset: Dataset, batch_size: int):
    """ Iterate through a dataset, yielding batches of data """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    for i, (batch_X, batch_y) in enumerate(loader):
        yield batch_X.cuda(), batch_y.cuda()

def iter_batch(X: Tensor, y: Tensor, physical_batch_size: int):
    """ Iterate over a single batch"""
    assert len(X) % physical_batch_size == 0
    for b in range(len(X) // physical_batch_size):
        start_idx = b * physical_batch_size
        end_idx = (b + 1) * physical_batch_size
        yield X[start_idx:end_idx], y[start_idx:end_idx]

def train_gd(directory, network, loss_fn, acc_fn, train_dataset, test_dataset, optimizer, batch_size, verbose=True, print_freq=1, physical_batch_size=DEFAULT_PHYS_BATCH_SIZE, should_save = save_every, sam=False, abs_tol=0.01):
    """ Train network """
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    train_loss, train_acc = [], []
    test_loss, test_acc = [], []
    
    start_time = time()
    perfect_train_acc = False
    
    epoch = 0 
    while not perfect_train_acc:
        network.eval()
        train_loss_epoch, train_acc_epoch = compute_losses(network, [loss_fn, acc_fn], train_dataset, DEFAULT_BATCH_SIZE)
        test_loss_epoch, test_acc_epoch = compute_losses(network, [loss_fn, acc_fn], test_dataset, DEFAULT_BATCH_SIZE)

        train_loss.append(train_loss_epoch)
        test_loss.append(test_loss_epoch)
        train_acc.append(train_acc_epoch)
        test_acc.append(test_acc_epoch)
        
        save_most_recent(directory,network,train_loss,test_loss,train_acc,test_acc)
        
        if should_save(epoch):
            torch.save(network.state_dict(), f"{directory}/snapshot_{epoch}")        
            
        if epoch > 0 and isclose(train_acc_epoch, 1.0, abs_tol=abs_tol):
            perfect_train_acc = True
            save_final(directory,network,train_loss,test_loss,train_acc,test_acc)
            
        if verbose and epoch % print_freq == 0:
            elapsed_time = time() - start_time
            print(f"{epoch}\t{elapsed_time:.1f}\t{train_loss[-1]:.3f}\t{train_acc[-1]:.3f}"
                  f"\t{test_loss[-1]:.3f}\t{test_acc[-1]:.3f}", flush=True)
               
        network.train()
        for i, (X, y) in enumerate(train_dataloader):
            optimizer.zero_grad()
            for (X_batch, y_batch) in iter_batch(X, y, min(physical_batch_size, batch_size)):
                loss = loss_fn(network(X_batch.cuda()), y_batch.cuda()) / batch_size
                loss.backward()
                    
            if sam:
                optimizer.first_step(zero_grad=True)
                for (X_batch, y_batch) in iter_batch(X, y, min(physical_batch_size, batch_size)):
                    loss = loss_fn(network(X_batch.cuda()), y_batch.cuda()) / batch_size
                    loss.backward()
                optimizer.second_step() 
            else:
                optimizer.step()
            
        epoch += 1
        
def train_gd_extra_iter(directory, network, loss_fn, acc_fn, train_dataset, test_dataset, optimizer, batch_size, verbose=True, print_freq=1, physical_batch_size=DEFAULT_PHYS_BATCH_SIZE, should_save = save_every, sam=False):
    """ Train network for a single extra iteration """
    if os.path.isfile(f"{directory}/snapshot_extra_iter"):
        return
    
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)  
    
    network.train()
    for i, (X, y) in enumerate(train_dataloader):
        optimizer.zero_grad()
        for (X_batch, y_batch) in iter_batch(X, y, min(physical_batch_size, batch_size)):
            loss = loss_fn(network(X_batch.cuda()), y_batch.cuda()) / batch_size
            loss.backward()

        optimizer.step()
        torch.save(network.state_dict(), f"{directory}/snapshot_extra_iter")     
        break
        
def trainable_parameters(network: nn.Module):
    for param in network.parameters():
        if param.requires_grad:
            yield param

def nparams(network: nn.Module):
    return len(parameters_to_vector(trainable_parameters(network)))

def compute_hvp(network: nn.Module, loss_fn: nn.Module,
                dataset: Dataset, vector: Tensor, batch_size: int = DEFAULT_BATCH_SIZE):
    """ Compute a Hessian-vector product """
    p = len(parameters_to_vector(network.parameters()))
    n = len(dataset)
    hvp = torch.zeros(p)
    vector = vector.cuda()
    for (X, y) in iterate_dataset(dataset, batch_size):
        loss = loss_fn(network(X), y) / n
        
        grads = torch.autograd.grad(loss, inputs=network.parameters(), create_graph=True) 
        dot = torch.nn.utils.parameters_to_vector(grads).mul(vector).sum()
       
        grads = [g.contiguous() for g in torch.autograd.grad(dot, network.parameters(), retain_graph=True)]
        hvp += torch.nn.utils.parameters_to_vector(grads).cpu().detach()
        
        torch.cuda.empty_cache()
       
    return hvp

def lanczos(matrix_vector, p: int, k: int):
    """ Use Lanczos algorithm to compute the leading eigenvalues and eigenvectors of a matrix / linear operator """
    def mv(vec: np.ndarray):
        gpu_vec = torch.tensor(vec, dtype=torch.float).cuda()
        return matrix_vector(gpu_vec).cpu().numpy()

    operator = LinearOperator((p, p), matvec=mv, dtype=np.float)
    evals, evecs = eigsh(operator, k)

    return torch.from_numpy(np.ascontiguousarray(evals[::-1])).float(), \
           torch.from_numpy(np.ascontiguousarray(np.flip(evecs, -1))).float()

def get_hessian_eigenvalues(network: nn.Module, loss_fn, dataset, ncomponents=6, gpu_batch_size=DEFAULT_PHYS_BATCH_SIZE):
    """ Compute the leading Hessian eigenvalues """
    hvp_delta = lambda delta: compute_hvp(network, loss_fn, dataset, delta, batch_size=gpu_batch_size).detach().cpu() 
    Lam, _ = lanczos(hvp_delta, nparams(network), k=ncomponents)
    return Lam

def get_hessian_leading_eigenvalue(network: nn.Module, loss_fn, dataset, gpu_batch_size=DEFAULT_PHYS_BATCH_SIZE):
    """ Compute the leading Hessian eigenvalue """
    return get_hessian_eigenvalues(network, loss_fn, dataset, gpu_batch_size=gpu_batch_size)[0]

def get_hessian_leading_eigenvalue_at_iterate(network: nn.Module, loss_fn, batch_size, dataset, disable_dropout=False, gpu_batch_size=DEFAULT_PHYS_BATCH_SIZE):
    """ Compute the leading Hessian eigenvalue at iterate """
    network.train()
    if disable_dropout:
        network.eval()
    
    with torch.backends.cudnn.flags(enabled=False):
        leading_eig = get_hessian_leading_eigenvalue(network, loss_fn, dataset, gpu_batch_size=min(batch_size,gpu_batch_size))
    return leading_eig

def interpolate_state_dicts(state_dict_1, state_dict_2, weight):
    """ Interpolate between model state dicts """
    return {key: (1 - weight) * state_dict_1[key] + weight * state_dict_2[key] for key in state_dict_1.keys()}

def get_hessian_leading_eigenvalue_between_iterates(network: nn.Module, directory, loss_fn, batch_size, dataset, num_interpolations = 8, disable_dropout=False, gpu_batch_size=DEFAULT_PHYS_BATCH_SIZE):
    """ Compute the leading Hessian eigenvalue between iterates"""
    state_dict_1 = torch.load(f"{directory}/snapshot_final")
    state_dict_2 = torch.load(f"{directory}/snapshot_extra_iter")
    
    weights = list([weight/(num_interpolations-1) for weight in range(num_interpolations)])
    leading_eigs_at_interpolations = [None for _ in range(num_interpolations)]
    
    for i, weight in enumerate(weights):
        state_dict = interpolate_state_dicts(state_dict_1, state_dict_2, weight)
        network.load_state_dict(state_dict)
        
        leading_eigs_at_interpolations[i] = get_hessian_leading_eigenvalue_at_iterate(network, loss_fn, batch_size, dataset, disable_dropout=disable_dropout, gpu_batch_size=gpu_batch_size)
    
    return max(leading_eigs_at_interpolations)
        

