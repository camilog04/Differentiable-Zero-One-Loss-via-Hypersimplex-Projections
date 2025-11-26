import lightgbm as lgb
import time
import os
import pickle

import torch.nn as nn
import torch


import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import torch.nn as nn
import torch

import logging
logging.basicConfig(
    filename="KDD12_hypersimplex.log", 
    level=logging.INFO
)
torch.set_default_tensor_type(torch.DoubleTensor)

from src.soft_binary_arg_max_ops import soft_binary_argmax

import numpy as np


params = {
    "metric": "auc",
    "num_leaves": 2048, #1024
    "max_depth": -1,
    "num_iterations": 3000,
    "learning_rate": 0.01,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "min_data_in_leaf": 20,
    "min_sum_hessian_in_leaf": 1e-3,
    "lambda_l2": 0.1,
    "num_thread": -1,
    "max_bin": 255,
} 

# Note that we have convert the original raw data into a pure libsvm format.
infile_train = "../data/KDD12/train.libsvm"
infile_valid = "../data/KDD12/valid.libsvm"


# Load data
train_data = lgb.Dataset(infile_train)
valid_data = lgb.Dataset(infile_valid)

def as_torch(array):
    if array.ndim == 1:
        return torch.tensor([array], dtype=torch.float64, requires_grad=True)
    else: 
        return torch.tensor(array, dtype=torch.float64, requires_grad=True)    
    
def hypersimplex_basis(n,p):
    # Ensure p is between 0 and 1
    assert 0 <= p <= 1, "p must be between 0 and 1"
    # Calculate the number of 1's
    num_ones = int(n * p)
    # Create a tensor with n zeros
    tensor = torch.zeros(n)
    # Set the first num_ones elements to 1
    tensor[:num_ones] = 1
    return tensor

def proportion_of_ones(tensor):
    return torch.sum(tensor == 1).item() / torch.numel(tensor)

def projected_hypersimplex_loss(pred, target):
    def hypersimplex_loss(target, pred, regularization='l2', regularization_strength=1.0):

        n = len(target)
        p = proportion_of_ones(target)

        basis = hypersimplex_basis(n, p)

        pred = soft_binary_argmax.apply(pred, 
                                        basis,
                                        "l2",
                                        0.001)
        #print(pred)

        MSE = nn.MSELoss()
        output = MSE(pred.view(n), target.view(n))
        #output.backward()
        return output
    
    target = target.get_label()
    target = as_torch(target).view(-1, 1)
    pred = as_torch(pred)
    # 3) compute loss
    loss = hypersimplex_loss(target, pred, regularization_strength=1e-2)
    #print(loss)
    
    #print(loss)
    # 4) compute grad
    # calculate gradient and convert to numpy
    loss_grads = torch.autograd.grad(loss, pred)[0] * 0.001 # rescales gradiant by the temperture
    

    loss_grads = loss_grads.to('cpu').detach().numpy().reshape(-1)
    
    # return gradient and ones instead of Hessian diagonal

    return loss_grads, np.ones(len(target))

params["objective"] = projected_hypersimplex_loss

# Training
res = {}
start = time.time()
logging.info("Hypersimplex traning starting (AUC, KDD12)...")
bst = lgb.train(
    params, train_data, 
    valid_sets=[valid_data], valid_names=["valid"],
    num_boost_round=50,
    callbacks=[lgb.record_evaluation(res), lgb.log_evaluation(period=10), lgb.early_stopping(stopping_rounds=10)])
end = time.time()