import torch
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import statistics
import logging
import pickle

import math
from math import gcd
from functools import reduce
from operator import mul

# from net.models import LeNet_5 as LeNet
# import util

import sys
sys.path.append('../')
import utils_gsp.padded_gsp as gsp_global
import utils_gsp.gpu_projection as gsp_gpu
import os

# logging.basicConfig(filename = 'logElem.log' , level=logging.DEBUG)

# Select Device
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else 'cpu')


# ==================================================================================================
def apply_global_gsp(model, sps, is_fw=False):
    matrix, val_mask, shape_l = concat_nnlayers(model, is_fw)
    try:
        type(matrix) == torch.Tensor
    except:
        print("The output of concat_nnlayers - 'matrix' is not a Torch Tensor!")
    xp_mat, ni_list = gsp_global.groupedsparseproj(matrix, val_mask, sps)
    rebuild_nnlayers(xp_mat, ni_list, shape_l, model, is_fw)


def concat_nnlayers(model, is_fw):

    shape_l = get_shape_l(model, is_fw)
    
    max_dim0 = max([x[0] for x in shape_l])
    max_dim1 = sum([x[1] for x in shape_l])
    
    matrix = torch.zeros(max_dim0, max_dim1, device=device)
    val_mask = torch.zeros(max_dim0, max_dim1, device=device)
    
    counter = 0
    dim1 = 0
    ni_list = []

    for name, param in model.named_parameters():
        if 'weight' in name:
            prev_dim0, cur_dim0 = shape_l[counter-1][0], shape_l[counter][0]
            prev_dim1, cur_dim1 = shape_l[counter-1][1], shape_l[counter][1]

            if is_fw:
                matrix[0:cur_dim0, dim1:dim1+cur_dim1] = param.data.T
            else:
                matrix[0:cur_dim0, dim1:dim1+cur_dim1] = param.data

            val_mask[0:cur_dim0, dim1:dim1+cur_dim1] = torch.ones(shape_l[counter]).to(device)
            
            dim1 += cur_dim1
            counter += 1
    return matrix, val_mask, shape_l

def get_shape_l(model, is_fw):
    """
    Get's the model layer shape tuples for LeNet 300 100 and LeNet*5 models.
    """
    params_d = {}
    weight_d = {}
    shape_list = []
    counter = 0
    for name, param in model.named_parameters(): 
        if 'weight' in name:
            shape_list.append(param.data.shape)    
            if param.dim() > 2:  
                if is_fw:
                    w_shape = param.shape
                    dim_1 = w_shape[0] * w_shape[1]
                    weight_d[counter] = param.detach().view(dim_1,-1).T
                else:
                    w_shape = param.shape
                    dim_1 = w_shape[0] * w_shape[1]
                    weight_d[counter] = param.detach().view(dim_1,-1)
            else:
                if is_fw:
                    weight_d[counter] = param.detach().T
                else:
                    weight_d[counter] = param.detach()
            counter += 1

    shape_l = [tuple(x.shape) for x in weight_d.values()]
    # print(shape_l)

    return shape_l


def rebuild_nnlayers(matrix, ni_list, shape_l, model, is_fw):
    counter = 0
    dim1 = 0
    ni_list = []

    for name, param in model.named_parameters():
        if 'weight' in name:
            prev_dim0, cur_dim0 = shape_l[counter-1][0], shape_l[counter][0]
            prev_dim1, cur_dim1 = shape_l[counter-1][1], shape_l[counter][1]
            
            if is_fw:
                param.data = matrix[0:cur_dim0, dim1:dim1+cur_dim1].T
            else:
                param.data = matrix[0:cur_dim0, dim1:dim1+cur_dim1]

            dim1 += cur_dim1
            counter += 1