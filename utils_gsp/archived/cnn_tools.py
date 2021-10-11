import torch
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import statistics
import logging
import pickle

from math import gcd
from functools import reduce
from operator import mul

import sys
sys.path.append('../')
import utils_gsp.padded_gsp as gsp_global
import utils_gsp.gpu_projection as gsp_gpu
import os

# logging.basicConfig(filename = 'logElem.log' , level=logging.DEBUG)

# Select Device
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else 'cpu')


## ================================================================================================= ##
# --------------------------------------------- Tools ---------------------------------------------- #

def apply_concat_gsp(model, sps):
    matrix, val_mask, shape_l = concat_nnlayers(model)
    
    try:
        xp_mat, ni_list = gsp_global.groupedsparseproj(matrix, val_mask, sps)
    except:
        output = gsp_global.groupedsparseproj(matrix, val_mask, sps)
        print(output)
        print( len(output) )
        print(type(output))

    rebuild_nnlayers(xp_mat, ni_list, shape_l, model)

    # return xp_mat, ni_list


def get_shape_l(model):
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
                w_shape = param.shape
                dim_1 = w_shape[0] * w_shape[1]
                weight_d[counter] = param.detach().view(dim_1,-1)
            else:
                weight_d[counter] = param.detach()
            counter += 1
    shape_l = [tuple(x.shape) for x in weight_d.values()]
    return weight_d, shape_l


def concat_nnlayers(model):
    weight_d, shape_l = get_shape_l(model)
    
    max_dim0 = max([x[0] for x in shape_l])
    max_dim1 = sum([x[1] for x in shape_l])
    
    matrix = torch.zeros(max_dim0, max_dim1, device=device)
    val_mask = torch.zeros(max_dim0, max_dim1, device=device)
    
    counter = 0
    dim1 = 0
    ni_list = []

    for val in weight_d.values():        
        prev_dim0, cur_dim0 = shape_l[counter-1][0], shape_l[counter][0]
        prev_dim1, cur_dim1 = shape_l[counter-1][1], shape_l[counter][1]
        
        matrix[0:cur_dim0, dim1:dim1+cur_dim1] = val
        val_mask[0:cur_dim0, dim1:dim1+cur_dim1] = torch.ones(shape_l[counter]).to(device)
        
        dim1 += cur_dim1
        counter += 1

    return matrix, val_mask, shape_l


def rebuild_nnlayers(xp_mat, ni_list, shape_l, model):
    sps_weight_d = {}
    counter = 0
    dim1 = 0
    ni_list = []

    for counter, shp in enumerate(shape_l):
        cur_dim0 = shape_l[counter][0]
        cur_dim1 = shape_l[counter][1]
        sps_weight_d[counter] = xp_mat[0:cur_dim0, dim1:dim1+cur_dim1]
        dim1 += cur_dim1
        counter += 1
    
    counter = 0
    for name, param in model.named_parameters(): 
        if 'weight' in name:
            w_shape = param.shape
            param.data = sps_weight_d[counter].view(w_shape)
            counter += 1

    # return model