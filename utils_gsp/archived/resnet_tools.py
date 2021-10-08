
# import os
import math

import torch
# import torch.nn as nn
# import torch.nn.parallel
# import torch.backends.cudnn as cudnn
# import torch.optim as optim
# import torch.utils.data as data
# import torchvision.transforms as transforms
# import torchvision.datasets as datasets
# import models.cifar as models
# import numpy as np
import torch.nn as nn
import utils_gsp.padded_gsp as gsp_global

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else 'cpu')

## ===================================================================================================== ##
## ------------------------ Set of Functions for Structural Pruning on Resnets: ------------------------
## ===================================================================================================== ##
def get_abs_sps(model):
    nonzero = total = 0
    for name, param in model.named_parameters():
        tensor = param.data
        nz_count = torch.count_nonzero(tensor)
        total_params = tensor.numel()
        nonzero += nz_count
        total += total_params
    abs_sps = 100 * (total-nonzero) / total
    return abs_sps, total, (total-nonzero)


def get_concat_filters(model):
    # Get the total Parameters Count of the model
    sps, tot_param, non_zeros = get_abs_sps(model)
    
    params_d = {}
    weight_d = {}
    shape_d = {}
    dim_list = []

    cat_weights = torch.empty(0, device=device)
    non_conv_elm = torch.empty(0, device=device)

    for name, param in model.named_parameters(): 
        params_d[name] = param
        if 'weight' in name and 'bn' not in name and 'downsample' not in name and 'fc' not in name:
            shape_d[name] = param.data.shape
            weight_d[name] = param
            w_shape = param.shape
            dim_1 = w_shape[0] * w_shape[1]
            dim_list.append( (w_shape[0], w_shape[1]) )
            filters = param.detach().view(dim_1,-1).T
            cat_weights = torch.cat((cat_weights, filters), dim=1)
        else:
            non_cv_w = param.detach().flatten()
            non_conv_elm = torch.cat((non_conv_elm, non_cv_w))


    norm_list = torch.norm(cat_weights, 1, dim=0)
    srt_l2_vals, indices = torch.sort(norm_list)

    max_norm = torch.max(norm_list)
    min_norm = torch.min(norm_list)
    
    non_cv_num_zero = non_conv_elm[non_conv_elm == 0].numel()
    
    concat_info_d = {
        'cat_weights': cat_weights,
        'srt_l2_vals': srt_l2_vals,
        'indices': indices,
        'max_norm': max_norm,
        'min_norm': min_norm,
        'tot_param': tot_param,
        'dim_list': dim_list,
        'non_cv_num_zero': non_cv_num_zero
    }
    return concat_info_d



def model_sps_at(threshold, cat_w, srt_l2_vals, concat_info_d):

    # Unpack the info about the norm of the sorted filters
    indices, tot_param = concat_info_d['indices'], concat_info_d['tot_param'] 
    max_norm, min_norm = concat_info_d['max_norm'], concat_info_d['min_norm']
    non_cv_0 = concat_info_d['non_cv_num_zero']
    
    # Prune with the threshold in the sorted concatenated tensor of filters norms
    srt_pruned = torch.where(srt_l2_vals < threshold, torch.tensor(0., device=device), srt_l2_vals)
    
    # Get the indices of the pruned filters
    fltr_ind_to_prune = indices[(srt_pruned == 0)]

    # Prune those filters in the concat filter array
    cat_w[:, fltr_ind_to_prune] = torch.zeros(cat_w.shape[0], 1, device=device)

    # Calculate the estimate of absolute sparsity after pruning
    sps_at_thresh = (cat_w[cat_w==0].numel() + non_cv_0) / tot_param
    
    return sps_at_thresh


def get_pruned_catweights(threshold, cat_w, srt_l2_vals, concat_info_d):
    """
    This function effectively same as model_sps_at(). Only this one returns the copy of the 
    pruned filter. The model_sps_at() function will mutate the list, hence we need to provide
    a copy of the concatenated weight tensor.
    """
    # Unpack the info about the norm of the sorted filters
    indices, tot_param = concat_info_d['indices'],  concat_info_d['tot_param'] 
    max_norm, min_norm = concat_info_d['max_norm'], concat_info_d['min_norm']
    
    # Prune with the threshold in the sorted concatenated tensor of filters norms
    srt_pruned = torch.where(srt_l2_vals < threshold, torch.tensor(0., device=device), srt_l2_vals)

    # Get the indices of the pruned filters
    fltr_ind_to_prune = indices[(srt_pruned == 0)]

    # Prune those filters in the concat filter array
    cat_w[:, fltr_ind_to_prune] = torch.zeros(cat_w.shape[0], 1, device=device)

    # Calculate the estimate of absolute sparsity after pruning
    sps_at_thresh = cat_w[cat_w==0].numel() / tot_param

    return cat_w


def get_threshold_bisection(concat_info_d, target_sps=0.90,  epsilon = 0.0009):
    
    cat_w = concat_info_d['cat_weights']
    srt_l2_vals = concat_info_d['srt_l2_vals']
    
    high = concat_info_d['max_norm'].item()
    low = concat_info_d['min_norm'].item()
    
    # Calculate the threshold estimate
    threshold = (high + low)/2.0
    print(f"Starting threshold: {threshold}")
    
    while abs( model_sps_at(threshold, cat_w.clone(), srt_l2_vals.clone(), concat_info_d) - target_sps ) >= epsilon and abs(high - low) > 1e-6:
        print( f" Low: {low:.4f} | high = {high:0.4f} | threshold: {threshold:.4f}")

        sps_at_thresh = model_sps_at(threshold, cat_w.clone(), srt_l2_vals.clone(), concat_info_d)

        if sps_at_thresh < target_sps:
            low = threshold
        else:
            high = threshold

        threshold = (high + low)/2.0
    return threshold


def put_back_cat(model, cat_weights, dim_list):
    dim1_start = 0
    ctr = 0
    for name, param in model.named_parameters(): 
        if 'weight' in name and 'bn' not in name and 'downsample' not in name and 'fc' not in name:

            dim1_int = math.prod(dim_list[ctr])
            dim1_end = dim1_start + dim1_int
            ker_shape = param.shape[2]
            # print(f"dim1_start is:  {dim1_start} | dim1_end is:  {dim1_end}  | dim1_int is: {dim1_int}")

            # slice of the current layer
            cur_layer = cat_weights[ :, dim1_start:dim1_end]

            # Get the original Shape of the Layer Filters
            l_shape = (dim_list[ctr][0], dim_list[ctr][1], ker_shape, ker_shape)

            # Reshape to original shape and put back filter
            cur_layer = cur_layer.reshape(l_shape)
            param.data = cur_layer

            dim1_start = dim1_end
            ctr += 1

def prune_resnet_sps(model, target_sps):
    concat_info_d  = get_concat_filters(model)
    threshold = get_threshold_bisection(concat_info_d, target_sps=target_sps, epsilon =  0.00001)

    cat_w_prn = get_pruned_catweights(threshold, 
                                    concat_info_d['cat_weights'].clone(), 
                                    concat_info_d['srt_l2_vals'].clone(), 
                                    concat_info_d)

    put_back_cat(model, cat_w_prn, concat_info_d['dim_list'])
    print( f" The total model sparsity is: {get_abs_sps(model)}")

## ===================================================================================================== ##
## ---------------------------- GLOBAL GSP WITH CONV LAYERS CONCATENATED: ------------------------------ ##
## ===================================================================================================== ##


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
    return xp_mat, ni_list


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


def get_shape_l(model):
    """
    Get's the model layer shape tuples for LeNet 300 100 and LeNet*5 models.
    """
    params_d = {}
    weight_d = {}
    shape_list = []
    counter = 0
    for name, param in model.named_parameters(): 
        if 'weight' in name and 'layer' in name and 'bn' not in name and 'downsample' not in name and 'fc' not in name:
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
        if 'weight' in name and 'layer' in name and 'bn' not in name and 'downsample' not in name and 'fc' not in name:
            w_shape = param.shape
            param.data = sps_weight_d[counter].view(w_shape)
            counter += 1


## ============================ RESNET LOGGING =====================================
# def prune_with_sps(model, sparsity, device=device):
#     weight_d = {}
#     shape_list = []

#     weight_tensor = torch.empty(0, device=device)
#     for name, param in model.named_parameters(): 
#         weight_tensor = torch.cat((weight_tensor, param.data.flatten()))

#     wpct_val =  len(weight_tensor) * sparsity
#     sorted_weights, indices = torch.sort(weight_tensor.abs())
#     threshold = sorted_weights[:math.ceil(wpct_val)+1][-1]

#     for name, p in model.named_parameters():
#         tensor = p.data
#         # print(f'Pruning with threshold : {threshold} for layer {name}')
#         new_mask = torch.where(abs(tensor) < threshold, torch.tensor(0.0, device=device), tensor)
#         p.data = new_mask

def prune_with_sps(model, sparsity):
    weight_d = {}
    shape_list = []
    device = next(model.parameters()).device

    weight_tensor = torch.empty(0, device=device)
    for name, param in model.named_parameters(): 
        weight_tensor = torch.cat((weight_tensor, param.data.detach().flatten()))

    wpct_val =  len(weight_tensor) * sparsity
    sorted_weights, indices = torch.sort(weight_tensor.abs())
    threshold = sorted_weights[:math.ceil(wpct_val)+1][-1]

    for name, p in model.named_parameters():
        tensor = p.data
        # print(f'Pruning with threshold : {threshold} for layer {name}')
        sparse_w = torch.where(abs(tensor) < threshold, torch.tensor(0.0, device=device), tensor)
        p.data = sparse_w

def get_mask(model, threshold=1e-8):
    masks = {}
    for name, p in model.named_parameters():
        if ('weight' in name and 'conv' in name and 'layer' in name):
            tensor = p.data
            masked_tensor = torch.where(abs(tensor) < threshold, torch.tensor(0.0, device=device), tensor)
            mask = torch.where(abs(tensor) < threshold, torch.tensor(0.0, device=device), torch.tensor(1.0, device=device))
            masks[name] = mask
            p.data = masked_tensor
    return masks


# def get_conv_linear_mask(model, threshold=1e-8, device=device):
#     masks = {}
#     for name, layer in model.named_modules():
#         if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
#             tensor = layer.weight.data
#             masked_tensor = torch.where(abs(tensor) < threshold, torch.tensor(0.0, device=device), tensor)
#             mask = torch.where(abs(tensor) < threshold, torch.tensor(0.0, device=device), torch.tensor(1.0, device=device))
#             masks[name] = mask
#             layer.weight.data = masked_tensor
#     return masks

def get_conv_linear_mask(model, threshold=1e-8):
    masks = dict()
    device = model.get_device
    for name, layer in model.named_modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            tensor = layer.weight.data
            # Force Values smaller than threshold to 0
            masked_tensor = torch.where(abs(tensor) < threshold, torch.tensor(0.0, device=device), tensor) 
            
            mask = torch.where(abs(tensor) < threshold, torch.tensor(0.0, device=device), torch.tensor(1.0, device=device))
            masks[layer] = mask
            layer.weight.data = masked_tensor
    return masks
