import torch 
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append('/data/users2/rohib/github/testing')
import utils_gsp.sps_tools as sps_tools
import utils_gsp.gpu_projection as gsp_gpu
import utils_gsp.padded_gsp as gsp_pad
import math

device = 'cuda' if torch.cuda.is_available() else 'cpu'


# =======================================================================================
# ================================== PRUNING CODE =======================================

def prune_gates(model, prune_sps=0.8):
    gate_cat_torch = get_cat_gates(model)
    mask_gate_cat = keep_topk( gate_cat_torch, prune_sps=prune_sps)
    masked_gate_d = get_masked_gate_list(model, mask_gate_cat)
    build_sparse_gates(model, masked_gate_d)


def get_cat_gates(model):
    gate_cat_torch = torch.tensor([]).to(next(model.parameters()).device)

    for name, layer in model.named_modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            data_torch = torch.abs(layer.gsp_gate.detach().clone().flatten())

            # Normalize the Layer-Gates
            dmin = data_torch.min(); dmax= data_torch.max()
            data_torch -= dmin
            data_torch /= (dmax - dmin)

            gate_cat_torch = torch.cat( (gate_cat_torch, data_torch) )

    return gate_cat_torch


def keep_topk( gate_cat_torch, prune_sps=0.8):
    mask_gate_cat = torch.zeros_like(gate_cat_torch, device=gate_cat_torch.device)
    k = math.floor(gate_cat_torch.shape[0] * (1 - prune_sps))
    # print(k)
    vals, ind = torch.topk(gate_cat_torch, k=k)
    mask_gate_cat[ind] = torch.ones_like(vals, device=vals.device)
    # mask_gate_cat[ind] = vals
    return mask_gate_cat


def get_masked_gate_list(model, mask_gate_cat):
    names = list()
    shapes_l = list()
    for name, layer in model.named_modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            shapes_l.append(layer.gsp_gate.shape[0])
            names.append(name)
            
    masked_gate_d = dict()
    start = 0
    for i, shape in enumerate(shapes_l):
        masked_gate_d[names[i]] = mask_gate_cat[ start:(start+shape)]
        start += shape
        # print(start)
    return masked_gate_d



def build_sparse_gates(model, masked_gate_d):
    for name, layer in model.named_modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            mask_gates = masked_gate_d[name].reshape(layer.gsp_gate.shape)
            
            if isinstance(layer, nn.Conv2d):
                assert (layer.gsp_gate.shape == mask_gates.shape), "Shape Mismatch in masked gate re-insertion"
                layer.gsp_gate.data = masked_gate_d[name].reshape(layer.gsp_gate.shape)
                layer.gsp_gate.requires_grad = False
            if isinstance(layer, nn.Linear):
                assert (layer.gsp_gate.shape == mask_gates.shape), "Shape Mismatch in masked gate re-insertion"
                layer.gsp_gate.data = torch.ones_like(layer.gsp_gate, device=layer.gsp_gate.device)
                layer.gsp_gate.requires_grad = False
