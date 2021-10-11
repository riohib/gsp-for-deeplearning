import torch
from torch import Tensor
import torch.nn as nn
from collections import OrderedDict
# from utils import load_state_dict_from_url
from typing import Type, Any, Callable, Union, List, Optional    
import numpy as np

import sys 
sys.path.append('/data/users2/rohib/github/testing')
import utils_gsp.sps_tools as sps_tools
import utils_gsp.gpu_projection as gsp_gpu

class GSP_Model:
    def __init__(self, model, sps=0.8) -> None:
        #  Class Variables for GSP
        self.model = model
        
        self.curr_epoch = 0
        self.curr_iter = 0
        
        self.start_gsp_epoch = 0
        self.gsp_training_mode = False
        self.sps = 0.0
        self.gsp_int = 0
        self.proj_filters = True
        
        self.logger = None
        self.masks = None
        self.train_loader_len = None
        self.device = next(self.model.parameters()).device
    
    
    def mask_out_parameters(self):
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                module.weight.data = module.weight.data * self.masks[module]
    
    def apply_gsp(self, sps=None):
        sps = self.sps if (sps == None) else sps # To use this method with sparsity as argument. Default will use self.sps.

        if self.gsp_training_mode and (self.curr_epoch > self.start_gsp_epoch) and (self.curr_iter % self.gsp_int == 0):
            print(f"Applying GSP!! GSP_Mode: {self.gsp_training_mode}")
            self._apply_gsp_to_layers(sps)
            # print(f"The total MODEL SPS: {self.get_model_sps():.2f}")
        
        if self.curr_iter == (self.train_loader_len-1):
            self.curr_iter = 0
            self.curr_epoch +=1 
        else:
            self.curr_iter += 1
    

    def force_apply_gsp(self, sps=None):
        sps = self.sps if (sps == None) else sps # To use this method with sparsity as argument. Default will use self.sps.
        print("Forced GSP Application!!")
        self._apply_gsp_to_layers(sps)


    def _apply_gsp_to_layers(self, sps):
        for name, layer in self.model.named_modules():
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                w_shape = layer.weight.shape
                gsp_in = layer.weight.data.detach().reshape(layer.weight.shape[0], -1)

                if self.proj_filters:
                    layer.weight.data = gsp_gpu.groupedsparseproj(gsp_in.T, sps = sps).T.reshape(w_shape)
                else:
                    layer.weight.data = gsp_gpu.groupedsparseproj(gsp_in, sps = sps).reshape(w_shape)
        

        if self.logger != None:
            self.logger.info(f"Applied GSP to all Layers! Epoch: {self.curr_epoch} | Iter: {self.curr_iter} | sps: {sps}" \
                             f"Filter-Proj: {self.proj_filters}")



    def get_model_sps(self):
        nonzero = total = 0
        for name, param in self.model.named_parameters():
            if 'mask' not in name:
                tensor = param.detach().clone()
                # nz_count.append(torch.count_nonzero(tensor))
                nz_count = torch.count_nonzero(tensor).item()
                total_params = tensor.numel()
                nonzero += nz_count
                total += total_params
        
        tensor = None
        # print(f"TOTAL: {total}")
        abs_sps = 100 * (total-nonzero) / total
        return abs_sps
    
    def print_model_sps(self):
        print(f"Model Sparsity: {self.get_model_sps():.2f} %")

    
    def get_layer_sps(self):
        for name, layer in self.model.named_modules():
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                layer_sps = sps_tools.sparsity(layer.weight.data.detach())
                print(f"Sparsity of layer: {name}: {layer_sps}")

   
    
    # =================================== Pruning Methods =======================================
    def prune_and_mask_model(self, sps):
        self._prune_model_with_sps(sps)
        masks_d, _ = self._mask_zeros()
        self._register_pre_hook_mask(masks_d)


    def _prune_model_with_sps(self, sps):
        # Get the threshold for top-k values
        weight_tensor = torch.empty(0, device=self.device)
        for name, param in self.model.named_parameters(): 
            weight_tensor = torch.cat((weight_tensor, param.data.flatten()))

        num_survive_w = weight_tensor.numel() * (1-sps)
        topk_val = int(np.floor(num_survive_w))
        vals, ind = torch.topk(weight_tensor.abs(), topk_val, sorted=True)
        threshold = vals.min()

        print(f'Pruning with threshold : {threshold} for layer {name}')

        if self.logger != None:
            self.logger.info(f'Pruning with threshold : {threshold} for layer {name}')
            
        for name, p in self.model.named_parameters():
            tensor = p.data
            # print(f'Pruning with threshold : {threshold} for layer {name}')
            sparse_w = torch.where(abs(tensor) < threshold, torch.tensor(0.0, device=self.device), tensor)
            p.data = sparse_w


    def _mask_zeros(self, threshold=1e-8):
        masks_d = dict()
        mask_l = list()
        for name, layer in self.model.named_modules():
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                tensor = layer.weight.data
                masked_tensor = torch.where(abs(tensor) < threshold, torch.tensor(0.0, device=self.device), tensor)
                mask = torch.where(abs(tensor) < threshold, torch.tensor(0.0, device=self.device), torch.tensor(1.0, device=self.device))
                masks_d[layer] = mask
                mask_l.append(mask)
                layer.weight.data = masked_tensor
        return masks_d, mask_l



    # =================================== Finetuning Methods =======================================
    def _register_pre_hook_mask(self, masks=None):
        # self.masks = None
        self.unregister_mask()
        if masks is not None:
            self.masks = masks
        assert self.masks is not None, 'Masks should be generated first.'

        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                module.mask = nn.Parameter(masks[module]).requires_grad_(False).to(module.weight.get_device())
                module.register_forward_pre_hook(self._forward_pre_hook)


    def unregister_mask(self):
        for module in self.model.modules():
            module._backward_hooks = OrderedDict()
            module._forward_pre_hooks = OrderedDict()

    @staticmethod
    def _forward_pre_hook(module, x):
        module.mask.requires_grad_(False)
        mask = module.mask
        module.weight.data.mul_(mask.to(module.weight.get_device()))