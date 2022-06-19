import torch
from torch import Tensor
import torch.nn as nn
from collections import OrderedDict
# from utils import load_state_dict_from_url
from typing import Type, Any, Callable, Union, List, Optional    
import numpy as np

import sys 
sys.path.append('/data/users2/rohib/github/testing')
# import utils_gsp.sps_tools as sps_tools
import utils_gsp.gpu_projection as gsp_gpu
import utils_gsp.gsp_general as gsp_gen

class GSP_Model:
    def __init__(self, model, sps=0.8) -> None:
        #  Class Variables for GSP
        self.model = model
        
        self.total_epochs = None
        self.curr_epoch = 0
        self.curr_iter = 0
        
        self.is_rand_prune = False

        self.gsp_training_mode = False
        self.start_gsp_epoch = 0
        self.gsp_int = 0
        self.sps = 0.0
        self.scheduled_sps_run = False
        self.project_model = False # Controls whether to project the whol model simultaneously.
        self.proj_filters = True  # If not projecting model, and just projecting layers, 
                                  # Controls whether to project filters or kernels per layer.
        self.logger = None
        self.masks = None
        self.train_loader_len = None
        self.device = next(self.model.parameters()).device
    
    
    def _create_sps_schedule(self):
        assert self.total_epochs is not None, "Total Epoch is None"

        ep_range = torch.arange(self.total_epochs)
        sps_schedule = 1- (self.total_epochs**(-ep_range/self.total_epochs))
        return sps_schedule
    
    def _linear_sps_schedule(self, start, final):
        sps_schedule = np.linspace(start, final, self.total_epochs)
        return sps_schedule

    def mask_out_parameters(self):
        """
        Multiply the parameters with the corresponding masks to zero out the masked parameters.
        """
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                module.weight.data = module.weight.data * self.masks[module]

    
    def apply_gsp(self, sps=None, schedule=None):
        if schedule == "linear":
            sps_schedule = self._linear_sps_schedule(0.2, 0.95)
            sps = sps_schedule[self.curr_epoch]
        else:
            sps = self.sps if (sps == None) else sps # To use this method with sparsity as argument. Default will use self.sps.
        
        self.sps = sps
        if self.gsp_training_mode and (self.curr_epoch > self.start_gsp_epoch) and (self.curr_iter % self.gsp_int == 0):
            if self.scheduled_sps_run:
                self.sps_schedule = self._create_sps_schedule()
                sps = self.sps_schedule[self.curr_epoch]

            if self.project_model == False:    
                print(f"Projecting layers with GSP!! | GSP_Mode: {self.gsp_training_mode} | Sparsity level: {sps}")
                self._apply_gsp_to_layers(sps)
            else:
                print(f"Projecting whole Model with GSP!! | GSP_Mode: {self.gsp_training_mode}")
                self._project_all_layers(sps)
        
        # Update Internal epoch and iterations
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
        

        if self.logger:
            self.logger.info(f"Applied GSP to all Layers! Epoch: {self.curr_epoch} | Iter: {self.curr_iter} \
                 | sps: {sps:.4f} | Filter-Proj: {self.proj_filters} ")


    def _project_all_layers(self, sps):
        layer_d = dict()
        shape_d = dict()
        ctr = 0
        for i, (name, layer) in enumerate(self.model.named_modules()):
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                shape_d[name] = layer.weight.shape
                layer_d[ctr] = layer.weight.data.detach().clone().flatten()
                ctr+=1

        xp_mat, ni_list = gsp_gen.GSP(layer_d, sps=sps)
        out_layers = gsp_gen.unpad_output_mat(xp_mat, ni_list)

        # rebuild_network
        ctr = 0
        for name, layer in self.model.named_modules():
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                layer.weight.data = out_layers[ctr].reshape(shape_d[name])
                ctr += 1
        
        if self.logger:
            self.logger.info(f" Applied generalGSP to Model! Epoch: {self.curr_epoch} | Iter: {self.curr_iter} | sps: {sps}" \
                             f" | Filter-Proj: {self.proj_filters} ")


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
                layer_sps = sparsity(layer.weight.data.detach())
                print(f"Sparsity of layer: {name}: {layer_sps}")

   
    
    # =================================== Pruning Methods =======================================
    def prune_and_mask_model(self, sps):
        self.logger.info(f"Value of is Rand: {self.is_rand_prune}")
        if self.is_rand_prune:
            self.logger.info(f"=> Random Pruning of the Network Undergoing! \n")
            self._random_prune(sps)

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


    def _random_prune(self, sps=0.95):
        for name, layer in self.model.named_modules():
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                num = layer.weight.data.numel()
                num_zero = int(round(num*sps))
                indices = torch.randperm(num)[:num_zero]
                layer.weight.data.flatten()[[indices]] = 0.


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

    # ====================================== HOYER SQUARE ==========================================
    # def hoyer_square(self):
    #     reg = 0.0
    #     reg = (torch.sum(torch.abs(param))**2)/torch.sum(param**2)

# ========================================================================================================
# class HoyerRegularizer:
#     def __init__(self, model, loss, decay) -> None:
#         self.model = model
#         self.decay = decay
#         self.loss = loss

    
def hoyer_square(model, decay, loss):
    reg = 0.0
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            reg += (torch.sum(torch.abs(module.weight))**2)/torch.sum(module.weight**2)
    orig_loss = loss
    loss = loss + decay * reg 
    return orig_loss, loss


def group_hoyer_sq(model, decay, loss):
    reg=0.0
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            param = module.weight
            reg += ( (torch.sum(torch.sqrt(torch.sum(param**2,(0,2,3))))**2) 
                    + (torch.sum(torch.sqrt(torch.sum(param**2,(1,2,3))))**2) )/torch.sum(param**2)    
    
        if isinstance(module, nn.Linear):
            param = module.weight
            reg += ( (torch.sum(torch.sqrt(torch.sum(param**2,0)))**2) 
                    + (torch.sum(torch.sqrt(torch.sum(param**2,1)))**2) )/torch.sum(param**2)

    orig_loss = loss
    loss = loss + decay * reg 
    return orig_loss, loss
    

# Functions
# --------------------------------------------------------------------------------------------------------
def sparsity(matrix):
    device = matrix.device
    matrix = matrix.detach().clone()
    ni = torch.tensor(matrix.shape[0], device=device)

    # Get Indices of columns with all-0 vectors.
    zero_col_ind = (abs(matrix - 0).sum(0) < 2.22e-16).nonzero().reshape(-1)  
    spx_c = (torch.sqrt(ni) - torch.norm(matrix,1, dim=0) / torch.norm(matrix,2, dim=0)) / (torch.sqrt(ni) - 1)

    if len(zero_col_ind) != 0:
        spx_c[zero_col_ind] = 1  # Sparsity = 1 if column already zero vector.
    
    if matrix.dim() > 1:   
        # sps_avg =  spx_c.sum() / matrix.shape[1]
        sps_avg = spx_c.mean()
    elif matrix.dim() == 1:  # If not a matrix but a column vector!
        sps_avg =  spx_c    
    return sps_avg