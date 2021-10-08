import torch
from torch import Tensor
import torch.nn as nn
from collections import OrderedDict
# from utils import load_state_dict_from_url
from typing import Type, Any, Callable, Union, List, Optional    

import sys 
sys.path.append('/data/users2/rohib/github/testing')
import utils_gsp.sps_tools as sps_tools
import utils_gsp.gpu_projection as gsp_gpu

class GSP_Model:
    def __init__(self, model, sps=0.8) -> None:
        #  Class Variables for GSP
        self.model = model
        self.gsp_training_mode = False
        self.sps = 0.0
        self.curr_epoch = 0
        self.curr_iter = 0
        self.start_gsp_epoch = 0
        self.gsp_int = 0
        self.logger = None
        self.masks = None
        self.train_loader_len = None
    
    
    def apply_gsp(self):
        if self.gsp_training_mode and (self.curr_epoch > self.start_gsp_epoch) and (self.curr_iter % self.gsp_int == 0):
            print("Applying GSP!!")
            self.apply_gsp_to_layers()
            # print(f"The total MODEL SPS: {self.get_model_sps():.2f}")
        
        if self.curr_iter == self.train_loader_len:
            self.curr_iter = 0
            self.curr_epoch +=1 
        else:
            self.curr_iter += 1
    

    def apply_gsp_to_layers(self):
        for name, layer in self.model.named_modules():
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                w_shape = layer.weight.shape
                gsp_in = layer.weight.data.detach().reshape(layer.weight.shape[0], -1)

                layer.weight.data = gsp_gpu.groupedsparseproj(gsp_in.T, sps = self.sps).T.reshape(w_shape)
                
                # if self.curr_epoch == 0 and self.curr_iter == 0:
                #     print(f"Sparsity of layer: {name}: {sps_tools.sparsity(layer.weight.data)}")
        # if self.curr_epoch == 0 and self.curr_iter == 0:
        self.logger.info(f"Applied GSP to all Layers! Epoch: {self.curr_epoch} | Iter: {self.curr_iter} | sps: {self.sps}")


    def get_model_sps(self):
        nonzero = total = 0
        for name, param in self.model.named_parameters():
            # print(name)
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

    
    def get_layer_sps(self):
        for name, layer in self.model.named_modules():
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                layer_sps = sps_tools.sparsity(layer.weight.data.detach())
                print(f"Sparsity of layer: {name}: {layer_sps}")



    # ============== Pruning Methods ==============
    def register_pre_hook_mask(self, masks=None):
        # self.masks = None
        self.unregister_mask()
        if masks is not None:
            self.masks = masks
        assert self.masks is not None, 'Masks should be generated first.'

        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                module.mask = nn.Parameter(masks[module]).requires_grad_(False).to(module.weight.get_device())
                module.register_forward_pre_hook(self.forward_pre_hook)


    def unregister_mask(self):
        for module in self.model.modules():
            module._backward_hooks = OrderedDict()
            module._forward_pre_hooks = OrderedDict()

    @staticmethod
    def forward_pre_hook(module, x):
        module.mask.requires_grad_(False)
        mask = module.mask
        module.weight.data.mul_(mask.to(module.weight.get_device()))

    
    
    
    # ============================ Pruning with Register Hook ============================
    # Pruning with Just HOOK (Not forward pre hook, more stable)
    # @staticmethod
    # def register_hook_mask(model, keep_masks):

    #     # Before I can zip() layers and pruning masks I need to make sure they match
    #     # one-to-one by removing all the irrelevant modules:
    #     prunable_layers = filter(
    #         lambda layer: isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear), 
    #         model.modules()
    #         )

    #     for layer, keep_mask in zip(prunable_layers, keep_masks):
    #         assert (layer.weight.shape == keep_mask.shape)

    #         def hook_factory(keep_mask):
    #             """
    #             The hook function can't be defined directly here because of Python's
    #             late binding which would result in all hooks getting the very last
    #             mask! Getting it through another function forces early binding.
    #             """

    #             def hook(grads):
    #                 return grads * keep_mask

    #             return hook

    #         layer.weight.data[keep_mask == 0.] = 0.
    #         layer.weight.register_hook(hook_factory(keep_mask))