import torch
from torch import Tensor
import torchvision.models as models
import torch.nn as nn

import sys 
sys.path.append('/data/users2/rohib/github/testing')
import utils_gsp.sps_tools as sps_tools
import utils_gsp.gpu_projection as gsp_gpu


def bind_gsp_methods_to_model(model):
    # Add required class variables to the model
    model.gsp_training_mode = False
    model.sps = 0.0
    model.curr_epoch = 0
    model.curr_iter = 0
    model.start_gsp_epoch = 0
    model.gsp_int = 0

    # Bind the required methods to the Model Instance
    # model._see_layers = _see_layers.__get__(model)
    model.get_sparsity = get_sparsity.__get__(model)
    model._apply_gsp_to_layers = _apply_gsp_to_layers.__get__(model)
    model._apply_gsp_to_modules = _apply_gsp_to_modules.__get__(model)
    model._apply_gsp = _apply_gsp.__get__(model)

    # Finally Bind the new Forward Method
    model._forward_impl = _forward_impl.__get__(model)

    return model


# def _see_layers(self):
#     for name, layer in self.layer1.named_modules():
#         if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
#             # print(layer)
#             pass

def get_sparsity(self):
    return sps_tools.get_abs_sps(self)[0].item()


def _apply_gsp_to_layers(self, name, layer, sps):
    gsp_in_d = {}

    # if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
        # layer_d[name] = layer
    w_shape = layer.weight.shape

    if 'downsample' in name:
        # print(layer.weight.shape)
        dim_1 ,  dim_2 = layer.weight.shape[0], layer.weight.shape[1]
        gsp_in_d[name] = layer.weight.data.detach().reshape(dim_1, -1)
    else:
        # print(layer.weight.shape[0])
        gsp_in_d[name] = layer.weight.data.detach().reshape(layer.weight.shape[0], -1)
    
    layer.weight.data = gsp_gpu.groupedsparseproj(gsp_in_d[name].T, self.sps).T.reshape(w_shape)
    # print()
    # print(f"Layer | requires_grad: {layer.weight.requires_grad}")


def _apply_gsp_to_modules(self, layer, sps):
    if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
        w_shape = layer.weight.shape
        dim_1 ,  dim_2 = layer.weight.shape[0], layer.weight.shape[1]
        gsp_in = layer.weight.data.detach().reshape(dim_1, -1)
        layer.weight.data = gsp_gpu.groupedsparseproj(gsp_in.T, self.sps).T.reshape(w_shape)
        # print(f"Module | requires_grad: {layer.weight.requires_grad}")


def _apply_gsp(self) -> None:
    self._apply_gsp_to_modules(self.conv1, self.sps)

    for name, layer in self.layer1.named_modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            self._apply_gsp_to_layers(name, layer, self.sps)
    
    for name, layer in self.layer2.named_modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            self._apply_gsp_to_layers(name, layer, self.sps)
    
    for name, layer in self.layer3.named_modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            self._apply_gsp_to_layers(name, layer, self.sps)
    
    for name, layer in self.layer4.named_modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            self._apply_gsp_to_layers(name, layer, self.sps)
    
    self._apply_gsp_to_modules(self.fc, self.sps)

    print("GSP applied to all layers!")


def _forward_impl(self, x: Tensor) -> Tensor:
    # See note [TorchScript super()]
    # print(f"Current iter: {self.curr_iter}")
    if self.gsp_training_mode and (self.curr_epoch > self.start_gsp_epoch) and (self.curr_iter % self.gsp_int == 0):
        self._apply_gsp()

    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)

    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)

    x = self.avgpool(x)
    x = torch.flatten(x, 1)
    x = self.fc(x)

    return x