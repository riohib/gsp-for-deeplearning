import torch
import torch.nn as nn

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else 'cpu')


def get_conv_linear_mask(model, threshold=1e-8, device=device):
    masks = {}
    for name, layer in model.named_modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            tensor = layer.weight.data
            masked_tensor = torch.where(abs(tensor) < threshold, torch.tensor(0.0, device=device), tensor)
            mask = torch.where(abs(tensor) < threshold, torch.tensor(0.0, device=device), torch.tensor(1.0, device=device))
            masks[name] = mask
            layer.weight.data = masked_tensor
    return masks


def bind_gsp_methods_to_model(model):
    model.register_mask = register_masks.__get__(model)

def register_masks(self, in_masks):
    self.masks = list()
    print(f"Type of self: {type(self)}")
    for name, layer in self.named_modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            assert (layer.weight.data.shape == in_masks[name]), f"Weight and mask shape mismatch in layer: {self}"
            self.masks[name] = in_masks[name]



def get_unmasked_weights(self):
    """Return the weights that are unmasked.
    :return dict, key->module, val->list of weights
    """
    assert self.masks is not None, 'Masks should be generated first.'
    res = dict()
    for m in self.masks.keys():
        res[m] = filter_weights(m.weight, self.masks[m])
    return res

def get_masked_weights(self):
    """Return the weights that are masked.
    :return dict, key->module, val->list of weights
    """
    assert self.masks is not None, 'Masks should be generated first.'
    res = dict()
    for m in self.masks.keys():
        res[m] = filter_weights(m.weight, 1-self.masks[m])
    return res

def register_mask(self, masks=None):
    # self.masks = None
    self.unregister_mask()
    if masks is not None:
        self.masks = masks
    assert self.masks is not None, 'Masks should be generated first.'
    for m in self.masks.keys():
        m.register_forward_pre_hook(self._forward_pre_hooks)

def unregister_mask(self):
    for m in self.model.modules():
        m._backward_hooks = OrderedDict()
        m._forward_pre_hooks = OrderedDict()

def _forward_pre_hooks(self, m, input):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        # import pdb; pdb.set_trace()
        mask = self.masks[m]
        m.weight.data.mul_(mask)
    else:
        raise NotImplementedError('Unsupported ' + m)