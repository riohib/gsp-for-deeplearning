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



# ============================== GSP TRAINING ====================================== #
def apply_prune_mask(net, keep_masks):
    prunable_layers = filter(
        lambda layer: isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear), 
        net.modules()
        )

    for layer, keep_mask in zip(prunable_layers, keep_masks):
        assert (layer.weight.shape == keep_mask.shape)

        def hook_factory(keep_mask):
            """
            The hook function can't be defined directly here because of Python's
            late binding which would result in all hooks getting the very last
            mask! Getting it through another function forces early binding.
            """

            def hook(grads):
                return grads * keep_mask

            return hook

        layer.weight.data[keep_mask == 0.] = 0.
        layer.weight.register_hook(hook_factory(keep_mask))



def gsp_forward_conv2d(self, x):
    if self.ctr == 0:
        print("Modified Conv2D forward")
                
    # print(f"Size of gsp_w: {self.gsp_w.shape} || Size of gsp_gate: {self.gsp_gate.shape}")
    # print(f" gsp_w: {self.gsp_w.shape} - device:{str(self.gsp_w.device)} || gsp_gate: {self.gsp_gate.shape} - device:{str(self.gsp_gate.device)}")
    # print(f" matmul: {(self.gsp_w @ self.gsp_gate).shape} || device: {str((self.gsp_w @ self.gsp_gate).device)}")

    self.gate_act = torch.sigmoid( self.gsp_w @ self.gsp_gate)
    self.reshaped_act = self.gate_act.reshape(self.gate_act.shape[0],1,1,1)
    # print(f" weightShp: {self.weight.shape} | reshaped_act: {self.reshaped_act.shape}")

    # prod = self.weight.data.detach() * self.reshaped_act.data.detach()
    # assert (self.weight.shape == prod.shape), f"Shape mismatch in layer: {self}"

    # print(f"shape conv2d.weight: {self.weight.shape} | shape reshaped_act: {self.reshaped_act.shape} | product: {(self.weight * self.reshaped_act).shape}")
    # print( f" gsp_w: {self.gsp_w .requires_grad} || gate: {self.gsp_gate.requires_grad} || weight req_grad:" \
    #     f"{self.weight.requires_grad} || reshaped_act req_grad: {self.reshaped_act.requires_grad}" )
    self.ctr += 1 
    return F.conv2d(x, self.weight * self.reshaped_act, self.bias,
            self.stride, self.padding, self.dilation, self.groups)



def gsp_forward_linear(self, x):
    if self.ctr == 0:
        print("Modified Linear forward")

    self.gate_act = torch.sigmoid( self.gsp_w @ self.gsp_gate)
    self.reshaped_act = self.gate_act.reshape(self.gate_act.shape[0],1)
    
    # print(f"shape conv2d.weight: {self.weight.shape} | shape reshaped_act: {self.reshaped_act.shape} | product: {(self.weight * self.reshaped_act).shape}")

    # prod = self.weight.data.detach() * self.reshaped_act.data.detach()
    # assert (self.weight.shape == prod.shape), f"Shape mismatch in layer: {self}"

    # print( f" gsp_w: {self.gsp_w .requires_grad} || gate: {self.gsp_gate.requires_grad} || weight req_grad:" \
    #     f"{self.weight.requires_grad} || reshaped_act req_grad: {self.reshaped_act.requires_grad}" )
    self.ctr += 1
    return F.linear(x, self.weight * self.reshaped_act, self.bias)




def initialize_gsp_layers(self):
    """
    Input: the model class itself! 
    self: the model class'
    """
    index =0 
    for name, layer in self.named_modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            # print(f"[{index}]: {layer}")
            layer.ctr = 0
            layer_shape = layer.weight.shape
            dim_1, dim_2 = layer.weight.shape[0], layer.weight.shape[1]
            
            gsp_w = layer.weight.data.detach().clone().reshape(dim_1,-1)
            layer.gsp_w = nn.Parameter(gsp_w, requires_grad=False)
            layer.gsp_gate = nn.Parameter(torch.zeros(layer.gsp_w.shape[1], device=device))

        # Modify the layer forward methods
        print("Binding the modified forward layers!")
        if isinstance(layer, nn.Conv2d):
            layer.forward = gsp_forward_conv2d.__get__(layer)

        if isinstance(layer, nn.Linear):
            layer.forward = gsp_forward_linear.__get__(layer)

        index +=1


def _apply_gsp_to_layers(self, name, layer, sps):
    # gsp_in_d = {}
    w_shape = layer.weight.shape

    if 'downsample' in name:
        dim_1, dim_2 = layer.weight.shape[0], layer.weight.shape[1]
        # gsp_in = layer.gsp_w.data.detach().reshape(dim_1, -1)
        gsp_in = layer.gsp_w.data.detach().reshape(dim_1, -1)
    else:
        # gsp_in = layer.gsp_w.data.detach().reshape(layer.weight.shape[0], -1)
        gsp_in = layer.gsp_w.data.detach().reshape(layer.weight.shape[0], -1)
    
    layer.gsp_w.data = gsp_gpu.groupedsparseproj(gsp_in.T, self.sps).T


def _apply_gsp_old(self) -> None:
    # self._apply_gsp_to_modules(self.conv1, self.sps)

    for name, layer in self.features.named_modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            self._apply_gsp_to_layers(name, layer, self.sps)
    
    for name, layer in self.classifier.named_modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            self._apply_gsp_to_layers(name, layer, self.sps)
    
    # print(f"GSP applied to all layers!  epoch: {self.curr_epoch} | iter: {self.curr_iter} | gsp_int: {self.gsp_int} ")

    if self.logger != None:
        self.logger.info(f"GSP applied to all layers! iter: {self.curr_iter}")
        print("GSP applied to all layers!")


def forward_gsp(self, x):
    if self.gsp_training_mode and (self.curr_epoch > self.start_gsp_epoch) and (self.curr_iter % self.gsp_int == 0):
        self._apply_gsp_gates()

    out = self.features(x)
    out = out.view(out.size(0), -1)
    out = self.classifier(out)
    return out


def bind_gsp_methods_to_model(model, args, apply_gsp=True):
    # Additional Class Variables for GSP
    # model.gsp_training_mode = apply_gsp
    # model.sps = 0.80
    # model.curr_epoch = 0
    # model.curr_iter = 1
    # model.start_gsp_epoch = -1
    # model.gsp_int = 1
    # model.logger = None

    model.initialize_gsp_layers = initialize_gsp_layers.__get__(model)
    model.forward = forward_gsp.__get__(model)
    model._apply_gsp = _apply_gsp.__get__(model)
    model._apply_gsp_to_layers = _apply_gsp_to_layers.__get__(model)

    # apply_backward_hook(model)
    model.initialize_gsp_layers()

#======================================================================================== #
# =============================== New GSP Filter gates ================================== #
#======================================================================================== #

def bind_new_gsp_methods_to_model(model, args, apply_gsp=True):
    model.initialize_new_gsp_layers = initialize_new_gsp_layers.__get__(model)
    
    model._apply_gsp_gates = _apply_gsp_gates.__get__(model)
    model.forward = _new_forward_gsp.__get__(model)
    model.initialize_new_gsp_layers()


def initialize_new_gsp_layers(self):
    """
    Input: the model class itself! 
    self: the model class'
    """
    index =0 
    for name, layer in self.named_modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            layer.ctr = 0
            no_of_filters = layer.weight.shape[0]

        # Modify the layer forward methods
        print("Binding the NEW forward layers!")
        if isinstance(layer, nn.Conv2d):
            gsp_gate = torch.ones(no_of_filters, device=device).reshape(no_of_filters,1,1,1) 
            torch.nn.init.xavier_uniform_(gsp_gate)
            layer.gsp_gate = nn.Parameter(gsp_gate, requires_grad=True)
            layer.forward = gate_gsp_forward_conv2D.__get__(layer)

        if isinstance(layer, nn.Linear):
            gsp_gate = torch.ones(no_of_filters, device=device).reshape(no_of_filters,1) 
            torch.nn.init.xavier_uniform_(gsp_gate)
            layer.gsp_gate = nn.Parameter(gsp_gate, requires_grad=True)
            layer.forward = gate_gsp_forward_linear.__get__(layer)

        index +=1


def gate_gsp_forward_conv2D(self, x):
    if self.ctr == 0:
        print("Modified NEW Conv2D forward")
    self.ctr += 1
    return F.conv2d(x, self.weight * self.gsp_gate, self.bias,
            self.stride, self.padding, self.dilation, self.groups)


def gate_gsp_forward_linear(self, x):
    if self.ctr == 0:
        print("Modified NEW Linear forward")
    self.ctr += 1
    return F.linear(x, self.weight * self.gsp_gate, self.bias)



def _apply_gsp_gates(self) -> None:
    self.gate_d = dict()
    count = 0

    for name, layer in self.named_modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            self.gate_d[count] = layer.gsp_gate.detach().clone().flatten()
            count += 1

    xp_mat, ni_list = gsp_pad.groupedsparseproj(self.gate_d, sps=0.85, precision=1e-6, linrat=0.9)
    sps_val = sps_tools.padded_sparsity(xp_mat, ni_list)
    
    print(f"GSP applied to all!  epoch: {self.curr_epoch} | iter: {self.curr_iter} | gsp_int: {self.gsp_int} | sps: {sps_val}")

    cnt = 0
    for name, layer in self.named_modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            # trimmed_gate = gate_d[cnt][0:ni_list[cnt]]
            dim0 = ni_list[cnt]
            if isinstance(layer, nn.Conv2d):
                layer.gsp_gate.data = xp_mat[0:dim0, cnt].reshape(dim0,1,1,1)
            if isinstance(layer, nn.Linear):
                layer.gsp_gate.data = xp_mat[0:dim0, cnt].reshape(dim0,1)
            # print(trimmed_gate.shape)
            cnt += 1

    if self.logger != None:
        self.logger.info(f"GSP applied to all layers! iter: {self.curr_iter}")
        print("GSP applied to all layers!")



def _new_forward_gsp(self, x):
    if self.gsp_training_mode and (self.curr_epoch > self.start_gsp_epoch) and (self.curr_iter % self.gsp_int == 0):
        self._apply_gsp_gates()

    out = self.features(x)
    out = out.view(out.size(0), -1)
    out = self.classifier(out)
    return out

#======================================================================================== #
# ================================= GSP FINETUNING ====================================== #
def get_abs_sps(model):
    nonzero = total = 0
    # print(f"TYPE: {type(model)}")

    for name, layer in model.named_modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            # print(name)
            tensor = layer.weight.detach().clone()
            # nz_count.append(torch.count_nonzero(tensor))
            nz_count = torch.count_nonzero(tensor).item()
            total_params = tensor.numel()
            nonzero += nz_count
            total += total_params
    
    # print(f"TOTAL: {total}")
    abs_sps = 100 * (total-nonzero) / total

    return abs_sps

def prune_filters(model, prune_sps = 0.9):
    prod_l = list()
    act_mat_d = dict()

    for name, layer in model.named_modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            # layer_d[name] = layer
            prod = torch.abs(layer.gsp_w @ layer.gsp_gate)
            # Normalize
            prod -= prod.min()
            prod /= prod.max()
            # print(f" gsp_w: {layer.gsp_w.shape} | gate: {layer.gsp_gate.shape} | prod: {prod.shape}")
            prod_l.append(prod)
            
            # Get TopK and create vector with topk values and zeros rest
            act_mat = torch.zeros_like(prod)
            k = math.floor(prod.shape[0] * (1 - prune_sps))
            vals, ind = torch.topk(prod, k=k)
            act_mat[ind] = vals

            if isinstance(layer, nn.Conv2d): 
                reshaped_act = act_mat.reshape(act_mat.shape[0],1,1,1)
                layer.weight.data = layer.weight.data * reshaped_act
            if isinstance(layer, nn.Linear):
                reshaped_act = act_mat.reshape(act_mat.shape[0],1)
                layer.weight.data = layer.weight.data * reshaped_act
            # print(f"LayerShp: {layer.weight.shape} | prod: {prod.shape} | act_mat: {reshaped_act.shape}")
            # print(f"Layer Data Shape: {}")
            act_mat_d[name] = act_mat


def forward_conv2d(self, x):
    return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

def forward_linear(self, x):
    return F.linear(x, self.weight, self.bias)


def bind_forward_methods(model):
    for name, layer in model.named_modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):            
            layer.gsp_w = None
            layer.gsp_gate = None
            layer.gate_act = None
            layer.reshaped_act = None

            print("Binding the regular forward layers!")
            if isinstance(layer, nn.Conv2d):
                layer.forward = forward_conv2d.__get__(layer)

            if isinstance(layer, nn.Linear):
                layer.forward = forward_linear.__get__(layer)


def setup_pruning_exp_vanilla(model, prune_sps = 0.9):
    
    prune_filters(model, prune_sps)
    print(f" SPS of model after pruning {get_abs_sps(model)}")

    bind_forward_methods(model) # Bind regular Forward Methods to the model

    _ , masks_l = sps_tools.get_conv_linear_mask(model)
    
    apply_prune_mask(model, masks_l) # Apply the mask to the model layer hooks




def setup_pruning_exp_block(model, prune_sps = 0.8):
    
    prune_filters(model, prune_sps)
    print(f" SPS of model after pruning {get_abs_sps(model)} | Prune SPS: {prune_sps}")

    # bind_forward_methods(model) # Bind regular Forward Methods to the model
    detach_gsp_gates(model)
    
    _ , masks_l = sps_tools.get_conv_linear_mask(model)
    
    apply_prune_mask(model, masks_l) # Apply the mask to the model layer hooks


def detach_gsp_gates(model):
    for name, layer in model.named_modules():
        if isinstance (layer, nn.Conv2d) or isinstance (layer, nn.Linear):
            layer.gsp_gate.requires_grad = False
            print(layer.gsp_gate.requires_grad)

#======================================================================================== #
#======================================================================================== #