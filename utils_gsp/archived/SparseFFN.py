import torch
import numpy as np

import sys
sys.path.append('../')
import utils_gsp.padded_gsp as gsp_global
import utils_gsp.gpu_projection as gsp_gpu
from utils_gsp.SparseModel import SparseModel 

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else 'cpu')
# --------------------------------------------------------------------------------------------------------
class LeNet(SparseFFN):
    def __init__(self):
        super(LeNet, self).__init__()
        linear = nn.Linear
        self.fc1 = linear(784, 300)
        self.fc2 = linear(300, 100)
        self.fc3 = linear(100, 10)

    def forward(self, x):
        try:
            x = x.view(-1, 784)
        except:
            extype, value, tb = sys.exc_info()
            traceback.print_exc()
            pdb.post_mortem(tb)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.log_softmax(self.fc3(x), dim=1)
        return x
    
# ------------------------------------------------------------------------------------------------------
class SparseFFN(SparseModel):
    def __init__(self, model, logger):
        super().__init__(model, logger)

    def apply_gsp(self, sps, gsp_func = gsp_gpu):
        """
        This function is for applying GSP layer-wise in a CNN or MLP or Resnet network in this repo.
        The GSP is applied layer-wise separately.
        """ 
        weight_d = {}
        shape_list = []
        counter = 0
        for name, param in self.model.named_parameters(): 
            if 'weight' in name:
                shape_list.append(param.data.shape)    
                if param.dim() > 2:  #Only two different dimension for LeNet-5
                    w_shape = param.shape
                    dim_1 = w_shape[0] * w_shape[1]
                    weight_d[counter] = param.detach().view(dim_1,-1)
                    param.data = gsp_func.groupedsparseproj(weight_d[counter], sps).view(w_shape)
                else:
                    param.data = gsp_func.groupedsparseproj(param.detach(), sps)
                counter += 1

    def apply_gsp_filterwise(self, sps, gsp_func = gsp_gpu):
        """
        This function is for applying GSP layer-wise in a CNN or MLP or Resnet network in this repo.
        The GSP is applied layer-wise separately.
        """ 
        weight_d = {}
        shape_list = []
        counter = 0
        for name, param in self.model.named_parameters(): 
            if 'weight' in name:
                shape_list.append(param.data.shape)    
                if param.dim() > 2:  #Only two different dimension for LeNet-5
                    w_shape = param.shape
                    dim_1 = w_shape[0] * w_shape[1]
                    weight_d[counter] = param.detach().view(dim_1,-1)
                    param.data = gsp_func.groupedsparseproj(weight_d[counter].T, sps).T.view(w_shape)
                else:
                    param.data = gsp_func.groupedsparseproj(param.detach().T, sps).T
                counter += 1


#   ========================================================================================== 
    def apply_global_gsp(self, sps, filterwise=False):
        matrix, val_mask, shape_l = self.concat_nnlayers(filterwise)
        try:
            type(matrix) == torch.Tensor
        except:
            print("The output of concat_nnlayers - 'matrix' is not a Torch Tensor!")

        xp_mat, ni_list = gsp_global.groupedsparseproj(matrix, val_mask, sps)
        
        # self.xp_mat = xp_mat
        # self.ni_list = ni_list

        self.rebuild_nnlayers(xp_mat, ni_list, shape_l, filterwise)


    def concat_nnlayers(self, filterwise):

        shape_l = self.get_shape_l(filterwise)
        
        max_dim0 = max([x[0] for x in shape_l])
        max_dim1 = sum([x[1] for x in shape_l])
        
        matrix = torch.zeros(max_dim0, max_dim1, device=device)
        val_mask = torch.zeros(max_dim0, max_dim1, device=device)
        
        counter = 0
        dim1 = 0
        ni_list = []

        for name, param in self.model.named_parameters():
            if 'weight' in name:
                prev_dim0, cur_dim0 = shape_l[counter-1][0], shape_l[counter][0]
                prev_dim1, cur_dim1 = shape_l[counter-1][1], shape_l[counter][1]

                weight_mat = param.data.T if filterwise else param.data

                matrix[0:cur_dim0, dim1:dim1+cur_dim1] = weight_mat
                val_mask[0:cur_dim0, dim1:dim1+cur_dim1] = torch.ones(shape_l[counter]).to(device)
                
                dim1 += cur_dim1
                counter += 1
        return matrix, val_mask, shape_l

    def get_shape_l(self, filterwise):
        """
        Get's the model layer shape tuples for LeNet 300 100 and LeNet*5 models.
        """
        shape_l = []
        counter = 0
        for name, param in self.model.named_parameters(): 
            if 'weight' in name:
                dim0, dim1 = tuple(param.data.shape)    
                if filterwise:
                    shape_l.append( (dim1, dim0) )
                else:
                    shape_l.append( (dim0, dim1) )
        return shape_l


    def rebuild_nnlayers(self, matrix, ni_list, shape_l, filterwise):
        counter = 0
        dim1 = 0
        ni_list = []

        for name, param in self.model.named_parameters():
            if 'weight' in name:
                prev_dim0, cur_dim0 = shape_l[counter-1][0], shape_l[counter][0]
                prev_dim1, cur_dim1 = shape_l[counter-1][1], shape_l[counter][1]
                
                sparse_weight = matrix[0:cur_dim0, dim1:dim1+cur_dim1]
                
                param.data = sparse_weight.T if filterwise else sparse_weight
                # if filterwise:
                #     param.data = matrix[0:cur_dim0, dim1:dim1+cur_dim1].T
                # else:
                #     param.data = matrix[0:cur_dim0, dim1:dim1+cur_dim1]

                dim1 += cur_dim1
                counter += 1