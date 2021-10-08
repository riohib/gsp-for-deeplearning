import torch
import numpy as np
from torch.nn.modules.module import Module


import sys
sys.path.append('../')
import utils_gsp.padded_gsp as gsp_global
import utils_gsp.gpu_projection as gsp_gpu

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else 'cpu')
#======================================================================================================
#====================================== Sparse Model Class ===========================================
#=====================================================================================================
class SparseModel(Module):
    def __init__(self, model, logger=None):
        self.model = model
        
        if logger == None:
            self.logger = logging.getLogger()
        else:
            self.logger = logger
    
    def get_layers(self, with_values=False):
        """
        Print the model Parameters.
        """
        self.logger.info(f"{'Param name':20} {'Shape':30} {'Type':15}")
        self.logger.info('-'*70)
        for name, param in self.model.named_parameters():
            self.logger.info(f'{name:20} {str(param.shape):30} {str(param.dtype):15}')
            if with_values:
                self.logger.info(param)
                
    def print_nonzeros(self):
        nonzero = total = 0
        for name, p in self.model.named_parameters():
            if 'mask' in name:
                continue
            tensor = p.data.cpu().numpy()
            nz_count = np.count_nonzero(tensor)
            total_params = np.prod(tensor.shape)
            nonzero += nz_count
            total += total_params
            self.logger.info(f'{name:20} | nonzeros = {nz_count:7} / {total_params:7} ({100 * nz_count / total_params:6.2f}%) | total_pruned = {total_params - nz_count :7} | shape = {tensor.shape}')
            if 'weight' in name:
                tensor = np.abs(tensor)
                if 'conv' in name:
                    dim0 = np.sum(np.sum(tensor, axis=0),axis=(1,2))
                    dim1 = np.sum(np.sum(tensor, axis=1),axis=(1,2))
                if 'fc' in name:
                    dim0 = np.sum(tensor, axis=0)
                    dim1 = np.sum(tensor, axis=1)
                nz_count0 = np.count_nonzero(dim0)
                nz_count1 = np.count_nonzero(dim1)
                self.logger.info(f'{name:20} | dim0 = {nz_count0:7} / {len(dim0):7} ({100 * nz_count0 / len(dim0):6.2f}%) | dim1 = {nz_count1:7} / {len(dim1):7} ({100 * nz_count1 / len(dim1):6.2f}%)')
        self.logger.info(f'alive: {nonzero}, pruned : {total - nonzero}, total: {total}, Compression rate : {total/nonzero:10.2f}x  ({100 * (total-nonzero) / total:6.2f}% pruned)')
        
    def nonzero_sps(self):
        nonzero = total = 0
        for name, param in self.model.named_parameters():
            tensor = param.data
            nz_count = torch.count_nonzero(tensor)
            total_params = tensor.numel()
            nonzero += nz_count
            total += total_params
        abs_sps = 100 * (total-nonzero) / total
        return abs_sps.item(), total, (total-nonzero).item()


    def get_layerwise_sps(self):
        """
        This function will output a list of layerwise sparsity of a LeNet-5 CNN model.
        The sparsity measure is the Hoyer Sparsity Measure.
        """
        counter = 0
        weight_d = {}
        sps_d = {}
        for name, param in self.model.named_parameters(): 
            if 'weight' in name:
                if param.dim() > 2:  #Only two different dimension for LeNet-5
                    w_shape = param.shape
                    dim_1 = w_shape[0] * w_shape[1]
                    weight_d[counter] = param.detach().view(dim_1,-1)
                    sps_d[name] = self.sparsity(weight_d[counter])
                else:
                    sps_d[name] = self.sparsity(param.data)
                counter += 1
        
        w_name_list = [x for x in sps_d.keys()] 
        return sps_d

    def get_neuronwise_sps(self):
        """
        This function will output a list of layerwise sparsity of a LeNet-5 CNN model.
        The sparsity measure is the Hoyer Sparsity Measure.
        """
        counter = 0
        weight_d = {}
        sps_d = {}
        for name, param in self.model.named_parameters(): 
            if 'weight' in name:
                if param.dim() > 2:  #Only two different dimension for LeNet-5
                    w_shape = param.shape
                    dim_1 = w_shape[0] * w_shape[1]
                    weight_d[counter] = param.detach().view(dim_1,-1)
                    sps_d[name] = self.sparsity(weight_d[counter].T)
                else:
                    sps_d[name] = self.sparsity(param.data.T)
                counter += 1
        
        w_name_list = [x for x in sps_d.keys()] 

        return sps_d


    def sparsity(self, matrix):
        ni = matrix.shape[0]
        # Get Indices of columns with all-0 vectors.
        zero_col_ind = (abs(matrix.sum(0) - 0) < 2.22e-16).nonzero().view(-1)  
        spx_c = (np.sqrt(ni) - torch.norm(matrix,1, dim=0) / torch.norm(matrix,2, dim=0)) / (np.sqrt(ni) - 1)
        if len(zero_col_ind) != 0:
            spx_c[zero_col_ind] = 1  # Sparsity = 1 if column already zero vector.
        
        if matrix.dim() > 1:   
            # sps_avg =  spx_c.sum() / matrix.shape[1]
            sps_avg = spx_c.mean()
        elif matrix.dim() == 1:  # If not a matrix but a column vector!
            sps_avg =  spx_c    
        return sps_avg


    def padded_sparsity(self, matrix, ni_list):
        """
        This Hoyer Sparsity Calculation is for matrices with the end of the columns that are padded. Hence,
        it needs the information of how much of each columns are elements and how much of them are padded.
        ni_list: Contains the number of values in each column (rest are padded with zero).
        """

        ni = matrix.shape[0]
        ni_tensor = torch.tensor(ni_list).to(device)

        # Get Indices of all zero vector columns.
        zero_col_ind = (abs(matrix.sum(0) - 0) < 2.22e-16).nonzero().view(-1)  
        spx_c = (torch.sqrt(ni_tensor) - torch.norm(matrix,1, dim=0) / torch.norm(matrix,2, dim=0)) / (torch.sqrt(ni_tensor) - 1)
        if len(zero_col_ind) != 0:
            spx_c[zero_col_ind] = 1  # Sparsity = 1 if column already zero vector.
        
        if matrix.dim() > 1:   
            sps_avg = spx_c.mean()
        elif matrix.dim() == 1:  # If not a matrix but a column vector!
            sps_avg =  spx_c    
        return sps_avg




# --------------------------------------------------------------------------------------------------------

class SparseResnet:
    def __init__(self, model):
        self.model = model
    
    def get_layers(self):
        params_d = {}
        for name, param in self.model.named_parameters(): 
            params_d[name] = param
        
        layer_list = [x for x in params_d.keys()]
        return layer_list

    def get_layerwise_sps(self):
        """
        This function will output a list of layerwise sparsity of a LeNet-5 CNN model.
        The sparsity measure is the Hoyer Sparsity Measure.
        """
        counter = 0
        weight_d = {}
        sps_d = {}
        for name, param in self.model.named_parameters(): 
            if 'weight' in name:
                if param.dim() > 2:  #Only two different dimension for LeNet-5
                    w_shape = param.shape
                    dim_1 = w_shape[0] * w_shape[1]
                    weight_d[counter] = param.detach().view(dim_1,-1)
                    sps_d[name] = self.sparsity(weight_d[counter]).item()
                else:
                    sps_d[name] = self.sparsity(param.data).item()
                counter += 1
        
        w_name_list = [x for x in sps_d.keys()] 
        return sps_d
    
    def sparsity(matrix):
        ni = matrix.shape[0]

        # Get Indices of all zero vector columns.
        zero_col_ind = (abs(matrix.sum(0) - 0) < 2.22e-16).nonzero().view(-1)  
        spx_c = (np.sqrt(ni) - torch.norm(matrix,1, dim=0) / torch.norm(matrix,2, dim=0)) / (np.sqrt(ni) - 1)
        if len(zero_col_ind) != 0:
            spx_c[zero_col_ind] = 1  # Sparsity = 1 if column already zero vector.
        
        if matrix.dim() > 1:   
            sps_avg =  spx_c.sum() / matrix.shape[1]
        elif matrix.dim() == 1:  # If not a matrix but a column vector!
            sps_avg =  spx_c    
        return sps_avg