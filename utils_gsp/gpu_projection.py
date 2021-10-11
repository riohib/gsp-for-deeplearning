# from DeepLearningExamples.PyTorch.Segmentation.nnUNet.utils.gpu_affinity import device
import torch
import numpy as np
from numpy import linalg as LA
import pickle
# import scipy.io
import logging
import pdb
import time



def sparsity(matrix):
    device=matrix.device
    matrix = matrix.detach().clone()
    ni = torch.tensor(matrix.shape[0], device=matrix.device)

    # Get Indices of columns with all-0 vectors.
    zero_col_ind = (abs(matrix.sum(0) - 0) < 2.22e-16).nonzero().reshape(-1)  
    spx_c = (torch.sqrt(ni) - torch.norm(matrix,1, dim=0) / torch.norm(matrix,2, dim=0)) / (torch.sqrt(ni) - 1)

    if len(zero_col_ind) != 0:
        spx_c[zero_col_ind] = 1  # Sparsity = 1 if column already zero vector.
    
    if matrix.dim() > 1:   
        # sps_avg =  spx_c.sum() / matrix.shape[1]
        sps_avg = spx_c.mean()
    elif matrix.dim() == 1:  # If not a matrix but a column vector!
        sps_avg =  spx_c    
    return sps_avg


def checkCritical(matrix, critval_list, precision=1e-6):
    device=matrix.device
    max_elems = torch.max(matrix, 0)[0]

    ind_crit_bool = (abs(matrix - max_elems) < precision)
    crit_points = matrix * ind_crit_bool

    num_crit_points = torch.sum(ind_crit_bool, dim=0)

    # Boolean of vector cols with non-trivial critical values
    crit_cols = torch.where((num_crit_points.float() > 1).cuda(device), torch.ones(matrix.shape[1], device=device), \
                            torch.zeros(matrix.shape[1], device=device))

    # getting non-trivial critical values
    critval_list = max_elems[crit_cols.bool()]

    return critval_list, max_elems


def gmu(matrix, xp_mat, mu=0):
    device=matrix.device
    vgmu = 0
    gradg = 0
    matrix = torch.abs(matrix)
    xp_mat = torch.zeros([matrix.shape[0], matrix.shape[1]]).to(device)
    glist = []

    gsp_iter = 0

    # ------------------------------- Previous For Loop --------------------------------------
    ni = matrix.shape[0]
    betai = 1 / (torch.sqrt(torch.tensor(ni, dtype=torch.float32, device=device)) - 1)

    xp_mat = matrix - (mu * betai)
    indtp = xp_mat > 0

    xp_mat.relu_()

    # outputs
    mnorm = torch.norm(xp_mat, dim=0)
    mnorm_inf = mnorm.clone()
    mnorm_inf[mnorm_inf == 0] = float("Inf")

    col_norm_mask = (mnorm > 0)

    # mat_mask =  (col_norm_mask.float().view(1,784) * torch.ones(300,1))
    mat_mask = (col_norm_mask.float().view(1, matrix.shape[1]) * torch.ones(matrix.shape[0], 1,device=device))

    nip = torch.sum(xp_mat > 0, dim=0)  # columnwise number of values > 0

    # needs the if condition mnorm> 0 (it's included)
    # Terms in the Gradient Calculation
    term2 = torch.pow(torch.sum(xp_mat, dim=0), 2)
    mnorm_inv = torch.pow(mnorm_inf, -1)
    mnorm_inv3 = torch.pow(mnorm_inf, -3)

    # The column vectors with norm mnorm == 0 zero, should not contribute to the gradient sum.
    # In the published algorithm, we only calculate gradients for condition: mnorm> 0
    # To vectorize, we include in the matrix columns where mnorm == 0, but we manually replace
    # the inf after divide by zero with 0, so that the grad of that column becomes 0 and
    # doesn't contribute to the sum.
    # mnorm_inv[torch.isinf(mnorm_inv)] = 0
    # mnorm_inv3[torch.isinf(mnorm_inv3)] = 0

    # Calculate Gradient
    gradg_mat = torch.pow(betai, 2) * (-nip * mnorm_inv + term2 * mnorm_inv3)
    gradg = torch.sum(gradg_mat)

    # vgmu calculation
    ## When indtp is not empty (the columns whose norm are not zero)
    # xp_mat /= mnorm
    xp_mat[:, col_norm_mask] /= mnorm[col_norm_mask]

    ## When indtp IS empty (the columns whose norm ARE zero)
    max_elem_rows = torch.argmax(matrix, dim=0)[~col_norm_mask]  # The Row Indices where maximum of that column occurs
    xp_mat[max_elem_rows, ~col_norm_mask] = 1

    # vgmu computation
    vgmu_mat = betai * torch.sum(xp_mat, dim=0)
    vgmu = torch.sum(vgmu_mat)

    return vgmu, xp_mat, gradg


def groupedsparseproj(matrix, sps, precision=1e-6, linrat=0.9):
    # sps = 0.9 ;  precision=1e-6; linrat=0.9
    device=matrix.device

    epsilon = 10e-15
    k = 0
    muup0 = 0
    r = matrix.shape[1]  # No of Columns

    critmu = torch.tensor([])
    critval_list = []

    vgmu = torch.zeros(1, device=device)
    # maxxi_list = []

    # These operations were inside the loop, but doesn't need to be.
    matrix_sign = torch.sign(matrix)
    pos_matrix = matrix_sign * matrix
    xp_mat = torch.zeros([matrix.shape[0], matrix.shape[1]]).to(device)
    ni = matrix.shape[0]

    # -------------------------------------------------------------------------------
    # Check Critical Points
    k = r * np.sqrt(ni) / (np.sqrt(ni) - 1)
    # check critical values of mu where g(mu) is discontinuous, that is,
    # where the two (or more) largest entries of x{i} are equal to one another.
    critical_val, max_xi = checkCritical(pos_matrix, critval_list)
    muup0 = max(max_xi * (np.sqrt(ni) - 1))
    critmu = torch.tensor(critval_list) * (np.sqrt(ni) - 1)

    k = k - r * sps
    vgmu, xp_mat, gradg = gmu(pos_matrix, xp_mat, 0)

    if vgmu < k:
        xp_mat = matrix
        gxpmu = vgmu
        numiter = 0
        return xp_mat
    else:
        numiter = 0; 
        mulow = 0; muup = muup0
        glow = vgmu
        # Initialization on mu using 0, it seems to work best because the
        # slope at zero is rather steep while it is gets falt for large mu
        newmu = 0; gnew = glow; gpnew = gradg  # g'(0)
        delta = muup - mulow

        while abs(gnew - k) > precision * r and numiter < 100:
            oldmu = newmu
            # % Newton:
            newmu = oldmu + (k - gnew) / (gpnew + epsilon)

            if (newmu >= muup) or (newmu <= mulow):  # If Newton goes out of the interval, use bisection
                newmu = (mulow + muup) / 2

            gnew, xnew, gpnew = gmu(matrix, xp_mat, newmu)

            if gnew < k:
                gup = gnew; xup = xnew; muup = newmu
            else:
                glow = gnew; mulow = xnew; mulow = newmu
                
            # Guarantees linear convergence
            if (muup - mulow) > linrat * delta and abs(oldmu - newmu) < (1 - linrat) * delta:
                newmu = (mulow + muup) / 2
                gnew, xnew, gpnew = gmu(matrix, xp_mat, newmu)

                if gnew < k:
                    gup = gnew; xup = xnew; muup = newmu
                else:
                    glow = gnew; mulow = xnew; mulow = newmu

                numiter += 1
            numiter += 1

            if critmu.shape[0] != 0 and abs(mulow - muup) < abs(newmu) * precision and \
                    min(abs(newmu - critmu)) < precision * newmu:
                print('The objective function is discontinuous around mu^*.')
                xp = xnew
                gxpmu = gnew

            xp_mat = xnew

        gxpmu = gnew

    alpha_mat = torch.matmul(xp_mat.T, pos_matrix)
    alpha = torch.diagonal(alpha_mat)
    xp_mat = alpha * (matrix_sign * xp_mat)
    
    outinfo = { 'numiter': numiter, 'gxpmu':gxpmu, 'newmu':newmu} #for Plotting
    
    return xp_mat