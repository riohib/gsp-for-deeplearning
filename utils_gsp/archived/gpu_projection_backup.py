# from DeepLearningExamples.PyTorch.Segmentation.nnUNet.utils.gpu_affinity import device
import torch
import numpy as np
from numpy import linalg as LA
import pickle
import scipy.io
import logging
import pdb
import time

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def sparsity(matrix):
    r = matrix.shape[1]  # no of vectors

    spx = 0
    spxList = []
    for i in range(r):
        if matrix[:, i].sum() == 0:
            spx = 1
            spxList.append(spx)
            print('here')
        else:
            ni = matrix.shape[0]
            spx = (np.sqrt(ni) - torch.norm(matrix[:, i], 1) / torch.norm(matrix[:, i], 2)) / (np.sqrt(ni) - 1)
            spxList.append(spx)
        spx = sum(spxList) / r

    return spx


def checkCritical(matrix, critval_list, precision=1e-6):
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
    global device
    # print(f"The device from GPU-proj matrix: {str(matrix.device)}")
    if str(matrix.device) != 'cpu':
        device = torch.device('cuda')
        # print("The device is cuda")
    if str(matrix.device) == 'cpu':
        device = torch.device('cpu')
        # print("The device is cpu")
    if str(matrix.device) == 'cpu':
        print("Device neither CPU nor CUDA! Please check!")

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
        numiter = 0
        mulow = 0
        glow = vgmu
        muup = muup0
        # Initialization on mu using 0, it seems to work best because the
        # slope at zero is rather steep while it is gets falt for large mu
        newmu = 0
        gnew = glow
        gpnew = gradg  # g'(0)
        delta = muup - mulow
        switch = True

        # pdb.set_trace()
        while abs(gnew - k) > precision * r and numiter < 100:
            oldmu = newmu
            # % Secant method:
            # % newmu = mulow + (k-glow)*(muup-mulow)/(gup-glow);

            # % Bisection:
            # % newmu = (muup+mulow)/2;
            # % Newton:
            newmu = oldmu + (k - gnew) / (gpnew + epsilon)

            if (newmu >= muup) or (newmu <= mulow):  # If Newton goes out of the interval, use bisection
                newmu = (mulow + muup) / 2

            # print( 'Value of numiter: ' + str(numiter))
            gnew, xnew, gpnew = gmu(matrix, xp_mat, newmu)

            if gnew < k:
                gup = gnew
                xup = xnew
                muup = newmu
            else:
                glow = gnew
                mulow = xnew
                mulow = newmu

            # Guarantees linear convergence
            if (muup - mulow) > linrat * delta and abs(oldmu - newmu) < (1 - linrat) * delta:
                newmu = (mulow + muup) / 2
                gnew, xnew, gpnew = gmu(matrix, xp_mat, newmu)

                if gnew < k:
                    gup = gnew
                    xup = xnew
                    muup = newmu
                else:
                    glow = gnew
                    mulow = xnew
                    mulow = newmu
                numiter += 1
            numiter += 1

            if critmu.shape[0] != 0 and abs(mulow - muup) < abs(newmu) * precision and \
                    min(abs(newmu - critmu)) < precision * newmu:
                print('The objective function is discontinuous around mu^*.')
                xp = xnew
                gxpmu = gnew
        try:
            xp_mat = xnew
            # print(' xp_mat = xnew')
        except:
            scipy.io.savemat('matrix.mat', mdict={'arr': matrix})

        gxpmu = gnew

    # pdb.set_trace()

    # alpha = torch.zeros([1, matrix.shape[1]], device=device)
    # for i in range(r):
    #     alpha[0, i] = torch.matmul(xp_mat[:, i], pos_matrix[:, i])
    #     xp_mat[:, i] = alpha[:, i] * (matrix_sign[:, i] * xp_mat[:, i])
    
    alpha_mat = torch.matmul(xp_mat.T, pos_matrix)
    alpha = torch.diagonal(alpha_mat)
    xp_mat = alpha * (matrix_sign * xp_mat)

    return xp_mat


def load_matrix_debug():
    with open("./matrices/matrix_1.pkl", "rb") as fpA:  # Pickling
        matrix = pickle.load(fpA)
        # matrix = matrix.detach()
        matrix = torch.from_numpy(matrix)
    return matrix

## ********************************************************************************** ##

# matrix = load_matrix_debug()
# matrix = matrix.to(device)
# start_time = time.time()
# sps = 0.9
# precision = 1e-6
# linrat = 0.9
# X = groupedsparseproj(matrix, sps, precision=1e-6, linrat=0.9)
# print("--- %s seconds ---" % (time.time() - start_time))


# r = 100
# n = 10000
# k = 0

# ## Data Loacing
# # mu, sigma = 0, 1 # mean and standard deviation
# # x = np.random.normal(mu, sigma, (10000, 100)) * 10

# with open('matnew.pkl', 'rb') as fin:
#     x = pickle.load(fin)
# ## ****************************************

# xPos = np.abs(x)

# for i in range(r):
#     k = k + np.sqrt(n) / (np.sqrt(n) - 1)

# sp = sparsity(x)

# print("The Sparsity of input set of vectors: " + str(sp))

# xp_mat = groupedsparseproj(x, 0.8)

# spNew = sparsity(xp_mat)

# print("The Output Sparsity: " + str(spNew))

