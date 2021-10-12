import torch
import numpy as np
from numpy import linalg as LA
import pickle
# import scipy.io
import logging
import pdb
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')




### ===================================================================================================
def pad_input_dict(in_dict):
    """
    This function is for the case when the layers are flattened in a column and structured as
    a dictionary. Each columns stored in the dictionary is  padded at the end with zeros and a
    matrix is created.
    """
    ni_list = [x.shape[0] for x in in_dict.values()]
    max_rows = max(ni_list)

    matrix = torch.zeros(max_rows, len(in_dict), device=device)

    for ind in range(len(in_dict)):
        matrix[:ni_list[ind],ind] = in_dict[ind]
    return matrix, ni_list


def unpad_output_mat(out_mat, ni_list):
    out_dict = {}
    for ind in range(out_mat.shape[1]):
        out_dict[ind] = out_mat[:ni_list[ind],ind]
    return out_dict


def checkCritical(pos_matrix, precision=1e-6):
    max_elems = torch.max(pos_matrix, 0)[0]

    ind_crit_bool = (abs(pos_matrix - max_elems) < precision)
    crit_points = pos_matrix * ind_crit_bool

    num_crit_points = torch.sum(ind_crit_bool, dim=0)

    # Boolean of vector cols with non-trivial critical values
    crit_cols = torch.where(num_crit_points.float() > 1, torch.ones(pos_matrix.shape[1], device=device), \
                            torch.zeros(pos_matrix.shape[1], device=device))
    # getting non-trivial critical values
    critval_list = max_elems[crit_cols.bool()]
    critval_all_col = max_elems * crit_cols

    return critval_list, max_elems, critval_all_col


def gmu(p_matrix, xp_mat, mu=0, *args):
    ni_tensor, inv_mask = args

    vgmu = 0
    gradg = 0
    ni_tlist = ni_tensor.int()
    
    p_matrix = torch.abs(p_matrix)
    glist = []

#----------------------------------------------------------------------------------------
    # ni_tensor
    betai = 1 / (torch.sqrt(ni_tensor) - 1)
    xp_mat = p_matrix - (mu * betai)
    indtp = xp_mat > 0
    xp_mat.relu_()


    # outputs
    mnorm = torch.norm(xp_mat, dim=0)
    mnorm_inf = mnorm.clone()
    mnorm_inf[mnorm_inf == 0] = float("Inf")
    col_norm_mask = (mnorm > 0)

#----------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------
    # mat_mask =  (col_norm_mask.float().view(1,784) * torch.ones(300,1))
    # mat_mask = (col_norm_mask.float().view(1, matrix.shape[1]) * torch.ones(matrix.shape[0], 1))

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

#----------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------
    # vgmu calculation
    ## When indtp is not empty (the columns whose norm are not zero)
    xp_mat *= inv_mask 
    xp_mat[:, col_norm_mask] /= mnorm[col_norm_mask]

    ### When indtp IS empty (the columns whose norm ARE zero)
    # The Row Indices where maximum of that column occurs
    max_elem_rows = torch.argmax(p_matrix, dim=0)[~col_norm_mask] 
    
    xp_mat[max_elem_rows, ~col_norm_mask] = 1

    # vgmu computation
    vgmu_mat = betai * torch.sum(xp_mat, dim=0)
    vgmu = torch.sum(vgmu_mat)

    return vgmu, xp_mat, gradg


# --------------------------------------------------------------------------------------------------- #
# ------------------------------------- groupedsparseproj ------------------------------------------- #
# --------------------------------------------------------------------------------------------------- #
def GSP(input_data, sps, precision=1e-6, linrat=0.9):
    # sps = 0.9 ;  precision=1e-6; linrat=0.9
    """
    This function will produce a sparse matrix of the matrices in the input_data. This a padded 
    implementaion of the GSP, hence this version is for use when we want to concatenate matrices
    of uneven dimensions. This version will pad the remaining elements with zero to get a 
    rectangular matrix.

    Inputs:
    1. input_data: Can be a dictionary with each value containing a 1D flattened version of a torch.Tensor.
                   Can be a matrix already concatenated beforehand. In such a case, the mask is required
                   with ones in the place where the matrix block is present in the wall block.
    2. inv_mask: torch.Tensor mask of ones in the positions where the values are in the bigger padded mat.
    3. sps : Sparsity constraint between [0,1].
    """
    epsilon = 10e-15
    k = 0
    muup0 = 0

    if type(input_data) == dict:
        matrix, ni_list = pad_input_dict(input_data)
        ni_tensor = torch.tensor(ni_list, device=device, dtype=torch.float32)

        # --------------- Create Mask ---------------------
        inv_mask = torch.zeros(matrix.shape, device=device, dtype=torch.float32)
        for i in range(matrix.shape[1]):
            inv_mask[:ni_list[i],i] = torch.ones(ni_list[i])
        # -------------------------------------------------

    elif type(input_data) == torch.Tensor:
        matrix = input_data
        ni_tensor = inv_mask.sum(dim=0)
        ni_list = ni_tensor.tolist()
        print("Torch Tensor")


    r = matrix.shape[1]  # No of Columns
    critmu = torch.tensor([])
    critval_list = []

    vgmu = torch.zeros(1, device=device)

    # These operations were inside the loop, but doesn't need to be.
    matrix_sign = torch.sign(matrix)
    pos_matrix = matrix_sign * matrix
    ni = matrix.shape[0]

#----------------------------------------------------------------------------------------
    k = sum(np.sqrt(ni_list)/(np.sqrt(ni_list)-1))


    # check critical values of mu where g(mu) is discontinuous, that is,
    # where the two (or more) largest entries of x{i} are equal to one another.
    critical_val, max_xi, cval_all_col = checkCritical(pos_matrix)

    # print(f"Input Matrix Shape: {matrix.shape}")
    # print(f"Shape of ni_tensor: {ni_tensor.shape}")
    # print(f"Shape of max_xi: {max_xi.shape}")

    muup0 = max(max_xi * (torch.sqrt(ni_tensor) - 1))

    # cval_all_col was extracted for the sole reason that we can multiply the critical
    # values withe the appropriate column ni below. Hence, it preserves the column information
    # of where the critical values came from.
    critmu = cval_all_col * (torch.sqrt(ni_tensor) - 1) 
    critmu = critmu[critmu > 1e-6] # we only need the critival values here, not the zeros in col.

    k = k - r * sps

    # -------------------- gmu --------------------
    xp_mat = torch.zeros([pos_matrix.shape[0], pos_matrix.shape[1]]).to(device)
    # gmu_args = {'xp_mat':xp_mat, 'ni_tensor':ni_tensor}
    
    vgmu, xp_mat, gradg = gmu(pos_matrix, xp_mat, 0, ni_tensor, inv_mask)

#----------------------------------------------------------------------------------------
    # if vgmu < k or abs(vgmu-k) < 1e-15:
    if vgmu < k:     
        xp_mat = matrix
        gxpmu = vgmu
        numiter = 0
        print("************ vgmu < k: returning without optimization!! ************")
        return xp_mat, ni_list
    else:
        numiter = 0
        mulow = 0
        glow = vgmu
        muup = muup0
        # Initialization on mu using 0, it seems to work best because the
        # slope at zero is rather steep while it is gets flat for large mu
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
            gnew, xnew, gpnew = gmu(pos_matrix, xp_mat, newmu, ni_tensor, inv_mask)
                                
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
                gnew, xnew, gpnew = gmu(pos_matrix, xp_mat, newmu, ni_tensor, inv_mask)

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
                # print('The objective function is discontinuous around mu^*.')
                xp = xnew
                gxpmu = gnew
        
        # ----- While Loop Ends -----
        try:
            xp_mat = xnew
        except:
            # pdb.set_trace()
            var_dict = {}
            if type(input_data) == dict:
                var_dict['in_dict'] = input_data
            else:
                var_dict['in_mat'] = input_data
            var_dict['gnew'] = gnew
            var_dict['k'] = k
            var_dict['precision'] = precision
            var_dict['r'] = r
            var_dict['numiter'] = numiter
            with open('problem_matrix_dict.pickle', 'wb') as handle:
                pickle.dump(var_dict, handle)

        gxpmu = gnew

    # -------------------------------------------
    # We need the column to column dot product between the two matrices xp_mat and pos_matrix
    # Hence, we resort to Matrix- Multiplication and then extract the diagonal elements.
    # This is equivalent to the above.
    alpha = torch.diag(torch.matmul(xp_mat.T, pos_matrix))
    xp_mat = alpha * (xp_mat * matrix_sign)
    # -------------------------------------------
 
    return xp_mat, ni_list



# ------------------------------------------------------------------------------------
# def load_matrix_debug(mat_tuple, is_dict):
#     matrix_1, matrix_2, matrix_3, matrix_4 = mat_tuple
#     with open(matrix_1, "rb") as fpA:  # Pickling
#         matrix_1 = pickle.load(fpA)
#     with open(matrix_2, "rb") as fpA:  # Pickling
#         matrix_2 = pickle.load(fpA)
#     with open(matrix_3, "rb") as fpA:  # Pickling
#         matrix_3 = pickle.load(fpA)
#     with open(matrix_4, "rb") as fpA:  # Pickling
#         matrix_4 = pickle.load(fpA)

#     if is_dict == True:
#         matrix_1 = torch.from_numpy(matrix_1).view(-1)
#         matrix_2 = torch.from_numpy(matrix_2).view(-1)
#         matrix_3 = torch.from_numpy(matrix_3).view(-1)
#         matrix_4 = torch.from_numpy(matrix_4).view(-1)
#         matrix = {0:matrix_1, 1:matrix_2, 2:matrix_3, 3:matrix_4}
#     else:
#         matrix_1 = torch.from_numpy(matrix_1)
#         matrix_2 = torch.from_numpy(matrix_2)
#         matrix_3 = torch.from_numpy(matrix_3)
#         matrix_4 = torch.from_numpy(matrix_4)
#         matrix = {0:matrix_1, 1:matrix_2, 2:matrix_3, 3:matrix_4}

#     return matrix


# # # ## ********************************************************************************** ##
# mat_tuple = ("./matrices/matrix_1.pkl", "./matrices/matrix_2.pkl", "./matrices/matrix_3.pkl", \
#              "./matrices/matrix_4.pkl")
# in_dict = load_matrix_debug(mat_tuple, is_dict=True)
# # in_mat = load_matrix_debug(mat_tuple, is_dict=False)
# # ## ********************************************************************************** ##

# start_time = time.time()
# sps = 0.9
# precision = 1e-6
# linrat = 0.9
# X, ni_list = groupedsparseproj(in_dict, sps, precision=1e-6, linrat=0.9)
# print("--- %s seconds ---" % (time.time() - start_time))

# out_dict = unpad_output_mat(X, ni_list)
