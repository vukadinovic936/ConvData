import numpy as np
import numpy as np
from scipy.linalg import toeplitz
import torch
from torch import nn,optim
import torch.nn.functional as F

def conv_to_fc(conv, inp_shape=(28,28), convtype="full"):

    K=np.flip(conv.weight.detach().numpy().squeeze())

    # number of columns and rows of the input 
    I_row_num, I_col_num = inp_shape


    # number of columns and rows of the filter
    if len(K.shape)==1:
        K=K.reshape((1,-1))
    K_row_num, K_col_num = K.shape

    #  calculate the output dimensions
    output_row_num = I_row_num + K_row_num - 1
    output_col_num = I_col_num + K_col_num - 1

    # zero pad the filter
    K_zero_padded = np.pad(K, ((output_row_num - K_row_num, 0),
                               (0, output_col_num - K_col_num)),
                            'constant', constant_values=0)

    # use each row of the zero-padded F to creat a toeplitz matrix. 
    #  Number of columns in this matrices are same as numbe of columns of input signal
    toeplitz_list = []
    for i in range(K_zero_padded.shape[0]-1, -1, -1): # iterate from last row to the first row
        c = K_zero_padded[i, :] # i th row of the F 
        r = np.r_[c[0], np.zeros(I_col_num-1)] # first row for the toeplitz fuction should be defined otherwise
                                                            # the result is wrong
        toeplitz_m = toeplitz(c,r) # this function is in scipy.linalg library
        toeplitz_list.append(toeplitz_m)

        # doubly blocked toeplitz indices: 
    #  this matrix defines which toeplitz matrix from toeplitz_list goes to which part of the doubly blocked
    c = range(1, K_zero_padded.shape[0]+1)
    r = np.r_[c[0], np.zeros(I_row_num-1, dtype=int)]
    doubly_indices = toeplitz(c, r)

    ## creat doubly blocked matrix with zero values
    toeplitz_shape = toeplitz_list[0].shape # shape of one toeplitz matrix
    h = toeplitz_shape[0]*doubly_indices.shape[0]
    w = toeplitz_shape[1]*doubly_indices.shape[1]
    doubly_blocked_shape = [h, w]
    doubly_blocked = np.zeros(doubly_blocked_shape)

    # instead of vectorizing I, we can flip the indices
    # we are keeping I constant, which is easier for torch implementation
    doubly_indices = np.flip(doubly_indices,axis=1)

    doubly_indices = np.flip(doubly_indices,axis=0)

    # tile toeplitz matrices for each row in the doubly blocked matrix
    b_h, b_w = toeplitz_shape # hight and withs of each block
    for i in range(doubly_indices.shape[0]):
        for j in range(doubly_indices.shape[1]):
            start_i = i * b_h
            start_j = j * b_w
            end_i = start_i + b_h
            end_j = start_j + b_w
            doubly_blocked[start_i: end_i, start_j:end_j] = toeplitz_list[doubly_indices[i,j]-1]
    
    if convtype=='valid':
        valid_row_num = I_row_num - K_row_num + 1
        valid_col_num = I_col_num - K_col_num + 1
        doubly_blocked=doubly_blocked[submatrix_indices(output_row_num,output_col_num,valid_row_num,valid_col_num)]

    W = doubly_blocked 
    b = conv.bias[0].detach().numpy() 
    fc = nn.Linear(W.shape[1], W.shape[0])
    with torch.no_grad():
        fc.weight = nn.Parameter(torch.from_numpy(W.astype('float32')))
        fc.bias = nn.Parameter(torch.from_numpy(b.astype('float32')))
    return fc

def submatrix_indices(large_matrix_rows,large_matrix_cols, n, m):


    # Check if large_matrix is large enough for the sub-matrix
    if large_matrix_rows < n or large_matrix_cols < m:
        raise ValueError("The large matrix is not large enough for the desired sub-matrix.")

    # Calculate the starting indices for the sub-matrix
    start_row = (large_matrix_rows - n) // 2
    start_col = (large_matrix_cols - m) // 2

    # Extract the sub-matrix
    x = np.arange(start_row, start_row+n)
    y = np.arange(start_col, start_col+m)
    xv, yv = np.meshgrid(x, y)
    xv=xv.flatten()
    yv=yv.flatten()

    w_indices=[]
    for g in range(len(xv)):
        i=xv[g]
        j=yv[g]

        w_indices.append(i*large_matrix_cols+j)
    return w_indices