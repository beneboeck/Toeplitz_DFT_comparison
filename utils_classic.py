import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

def b(idx,K_list):
    b_list = np.zeros(idx+1)
    b_list[0] = 1
    for i in range(1,idx+1):
        b_list[i] = np.sum(np.flip(np.array(K_list[1:i+1])) * b_list[:i])
    return b_list[idx]

def g(idx,K_list,b_list):
    g = 0
    for i in range(1,idx+1):
        g += K_list[len(K_list)-i] * b_list[idx - i]
    return g

def bound(K_list):
    bound = 0
    b_list = np.zeros(len(K_list))
    for i in range(len(K_list)):
        b_list[i] = b(i,K_list)
    for i in range(1,len(K_list)):
        bound += (len(K_list) - i) * g(i,K_list,b_list)**2
    return bound

def adjustingK(K,f,result):
    alpha_0 = result.x[0]
    idx_bounding = np.squeeze(np.where(np.abs(result.x[1:]) > 0.9 * K[1:] * alpha_0)).reshape(-1) + np.array(1)
    idx_interior = np.squeeze(np.where(np.abs(result.x[1:]) < 0.9 * K[1:] * alpha_0)).reshape(-1) + np.array(1)


    len_bounding = len(idx_bounding)
    len_interior = len(idx_interior)
    derivatives = optimize.approx_fprime(result.x, f, epsilon=10e-8)
    while (len_interior > 0) & (len_bounding > 0):
        max_bounding_idx = idx_bounding[np.argmax(np.abs(derivatives[idx_bounding]))]
        max_interior_idx = idx_interior[np.argmax(K[idx_interior] * alpha_0 - np.abs(result.x[idx_interior]))]

        K[max_interior_idx] = (0.7 * (K[max_interior_idx] * result.x[0] - np.abs(result.x[max_interior_idx])) + np.abs(result.x[max_interior_idx])) / result.x[0]
        K_test = K
        while bound(K_test) < 1:
            K_test[max_bounding_idx] = 1.01 * K_test[max_bounding_idx]
        K_test[max_bounding_idx] = 1 / 1.01 * K_test[max_bounding_idx]
        K = K_test

        idx_bounding = np.delete(idx_bounding,np.where(idx_bounding == max_bounding_idx))
        idx_interior = np.delete(idx_interior,np.where(idx_interior ==  max_interior_idx))
        len_bounding = len(idx_bounding)
        len_interior = len(idx_interior)

    return K

