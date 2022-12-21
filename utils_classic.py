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

def adjustingK(K,f,result,sCov, B_mask, C_mask, N):
    alpha_0 = result.x[0]
    idx_bounding = np.squeeze(np.where(np.abs(result.x[1:]) > 0.9 * K[1:] * alpha_0)).reshape(-1) + np.array(1)
    idx_interior = np.squeeze(np.where(np.abs(result.x[1:]) < 0.9 * K[1:] * alpha_0)).reshape(-1) + np.array(1)


    len_bounding = len(idx_bounding)
    len_interior = len(idx_interior)
    args = (sCov, B_mask, C_mask, N)
    derivatives = optimize.approx_fprime(result.x, f, 10e-8,*args)
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

def generating_Gamma(alpha,B_mask,C_mask,N):
    alpha_prime = np.concatenate((np.array([0]), np.flip(np.array(alpha[1:]))))
    values = np.concatenate((alpha, np.flip(np.array(alpha[1:]))))
    i, j = np.ones((N, N)).nonzero()
    values = values[j - i].reshape(N, N)
    B = values * B_mask
    values_prime = np.concatenate((alpha_prime, np.flip(np.array(alpha_prime[1:]))))
    i, j = np.ones((N, N)).nonzero()
    values_prime2 = values_prime[j - i].reshape(N, N)
    C = np.conj(values_prime2 * C_mask)
    alpha_0 = B[0, 0]
    Gamma = 1 / alpha_0[None, None] * (np.matmul(B, np.conj(B).T) - np.matmul(C, np.conj(C).T))
    return Gamma

def generating_reduced_Gamma(alpha,B_mask,C_mask,N):
    z = np.zeros(N//2)
    alpha_full = np.concatenate((alpha,z))
    alpha_prime = np.concatenate((np.array([0]), np.flip(np.array(alpha_full[1:]))))
    values = np.concatenate((alpha_full, np.flip(np.array(alpha_full[1:]))))
    i, j = np.ones((N, N)).nonzero()
    values = values[j - i].reshape(N, N)
    B = values * B_mask
    values_prime = np.concatenate((alpha_prime, np.flip(np.array(alpha_prime[1:]))))
    i, j = np.ones((N, N)).nonzero()
    values_prime2 = values_prime[j - i].reshape(N, N)
    C = np.conj(values_prime2 * C_mask)
    alpha_0 = B[0, 0]
    Gamma = 1 / alpha_0[None, None] * (np.matmul(B, np.conj(B).T) - np.matmul(C, np.conj(C).T))
    return Gamma

def f(alpha,*args):
    sCov, B_mask, C_mask, N = args
    Gamma = generating_Gamma(alpha,B_mask,C_mask,N)
    return - np.log(np.linalg.det(Gamma)) + np.trace(Gamma @ sCov)

def f_con(alpha,*args):
    sCov, B_mask, C_mask, N = args
    Gamma = generating_reduced_Gamma(alpha,B_mask,C_mask,N)
    return - np.log(np.linalg.det(Gamma)) + np.trace(Gamma @ sCov)

def f_eig(eig,*args):
    sCov, U = args
    return - np.sum(np.log(eig)) + np.trace(U @ np.diag(eig) @ U.T @ sCov)

def SVD_comparison(N,C,C_est):
    U, S, VH = np.linalg.svd(C)
    tot_e = np.sum(S ** 2)
    boundary = 0.9 * tot_e
    n_eig = 0
    e = 0
    for i in range(N):
        if e < boundary:
            n_eig += 1
            e += S[i] ** 2
        else:
            n_eig -= 1
            e -= S[i - 1] ** 2
            break

    U_est, _, _ = np.linalg.svd(C_est)

    mse_est = []

    for eig in range(int(n_eig)):
        mse_est.append(np.min([np.sum(np.abs(-U[:, eig][:, None] - U_est) ** 2, axis=0),np.sum(np.abs(U[:, eig][:, None] - U_est) ** 2, axis=0)]))
        arg_est = np.argmin([np.sum(np.abs(-U[:, eig][:, None] - U_est) ** 2, axis=0),np.sum(np.abs(U[:, eig][:, None] - U_est) ** 2, axis=0)]) % (N - eig)
        U_est = np.delete(U_est, arg_est, 1)


    mse_est = np.mean(mse_est)
    return mse_est