import numpy as np
from scipy import optimize
from utils_classic import *
import K_constraints as K_c

def OAS_estimator(sCov,N,n_samples):
    F = np.trace(sCov) / N * np.eye(N)
    rho = min(((1 - 2 / N) * np.trace(sCov @ sCov) + np.trace(sCov) ** 2) / (
                (n_samples + 1 - 2 / N) * (np.trace(sCov @ sCov) - np.trace(sCov) ** 2 / N)), 1)
    OAS_C = (1 - rho) * sCov + rho * F
    return OAS_C

def DFT_estimator(sCov,DFT):
    DFT_Cov = np.conjugate(DFT).T @ np.diag(np.diag(DFT @ sCov @ np.conjugate(DFT).T)) @ DFT
    DFT_Cov = np.real(DFT_Cov)
    return DFT_Cov

def ToepCube_estimator(sCov,N,constraints,K,B_mask,C_mask):
    init_values = np.zeros(N)
    init_values[0] = np.random.uniform(low=1, high=20)
    for n in range(1, N):
        init_values[n] = np.random.uniform(low=- K[n] * init_values[0] + 0.0001, high=K[n] * init_values[0] - 0.0001)

   # norm = constraint_frob(init_values, sCov, B_mask, C_mask, N)

    result = optimize.minimize(f, init_values,args = (sCov, B_mask, C_mask, N), method="SLSQP",constraints=constraints)
    Gamma_est = generating_Gamma(result.x,B_mask,C_mask,N)
    return Gamma_est

def ToepConCube_estimator(sCov,N,constraints,K,B_mask,C_mask):
    init_values = np.zeros(N//2)
    init_values[0] = np.random.uniform(low=1, high=20)
    for n in range(1, N//2):
        init_values[n] = np.random.uniform(low=- K[n] * init_values[0] + 0.0001, high=K[n] * init_values[0] - 0.0001)
    result = optimize.minimize(f_con, init_values,args = (sCov, B_mask, C_mask, N), method="SLSQP",constraints=constraints)
    Gamma_est = generating_reduced_Gamma(result.x,B_mask,C_mask,N)
    return Gamma_est

def ToepCubeEig_estimator(U,constraints,S_toeplitz,sCov):
    result_eig1 = optimize.minimize(f_eig, S_toeplitz,args=(sCov,U), method="SLSQP", constraints=constraints)
    Gamma_est_eig = U @ np.diag(result_eig1.x) @ U.T
    return Gamma_est_eig

def ToepCuboid_estimator(N,constraints,K,sCov, B_mask, C_mask):
    init_values = np.zeros(N)
    init_values[0] = np.random.uniform(low=1, high=20)
    for n in range(1, N):
        init_values[n] = np.random.uniform(low=- K[n] * init_values[0] + 0.0001, high=K[n] * init_values[0] - 0.0001)
    result = optimize.minimize(f, init_values,args = (sCov, B_mask, C_mask, N), method="SLSQP", constraints=constraints)
    K = adjustingK(K,f, result,sCov, B_mask, C_mask, N)
    constraints = K_c.generating_constraints(K, N)
    alpha_0 = result.x[0]
    idx_bounding = np.squeeze(np.where(np.abs(result.x[1:]) > 0.9 * K[1:] * alpha_0)).reshape(-1) + np.array(1)
    idx_interior = np.squeeze(np.where(np.abs(result.x[1:]) < 0.9 * K[1:] * alpha_0)).reshape(-1) + np.array(1)
    len_bounding = len(idx_bounding)
    len_interior = len(idx_interior)
    counter = 0
    while (len_interior > 0) & (len_bounding > 0):
        print(counter)
        counter += 1
        if counter == 200:
            break
        result2 = optimize.minimize(f, result.x,args = (sCov, B_mask, C_mask, N), method="SLSQP", constraints=constraints)
        result = result2
        K = adjustingK(K,f, result,sCov, B_mask, C_mask, N)
        idx_bounding = np.squeeze(np.where(np.abs(result.x[1:]) > 0.9 * K[1:] * alpha_0)).reshape(-1) + np.array(1)
        idx_interior = np.squeeze(np.where(np.abs(result.x[1:]) < 0.9 * K[1:] * alpha_0)).reshape(-1) + np.array(1)
        len_bounding = len(idx_bounding)
        len_interior = len(idx_interior)

    Gamma_est2 = generating_Gamma(result.x,B_mask,C_mask,N)
    return Gamma_est2

def ToepConCuboid_estimator(N,constraints,K,sCov, B_mask, C_mask):
    init_values = np.zeros(N//2)
    init_values[0] = np.random.uniform(low=1, high=20)
    for n in range(1, N//2):
        init_values[n] = np.random.uniform(low=- K[n] * init_values[0] + 0.0001, high=K[n] * init_values[0] - 0.0001)
    result = optimize.minimize(f_con, init_values,args = (sCov, B_mask, C_mask, N), method="SLSQP", constraints=constraints)
    K = adjustingK(K,f_con, result,sCov, B_mask, C_mask, N)
    constraints = K_c.generating_constraints(K, N//2)
    alpha_0 = result.x[0]
    idx_bounding = np.squeeze(np.where(np.abs(result.x[1:]) > 0.9 * K[1:] * alpha_0)).reshape(-1) + np.array(1)
    idx_interior = np.squeeze(np.where(np.abs(result.x[1:]) < 0.9 * K[1:] * alpha_0)).reshape(-1) + np.array(1)
    len_bounding = len(idx_bounding)
    len_interior = len(idx_interior)
    counter = 0
    while (len_interior > 0) & (len_bounding > 0):
        print(counter)
        counter += 1
        if counter == 200:
            break
        result2 = optimize.minimize(f_con, result.x,args = (sCov, B_mask, C_mask, N), method="SLSQP", constraints=constraints)
        result = result2
        K = adjustingK(K,f_con, result,sCov, B_mask, C_mask, N)
        idx_bounding = np.squeeze(np.where(np.abs(result.x[1:]) > 0.9 * K[1:] * alpha_0)).reshape(-1) + np.array(1)
        idx_interior = np.squeeze(np.where(np.abs(result.x[1:]) < 0.9 * K[1:] * alpha_0)).reshape(-1) + np.array(1)
        len_bounding = len(idx_bounding)
        len_interior = len(idx_interior)

    Gamma_est2 = generating_reduced_Gamma(result.x,B_mask,C_mask,N)
    return Gamma_est2

def ToepFrob_estimator(sCov,N,constraints,K,B_mask,C_mask):
    init_values = np.zeros(N)
    init_values[0] = np.random.uniform(low=1, high=20)
    for n in range(1, N):
        init_values[n] = np.random.uniform(low=- K[n] * init_values[0] + 0.0001, high=K[n] * init_values[0] - 0.0001)
    result = optimize.minimize(f, init_values,args = (sCov, B_mask, C_mask, N), method="SLSQP",constraints=constraints)
    Gamma_est = generating_Gamma(result.x,B_mask,C_mask,N)
    return Gamma_est

def ToepConFrob_estimator(sCov,N,constraints,K,B_mask,C_mask):
    init_values = np.zeros(N//2)
    init_values[0] = np.random.uniform(low=1, high=20)
    for n in range(1, N//2):
        init_values[n] = np.random.uniform(low=- K[n] * init_values[0] + 0.0001, high=K[n] * init_values[0] - 0.0001)
    result = optimize.minimize(f_con, init_values,args = (sCov, B_mask, C_mask, N), method="SLSQP",constraints=constraints)
    Gamma_est = generating_reduced_Gamma(result.x,B_mask,C_mask,N)
    return Gamma_est

def ToepConCauchy_estimator(sCov,N,constraints,K,B_mask,C_mask):
    init_values = np.zeros(N//2)
    init_values[0] = np.random.uniform(low=1, high=20)
    for n in range(1, N//2):
        init_values[n] = np.random.uniform(low=- K[n] * init_values[0] + 0.0001, high=K[n] * init_values[0] - 0.0001)
    result = optimize.minimize(f_con, init_values,args = (sCov, B_mask, C_mask, N), method="SLSQP",constraints=constraints)
    Gamma_est = generating_reduced_Gamma(result.x,B_mask,C_mask,N)
    return Gamma_est

def ToepConSmoothMax_estimator(sCov,N,constraints,K,B_mask,C_mask):
    init_values = np.zeros(N//2)
    init_values[0] = np.random.uniform(low=1, high=20)
    for n in range(1, N//2):
        init_values[n] = np.random.uniform(low=- K[n] * init_values[0] + 0.0001, high=K[n] * init_values[0] - 0.0001)
    result = optimize.minimize(f_con, init_values,args = (sCov, B_mask, C_mask, N), method="SLSQP",constraints=constraints)
    Gamma_est = generating_reduced_Gamma(result.x,B_mask,C_mask,N)
    return Gamma_est

def StieltjesCon_estimator(sCov,N,constraints,K,B_mask,C_mask):
    init_values = np.zeros(N//2)
    init_values[0] = np.random.uniform(low=1, high=20)
    for n in range(1, N//2):
        init_values[n] = np.random.uniform(low=- init_values[0]/N, high=0)
    result = optimize.minimize(f_con, init_values,args = (sCov, B_mask, C_mask, N), method="SLSQP",constraints=constraints)
    Gamma_est = generating_reduced_Gamma(result.x,B_mask,C_mask,N)
    a = f_con(result.x,sCov, B_mask, C_mask, N)
    return Gamma_est

def Stieltjes_estimator(sCov,N,constraints,K,B_mask,C_mask):
    init_values = np.zeros(N)
    init_values[0] = np.random.uniform(low=1, high=20)
    for n in range(1, N):
        init_values[n] = np.random.uniform(low=- init_values[0] / N, high=0)
    result = optimize.minimize(f, init_values,args = (sCov, B_mask, C_mask, N), method="SLSQP",constraints=constraints)
    Gamma_est = generating_Gamma(result.x,B_mask,C_mask,N)
    return Gamma_est