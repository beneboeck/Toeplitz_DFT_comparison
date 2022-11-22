import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from utils_classic import *
import csv
import datetime
import os
import math
from eig_constraints import *
from os.path import exists

import K_constraints as K_c

K_dic = {
    '2':1,
    '3':0.48725,
    '4':0.32328,
    '5':0.24209,
    '6':0.19357,
    '7':0.16127,
    '8':0.13822,
    '10':0.10750,
    '16':0.06451,
    '32':0.03122,
    '64':0.01536,
    '100':0.00978,
    '128':0.007624,
    '256':0.0037975,
}

IMPROVING = True
N = 32
K = K_dic[str(N)] * np.ones(N)


now = datetime.datetime.now()
date = str(now)[:10]
time = str(now)[11:16]
time = time[:2] + '_' + time[3:]

overall_path = '../data/'
dir_path = '../data/time_' + time
os.mkdir (dir_path)


csv_file = open('../data/time_' + time + '/Autoregressive_DIM32_RUNS20_r0_6.txt','w')
#csv_file = open('./test.txt','w')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(['n_samples', 'mse_samplesCov', 'mse_OAS', 'mse_toep1', 'mse_toep2','mse_eig1','mse_eig2'])

rand_matrix = np.random.randn(N, N)
B_mask = np.tril(rand_matrix)
B_mask[B_mask != 0] = 1
C_mask = np.tril(rand_matrix, k=-1)
C_mask[C_mask != 0] = 1

def generating_Gamma(alpha):
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

def f(alpha):
    Gamma = generating_Gamma(alpha)
    #print('...')
    #print(Gamma)
    return - np.log(np.linalg.det(Gamma)) + np.trace(Gamma @ sCov)

def f_eig(eig):
    return - np.sum(np.log(eig)) + np.trace(U_Toeplitz @ np.diag(eig) @ U_Toeplitz.T @ sCov)

def f_eig2(eig):
    return - np.sum(np.log(eig)) + np.trace(U_Toeplitz2 @ np.diag(eig) @ U_Toeplitz2.T @ sCov)

constraints = K_c.generating_constraints(K,N)
constraints_Eig = generating_constraints_eig(N)

# MODEL
#AUTOREGRESSIVES MODEL GAUS
#r = 0.8
#C = np.zeros((N,N))
#for i in range(N):
#    for j in range(N):
#        C[i,j] = r**(np.abs(j-i))

#Brownian Motion (see shrinkage estimator original paper)
#H = 0.8
#C = np.zeros((N,N))
#for i in range(N):
#    for j in range(N):
#        C[i,j] = 0.5 * ( (np.abs(j-i) + 1)**(2*H) - 2 * np.abs(j-i)**(2*H) + np.abs((np.abs(j-i) - 1))**(2*H) )

# GEZOGEN VON GAMMA NACH DER FORMEL
init_values = np.zeros(N)
init_values[0] = np.random.uniform(low=1, high=20)
for n in range(1, N):
    init_values[n] = np.random.uniform(low=- K[n] * init_values[0] + 0.0001, high=K[n] * init_values[0] - 0.0001)
Gamma = generating_Gamma(init_values)
C = np.linalg.inv(Gamma)

#N_SAMPLES = [4,8,10,16,32,64,100]
N_SAMPLES = [16]
#N_SAMPLES = [16]
RUNS = 5
print(f'N: {N}, RUNS: {RUNS}')
MSE_sCov_n = []
MSE_toeplitz_n = []
MSE_OAS_n = []
MSE_toeplitz2_n = []
MSE_DFT_n = []

MSE_SVD_sCov_n = []
MSE_SVD_toeplitz_n = []
MSE_SVD_OAS_n = []
MSE_SVD_toeplitz2_n = []
MSE_SVD_DFT_n = []

MSE_toeplitz_eig_n = []
MSE_toeplitz2_eig_n = []
MSE_toeplitz3_eig_n = []

MSE_toeplitz_DFT_eig_n = []

#samples_list = np.zeros((RUNS,N_SAMPLES[0],N))

n_outliers = 0
skipping = False

# DFT matrix
DFT = np.zeros((N, N), dtype=np.cfloat)
for m in range(N):
    for n in range(N):
        DFT[m, n] = 1 / np.sqrt(N) * np.exp(-1j * 2 * math.pi * (m * n) / N)

#SVD echte Cov

U,S,VH = np.linalg.svd(C)
tot_e = np.sum(S**2)
boundary = 0.9 * tot_e
n_eig = 0
e = 0
for i in range(N):
    if e < boundary:
        n_eig += 1
        e += S[i]**2
    else:
        n_eig -= 1
        e -= S[i-1]**2
        break


for n_samples in N_SAMPLES:

    MSE_sCov = []
    MSE_toeplitz = []
    MSE_OAS = []
    MSE_toeplitz2 = []
    MSE_DFT = []

    MSE_SVD_sCov = []
    MSE_SVD_toeplitz = []
    MSE_SVD_OAS = []
    MSE_SVD_toeplitz2 = []
    MSE_SVD_DFT = []

    MSE_toeplitz_eig = []
    MSE_toeplitz2_eig = []
    MSE_toeplitz3_eig = []

    MSE_toeplitz_DFT_eig = []


    for run in range(RUNS):
        K = K_dic[str(N)] * np.ones(N)
        if run%5 == 0:
            print(f'run {run}')
        samples = np.random.multivariate_normal(np.zeros(N),C,n_samples)
        #samples_list[run,:,:] = samples

        #first comparison - sample Cov
        sCov = 1/n_samples * (samples.T @ samples)

        #second comparison - oracle approximating Shrinkage Estimator
        F = np.trace(sCov)/N * np.eye(N)
        rho = min(((1 - 2/N) * np.trace(sCov @ sCov) + np.trace(sCov)**2)/((n_samples + 1 - 2/N) * (np.trace(sCov @ sCov) - np.trace(sCov)**2/N)),1)
        OAS_C = (1 - rho) * sCov + rho * F

        #third comparison - DFT-estimator
        DFT_Cov = np.conjugate(DFT).T @ np.diag(np.diag(DFT @ sCov @ np.conjugate(DFT).T)) @ DFT
        DFT_Cov = np.real(DFT_Cov)



        # my method
        init_values = np.zeros(N)
        init_values[0] = np.random.uniform(low=1, high=20)
        for n in range(1,N):
            init_values[n] = np.random.uniform(low = - K[n] * init_values[0] + 0.0001, high = K[n] * init_values[0] - 0.0001)
        result = optimize.minimize(f, init_values, method="SLSQP",constraints=constraints)
        Gamma_est = generating_Gamma(result.x)

        if np.sum((np.linalg.inv(Gamma_est) - C)**2) > 1000:
            n_outliers += 1
            skipping = True
            w,v = np.linalg.eigh(Gamma_est)
            print(w)
            print(run)
            print(np.linalg.det(np.linalg.inv(Gamma_est)))
            print(np.linalg.det(Gamma_est))
            print(result.x)

        if skipping == False:
            MSE_sCov.append(np.sum((sCov - C) ** 2))
            MSE_toeplitz.append(np.sum((np.linalg.inv(Gamma_est) - C) ** 2))
            MSE_OAS.append(np.sum((OAS_C - C)**2))
            MSE_DFT.append(np.sum((DFT_Cov - C) ** 2))

            U_sCov,_,_ = np.linalg.svd(sCov)
            U_toeplitz,S_toeplitz,_ = np.linalg.svd(Gamma_est)
            U_OAS,_,_ = np.linalg.svd(OAS_C)

            U_Toeplitz = U_toeplitz
            U_DFT,_,_ = np.linalg.svd(DFT_Cov)

            mse_scov = []
            mse_toeplitz = []
            mse_oas = []
            mse_dft = []

            for eig in range(int(n_eig)):

                mse_scov.append(np.min([np.sum(np.abs(-U[:,eig][:,None] - U_sCov)**2,axis=0),np.sum(np.abs(U[:,eig][:,None] - U_sCov)**2,axis=0)]))
                mse_toeplitz.append(np.min([np.sum(np.abs(-U[:, eig][:,None] - U_toeplitz) ** 2,axis=0),np.sum(np.abs(U[:, eig][:,None] - U_toeplitz)**2,axis=0)]))
                mse_oas.append(np.min([np.sum(np.abs(-U[:, eig][:,None] - U_OAS) ** 2,axis=0),np.sum(np.abs(U[:, eig][:,None] - U_OAS) ** 2,axis=0)]))
                mse_dft.append(np.min([np.sum(np.abs(-U[:, eig][:, None] - U_DFT) ** 2, axis=0),np.sum(np.abs(U[:, eig][:, None] - U_DFT) ** 2, axis=0)]))

                arg_sCov = np.argmin([np.sum(np.abs(-U[:,eig][:,None] - U_sCov)**2,axis=0),np.sum(np.abs(U[:,eig][:,None] - U_sCov)**2,axis=0)]) % (N-eig)
                arg_toep = np.argmin([np.sum(np.abs(-U[:, eig][:,None] - U_toeplitz) ** 2,axis=0),np.sum(np.abs(U[:, eig][:,None] - U_toeplitz)**2,axis=0)]) % (N-eig)
                arg_oas = np.argmin([np.sum(np.abs(-U[:, eig][:,None] - U_OAS) ** 2,axis=0),np.sum(np.abs(U[:, eig][:,None] - U_OAS) ** 2,axis=0)]) % (N-eig)
                arg_dft = np.argmin([np.sum(np.abs(-U[:, eig][:, None] - U_DFT) ** 2, axis=0),np.sum(np.abs(U[:, eig][:, None] - U_DFT) ** 2, axis=0)]) % (N - eig)

                U_sCov = np.delete(U_sCov,arg_sCov,1)
                U_toeplitz = np.delete(U_toeplitz, arg_toep, 1)
                U_OAS = np.delete(U_OAS, arg_oas, 1)
                U_DFT = np.delete(U_DFT, arg_dft, 1)

            mse_scov = np.mean(mse_scov)
            mse_toeplitz = np.mean(mse_toeplitz)
            mse_oas = np.mean(mse_oas)
            mse_dft = np.mean(mse_dft)

            result_eig1 = optimize.minimize(f_eig, S_toeplitz, method="SLSQP", constraints=constraints_Eig)
            Gamma_est_eig = U_Toeplitz @ np.diag(result_eig1.x) @ U_Toeplitz.T

            MSE_toeplitz_eig.append(np.sum((np.linalg.inv(Gamma_est_eig) - C) ** 2))

        #print(f'MSE  Toeplitz: {np.sum((np.linalg.inv(Gamma_est) - C)**2)}')


        if IMPROVING & (skipping == False):
            #print('start')
            K = adjustingK(K,f, result)
            constraints = K_c.generating_constraints(K, N)
            alpha_0 = result.x[0]
            idx_bounding = np.squeeze(np.where(np.abs(result.x[1:]) > 0.9 * K[1:] * alpha_0)).reshape(-1) + np.array(1)
            idx_interior = np.squeeze(np.where(np.abs(result.x[1:]) < 0.9 * K[1:] * alpha_0)).reshape(-1) + np.array(1)
            len_bounding = len(idx_bounding)
            len_interior = len(idx_interior)
            derivatives = optimize.approx_fprime(result.x, f, epsilon=10e-8)
            counter = 0
            while (len_interior > 0) & (len_bounding > 0):
                print(counter)
                counter += 1
                if counter == 200:
                    break
                #print(result.x[0])
                result2 = optimize.minimize(f, result.x, method="SLSQP", constraints=constraints)
                #print('RESULTS')
                #print(result.fun)
                #print(result2.fun)
                result = result2
                K = adjustingK(K,f, result)
                idx_bounding = np.squeeze(np.where(np.abs(result.x[1:]) > 0.9 * K[1:] * alpha_0)).reshape(-1) + np.array(1)
                idx_interior = np.squeeze(np.where(np.abs(result.x[1:]) < 0.9 * K[1:] * alpha_0)).reshape(-1) + np.array(1)
                len_bounding = len(idx_bounding)
                len_interior = len(idx_interior)
                derivatives = optimize.approx_fprime(result.x, f, epsilon=10e-8)

            Gamma_est2 = generating_Gamma(result.x)
            U_toeplitz2, S_toeplitz2, _ = np.linalg.svd(Gamma_est2)
            U_Toeplitz2 = U_toeplitz2
            mse_toeplitz2 = []
            for eig in range(int(n_eig)):
                mse_toeplitz2.append(np.min([np.min(np.sum(np.abs(-U[:, eig][:,None] - U_toeplitz2) ** 2,axis=0)),np.min(np.sum(np.abs(U[:, eig][:,None] - U_toeplitz2)**2,axis=0))]))
                arg_toep2 = np.argmin([np.sum(np.abs(-U[:, eig][:, None] - U_toeplitz2) ** 2, axis=0),np.sum(np.abs(U[:, eig][:, None] - U_toeplitz2) ** 2, axis=0)]) % (N - eig)
                U_toeplitz2 = np.delete(U_toeplitz2, arg_toep2, 1)
            mse_toeplitz2 = np.mean(mse_toeplitz2)

            result_eig2 = optimize.minimize(f_eig2, S_toeplitz2, method="SLSQP", constraints=constraints_Eig)
            Gamma_est_eig2 = U_Toeplitz2 @ np.diag(result_eig2.x) @ U_Toeplitz2.T

            DFT_Gamma_eig2 = np.conjugate(DFT).T @ np.diag(np.diag(DFT @ Gamma_est_eig2 @ np.conjugate(DFT).T)) @ DFT
            DFT_Gamma_eig2 = np.real(DFT_Gamma_eig2)

            MSE_toeplitz2_eig.append(np.sum((np.linalg.inv(Gamma_est_eig2) - C) ** 2))
            MSE_toeplitz_DFT_eig.append(np.sum((np.linalg.inv(DFT_Gamma_eig2) - C) ** 2))
            #result_eig3 = optimize.minimize(f_eig, S_toeplitz2, method="SLSQP", constraints=constraints_Eig)
            #Gamma_est_eig3 = U_Toeplitz2 @ np.diag(result_eig3.x) @ U_Toeplitz2.T

            #MSE_toeplitz3_eig.append(np.sum((np.linalg.inv(Gamma_est_eig3) - C) ** 2))

            MSE_SVD_sCov.append(mse_scov)
            MSE_SVD_toeplitz.append(mse_toeplitz)
            MSE_SVD_OAS.append(mse_oas)
            MSE_SVD_toeplitz2.append(mse_toeplitz2)
            MSE_SVD_DFT.append(mse_dft)

            MSE_toeplitz2.append(np.sum((np.linalg.inv(Gamma_est2) - C) ** 2))
            MSE_toeplitz2_eig.append(np.sum((np.linalg.inv(Gamma_est_eig2) - C) ** 2))
            #MSE_toeplitz3_eig.append(np.sum((np.linalg.inv(Gamma_est_eig3) - C) ** 2))



        skipping = False
    print(f'MSE of sCov and real Cov: {np.mean(MSE_sCov):.4f}')
    print(f'MSE of Toep and real Cov: {np.mean(MSE_toeplitz):.4f}')
    print(f'MSE of Toep2 and real Cov: {np.mean(MSE_toeplitz2):.4f}')
    print(f'MSE of OAS and real Cov: {np.mean(MSE_OAS):.4f}')
    print(f'MSE of DFT and real Cov: {np.mean(MSE_DFT):.4f}')
    print(f'MSE of ToepEig and real Cov: {np.mean(MSE_toeplitz_eig):.4f}')
    print(f'MSE of ToepEig2 and real Cov: {np.mean(MSE_toeplitz2_eig):.4f}')
    print(f'MSE of ToepEigDFT and real Cov: {np.mean(MSE_toeplitz_DFT_eig):.4f}')
    #print(f'MSE of ToepEig3 and real Cov: {np.mean(MSE_toeplitz3_eig):.4f}')

    print(f'\nMSE of SVD sCov and real Cov: {np.mean(MSE_SVD_sCov):.4f}')
    print(f'\nMSE of SVD DFT and real Cov: {np.mean(MSE_SVD_DFT):.4f}')
    print(f'MSE of SVD Toep and real Cov: {np.mean(MSE_SVD_toeplitz):.4f}')
    print(f'MSE of SVD Toep2 and real Cov: {np.mean(MSE_SVD_toeplitz2):.4f}')
    print(f'MSE of SVD OAS and real Cov: {np.mean(MSE_SVD_OAS):.4f}')
    print(f'Outliers: {n_outliers}')

    MSE_sCov_n.append(np.mean(MSE_sCov))
    MSE_toeplitz_n.append(np.mean(MSE_toeplitz))
    MSE_OAS_n.append(np.mean(MSE_OAS))
    MSE_DFT_n.append(np.mean(MSE_DFT))
    MSE_toeplitz2_n.append(np.mean(MSE_toeplitz2))
    MSE_toeplitz_eig_n.append(np.mean(MSE_toeplitz_eig))
    MSE_toeplitz2_eig_n.append(np.mean(MSE_toeplitz2_eig))
    #MSE_toeplitz3_eig_n.append(np.mean(MSE_toeplitz3_eig))
    MSE_toeplitz_DFT_eig_n.append(MSE_toeplitz_DFT_eig)

    MSE_SVD_sCov_n.append(np.mean(MSE_SVD_sCov))
    MSE_SVD_toeplitz_n.append(np.mean(MSE_SVD_toeplitz))
    MSE_SVD_OAS_n.append(np.mean(MSE_SVD_OAS))
    MSE_SVD_toeplitz2_n.append(np.mean(MSE_SVD_toeplitz2))
    MSE_SVD_DFT_n.append(np.mean(MSE_SVD_DFT))

MSE_sCov_n = np.array(MSE_sCov_n)
MSE_toeplitz_n = np.array(MSE_toeplitz_n)
MSE_OAS_n = np.array(MSE_OAS_n)
MSE_toeplitz2_n = np.array(MSE_toeplitz2_n)
MSE_toeplitz_eig_n = np.array(MSE_toeplitz_eig_n)
MSE_toeplitz2_eig_n = np.array(MSE_toeplitz2_eig_n)
MSE_DFT_n = np.array(MSE_DFT_n)
MSE_toeplitz_DFT_eig_n = np.array(MSE_toeplitz_DFT_eig_n)
#MSE_toeplitz3_eig_n = np.array(MSE_toeplitz3_eig_n)

MSE_SVD_sCov_n = np.array(MSE_SVD_sCov_n)
MSE_SVD_toeplitz_n = np.array(MSE_SVD_toeplitz_n)
MSE_SVD_OAS_n = np.array(MSE_SVD_OAS_n)
MSE_SVD_toeplitz2_n = np.array(MSE_SVD_toeplitz2_n)
MSE_SVD_DFT_n = np.array(MSE_SVD_DFT_n)

# csv_writer.writerow(N_SAMPLES)
# #csv_writer.writerow(MSE_SVD_sCov_n)
# #csv_writer.writerow(MSE_SVD_OAS_n)
# #csv_writer.writerow(MSE_SVD_toeplitz_n)
# #csv_writer.writerow(MSE_SVD_toeplitz2_n)
#
# #n_samples','underlying_process', 'mse_samplesCov', 'mse_OAS', 'mse_toep1', 'mse_toep2','mse_eig1','mse_eig2
# csv_writer.writerow(MSE_sCov_n)
# csv_writer.writerow(MSE_OAS_n)
# csv_writer.writerow(MSE_toeplitz_n)
# csv_writer.writerow(MSE_toeplitz2_n)
# csv_writer.writerow(MSE_toeplitz_eig_n)
# csv_writer.writerow(MSE_toeplitz2_eig_n)
# csv_file.close()
#
# np.save('../data/time_' + time + '\Cov_real',C)
# np.save('../data/time_' + time + '\samples',samples_list)

plt.plot(N_SAMPLES,MSE_sCov_n,label = 'sCov')
plt.plot(N_SAMPLES,MSE_toeplitz_n,label = 'Toeplitz')
plt.plot(N_SAMPLES,MSE_OAS_n,label = 'OAS')
plt.plot(N_SAMPLES,MSE_DFT_n,label = 'DFT')
plt.plot(N_SAMPLES,MSE_toeplitz2_n,label = 'Toeplitz2')
plt.plot(N_SAMPLES,MSE_toeplitz_eig_n,label = 'ToepEig')
plt.plot(N_SAMPLES,MSE_toeplitz2_eig_n,label = 'ToepEig2')
plt.legend()
plt.ylabel('MSE')
plt.xlabel('N_SAMPLES')
#plt.title(f'Dimension: {N}, r-value: {r}')
plt.show()

plt.plot(N_SAMPLES,MSE_SVD_sCov_n,label = 'sCov')
plt.plot(N_SAMPLES,MSE_SVD_toeplitz_n,label = 'Toeplitz')
plt.plot(N_SAMPLES,MSE_SVD_OAS_n,label = 'OAS')
plt.plot(N_SAMPLES,MSE_SVD_DFT_n,label = 'DFT')
plt.plot(N_SAMPLES,MSE_SVD_toeplitz2_n,label = 'Toeplitz2')
plt.legend()
plt.ylabel('MSE SVD')
plt.xlabel('N_SAMPLES')
#plt.title(f'Dimension: {N}, r-value: {r}')
plt.show()