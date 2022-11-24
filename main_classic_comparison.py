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
from estimators import *
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
N = 16
K = K_dic[str(N)] * np.ones(N)
K_reduced = K_dic[str(N//2)] * np.ones(N//2)


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

constraints = K_c.generating_constraints(K,N)
constraints_reduced = K_c.generating_constraints(K_reduced,N//2)
constraints_Eig = generating_constraints_eig(N)

# MODEL
#AUTOREGRESSIVES MODEL GAUS
#r = 0.4
#C = np.zeros((N,N))
#for i in range(N):
#    for j in range(N):
#        C[i,j] = r**(np.abs(j-i))

#Brownian Motion (see shrinkage estimator original paper)
H = 0.8
C = np.zeros((N,N))
for i in range(N):
    for j in range(N):
        C[i,j] = 0.5 * ( (np.abs(j-i) + 1)**(2*H) - 2 * np.abs(j-i)**(2*H) + np.abs((np.abs(j-i) - 1))**(2*H) )

# GEZOGEN VON GAMMA NACH DER FORMEL
#init_values = np.zeros(N)
#init_values[0] = np.random.uniform(low=1, high=20)
#for n in range(1, N):
#    init_values[n] = np.random.uniform(low=- K[n] * init_values[0] + 0.0001, high=K[n] * init_values[0] - 0.0001)
#Gamma = generating_Gamma(init_values,B_mask,C_mask,N)
#C = np.linalg.inv(Gamma)

#N_SAMPLES = [4,8,10,16,32,64,100]
N_SAMPLES = [8]
RUNS = 10
print(f'N: {N}, RUNS: {RUNS}')
MSE_sCov_n = []
MSE_ToepCube_n = []
MSE_OAS_n = []
MSE_ToepCuboid_n = []
MSE_DFT_n = []
MSE_ToepCubeCon_n = []
MSE_ToepCuboidCon_n = []
MSE_ToepFrob_n = []

MSE_SVD_sCov_n = []
MSE_SVD_ToepCube_n = []
MSE_SVD_OAS_n = []
MSE_SVD_ToepCuboid_n = []
MSE_SVD_DFT_n = []
MSE_SVD_ToepCubeCon_n = []
MSE_SVD_ToepCuboidCon_n = []

MSE_ToepCubeEig_n = []
MSE_ToepCuboidEig_n = []
MSE_ToepCubeConEig_n = []
MSE_ToepCuboidConEig_n = []

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
    MSE_ToepCube = []
    MSE_OAS = []
    MSE_ToepCuboid = []
    MSE_DFT = []
    MSE_ToepCubeCon = []
    MSE_ToepCuboidCon = []
    MSE_ToepFrob = []

    MSE_SVD_sCov = []
    MSE_SVD_ToepCube = []
    MSE_SVD_OAS = []
    MSE_SVD_ToepCuboid = []
    MSE_SVD_DFT = []
    MSE_SVD_ToepCubeCon = []
    MSE_SVD_ToepCuboidCon = []

    MSE_ToepCubeEig = []
    MSE_ToepCuboidEig = []
    MSE_ToepCubeConEig = []
    MSE_ToepCuboidConEig = []


    for run in range(RUNS):
        K = K_dic[str(N)] * np.ones(N)
        if run%5 == 0:
            print(f'run {run}')
        samples = np.random.multivariate_normal(np.zeros(N),C,n_samples)

        #first comparison - sample Cov
        sCov = 1/n_samples * (samples.T @ samples)
        U_sCov, _, _ = np.linalg.svd(sCov)

        # some further constraint
        c_frob_full = {
            'fun': constraint_frob,
            'type': 'ineq',
            'args': (sCov, B_mask, C_mask, N)
        }
        c_cauchy_reduced = {
            'fun': constraint_reduced_cauchy,
            'type': 'ineq',
            'args': (sCov, B_mask, C_mask, N)
        }

        #second comparison - oracle approximating Shrinkage Estimator
        OAS_C = OAS_estimator(sCov,N,n_samples)
        U_OAS, _, _ = np.linalg.svd(OAS_C)

        #third comparison - DFT-estimator
        DFT_Cov = DFT_estimator(sCov,DFT)
        U_DFT, _, _ = np.linalg.svd(DFT_Cov)

        # my method (1) ToepCube Estimator
        ToepCube_est = ToepCube_estimator(sCov,N,constraints,K,B_mask,C_mask)
        U_ToepCube, S_ToepCube, _ = np.linalg.svd(ToepCube_est)

        # my method (2) ToepCubeEig Estimator
        ToepCubeEig_est = ToepCubeEig_estimator(U_ToepCube, constraints_Eig, S_ToepCube, sCov)

        if IMPROVING:
            # my method (3) ToepCuboid Estimator
            ToepCuboid_est = ToepCuboid_estimator(N, constraints, K, sCov, B_mask, C_mask)
            U_ToepCuboid, S_ToepCuboid, _ = np.linalg.svd(ToepCuboid_est)

            # my method (4) ToepCuboidEig Estimator
            ToepCuboidEig_est = ToepCubeEig_estimator(U_ToepCuboid, constraints_Eig, S_ToepCuboid, sCov)

            # my method (7) ToepConCuboid Estimator
            ToepConCuboid_est = ToepConCuboid_estimator(N, constraints_reduced, K_reduced, sCov, B_mask, C_mask)
            U_ToepConCuboid, S_ToepConCuboid, _ = np.linalg.svd(ToepConCuboid_est)

            # my method (8) ToepConCuboidEig Estimator
            ToepCuboidConEig_est = ToepCubeEig_estimator(U_ToepConCuboid, constraints_Eig, S_ToepConCuboid, sCov)

        # my method (5) ToepConCube Estimator
        K_reduced = K_dic[str(N//2)] * np.ones(N//2)
        ToepConCube_est = ToepConCube_estimator(sCov, N, constraints_reduced, K_reduced, B_mask, C_mask)
        U_ToepConCube, S_ToepConCube, _ = np.linalg.svd(ToepConCube_est)

        # my method (6) ToepConCubeEig Estimator
        ToepConCubeEig_est = ToepCubeEig_estimator(U_ToepConCube, constraints_Eig, S_ToepConCube, sCov)

        # my method (9) ToepFrob Estimator
        ToepFrob_est = ToepFrob_estimator(sCov,N,[con_alpha0,c_frob_full],K,B_mask,C_mask)
        U_ToepFrob, S_ToepFrob, _ = np.linalg.svd(ToepFrob_est)

        # my method (10) ToepConCauchy Estimator


        MSE_sCov.append(np.sum((sCov - C) ** 2))
        MSE_ToepCube.append(np.sum((np.linalg.inv(ToepCube_est) - C) ** 2))
        MSE_OAS.append(np.sum((OAS_C - C) ** 2))
        MSE_DFT.append(np.sum((DFT_Cov - C) ** 2))
        MSE_ToepCubeEig.append(np.sum((np.linalg.inv(ToepCubeEig_est) - C) ** 2))
        MSE_ToepCubeCon.append(np.sum((np.linalg.inv(ToepConCube_est) - C) ** 2))
        MSE_ToepCubeConEig.append(np.sum((np.linalg.inv(ToepConCubeEig_est) - C) ** 2))
        MSE_ToepFrob.append(np.sum((np.linalg.inv(ToepFrob_est) - C) ** 2))

        if IMPROVING:
            MSE_ToepCuboid.append(np.sum((np.linalg.inv(ToepCuboid_est) - C) ** 2))
            MSE_ToepCuboidEig.append(np.sum((np.linalg.inv(ToepCuboidEig_est) - C) ** 2))
            MSE_ToepCuboidCon.append(np.sum((np.linalg.inv(ToepConCuboid_est) - C) ** 2))
            MSE_ToepCuboidConEig.append(np.sum((np.linalg.inv(ToepCuboidConEig_est) - C) ** 2))

        mse_scov = []
        mse_ToepCube = []
        mse_oas = []
        mse_dft = []
        mse_ToepCuboid = []
        mse_ToepCubeCon = []
        mse_ToepCuboidCon = []
        mse_ToepFrob = []

        for eig in range(int(n_eig)):
            mse_scov.append(np.min([np.sum(np.abs(-U[:,eig][:,None] - U_sCov)**2,axis=0),np.sum(np.abs(U[:,eig][:,None] - U_sCov)**2,axis=0)]))
            mse_ToepCube.append(np.min([np.sum(np.abs(-U[:, eig][:,None] - U_ToepCube) ** 2,axis=0),np.sum(np.abs(U[:, eig][:,None] - U_ToepCube)**2,axis=0)]))
            mse_oas.append(np.min([np.sum(np.abs(-U[:, eig][:,None] - U_OAS) ** 2,axis=0),np.sum(np.abs(U[:, eig][:,None] - U_OAS) ** 2,axis=0)]))
            mse_dft.append(np.min([np.sum(np.abs(-U[:, eig][:, None] - U_DFT) ** 2, axis=0),np.sum(np.abs(U[:, eig][:, None] - U_DFT) ** 2, axis=0)]))
            mse_ToepCubeCon.append(np.min([np.sum(np.abs(-U[:, eig][:, None] - U_ToepConCube) ** 2, axis=0),np.sum(np.abs(U[:, eig][:, None] - U_ToepConCube) ** 2, axis=0)]))
            mse_ToepFrob.append(np.min([np.sum(np.abs(-U[:, eig][:, None] - U_ToepFrob) ** 2, axis=0),np.sum(np.abs(U[:, eig][:, None] - U_ToepFrob) ** 2, axis=0)]))

            arg_sCov = np.argmin([np.sum(np.abs(-U[:,eig][:,None] - U_sCov)**2,axis=0),np.sum(np.abs(U[:,eig][:,None] - U_sCov)**2,axis=0)]) % (N-eig)
            arg_toep = np.argmin([np.sum(np.abs(-U[:, eig][:,None] - U_ToepCube) ** 2,axis=0),np.sum(np.abs(U[:, eig][:,None] - U_ToepCube)**2,axis=0)]) % (N-eig)
            arg_oas = np.argmin([np.sum(np.abs(-U[:, eig][:,None] - U_OAS) ** 2,axis=0),np.sum(np.abs(U[:, eig][:,None] - U_OAS) ** 2,axis=0)]) % (N-eig)
            arg_dft = np.argmin([np.sum(np.abs(-U[:, eig][:, None] - U_DFT) ** 2, axis=0),np.sum(np.abs(U[:, eig][:, None] - U_DFT) ** 2, axis=0)]) % (N - eig)
            arg_toep3 = np.argmin([np.sum(np.abs(-U[:, eig][:, None] - U_ToepConCube) ** 2, axis=0),np.sum(np.abs(U[:, eig][:, None] - U_ToepConCube) ** 2, axis=0)]) % (N - eig)
            arg_frob = np.argmin([np.sum(np.abs(-U[:, eig][:, None] - U_ToepFrob) ** 2, axis=0),np.sum(np.abs(U[:, eig][:, None] - U_ToepFrob) ** 2, axis=0)]) % (N - eig)

            U_sCov = np.delete(U_sCov,arg_sCov,1)
            U_ToepCube = np.delete(U_ToepCube, arg_toep, 1)
            U_OAS = np.delete(U_OAS, arg_oas, 1)
            U_DFT = np.delete(U_DFT, arg_dft, 1)
            U_ToepConCube = np.delete(U_ToepConCube, arg_toep3, 1)
            U_ToepFrob = np.delete(U_ToepFrob, arg_frob, 1)

            if IMPROVING:
                mse_ToepCuboid.append(np.min([np.min(np.sum(np.abs(-U[:, eig][:, None] - U_ToepCuboid) ** 2, axis=0)),np.min(np.sum(np.abs(U[:, eig][:, None] - U_ToepCuboid) ** 2, axis=0))]))
                arg_toep2 = np.argmin([np.sum(np.abs(-U[:, eig][:, None] - U_ToepCuboid) ** 2, axis=0),np.sum(np.abs(U[:, eig][:, None] - U_ToepCuboid) ** 2, axis=0)]) % (N - eig)
                U_ToepCuboid = np.delete(U_ToepCuboid, arg_toep2, 1)

                mse_ToepCuboidCon.append(np.min([np.min(np.sum(np.abs(-U[:, eig][:, None] - U_ToepConCuboid) ** 2, axis=0)),np.min(np.sum(np.abs(U[:, eig][:, None] - U_ToepConCuboid) ** 2, axis=0))]))
                arg_toep2 = np.argmin([np.sum(np.abs(-U[:, eig][:, None] - U_ToepConCuboid) ** 2, axis=0),np.sum(np.abs(U[:, eig][:, None] - U_ToepConCuboid) ** 2, axis=0)]) % (N - eig)
                U_ToepConCuboid = np.delete(U_ToepConCuboid, arg_toep2, 1)

        mse_scov = np.mean(mse_scov)
        mse_ToepCube = np.mean(mse_ToepCube)
        mse_oas = np.mean(mse_oas)
        mse_dft = np.mean(mse_dft)
        mse_ToepCubeCon = np.mean(mse_ToepCubeCon)

        if IMPROVING:
            mse_ToepCuboid = np.mean(mse_ToepCuboid)
            mse_ToepCuboidCon = np.mean(mse_ToepCuboidCon)

        MSE_SVD_sCov.append(mse_scov)
        MSE_SVD_ToepCube.append(mse_ToepCube)
        MSE_SVD_OAS.append(mse_oas)
        MSE_SVD_DFT.append(mse_dft)
        MSE_SVD_ToepCubeCon.append(mse_ToepCubeCon)

        if IMPROVING:
            MSE_SVD_ToepCuboid.append(mse_ToepCuboid)
            MSE_SVD_ToepCuboidCon.append(mse_ToepCuboidCon)


    print(f'MSE of sCov and real Cov: {np.mean(MSE_sCov):.4f}')
    print(f'MSE of OAS and real Cov: {np.mean(MSE_OAS):.4f}')
    print(f'MSE of DFT and real Cov: {np.mean(MSE_DFT):.4f}')
    print(f'MSE of ToepCube and real Cov: {np.mean(MSE_ToepCube):.4f}')
    print(f'MSE of ToepCuboid and real Cov: {np.mean(MSE_ToepCuboid):.4f}')
    print(f'MSE of ToepCubeCon and real Cov: {np.mean(MSE_ToepCubeCon):.4f}')
    print(f'MSE of ToepConCuboid and real Cov: {np.mean(MSE_ToepCuboidCon):.4f}')
    print(f'MSE of ToepFrob and real Cov: {np.mean(MSE_ToepFrob):.4f}')

    print(f'MSE of ToepCubeEig and real Cov: {np.mean(MSE_ToepCubeEig):.4f}')
    print(f'MSE of ToepCuboidEig and real Cov: {np.mean(MSE_ToepCuboidEig):.4f}')
    print(f'MSE of ToepCubeConEig and real Cov: {np.mean(MSE_ToepCubeConEig):.4f}')
    print(f'MSE of ToepCuboidConEig and real Cov: {np.mean(MSE_ToepCuboidConEig):.4f}')

    print(f'\nMSE of SVD sCov and real Cov: {np.mean(MSE_SVD_sCov):.4f}')
    print(f'\nMSE of SVD DFT and real Cov: {np.mean(MSE_SVD_DFT):.4f}')
    print(f'MSE of SVD ToepCube and real Cov: {np.mean(MSE_SVD_ToepCube):.4f}')
    print(f'MSE of SVD ToepCuboid and real Cov: {np.mean(MSE_SVD_ToepCuboid):.4f}')
    print(f'MSE of SVD OAS and real Cov: {np.mean(MSE_SVD_OAS):.4f}')
    print(f'MSE of SVD ToepCubeCon and real Cov: {np.mean(MSE_SVD_ToepCubeCon):.4f}')
    print(f'MSE of SVD ToepCuboidCon and real Cov: {np.mean(MSE_SVD_ToepCuboidCon):.4f}')
    print(f'Outliers: {n_outliers}')

    MSE_sCov_n.append(np.mean(MSE_sCov))
    MSE_OAS_n.append(np.mean(MSE_OAS))
    MSE_DFT_n.append(np.mean(MSE_DFT))
    MSE_ToepCube_n.append(np.mean(MSE_ToepCube))
    MSE_ToepCuboid_n.append(np.mean(MSE_ToepCuboid))
    MSE_ToepCubeEig_n.append(np.mean(MSE_ToepCubeEig))
    MSE_ToepCuboidEig_n.append(np.mean(MSE_ToepCuboidEig))
    MSE_ToepCubeCon_n.append(np.mean(MSE_ToepCubeCon))
    MSE_ToepCuboidCon_n.append(np.mean(MSE_ToepCuboidCon))
    MSE_ToepCubeConEig_n.append(np.mean(MSE_ToepCubeConEig))
    MSE_ToepCuboidConEig_n.append(np.mean(MSE_ToepCuboidConEig))
    MSE_ToepFrob_n.append(np.mean(MSE_ToepFrob))

    MSE_SVD_sCov_n.append(np.mean(MSE_SVD_sCov))
    MSE_SVD_OAS_n.append(np.mean(MSE_SVD_OAS))
    MSE_SVD_ToepCube_n.append(np.mean(MSE_SVD_ToepCube))
    MSE_SVD_ToepCuboid_n.append(np.mean(MSE_SVD_ToepCuboid))
    MSE_SVD_ToepCubeCon_n.append(np.mean(MSE_SVD_ToepCubeCon))
    MSE_SVD_ToepCuboidCon_n.append(np.mean(MSE_SVD_ToepCuboidCon))
    MSE_SVD_DFT_n.append(np.mean(MSE_SVD_DFT))

MSE_sCov_n = np.array(MSE_sCov_n)
MSE_toeplitz_n = np.array(MSE_ToepCube_n)
MSE_OAS_n = np.array(MSE_OAS_n)
MSE_ToepCuboid_n = np.array(MSE_ToepCuboid_n)
MSE_ToepCubeEig_n = np.array(MSE_ToepCubeEig_n)
MSE_ToepCuboidEig_n = np.array(MSE_ToepCuboidEig_n)
MSE_DFT_n = np.array(MSE_DFT_n)
MSE_ToepFrob_n = np.array(MSE_ToepFrob_n)

MSE_SVD_sCov_n = np.array(MSE_SVD_sCov_n)
MSE_SVD_OAS_n= np.array(MSE_SVD_OAS_n)
MSE_SVD_ToepCube_n= np.array(MSE_SVD_ToepCube_n)
MSE_SVD_ToepCuboid_n= np.array(MSE_SVD_ToepCuboid_n)
MSE_SVD_ToepCubeCon_n= np.array(MSE_SVD_ToepCubeCon_n)
MSE_SVD_ToepCuboidCon_n= np.array(MSE_SVD_ToepCuboidCon_n)
MSE_SVD_DFT_n= np.array(MSE_SVD_DFT_n)

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

# plt.plot(N_SAMPLES,MSE_sCov_n,label = 'sCov')
# plt.plot(N_SAMPLES,MSE_toeplitz_n,label = 'Toeplitz')
# plt.plot(N_SAMPLES,MSE_OAS_n,label = 'OAS')
# plt.plot(N_SAMPLES,MSE_DFT_n,label = 'DFT')
# plt.plot(N_SAMPLES,MSE_toeplitz2_n,label = 'Toeplitz2')
# plt.plot(N_SAMPLES,MSE_toeplitz_eig_n,label = 'ToepEig')
# plt.plot(N_SAMPLES,MSE_toeplitz2_eig_n,label = 'ToepEig2')
# plt.legend()
# plt.ylabel('MSE')
# plt.xlabel('N_SAMPLES')
# #plt.title(f'Dimension: {N}, r-value: {r}')
# plt.show()
#
# plt.plot(N_SAMPLES,MSE_SVD_sCov_n,label = 'sCov')
# plt.plot(N_SAMPLES,MSE_SVD_toeplitz_n,label = 'Toeplitz')
# plt.plot(N_SAMPLES,MSE_SVD_OAS_n,label = 'OAS')
# plt.plot(N_SAMPLES,MSE_SVD_DFT_n,label = 'DFT')
# plt.plot(N_SAMPLES,MSE_SVD_toeplitz2_n,label = 'Toeplitz2')
# plt.legend()
# plt.ylabel('MSE SVD')
# plt.xlabel('N_SAMPLES')
# #plt.title(f'Dimension: {N}, r-value: {r}')
# plt.show()