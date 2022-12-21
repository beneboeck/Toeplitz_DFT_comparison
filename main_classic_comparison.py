import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from utils_classic import *
import csv
import datetime
import os
import math
from os.path import exists
from estimators import *
import K_constraints as K_c
import estimator_classes as ec
import cov_generators as cg

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

# GLOBAL VARIABLES
N = 16
N_SAMPLES = [4,8,64]
RUNS = 2
K = K_dic[str(N)] * np.ones(N)
K_reduced = K_dic[str(N//2)] * np.ones(N//2)

# ESTIMATORS
#sCov (check), UMVUE (check), DFT (check), ToepEM, PCA, Sparse, OAS (check), OAS_UMVUE (check)
#ToepCube (check), ToepCuboid (check), ToepFrob (check), ToepL1 (check), ToepSmooth (check), ToepStieltjes (check)
com_est = ['UMVUE','DFT','OAS','OAS_UMVUE']
my_est = ['ToepCube','ToepCuboid','ToepFrob','ToepL1','ToepSmooth','ToepStieltjes']

com_estimators_l = [True,True,True,True]
com_estimators = [ec.UMVUE_estimator,ec.DFT_estimator,ec.OAS_estimator,ec.OAS_UMVUE_estimator]
my_estimators_l = [True,False,False,False,False,False]
my_estimators = [ec.ToepCube_estimator,ec.ToepCuboid_estimator,ec.ToepFrob_estimator,ec.ToepL1_estimator,ec.ToepSmoothMax_estimator,ec.ToepStieltjes_estimator]

# Cov MODEL
#C = cg.generate_AR2(N,0.7,0.6,0.3)
C = cg.generate_MA1(N,0.7,0.7)
#C = cg.generate_ARMA11(N,0.7,0.8,0.4)
#C = cg.generate_brownian(N,0.8)

print(f'N: {N}, RUNS: {RUNS}')

for n_samples in N_SAMPLES:

    my_MSEs = [[[], [], [], []], [[], [], [], []], [[], [], [], []], [[], [], [], []], [[], [], [], []],[[], [], [], []]]
    my_SVD_MSEs = [[[], []], [[], []], [[], []], [[], []], [[], []], [[], []]]
    my_outliers = [0, 0, 0, 0, 0, 0]
    com_MSEs = [[], [], [], []]
    com_SVD_MSEs = [[], [], [], []]
    MSE_sCov = []

    for run in range(RUNS):

        samples = np.random.multivariate_normal(np.zeros(N), C, n_samples)
        sCov = 1 / n_samples * (samples.T @ samples)
        MSE_sCov.append(np.linalg.norm(sCov - C) ** 2)

        for idx,l in enumerate(com_estimators_l):
            if l == True:
                C_estimator = com_estimators[idx](N,n_samples,sCov,K,K_reduced)
                C_est = C_estimator.optimize()
                com_MSEs[idx].append(np.linalg.norm(C_est - C)**2)
                com_SVD_MSEs[idx].append(SVD_comparison(N, C, C_est))

        for idx,l in enumerate(my_estimators_l):
            if l == True:
                C_estimator = my_estimators[idx](N,n_samples,sCov,K,K_reduced)
                K = K_dic[str(N)] * np.ones(N)
                K_reduced = K_dic[str(N // 2)] * np.ones(N // 2)
                Gamma_est, GammaEig_est = C_estimator.optimize()
                GammaCon_est, GammaConEig_est = C_estimator.optimize_red()

                my_MSEs[idx][0].append(np.linalg.norm(np.linalg.inv(Gamma_est) - C) ** 2)
                my_MSEs[idx][1].append(np.linalg.norm(np.linalg.inv(GammaEig_est) - C) ** 2)
                my_MSEs[idx][2].append(np.linalg.norm(np.linalg.inv(GammaCon_est) - C) ** 2)
                my_MSEs[idx][3].append(np.linalg.norm(np.linalg.inv(GammaConEig_est) - C) ** 2)

                my_SVD_MSEs[idx][0].append(SVD_comparison(N, C, np.linalg.inv(Gamma_est)))
                my_SVD_MSEs[idx][1].append(SVD_comparison(N, C, np.linalg.inv(GammaCon_est)))

                if 3 * MSE_sCov[-1] < np.sum(my_MSEs[idx][:][-1]):
                    my_outliers[idx] += 1

    print(f'SAMPLES: {n_samples}\n')

    for idx,l in enumerate(com_estimators_l):
        if l == True:
            print(com_est[idx])
            print(f'MSE: {np.mean(com_MSEs[idx]):.2f}')
            print(f'SVD MSE: {np.mean(com_SVD_MSEs[idx]):.2f}\n')

    for idx,l in enumerate(my_estimators_l):
        if l == True:
            print(my_est[idx])
            print(f'MSE: {np.mean(my_MSEs[idx][0]):.2f} - {np.mean(my_MSEs[idx][1]):.2f} - {np.mean(my_MSEs[idx][2]):.2f} - {np.mean(my_MSEs[idx][3]):.2f}')
            print(f'SVD MSE: {np.mean(my_SVD_MSEs[idx][0]):.2f} - {np.mean(my_SVD_MSEs[idx][1]):.2f}\n')