import csv
import datetime
import os
import math
from eig_constraints import *
from os.path import exists
from estimators import *
import K_constraints as K_c
import cvxpy as cp
from utils_classic import *

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

N = 16
K = K_dic[str(N)] * np.ones(N-1)

rand_matrix = np.random.randn(N, N)
B_mask = np.tril(rand_matrix)
B_mask[B_mask != 0] = 1
C_mask = np.tril(rand_matrix, k=-1)
C_mask[C_mask != 0] = 1

#AUTOREGRESSIVES MODEL GAUS
r = 0.6
C = np.zeros((N,N))
for i in range(N):
    for j in range(N):
        C[i,j] = r**(np.abs(j-i))

N_SAMPLES = 6

samples = np.random.multivariate_normal(np.zeros(N),C,N_SAMPLES)

#first comparison - sample Cov
sCov = 1/N_SAMPLES * (samples.T @ samples)
alpha = cp.Variable(N)

constraint1 = [alpha[1:] <= K * alpha[0]]
constraint2 = [alpha[1:] >= - K * alpha[0]]
constraint3 = [alpha[0] >= 0.001]



def generate_gamma(alpha):
    #alpha (15,)
    alpha_prime = cp.hstack([alpha[0][None,None],alpha[:0:-1][None]])
    #alpha_prime (1,16)
    values = cp.hstack([alpha[None],alpha[:0:-1][None]])
    i, j = np.ones((N, N)).nonzero()
    values = cp.reshape(values[0,j-i],(N,N))
    B = cp.multiply(values,B_mask)
    values_prime = cp.hstack([alpha_prime,alpha_prime[0,:0:-1][None]])
    i, j = np.ones((N, N)).nonzero()
    values_prime2 = cp.reshape(values_prime[0,j - i],(N, N))
    C = cp.multiply(values_prime2,C_mask)
    alpha_0 = B[0, 0]
    Gamma = 1 / alpha_0[None, None] * (B @ B.T - C @ C.T)
    return Gamma


#print(alpha[0].shape)
#print(alpha[:0:-1][None].shape)
#print(cp.hstack([alpha[0][None,None],alpha[:0:-1][None]]))
#print(generate_gamma(alpha))


objective = cp.Minimize(cp.sum(alpha))




prob = cp.Problem(objective,constraint1 + constraint2 + constraint3)

prob.solve()

print(alpha.value)
