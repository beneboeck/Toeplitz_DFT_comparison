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

K_list = list(0.06451 * np.ones(16))
K_list[0] = 0
K_list[3] = K_list[3]/2
K_list[4] = 2 * K_list[4]
print(K_list)
print('...')
print(bound(K_list))



def alpha_b_upper_bound(K,idx):
    coeff = coeff_per_d[idx,:]
    upper_bound = 0

    for i in range(len(coeff)):
        upper_bound += coeff[i] * K**(i+1)

    return upper_bound

def g_squared_upper_bound(K,idx):
    upper_bound = 0
    for j in range(1,idx+1):
        upper_bound += alpha_b_upper_bound(K,idx - j)

    return upper_bound**2

def Frob_norm_upper_bound(K):
    upper_bound = 0
    for i in range(1,N):
        upper_bound += (N-i) * g_squared_upper_bound(K,i)
    return upper_bound


K_dic = {
    '2':1,
    '3':0.48725,
    '4':0.32328,
    '5':0.24209,
    '6':0.19357,
    '7':0.16127,
    '8':0.13822,
    '10':.10750,
    '16':0.06451,
    '32':0.03122,
    '64':0.01536,
    '100': 0.00978,
    '128':0.007624,
    '256':0.0037975,
}

N = 256

# First we compute the number of coefficients per dimension in some \alpha_m * b_n (dimensions go from 0 up to n-1 )
# rows are n, the columns are the dimensions, entries are the number of coefficients

coeff_per_d = np.zeros((N,N))
coeff_per_d[0,0] = 1
for j in range(1,N):
    for i in range(j,N):
        coeff_per_d[i,j] = np.sum(coeff_per_d[0:i,j-1])
print(coeff_per_d)

print(Frob_norm_upper_bound(0.0037975))
