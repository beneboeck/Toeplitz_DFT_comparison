import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

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
    '32':0.03122
}

N = 32
K = K_dic[str(N)]

rand_matrix = np.random.randn(N, N)
B_mask = np.tril(rand_matrix)
B_mask[B_mask != 0] = 1
C_mask = np.tril(rand_matrix, k=-1)
C_mask[C_mask != 0] = 1

def generating_Gamma(alpha):
    alpha_prime = np.concatenate((np.array([0]), np.flip(np.array(alpha[1:]))))
    values = np.concatenate((np.array([1]),alpha[1:], np.flip(np.array(alpha[1:]))))
    i, j = np.ones((N, N)).nonzero()
    values = values[j - i].reshape(N, N)
    B = values * B_mask
    values_prime = np.concatenate((alpha_prime, np.flip(np.array(alpha_prime[1:]))))
    i, j = np.ones((N, N)).nonzero()
    values_prime2 = values_prime[j - i].reshape(N, N)
    C = np.conj(values_prime2 * C_mask)
    #print('C')
    #print(C)
    Gamma = alpha[0] * (np.matmul(B, np.conj(B).T) - np.matmul(C, np.conj(C).T))
    return Gamma

def f(alpha):
    Gamma = generating_Gamma(alpha)
    #print('...')
    #print(Gamma)
    return - np.log(np.linalg.det(Gamma)) + np.trace(Gamma @ sCov)

def constraint0(alpha):
    return [alpha[0]]

def constraintU1(alpha):
    return [K- alpha[1]]
def constraintL1(alpha):
    return [alpha[1] + K]

def constraintU2(alpha):
    return [K- alpha[2]]
def constraintL2(alpha):
    return [alpha[2] + K]

def constraintU3(alpha):
    return [K - alpha[3]]
def constraintL3(alpha):
    return [alpha[3] + K]

def constraintU4(alpha):
    return [K - alpha[4]]
def constraintL4(alpha):
    return [alpha[4] + K]

def constraintU5(alpha):
    return [K - alpha[5]]
def constraintL5(alpha):
    return [alpha[5] + K]

def constraintU6(alpha):
    return [K - alpha[6]]
def constraintL6(alpha):
    return [alpha[6] + K]

def constraintU7(alpha):
    return [K - alpha[7]]
def constraintL7(alpha):
    return [alpha[7] + K]

def constraintU8(alpha):
    return [K - alpha[8]]
def constraintL8(alpha):
    return [alpha[8] + K]

def constraintU9(alpha):
    return [K - alpha[9]]
def constraintL9(alpha):
    return [alpha[9] + K]

def constraintU10(alpha):
    return [K - alpha[10]]
def constraintL10(alpha):
    return [alpha[10] + K]

def constraintU11(alpha):
    return [K - alpha[11]]
def constraintL11(alpha):
    return [alpha[11] + K]

def constraintU12(alpha):
    return [K - alpha[12]]
def constraintL12(alpha):
    return [alpha[12] + K]

def constraintU13(alpha):
    return [K - alpha[13]]
def constraintL13(alpha):
    return [alpha[13] + K]

def constraintU14(alpha):
    return [K - alpha[14]]
def constraintL14(alpha):
    return [alpha[14] + K]

def constraintU15(alpha):
    return [K - alpha[15]]
def constraintL15(alpha):
    return [alpha[15] + K]

def constraintU16(alpha):
    return [K - alpha[16]]
def constraintL16(alpha):
    return [alpha[16] + K]

def constraintU17(alpha):
    return [K - alpha[17]]
def constraintL17(alpha):
    return [alpha[17] + K]

def constraintU18(alpha):
    return [K - alpha[18]]
def constraintL18(alpha):
    return [alpha[18] + K]

c0_dic = {
    'fun': constraint0,
    'type': 'ineq',
}
cL1_dic = {
    'fun': constraintL1,
    'type': 'ineq',
}
cU1_dic = {
    'fun': constraintU1,
    'type': 'ineq',
}
cL2_dic = {
    'fun': constraintL2,
    'type': 'ineq',
}
cU2_dic = {
    'fun': constraintU2,
    'type': 'ineq',
}
cL3_dic = {
    'fun': constraintL3,
    'type': 'ineq',
}
cU3_dic = {
    'fun': constraintU3,
    'type': 'ineq',
}
cL4_dic = {
    'fun': constraintL4,
    'type': 'ineq',
}
cU4_dic = {
    'fun': constraintU4,
    'type': 'ineq',
}
cL5_dic = {
    'fun': constraintL5,
    'type': 'ineq',
}
cU5_dic = {
    'fun': constraintU5,
    'type': 'ineq',
}
cL6_dic = {
    'fun': constraintL6,
    'type': 'ineq',
}
cU6_dic = {
    'fun': constraintU6,
    'type': 'ineq',
}
cL7_dic = {
    'fun': constraintL7,
    'type': 'ineq',
}
cU7_dic = {
    'fun': constraintU7,
    'type': 'ineq',
}
cL8_dic = {
    'fun': constraintL8,
    'type': 'ineq',
}
cU8_dic = {
    'fun': constraintU8,
    'type': 'ineq',
}
cL9_dic = {
    'fun': constraintL9,
    'type': 'ineq',
}
cU9_dic = {
    'fun': constraintU9,
    'type': 'ineq',
}
cL10_dic = {
    'fun': constraintL10,
    'type': 'ineq',
}
cU10_dic = {
    'fun': constraintU10,
    'type': 'ineq',
}
cL11_dic = {
    'fun': constraintL11,
    'type': 'ineq',
}
cU11_dic = {
    'fun': constraintU11,
    'type': 'ineq',
}
cL12_dic = {
    'fun': constraintL12,
    'type': 'ineq',
}
cU12_dic = {
    'fun': constraintU12,
    'type': 'ineq',
}
cL13_dic = {
    'fun': constraintL13,
    'type': 'ineq',
}
cU13_dic = {
    'fun': constraintU13,
    'type': 'ineq',
}
cL14_dic = {
    'fun': constraintL14,
    'type': 'ineq',
}
cU14_dic = {
    'fun': constraintU14,
    'type': 'ineq',
}
cL15_dic = {
    'fun': constraintL15,
    'type': 'ineq',
}
cU15_dic = {
    'fun': constraintU15,
    'type': 'ineq',
}
cL16_dic = {
    'fun': constraintL16,
    'type': 'ineq',
}
cU16_dic = {
    'fun': constraintU16,
    'type': 'ineq',
}
constraints=[c0_dic,cL1_dic,cU1_dic,cL2_dic,cU2_dic,cL3_dic,cU3_dic,cL4_dic,cU4_dic,cL5_dic,cU5_dic,cL6_dic,cU6_dic,cL7_dic,cU7_dic,cL8_dic,cU8_dic,cL9_dic,cU9_dic,cL10_dic,cU10_dic,cL11_dic,cU11_dic,cL12_dic,cU12_dic,cL13_dic,cU13_dic,cL14_dic,cU14_dic,cL15_dic,cU15_dic,cL16_dic,cU16_dic]
constraints = constraints[:2*N - 1]
# MODEL
N_SAMPLES = 8
RUNS = 200
#AUTOREGRESSIVES MODEL GAUS
r = 0.7
C = np.zeros((N,N))
for i in range(N):
    for j in range(N):
        C[i,j] = r**(np.abs(j-i))

MSE_sCov = []
MSE_toeplitz = []
MSE_OAS = []
for run in range(RUNS):
    if run%50 == 0:
        print(f'run {run}')
    samples = np.random.multivariate_normal(np.zeros(N),C,N_SAMPLES)


    #first comparison - sample Cov
    sCov = 1/N_SAMPLES * (samples.T @ samples)

    #second comparison - oracle approximating Shrinkage Estimator
    F = np.trace(sCov)/N * np.eye(N)
    rho = min(((1 - 2/N) * np.trace(sCov @ sCov) + np.trace(sCov)**2)/((N_SAMPLES + 1 - 2/N) * (np.trace(sCov @ sCov) - np.trace(sCov)/N)),1)
    OAS_C = (1 - rho) * sCov + rho * F

    # my method
    init_values = np.zeros(N)
    init_values[0] = np.random.uniform(low=1, high=20)
    for n in range(1,N):
        init_values[n] = np.random.uniform(low = - K * init_values[0] + 0.01, high = K * init_values[0] - 0.01 )
    result = optimize.minimize(f, init_values, method="SLSQP",constraints=constraints)
    Gamma_est = generating_Gamma(result.x)

    MSE_sCov.append(np.sum((sCov - C)**2))
    MSE_toeplitz.append(np.sum((np.linalg.inv(Gamma_est) - C)**2))
    if np.sum((np.linalg.inv(Gamma_est) - C)**2) > 100:
        print(run)
        print(np.linalg.det(np.linalg.inv(Gamma_est)))
        print(np.linalg.det(Gamma_est))
        raise ValueError
    MSE_OAS.append(np.sum((OAS_C - C)**2))

print(f'MSE of sCov and real Cov: {np.mean(MSE_sCov):.4f}')
print(f'MSE of Toep and real Cov: {np.mean(MSE_toeplitz):.4f}')
print(f'MSE of OAS and real Cov: {np.mean(MSE_OAS):.4f}')
# MSE_sCov = []
# MSE_toeplitz = []
# C = [[1,0.4,0.2,0.1],[0.4,1,0.4,0.2],[0.2,0.4,1,0.4],[0.1,0.2,0.4,1]]
# for k in range(1):
#     x_new = np.random.multivariate_normal([0,0,0,0],C,16)
#     sCov = 1/16 * (x_new.T @ x_new)
#     alpha_vec = []
#     for i in range(200):
#         alpha_0 = np.random.uniform(low=0.05,high=20)
#         alpha_1 = np.random.uniform(low = - K * alpha_0 + 0.01, high = K * alpha_0 - 0.01 )
#         alpha_2 = np.random.uniform(low=- K * alpha_0 + 0.01, high=K * alpha_0 - 0.01)
#         alpha_3 = np.random.uniform(low=- K * alpha_0 + 0.01, high=K * alpha_0 - 0.01)
#
#         result = optimize.minimize(f, np.array([alpha_0, alpha_1, alpha_2, alpha_3]), method="SLSQP",constraints=constraints)
#         alpha_vec.append(result.x)
#
#     print('MEAN')
#     print(np.sum(np.abs(np.array(alpha_vec) - np.mean(np.array(alpha_vec),axis=0))**2))
#     #result2 = optimize.minimize(f, np.array([3, 0.4, -0.9, 0.05]), method="SLSQP",constraints=[constraint1_dic, constraint2_dic, constraint3_dic, constraint4_dic, constraint5_dic, constraint6_dic, constraint7_dic])
#     #print(result)
#     #print(type(result))
#     #print(result.x)
#     Gamma_est = generating_Gamma(result.x)
#     #print(result.x[0])
#     #Gamma_est2 = generating_Gamma(result2.x)
#     #print(result2.x[0])
#     #print(Gamma_est - Gamma_est2)
#     #print('..')
#     #print(C)
#     #print(np.sum((np.linalg.inv(Gamma_est) - C)**2))
#     #print(np.sum((sCov - C)**2))
#     MSE_sCov.append(np.sum((sCov - C)**2))
#     MSE_toeplitz.append(np.sum((np.linalg.inv(Gamma_est) - C)**2))
#
# print(np.mean(MSE_sCov))
# print(np.mean(MSE_toeplitz))
# accumulator = []
#
# def f(x):
#     accumulator.append(x)
#     return (x[0] - 2)**2 + (x[1] - 3)**2
#
# def constraint1(x):
#     return [x[0]]
# def constraint2(x):
#     return [x[1]]
# def constraint3(x):
#     return [1 - x[0] - x[1]]
#
# constraint1_dic = {
#     'fun': constraint1,
#     'type': 'ineq',
# }
# constraint2_dic = {
#     'fun': constraint2,
#     'type': 'ineq',
# }
# constraint3_dic = {
#     'fun': constraint3,
#     'type': 'ineq',
# }
#
# result = optimize.minimize(f, np.array([0, 0]), method="SLSQP",
#                      constraints=[constraint1_dic,constraint2_dic,constraint3_dic])
# print(result)
# #print(accumulator)
#
# print('----------')
# C = np.array([[1,0.1],[0.1,1]])
# L,U = np.linalg.eigh(C)
# print(U @ np.diag(L) @ U.T)
# x = np.random.randn(10,2)
# x_new = np.einsum('ij,mj->mi',np.diag(np.sqrt(L)) @ U,x)
# sCov = 1/10 * np.mean(np.einsum('ij,ik->ijk',x_new,x_new),axis=0)
# print(sCov)
# def f(theta):
#     Gamma = np.array([[theta[0],theta[1]],[theta[2],theta[3]]])
#
#     return - 5 * np.log(np.linalg.det(Gamma)) + 5 * np.trace(Gamma @ sCov)
#
# result = optimize.minimize(f, np.array([1,0,0,1]), method="SLSQP")
# print(result)
# print(np.linalg.inv(sCov))