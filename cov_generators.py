import numpy as np
from scipy.linalg import toeplitz


def generate_AR1(N,sigma,a):
    C = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            C[i, j] = a ** (np.abs(i - j))
    C = (sigma ** 2) / (1 - a ** 2) * C
    return C

def generate_AR2(N,sigma,a1,a2):
    if (np.abs(a2) > 1) | ((a2 - a1) > 1) | (a2 + a1 > 1):
        print('instable AR(2) process')

    r = np.zeros(N)
    r[0] = ((1 - a2) * sigma ** 2) / ((1 + a2) * (1 - a1 - a2) * (1 + a1 - a2))
    r[1] = a1 / (1 - a2) * r[0]
    for i in range(2, N):
        r[i] = a1 * r[i - 1] + a2 * r[i - 2]

    C = toeplitz(r, r)
    return C

def generate_MA1(N,sigma,b):
    r = np.zeros(N)
    r[0] = (1 + b ** 2) * sigma ** 2
    r[1] = b * sigma ** 2
    C = toeplitz(r, r)
    return C

def generate_ARMA11(N,sigma,a,b):
    r = np.zeros(N)
    r[0] = sigma ** 2 * (1 + ((b + a) ** 2) / (1 - a ** 2))
    r[1] = sigma ** 2 * (b + a + ((b + a) ** 2 * a) / (1 - a ** 2))
    for k in range(2, N):
        r[k] = a * r[k - 1]

    C = toeplitz(r, r)
    return C

def generate_brownian(N,H):
    C = np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            C[i,j] = 0.5 * ( (np.abs(j-i) + 1)**(2*H) - 2 * np.abs(j-i)**(2*H) + np.abs((np.abs(j-i) - 1))**(2*H) )
    return C