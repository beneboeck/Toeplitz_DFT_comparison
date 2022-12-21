import numpy as np
from scipy.linalg import toeplitz

N = 8
## AR(1)

sigma = 1
a = 0.6
C = np.zeros((N,N))
for i in range(N):
    for j in range(N):
        C[i,j] = a**(np.abs(i-j))
C = (sigma**2)/(1 - a**2) * C
Gamma = np.linalg.inv(C)
print(C[:4,:4])
print(Gamma[:4,:4])

## AR(2)
sigma = 1
a1 = 0.6
a2 = 0.3
if (np.abs(a2) > 1) | ((a2 - a1) > 1) | (a2 + a1 > 1):
    print('instable AR(2) process')

r = np.zeros(N)
r[0] = ((1 - a2) * sigma**2)/((1 + a2) * (1 - a1 - a2) * (1 + a1 - a2))
r[1] = a1/(1 - a2) * r[0]
for i in range(2,N):
    r[i] = a1 * r[i-1] + a2 * r[i-2]

C = toeplitz(r,r)
Gamma = np.linalg.inv(C)
print(C[:5,:5])
print(Gamma[:5,:5])

## MA(1)
sigma = 1
b = 0.8
r = np.zeros(N)
r[0] = (1 + b**2) * sigma**2
r[1] = b * sigma**2
C = toeplitz(r,r)
Gamma = np.linalg.inv(C)
print(C[:5,:5])
print(Gamma[:5,:5])
print(Gamma[0,:])

## ARMA(1,1)
sigma = 1
a = 0.8
b = 0.4
r = np.zeros(N)
r[0] = sigma**2 * (1 + ((b + a)**2)/(1 - a**2))
r[1] = sigma**2 * (b + a + ((b + a)**2 * a)/(1 - a**2))
for k in range(2,N):
    r[k] = a * r[k-1]

C = toeplitz(r,r)
Gamma = np.linalg.inv(C)
print(C[:5,:5])
print(Gamma[:5,:5])
print(Gamma[0,:])