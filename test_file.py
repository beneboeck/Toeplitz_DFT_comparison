from scipy import optimize
import numpy as np

def f(x):
    return np.linalg.norm(x)**2

cons = tuple({'type':'ineq','fun': (lambda x,i=i: x[i] - 1)} for i in range(3))

x_init = [2,2,2]

result = optimize.minimize(f, x_init, method="SLSQP",constraints=cons)
print(result.x)