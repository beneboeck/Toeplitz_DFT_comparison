import cvxpy as cp
import numpy as np

x = cp.Variable(1)
y = cp.Variable(1)

bound_x = [0 <= x]


#cons1 = [5*x + 3*y <= 10]
#cons2 = [2*x + 7*y <= 9]

b = np.array([10,9])
A = np.array([[5,3],[2,7]])
objective  = cp.Maximize(5 * x + 2 * y + np.trace(A))

def cons2(x,i):
    print(i)
    return A @ x

cons2_l = [cons2(x,2) <= b]

prob = cp.Problem(objective,cons2_l + bound_x)

prob.solve()

print(objective.value)



# m  = 20
# n = 15
# np.random.seed(1)
# A = np.random.randn(m,n)
# b = np.random.randn(m)
#
# x = cp.Variable(n)
# cost = cp.sum_squares(A @ x - b)
# prob = cp.Problem(cp.Minimize(cost))
# prob.solve()
# print(prob.value)