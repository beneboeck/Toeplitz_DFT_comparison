import numpy as np
from scipy import optimize
import scipy as sc
import copy as c
from utils_classic import *
import math

# my estimators
class ToepCube_estimator():
    def __init__(self,N,n_samples,sCov,K,K_reduced):
        self.sCov = sCov
        self.N = N
        self.K = K
        self.K_red = K_reduced
        rand_matrix = np.random.randn(N, N)
        self.B_mask = np.tril(rand_matrix)
        self.B_mask[self.B_mask != 0] = 1
        self.C_mask = np.tril(rand_matrix, k=-1)
        self.C_mask[self.C_mask != 0] = 1

    def generate_constraints(self):
        constraints = []
        constraints.append({'fun':lambda alpha: alpha[0] - 0.01,'type': 'ineq'})
        constraints.extend({'fun':lambda alpha,i=i: self.K[i] * alpha[0] - alpha[i],'type': 'ineq'} for i in range(1,self.N))
        constraints.extend({'fun': lambda alpha,i=i: alpha[i] + self.K[i] * alpha[0],'type': 'ineq'} for i in range(1, self.N))
        return constraints

    def generate_constraints_red(self):
        constraints = []
        constraints.append({'fun':lambda alpha: alpha[0] - 0.01,'type': 'ineq'})
        constraints.extend({'fun':lambda alpha,i=i: self.K_red[i] * alpha[0] - alpha[i],'type': 'ineq'} for i in range(1,self.N//2))
        constraints.extend({'fun': lambda alpha,i=i: alpha[i] + self.K_red[i] * alpha[0],'type': 'ineq'} for i in range(1, self.N//2))
        return constraints

    def generate_constraints_eig(self):
        constraints = []
        constraints.extend({'fun': lambda eig, i=i: eig[i] - 0.00001, 'type': 'ineq'} for i in range(0, self.N))
        return constraints

    def generate_Gamma(self,alpha):
        alpha_prime = np.concatenate((np.array([0]), np.flip(np.array(alpha[1:]))))
        values = np.concatenate((alpha, np.flip(np.array(alpha[1:]))))
        i, j = np.ones((self.N, self.N)).nonzero()
        values = values[j - i].reshape(self.N, self.N)
        B = values * self.B_mask
        values_prime = np.concatenate((alpha_prime, np.flip(np.array(alpha_prime[1:]))))
        values_prime2 = values_prime[j - i].reshape(self.N, self.N)
        C = np.conj(values_prime2 * self.C_mask)
        alpha_0 = B[0, 0]
        Gamma = 1 / alpha_0[None, None] * (np.matmul(B, np.conj(B).T) - np.matmul(C, np.conj(C).T))
        return Gamma

    def objective(self,alpha):
        Gamma = self.generate_Gamma(alpha)
        return - np.log(np.linalg.det(Gamma)) + np.trace(Gamma @ self.sCov)

    def objective_eig(self,eig,*args):
        U = args[0]
        return - np.sum(np.log(eig)) + np.trace(U @ np.diag(eig) @ U.T @ self.sCov)

    def objective_con(self,alpha):
        alpha = np.concatenate((alpha,np.zeros(self.N//2)))
        Gamma = self.generate_Gamma(alpha)
        return - np.log(np.linalg.det(Gamma)) + np.trace(Gamma @ self.sCov)

    def optimize(self):
        init_values = np.zeros(self.N)
        init_values[0] = np.random.uniform(low=1, high=20)
        for n in range(1, self.N):
            init_values[n] = np.random.uniform(low=- self.K[n] * init_values[0] + 0.01,high=self.K[n] * init_values[0] - 0.01)
        result = optimize.minimize(self.objective, init_values, method="SLSQP",constraints=self.generate_constraints())
        Gamma_est = self.generate_Gamma(result.x)
        U_ToepCube, S_ToepCube, _ = np.linalg.svd(Gamma_est)

        result_eig1 = optimize.minimize(self.objective_eig, S_ToepCube,args=U_ToepCube, method="SLSQP", constraints=self.generate_constraints_eig())
        Gamma_est_eig = U_ToepCube @ np.diag(result_eig1.x) @ U_ToepCube.T

        return Gamma_est,Gamma_est_eig

    def optimize_red(self):
        init_values = np.zeros(self.N//2)
        init_values[0] = np.random.uniform(low=1, high=20)
        for n in range(1, self.N//2):
            init_values[n] = np.random.uniform(low=- self.K_red[n] * init_values[0] + 0.01,high=self.K_red[n] * init_values[0] - 0.01)
        result = optimize.minimize(self.objective_con, init_values, method="SLSQP",constraints=self.generate_constraints_red())
        Gamma_est = self.generate_Gamma(np.concatenate((result.x,np.zeros(self.N//2))))
        U_ToepCube, S_ToepCube, _ = np.linalg.svd(Gamma_est)

        result_eig1 = optimize.minimize(self.objective_eig, S_ToepCube,args=U_ToepCube, method="SLSQP", constraints=self.generate_constraints_eig())
        Gamma_est_eig = U_ToepCube @ np.diag(result_eig1.x) @ U_ToepCube.T

        return Gamma_est,Gamma_est_eig

class ToepCuboid_estimator():
    def __init__(self,N,n_samples,sCov,K,K_reduced):
        self.sCov = sCov
        self.N = N
        self.K = K
        self.K_red = K_reduced
        rand_matrix = np.random.randn(N, N)
        self.B_mask = np.tril(rand_matrix)
        self.B_mask[self.B_mask != 0] = 1
        self.C_mask = np.tril(rand_matrix, k=-1)
        self.C_mask[self.C_mask != 0] = 1

    def generate_constraints(self):
        constraints = []
        constraints.append({'fun':lambda alpha: alpha[0] - 0.01,'type': 'ineq'})
        constraints.extend({'fun':lambda alpha,i=i: self.K[i] * alpha[0] - alpha[i],'type': 'ineq'} for i in range(1,self.N))
        constraints.extend({'fun': lambda alpha,i=i: alpha[i] + self.K[i] * alpha[0],'type': 'ineq'} for i in range(1, self.N))
        return constraints

    def generate_constraints_con(self):
        constraints = []
        constraints.append({'fun':lambda alpha: alpha[0] - 0.01,'type': 'ineq'})
        constraints.extend({'fun':lambda alpha,i=i: self.K_red[i] * alpha[0] - alpha[i],'type': 'ineq'} for i in range(1,self.N//2))
        constraints.extend({'fun': lambda alpha,i=i: alpha[i] + self.K_red[i] * alpha[0],'type': 'ineq'} for i in range(1, self.N//2))
        return constraints

    def generate_constraints_eig(self):
        constraints = []
        constraints.extend({'fun': lambda eig, i=i: eig[i] - 0.00001, 'type': 'ineq'} for i in range(0, self.N))
        return constraints

    def adjustingK(self,result):
        alpha_0 = result.x[0]
        idx_bounding = np.squeeze(np.where(np.abs(result.x[1:]) > 0.9 * self.K[1:] * alpha_0)).reshape(-1) + np.array(1)
        idx_interior = np.squeeze(np.where(np.abs(result.x[1:]) < 0.9 * self.K[1:] * alpha_0)).reshape(-1) + np.array(1)

        len_bounding = len(idx_bounding)
        len_interior = len(idx_interior)
        derivatives = optimize.approx_fprime(result.x, self.objective, 10e-8)
        while (len_interior > 0) & (len_bounding > 0):
            max_bounding_idx = idx_bounding[np.argmax(np.abs(derivatives[idx_bounding]))]
            max_interior_idx = idx_interior[np.argmax(self.K[idx_interior] * alpha_0 - np.abs(result.x[idx_interior]))]

            self.K[max_interior_idx] = (0.7 * (self.K[max_interior_idx] * result.x[0] - np.abs(result.x[max_interior_idx])) + np.abs(
                result.x[max_interior_idx])) / result.x[0]
            K_test = self.K
            while bound(K_test) < 1:
                K_test[max_bounding_idx] = 1.01 * K_test[max_bounding_idx]
            K_test[max_bounding_idx] = 1 / 1.01 * K_test[max_bounding_idx]
            self.K = K_test

            idx_bounding = np.delete(idx_bounding, np.where(idx_bounding == max_bounding_idx))
            idx_interior = np.delete(idx_interior, np.where(idx_interior == max_interior_idx))
            len_bounding = len(idx_bounding)
            len_interior = len(idx_interior)

    def adjustingK_red(self,result):
        alpha_0 = result.x[0]
        idx_bounding = np.squeeze(np.where(np.abs(result.x[1:]) > 0.9 * self.K_red[1:] * alpha_0)).reshape(-1) + np.array(1)
        idx_interior = np.squeeze(np.where(np.abs(result.x[1:]) < 0.9 * self.K_red[1:] * alpha_0)).reshape(-1) + np.array(1)

        len_bounding = len(idx_bounding)
        len_interior = len(idx_interior)
        derivatives = optimize.approx_fprime(result.x, self.objective_con, 10e-8)
        while (len_interior > 0) & (len_bounding > 0):
            max_bounding_idx = idx_bounding[np.argmax(np.abs(derivatives[idx_bounding]))]
            max_interior_idx = idx_interior[np.argmax(self.K_red[idx_interior] * alpha_0 - np.abs(result.x[idx_interior]))]

            self.K_red[max_interior_idx] = (0.7 * (self.K_red[max_interior_idx] * result.x[0] - np.abs(result.x[max_interior_idx])) + np.abs(result.x[max_interior_idx])) / result.x[0]
            K_test = self.K_red
            while bound(K_test) < 1:
                K_test[max_bounding_idx] = 1.01 * K_test[max_bounding_idx]
            K_test[max_bounding_idx] = 1 / 1.01 * K_test[max_bounding_idx]
            self.K_red = K_test

            idx_bounding = np.delete(idx_bounding, np.where(idx_bounding == max_bounding_idx))
            idx_interior = np.delete(idx_interior, np.where(idx_interior == max_interior_idx))
            len_bounding = len(idx_bounding)
            len_interior = len(idx_interior)

    def generate_Gamma(self,alpha):
        alpha_prime = np.concatenate((np.array([0]), np.flip(np.array(alpha[1:]))))
        values = np.concatenate((alpha, np.flip(np.array(alpha[1:]))))
        i, j = np.ones((self.N, self.N)).nonzero()
        values = values[j - i].reshape(self.N, self.N)
        B = values * self.B_mask
        values_prime = np.concatenate((alpha_prime, np.flip(np.array(alpha_prime[1:]))))
        values_prime2 = values_prime[j - i].reshape(self.N, self.N)
        C = np.conj(values_prime2 * self.C_mask)
        alpha_0 = B[0, 0]
        Gamma = 1 / alpha_0[None, None] * (np.matmul(B, np.conj(B).T) - np.matmul(C, np.conj(C).T))
        return Gamma

    def objective(self,alpha):
        Gamma = self.generate_Gamma(alpha)
        return - np.log(np.linalg.det(Gamma)) + np.trace(Gamma @ self.sCov)

    def objective_eig(self,eig,*args):
        U = args[0]
        return - np.sum(np.log(eig)) + np.trace(U @ np.diag(eig) @ U.T @ self.sCov)

    def objective_con(self,alpha):
        alpha = np.concatenate((alpha,np.zeros(self.N//2)))
        Gamma = self.generate_Gamma(alpha)
        return - np.log(np.linalg.det(Gamma)) + np.trace(Gamma @ self.sCov)

    def optimize(self):
        init_values = np.zeros(self.N)
        init_values[0] = np.random.uniform(low=1, high=20)
        for n in range(1, self.N):
            init_values[n] = np.random.uniform(low=- self.K[n] * init_values[0] + 0.01,high=self.K[n] * init_values[0] - 0.01)
        result = optimize.minimize(self.objective, init_values, method="SLSQP",constraints=self.generate_constraints())
        self.adjustingK(result)
        alpha_0 = result.x[0]
        idx_bounding = np.squeeze(np.where(np.abs(result.x[1:]) > 0.9 * self.K[1:] * alpha_0)).reshape(-1) + np.array(1)
        idx_interior = np.squeeze(np.where(np.abs(result.x[1:]) < 0.9 * self.K[1:] * alpha_0)).reshape(-1) + np.array(1)
        len_bounding = len(idx_bounding)
        len_interior = len(idx_interior)
        counter = 0
        while (len_interior > 0) & (len_bounding > 0):
            print(counter)
            counter += 1
            if counter == 200:
                break
            result2 = optimize.minimize(self.objective, result.x, method="SLSQP",constraints=self.generate_constraints())
            result = result2
            self.adjustingK(result)
            idx_bounding = np.squeeze(np.where(np.abs(result.x[1:]) > 0.9 * self.K[1:] * alpha_0)).reshape(-1) + np.array(1)
            idx_interior = np.squeeze(np.where(np.abs(result.x[1:]) < 0.9 * self.K[1:] * alpha_0)).reshape(-1) + np.array(1)
            len_bounding = len(idx_bounding)
            len_interior = len(idx_interior)

        Gamma_est = self.generate_Gamma(result.x)
        U_ToepCube, S_ToepCube, _ = np.linalg.svd(Gamma_est)

        result_eig1 = optimize.minimize(self.objective_eig, S_ToepCube,args=U_ToepCube, method="SLSQP", constraints=self.generate_constraints_eig())
        Gamma_est_eig = U_ToepCube @ np.diag(result_eig1.x) @ U_ToepCube.T

        return Gamma_est,Gamma_est_eig

    def optimize_red(self):
        init_values = np.zeros(self.N//2)
        init_values[0] = np.random.uniform(low=1, high=20)
        for n in range(1, self.N//2):
            init_values[n] = np.random.uniform(low=- self.K_red[n] * init_values[0] + 0.01,high=self.K_red[n] * init_values[0] - 0.01)
        result = optimize.minimize(self.objective_con, init_values, method="SLSQP", constraints=self.generate_constraints_con())
        self.adjustingK_red(result)
        alpha_0 = result.x[0]
        idx_bounding = np.squeeze(np.where(np.abs(result.x[1:]) > 0.9 * self.K_red[1:] * alpha_0)).reshape(-1) + np.array(1)
        idx_interior = np.squeeze(np.where(np.abs(result.x[1:]) < 0.9 * self.K_red[1:] * alpha_0)).reshape(-1) + np.array(1)
        len_bounding = len(idx_bounding)
        len_interior = len(idx_interior)
        counter = 0
        while (len_interior > 0) & (len_bounding > 0):
            print(counter)
            counter += 1
            if counter == 200:
                break
            result2 = optimize.minimize(self.objective_con, result.x, method="SLSQP",constraints=self.generate_constraints_con())
            result = result2
            self.adjustingK_red(result)
            idx_bounding = np.squeeze(np.where(np.abs(result.x[1:]) > 0.9 * self.K_red[1:] * alpha_0)).reshape(-1) + np.array(1)
            idx_interior = np.squeeze(np.where(np.abs(result.x[1:]) < 0.9 * self.K_red[1:] * alpha_0)).reshape(-1) + np.array(1)
            len_bounding = len(idx_bounding)
            len_interior = len(idx_interior)

        Gamma_est = self.generate_Gamma(np.concatenate((result.x,np.zeros(self.N//2))))
        U_ToepCube, S_ToepCube, _ = np.linalg.svd(Gamma_est)

        result_eig1 = optimize.minimize(self.objective_eig, S_ToepCube, args=U_ToepCube, method="SLSQP",constraints=self.generate_constraints_eig())
        Gamma_est_eig = U_ToepCube @ np.diag(result_eig1.x) @ U_ToepCube.T

        return Gamma_est, Gamma_est_eig

class ToepFrob_estimator():
    def __init__(self,N,n_samples,sCov,K,K_reduced):
        self.sCov = sCov
        self.N = N
        self.K = K
        self.K_red = K_reduced
        rand_matrix = np.random.randn(N, N)
        self.B_mask = np.tril(rand_matrix)
        self.B_mask[self.B_mask != 0] = 1
        self.C_mask = np.tril(rand_matrix, k=-1)
        self.C_mask[self.C_mask != 0] = 1

    def generate_constraints(self):
        constraints = []
        constraints.append({'fun':lambda alpha: alpha[0] - 0.01,'type': 'ineq'})
        def constraint_frob(alpha):
            alpha_prime = np.concatenate((np.array([0]), np.flip(np.array(alpha[1:]))))
            values = np.concatenate((alpha, np.flip(np.array(alpha[1:]))))
            i, j = np.ones((self.N, self.N)).nonzero()
            values = values[j - i].reshape(self.N, self.N)
            B = values * self.B_mask
            values_prime = np.concatenate((alpha_prime, np.flip(np.array(alpha_prime[1:]))))
            i, j = np.ones((self.N, self.N)).nonzero()
            values_prime2 = values_prime[j - i].reshape(self.N, self.N)
            C = np.conj(values_prime2 * self.C_mask)
            matrix = np.conj(C).T @ np.linalg.inv(np.conj(B).T)
            frob_norm = np.sum(np.abs(matrix) ** 2)
            return 1 - frob_norm
        constraints.append({'fun': constraint_frob, 'type': 'ineq'})

        #def constraint_det(alpha):
        #    Gamma = self.generate_Gamma(alpha)
        #    return 10 ** 9 - np.linalg.det(Gamma)
        #constraints.append({'fun': constraint_det, 'type': 'ineq'})
        return constraints

    def generate_constraints_red(self):
        constraints = []
        constraints.append({'fun':lambda alpha: alpha[0] - 0.01,'type': 'ineq'})
        def constraint_frob(alpha):
            alpha = np.concatenate((alpha,np.zeros(self.N//2)))
            alpha_prime = np.concatenate((np.array([0]), np.flip(np.array(alpha[1:]))))
            values = np.concatenate((alpha, np.flip(np.array(alpha[1:]))))
            i, j = np.ones((self.N, self.N)).nonzero()
            values = values[j - i].reshape(self.N, self.N)
            B = values * self.B_mask
            values_prime = np.concatenate((alpha_prime, np.flip(np.array(alpha_prime[1:]))))
            i, j = np.ones((self.N, self.N)).nonzero()
            values_prime2 = values_prime[j - i].reshape(self.N, self.N)
            C = np.conj(values_prime2 * self.C_mask)
            matrix = np.conj(C).T @ np.linalg.inv(np.conj(B).T)
            frob_norm = np.sum(np.abs(matrix) ** 2)
            return 1 - frob_norm
        constraints.append({'fun': constraint_frob, 'type': 'ineq'})
        #def constraint_det(alpha):
        #    summands = np.zeros(len(alpha))
        #    for i in range(len(alpha)):
        #        summands[i] = summands[i-1] + (alpha[i]**2)/alpha[0]
        #    det = np.prod(summands) ** 2
        #    print(det)
        #    print(np.linalg.det(self.generate_Gamma(np.concatenate((alpha,np.zeros(self.N//2))))))
        #    return 10**8 - det
        #constraints.append({'fun': constraint_det, 'type': 'ineq'})
        return constraints

    def generate_constraints_eig(self):
        constraints = []
        constraints.extend({'fun': lambda eig, i=i: eig[i] - 0.00001, 'type': 'ineq'} for i in range(0, self.N))
        return constraints

    def generate_Gamma(self,alpha):
        alpha_prime = np.concatenate((np.array([0]), np.flip(np.array(alpha[1:]))))
        values = np.concatenate((alpha, np.flip(np.array(alpha[1:]))))
        i, j = np.ones((self.N, self.N)).nonzero()
        values = values[j - i].reshape(self.N, self.N)
        B = values * self.B_mask
        values_prime = np.concatenate((alpha_prime, np.flip(np.array(alpha_prime[1:]))))
        values_prime2 = values_prime[j - i].reshape(self.N, self.N)
        C = np.conj(values_prime2 * self.C_mask)
        alpha_0 = B[0, 0]
        Gamma = 1 / alpha_0[None, None] * (np.matmul(B, np.conj(B).T) - np.matmul(C, np.conj(C).T))
        return Gamma

    def objective(self,alpha):
        Gamma = self.generate_Gamma(alpha)
        return - np.log(np.linalg.det(Gamma)) + np.trace(Gamma @ self.sCov)

    def objective_eig(self,eig,*args):
        U = args[0]
        return - np.sum(np.log(eig)) + np.trace(U @ np.diag(eig) @ U.T @ self.sCov)

    def objective_con(self,alpha):
        alpha = np.concatenate((alpha,np.zeros(self.N//2)))
        Gamma = self.generate_Gamma(alpha)
        ####
        #Gamma_full = np.zeros((len(alpha),len(alpha)))
        #Gamma2 = np.zeros((len(alpha)//2, len(alpha)//2))
        #for i in range(len(alpha)//2):

        #    for j in range(i+1):
        #        sum = 0
        #        for k in range(j+1):
        #            sum = sum + alpha[k] * alpha[k+i-j]/alpha[0]
        #        Gamma2[i, j] = sum
        #        Gamma2[j,i] = Gamma2[i,j]
        #Gamma_full[:len(alpha)//2,:len(alpha)//2] = Gamma2

        #Gamma_full[len(alpha)//2:,len(alpha)//2:] = np.flipud(np.eye(len(alpha)//2)) @ Gamma2 @ np.flipud(np.eye(len(alpha)//2))
                #Gamma2[j,i] = Gamma[i,j]
                #Gamma2[len(alpha)-i-1,len(alpha)-j-1] = Gamma[i,j]


        #print(- np.log(np.linalg.det(Gamma)) + np.trace(Gamma @ self.sCov))

        return - np.log(np.linalg.det(Gamma)) + np.trace(Gamma @ self.sCov)

    def optimize(self):
        init_values = np.zeros(self.N)
        init_values[0] = np.random.uniform(low=1, high=2)
        for n in range(1, self.N):
            init_values[n] = np.random.uniform(low=- self.K[n] * init_values[0] + 0.01,high=self.K[n] * init_values[0] - 0.01)
        result = optimize.minimize(self.objective, init_values, method="SLSQP",constraints=self.generate_constraints())
        Gamma_est = self.generate_Gamma(result.x)
        U_ToepCube, S_ToepCube, _ = np.linalg.svd(Gamma_est)

        result_eig1 = optimize.minimize(self.objective_eig, S_ToepCube,args=U_ToepCube, method="SLSQP", constraints=self.generate_constraints_eig())
        Gamma_est_eig = U_ToepCube @ np.diag(result_eig1.x) @ U_ToepCube.T

        return Gamma_est,Gamma_est_eig

    def optimize_red(self):
        init_values = np.zeros(self.N//2)
        init_values[0] = np.random.uniform(low=1, high=2)
        for n in range(1, self.N//2):
            init_values[n] = np.random.uniform(low=- self.K_red[n] * init_values[0] + 0.01,high=self.K_red[n] * init_values[0] - 0.01)
        result = optimize.minimize(self.objective_con, init_values, method="SLSQP",constraints=self.generate_constraints_red())
        Gamma_est = self.generate_Gamma(np.concatenate((result.x,np.zeros(self.N//2))))
        U_ToepCube, S_ToepCube, _ = np.linalg.svd(Gamma_est)

        result_eig1 = optimize.minimize(self.objective_eig, S_ToepCube,args=U_ToepCube, method="SLSQP", constraints=self.generate_constraints_eig())
        Gamma_est_eig = U_ToepCube @ np.diag(result_eig1.x) @ U_ToepCube.T

        return Gamma_est,Gamma_est_eig

class ToepL1_estimator():
    def __init__(self,N,n_samples,sCov,K,K_reduced):
        self.sCov = sCov
        self.N = N
        self.K = K
        self.K_red = K_reduced
        rand_matrix = np.random.randn(N, N)
        self.B_mask = np.tril(rand_matrix)
        self.B_mask[self.B_mask != 0] = 1
        self.C_mask = np.tril(rand_matrix, k=-1)
        self.C_mask[self.C_mask != 0] = 1

    def generate_constraints(self):
        constraints = []
        constraints.append({'fun':lambda alpha: alpha[0] - 0.01,'type': 'ineq'})
        def constraint_frob(alpha):
            alpha_prime = np.concatenate((np.array([0]), np.flip(np.array(alpha[1:]))))
            values = np.concatenate((alpha, np.flip(np.array(alpha[1:]))))
            i, j = np.ones((self.N, self.N)).nonzero()
            values = values[j - i].reshape(self.N, self.N)
            B = values * self.B_mask
            values_prime = np.concatenate((alpha_prime, np.flip(np.array(alpha_prime[1:]))))
            i, j = np.ones((self.N, self.N)).nonzero()
            values_prime2 = values_prime[j - i].reshape(self.N, self.N)
            C = np.conj(values_prime2 * self.C_mask)
            matrix = np.conj(C).T @ np.linalg.inv(np.conj(B).T)
            l1_norm = np.sum(np.abs(matrix[0,:]))
            return 0.99 - l1_norm
        constraints.append({'fun': constraint_frob, 'type': 'ineq'})
        return constraints

    def generate_constraints_red(self):
        constraints = []
        constraints.append({'fun':lambda alpha: alpha[0] - 0.01,'type': 'ineq'})
        def constraint_frob(alpha):
            alpha = np.concatenate((alpha,np.zeros(self.N//2)))
            alpha_prime = np.concatenate((np.array([0]), np.flip(np.array(alpha[1:]))))
            values = np.concatenate((alpha, np.flip(np.array(alpha[1:]))))
            i, j = np.ones((self.N, self.N)).nonzero()
            values = values[j - i].reshape(self.N, self.N)
            B = values * self.B_mask
            values_prime = np.concatenate((alpha_prime, np.flip(np.array(alpha_prime[1:]))))
            i, j = np.ones((self.N, self.N)).nonzero()
            values_prime2 = values_prime[j - i].reshape(self.N, self.N)
            C = np.conj(values_prime2 * self.C_mask)
            matrix = np.conj(C).T @ np.linalg.inv(np.conj(B).T)
            l1_norm = np.sum(np.abs(matrix[0,:]))
            return 0.99 - l1_norm
        constraints.append({'fun': constraint_frob, 'type': 'ineq'})
        return constraints

    def generate_constraints_eig(self):
        constraints = []
        constraints.extend({'fun': lambda eig, i=i: eig[i]- 0.00001, 'type': 'ineq'} for i in range(0, self.N))
        return constraints

    def generate_Gamma(self,alpha):
        alpha_prime = np.concatenate((np.array([0]), np.flip(np.array(alpha[1:]))))
        values = np.concatenate((alpha, np.flip(np.array(alpha[1:]))))
        i, j = np.ones((self.N, self.N)).nonzero()
        values = values[j - i].reshape(self.N, self.N)
        B = values * self.B_mask
        values_prime = np.concatenate((alpha_prime, np.flip(np.array(alpha_prime[1:]))))
        values_prime2 = values_prime[j - i].reshape(self.N, self.N)
        C = np.conj(values_prime2 * self.C_mask)
        alpha_0 = B[0, 0]
        Gamma = 1 / alpha_0[None, None] * (np.matmul(B, np.conj(B).T) - np.matmul(C, np.conj(C).T))
        return Gamma

    def objective(self,alpha):
        Gamma = self.generate_Gamma(alpha)
        return - np.log(np.linalg.det(Gamma)) + np.trace(Gamma @ self.sCov)

    def objective_eig(self,eig,*args):
        U = args[0]
        return - np.sum(np.log(eig)) + np.trace(U @ np.diag(eig) @ U.T @ self.sCov)

    def objective_con(self,alpha):
        alpha = np.concatenate((alpha,np.zeros(self.N//2)))
        Gamma = self.generate_Gamma(alpha)
        return - np.log(np.linalg.det(Gamma)) + np.trace(Gamma @ self.sCov)

    def optimize(self):
        init_values = np.zeros(self.N)
        init_values[0] = np.random.uniform(low=1, high=20)
        for n in range(1, self.N):
            init_values[n] = np.random.uniform(low=- self.K[n] * init_values[0] + 0.01,high=self.K[n] * init_values[0] - 0.01)
        result = optimize.minimize(self.objective, init_values, method="SLSQP",constraints=self.generate_constraints())
        Gamma_est = self.generate_Gamma(result.x)
        U_ToepCube, S_ToepCube, _ = np.linalg.svd(Gamma_est)

        result_eig1 = optimize.minimize(self.objective_eig, S_ToepCube,args=U_ToepCube, method="SLSQP", constraints=self.generate_constraints_eig())
        Gamma_est_eig = U_ToepCube @ np.diag(result_eig1.x) @ U_ToepCube.T

        return Gamma_est,Gamma_est_eig

    def optimize_red(self):
        init_values = np.zeros(self.N//2)
        init_values[0] = np.random.uniform(low=1, high=20)
        for n in range(1, self.N//2):
            init_values[n] = np.random.uniform(low=- self.K_red[n] * init_values[0] + 0.01,high=self.K_red[n] * init_values[0] - 0.01)
        result = optimize.minimize(self.objective_con, init_values, method="SLSQP",constraints=self.generate_constraints_red())
        Gamma_est = self.generate_Gamma(np.concatenate((result.x,np.zeros(self.N//2))))
        U_ToepCube, S_ToepCube, _ = np.linalg.svd(Gamma_est)

        result_eig1 = optimize.minimize(self.objective_eig, S_ToepCube,args=U_ToepCube, method="SLSQP", constraints=self.generate_constraints_eig())
        Gamma_est_eig = U_ToepCube @ np.diag(result_eig1.x) @ U_ToepCube.T

        return Gamma_est,Gamma_est_eig

class ToepSmoothMax_estimator():
    def __init__(self,N,n_samples,sCov,K,K_reduced):
        self.sCov = sCov
        self.N = N
        self.K = K
        self.K_red = K_reduced
        rand_matrix = np.random.randn(N, N)
        self.B_mask = np.tril(rand_matrix)
        self.B_mask[self.B_mask != 0] = 1
        self.C_mask = np.tril(rand_matrix, k=-1)
        self.C_mask[self.C_mask != 0] = 1

    def generate_constraints(self):
        constraints = []
        constraints.append({'fun':lambda alpha: alpha[0] - 0.01,'type': 'ineq'})
        def constraint_frob(alpha):
            alpha_prime = np.concatenate((np.array([0]), np.flip(np.array(alpha[1:]))))
            values = np.concatenate((alpha, np.flip(np.array(alpha[1:]))))
            i, j = np.ones((self.N, self.N)).nonzero()
            values = values[j - i].reshape(self.N, self.N)
            B = values * self.B_mask
            values_prime = np.concatenate((alpha_prime, np.flip(np.array(alpha_prime[1:]))))
            i, j = np.ones((self.N, self.N)).nonzero()
            values_prime2 = values_prime[j - i].reshape(self.N, self.N)
            C = np.conj(values_prime2 * self.C_mask)

            smooth_max = np.sum(np.abs(alpha[1:]) * np.exp(8 * np.abs(alpha[1:]))) / np.sum(
                np.exp(8 * np.abs(alpha[1:])))
            bound_on_B = ((smooth_max / alpha[0]) + 1) ** (len(alpha) - 1) / alpha[0]
            l1_norm = np.sum(np.abs(C[:, 0]) ** 2) * bound_on_B ** 2
            return 0.999 - l1_norm
        constraints.append({'fun': constraint_frob, 'type': 'ineq'})
        return constraints

    def generate_constraints_red(self):
        constraints = []
        constraints.append({'fun':lambda alpha: alpha[0] - 0.01,'type': 'ineq'})
        def constraint_frob(alpha):
            alpha_full = np.concatenate((alpha, np.zeros(self.N // 2)))
            alpha_prime = np.concatenate((np.array([0]), np.flip(np.array(alpha_full[1:]))))
            values = np.concatenate((alpha_full, np.flip(np.array(alpha_full[1:]))))
            i, j = np.ones((self.N, self.N)).nonzero()
            values = values[j - i].reshape(self.N, self.N)
            B = values * self.B_mask
            values_prime = np.concatenate((alpha_prime, np.flip(np.array(alpha_prime[1:]))))
            i, j = np.ones((self.N, self.N)).nonzero()
            values_prime2 = values_prime[j - i].reshape(self.N, self.N)
            C = np.conj(values_prime2 * self.C_mask)

            smooth_max = np.sum(np.abs(alpha_full[1:]) * np.exp(8 * np.abs(alpha_full[1:]))) / np.sum(
                np.exp(8 * np.abs(alpha_full[1:])))
            bound_on_B = ((smooth_max / alpha_full[0]) + 1) ** (len(alpha_full) - 1) / alpha_full[0]
            l1_norm = np.sum(np.abs(C[:, 0]) ** 2) * bound_on_B ** 2
            return 0.999 - l1_norm
        constraints.append({'fun': constraint_frob, 'type': 'ineq'})
        return constraints

    def generate_constraints_eig(self):
        constraints = []
        constraints.extend({'fun': lambda eig, i=i: eig[i]- 0.00001, 'type': 'ineq'} for i in range(0, self.N))
        return constraints

    def generate_Gamma(self,alpha):
        alpha_prime = np.concatenate((np.array([0]), np.flip(np.array(alpha[1:]))))
        values = np.concatenate((alpha, np.flip(np.array(alpha[1:]))))
        i, j = np.ones((self.N, self.N)).nonzero()
        values = values[j - i].reshape(self.N, self.N)
        B = values * self.B_mask
        values_prime = np.concatenate((alpha_prime, np.flip(np.array(alpha_prime[1:]))))
        values_prime2 = values_prime[j - i].reshape(self.N, self.N)
        C = np.conj(values_prime2 * self.C_mask)
        alpha_0 = B[0, 0]
        Gamma = 1 / alpha_0[None, None] * (np.matmul(B, np.conj(B).T) - np.matmul(C, np.conj(C).T))
        return Gamma

    def objective(self,alpha):
        Gamma = self.generate_Gamma(alpha)
        return - np.log(np.linalg.det(Gamma)) + np.trace(Gamma @ self.sCov)

    def objective_eig(self,eig,*args):
        U = args[0]
        return - np.sum(np.log(eig)) + np.trace(U @ np.diag(eig) @ U.T @ self.sCov)

    def objective_con(self,alpha):
        alpha = np.concatenate((alpha,np.zeros(self.N//2)))
        Gamma = self.generate_Gamma(alpha)
        return - np.log(np.linalg.det(Gamma)) + np.trace(Gamma @ self.sCov)

    def optimize(self):
        init_values = np.zeros(self.N)
        init_values[0] = np.random.uniform(low=1, high=20)
        for n in range(1, self.N):
            init_values[n] = np.random.uniform(low=- self.K[n] * init_values[0] + 0.01,high=self.K[n] * init_values[0] - 0.01)
        result = optimize.minimize(self.objective, init_values, method="SLSQP",constraints=self.generate_constraints())
        Gamma_est = self.generate_Gamma(result.x)
        U_ToepCube, S_ToepCube, _ = np.linalg.svd(Gamma_est)

        result_eig1 = optimize.minimize(self.objective_eig, S_ToepCube,args=U_ToepCube, method="SLSQP", constraints=self.generate_constraints_eig())
        Gamma_est_eig = U_ToepCube @ np.diag(result_eig1.x) @ U_ToepCube.T

        return Gamma_est,Gamma_est_eig

    def optimize_red(self):
        init_values = np.zeros(self.N//2)
        init_values[0] = np.random.uniform(low=1, high=20)
        for n in range(1, self.N//2):
            init_values[n] = np.random.uniform(low=- self.K_red[n] * init_values[0] + 0.01,high=self.K_red[n] * init_values[0] - 0.01)
        result = optimize.minimize(self.objective_con, init_values, method="SLSQP",constraints=self.generate_constraints_red())
        Gamma_est = self.generate_Gamma(np.concatenate((result.x,np.zeros(self.N//2))))
        U_ToepCube, S_ToepCube, _ = np.linalg.svd(Gamma_est)

        result_eig1 = optimize.minimize(self.objective_eig, S_ToepCube,args=U_ToepCube, method="SLSQP", constraints=self.generate_constraints_eig())
        Gamma_est_eig = U_ToepCube @ np.diag(result_eig1.x) @ U_ToepCube.T

        return Gamma_est,Gamma_est_eig

class ToepStieltjes_estimator():
    def __init__(self,N,n_samples,sCov,K,K_reduced):
        self.sCov = sCov
        self.N = N
        self.K = K
        self.K_red = K_reduced
        rand_matrix = np.random.randn(N, N)
        self.B_mask = np.tril(rand_matrix)
        self.B_mask[self.B_mask != 0] = 1
        self.C_mask = np.tril(rand_matrix, k=-1)
        self.C_mask[self.C_mask != 0] = 1

    def generate_constraints(self):
        constraints = []
        constraints.append({'fun':lambda alpha: alpha[0] - 0.01,'type': 'ineq'})

        for k in range(self.N):
            for l in range(self.N):
                def constraint_off(alpha,k=k,l=l):
                    alpha_prime = np.concatenate((np.array([0]), np.flip(np.array(alpha[1:]))))
                    values = np.concatenate((alpha, np.flip(np.array(alpha[1:]))))
                    i, j = np.ones((self.N, self.N)).nonzero()
                    values = values[j - i].reshape(self.N, self.N)
                    B = values * self.B_mask
                    values_prime = np.concatenate((alpha_prime, np.flip(np.array(alpha_prime[1:]))))
                    values_prime2 = values_prime[j - i].reshape(self.N, self.N)
                    C = np.conj(values_prime2 * self.C_mask)
                    alpha_0 = B[0, 0]
                    Gamma = 1 / alpha_0[None, None] * (np.matmul(B, np.conj(B).T) - np.matmul(C, np.conj(C).T))
                    return - Gamma[k,l]
                if k != l:
                    constraints.append({'fun': constraint_off,'type':'ineq'})

        for k in range(self.N):
            def constraint_g(alpha, k=k):
                N_h = len(alpha)
                alpha = np.concatenate((alpha, np.zeros(N_h)))
                N = len(alpha)
                rand_matrix = np.random.randn(N, N)
                B_mask = np.tril(rand_matrix)
                B_mask[B_mask != 0] = 1
                C_mask = np.tril(rand_matrix, k=-1)
                C_mask[C_mask != 0] = 1
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
                fac = np.sum(np.abs(Gamma[k, :])) - np.abs(Gamma[k, k])
                return Gamma[k, k] - fac
            constraints.append({'fun': constraint_g, 'type': 'ineq'})
        return constraints

    def generate_constraints_red(self):
        constraints = []
        constraints.append({'fun':lambda alpha: alpha[0] - 0.01,'type': 'ineq'})
        for k in range(self.N):
            for l in range(self.N):
                def constraint_off(alpha,k=k,l=l):
                    alpha = np.concatenate((alpha,np.zeros(self.N//2)))
                    alpha_prime = np.concatenate((np.array([0]), np.flip(np.array(alpha[1:]))))
                    values = np.concatenate((alpha, np.flip(np.array(alpha[1:]))))
                    i, j = np.ones((self.N, self.N)).nonzero()
                    values = values[j - i].reshape(self.N, self.N)
                    B = values * self.B_mask
                    values_prime = np.concatenate((alpha_prime, np.flip(np.array(alpha_prime[1:]))))
                    values_prime2 = values_prime[j - i].reshape(self.N, self.N)
                    C = np.conj(values_prime2 * self.C_mask)
                    alpha_0 = B[0, 0]
                    Gamma = 1 / alpha_0[None, None] * (np.matmul(B, np.conj(B).T) - np.matmul(C, np.conj(C).T))
                    return - Gamma[k,l]
                if k != l:
                    constraints.append({'fun': constraint_off,'type':'ineq'})

        for k in range(self.N):
            def constraint_g(alpha,k=k):
                alpha = np.concatenate((alpha,np.zeros(self.N//2)))
                N_h = len(alpha)
                alpha = np.concatenate((alpha, np.zeros(N_h)))
                N = len(alpha)
                rand_matrix = np.random.randn(N, N)
                B_mask = np.tril(rand_matrix)
                B_mask[B_mask != 0] = 1
                C_mask = np.tril(rand_matrix, k=-1)
                C_mask[C_mask != 0] = 1
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
                fac = np.sum(np.abs(Gamma[k, :])) - np.abs(Gamma[k, k])
                return Gamma[k, k] - fac
            constraints.append({'fun': constraint_g,'type':'ineq'})
        return constraints

    def generate_constraints_eig(self):
        constraints = []
        constraints.extend({'fun': lambda eig, i=i: eig[i]- 0.00001, 'type': 'ineq'} for i in range(0, self.N))
        return constraints

    def generate_Gamma(self,alpha):
        alpha_prime = np.concatenate((np.array([0]), np.flip(np.array(alpha[1:]))))
        values = np.concatenate((alpha, np.flip(np.array(alpha[1:]))))
        i, j = np.ones((self.N, self.N)).nonzero()
        values = values[j - i].reshape(self.N, self.N)
        B = values * self.B_mask
        values_prime = np.concatenate((alpha_prime, np.flip(np.array(alpha_prime[1:]))))
        values_prime2 = values_prime[j - i].reshape(self.N, self.N)
        C = np.conj(values_prime2 * self.C_mask)
        alpha_0 = B[0, 0]
        Gamma = 1 / alpha_0[None, None] * (np.matmul(B, np.conj(B).T) - np.matmul(C, np.conj(C).T))
        return Gamma

    def objective(self,alpha):
        Gamma = self.generate_Gamma(alpha)
        return - np.log(np.linalg.det(Gamma)) + np.trace(Gamma @ self.sCov)

    def objective_eig(self,eig,*args):
        U = args[0]
        return - np.sum(np.log(eig)) + np.trace(U @ np.diag(eig) @ U.T @ self.sCov)

    def objective_con(self,alpha):
        alpha = np.concatenate((alpha,np.zeros(self.N//2)))
        Gamma = self.generate_Gamma(alpha)
        return - np.log(np.linalg.det(Gamma)) + np.trace(Gamma @ self.sCov)

    def optimize(self):
        init_values = np.zeros(self.N)
        init_values[0] = np.random.uniform(low=1, high=20)
        for n in range(1, self.N):
            init_values[n] = np.random.uniform(low=- self.K[n] * init_values[0] + 0.01,high=self.K[n] * init_values[0] - 0.01)
        result = optimize.minimize(self.objective, init_values, method="SLSQP",constraints=self.generate_constraints())
        Gamma_est = self.generate_Gamma(result.x)
        U_ToepCube, S_ToepCube, _ = np.linalg.svd(Gamma_est)

        result_eig1 = optimize.minimize(self.objective_eig, S_ToepCube,args=U_ToepCube, method="SLSQP", constraints=self.generate_constraints_eig())
        Gamma_est_eig = U_ToepCube @ np.diag(result_eig1.x) @ U_ToepCube.T

        return Gamma_est,Gamma_est_eig

    def optimize_red(self):
        init_values = np.zeros(self.N//2)
        init_values[0] = np.random.uniform(low=1, high=20)
        for n in range(1, self.N//2):
            init_values[n] = np.random.uniform(low=- self.K_red[n] * init_values[0] + 0.01,high=self.K_red[n] * init_values[0] - 0.01)
        result = optimize.minimize(self.objective_con, init_values, method="SLSQP",constraints=self.generate_constraints_red())
        Gamma_est = self.generate_Gamma(np.concatenate((result.x,np.zeros(self.N//2))))
        U_ToepCube, S_ToepCube, _ = np.linalg.svd(Gamma_est)

        result_eig1 = optimize.minimize(self.objective_eig, S_ToepCube,args=U_ToepCube, method="SLSQP", constraints=self.generate_constraints_eig())
        Gamma_est_eig = U_ToepCube @ np.diag(result_eig1.x) @ U_ToepCube.T

        return Gamma_est,Gamma_est_eig


## comparison estimators
class UMVUE_estimator():
    def __init__(self,N,N_SAMPLES,sCov,K,K_reduced):
        self.sCov = sCov

    def optimize(self):
        c = np.zeros(self.sCov.shape[0])
        for i in range(len(c)):
            c[i] = np.mean(np.concatenate((np.diag(self.sCov,k=i),np.diag(self.sCov,k=-i))))

        C_est = sc.linalg.toeplitz(c,c)
        return C_est

class OAS_estimator():
    def __init__(self,N,N_SAMPLES,sCov,K,K_reduced):
        self.sCov = sCov
        self.N = N
        self.N_SAMPLES = N_SAMPLES

    def optimize(self):
        F = np.trace(self.sCov) / self.N * np.eye(self.N)
        rho = min(((1 - 2 / self.N) * np.trace(self.sCov @ self.sCov) + np.trace(self.sCov) ** 2) / (
                (self.N_SAMPLES + 1 - 2 / self.N) * (np.trace(self.sCov @ self.sCov) - np.trace(self.sCov) ** 2 / self.N)), 1)
        return (1 - rho) * self.sCov + rho * F

class DFT_estimator():
    def __init__(self,N,N_SAMPLES,sCov,K,K_reduced):
        self.sCov = sCov
        self.N = N

        self.DFT = np.zeros((N, N), dtype=np.cfloat)
        for m in range(N):
            for n in range(N):
                self.DFT[m, n] = 1 / np.sqrt(N) * np.exp(-1j * 2 * math.pi * (m * n) / N)

    def optimize(self):
        DFT_Cov = np.conjugate(self.DFT).T @ np.diag(np.diag(self.DFT @ self.sCov @ np.conjugate(self.DFT).T)) @ self.DFT
        DFT_Cov = np.real(DFT_Cov)
        return DFT_Cov

class OAS_UMVUE_estimator():
    def __init__(self,N,N_SAMPLES,sCov,K,K_reduced):
        self.sCov = sCov
        self.N = N
        self.N_SAMPLES = N_SAMPLES

    def optimize(self):
        c = np.zeros(self.sCov.shape[0])
        for i in range(len(c)):
            c[i] = np.mean(np.concatenate((np.diag(self.sCov, k=i), np.diag(self.sCov, k=-i))))
        C_est = sc.linalg.toeplitz(c, c)

        F = np.trace(C_est) / self.N * np.eye(self.N)
        rho = min(((1 - 2 / self.N) * np.trace(C_est @ C_est) + np.trace(C_est) ** 2) / ((self.N_SAMPLES + 1 - 2 / self.N) * (np.trace(C_est @ C_est) - np.trace(C_est) ** 2 / self.N)), 1)
        return (1 - rho) * C_est + rho * F