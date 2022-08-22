import time
from functools import wraps
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
import scipy
import math
import matplotlib.pyplot as plt

def create_DFT(n_ant):
    F = np.zeros((n_ant,n_ant),dtype=np.complex)
    for i in range(n_ant):
        for j in range(n_ant):
            F[i,j] = 1/(np.sqrt(n_ant)) * np.exp( 1j * 2 * math.pi * i * j/n_ant)
    return F


def network_architecture_search_VAE():
    LD = np.random.choice([32,40,48,56]).item()
    conv_layer = np.random.choice([0,1,2,3]).item()
    total_layer = np.random.choice([3,4,5]).item()
    out_channel = np.random.choice([64,128]).item()
    k_size = np.random.choice([7,9]).item()
    cov_type = np.random.choice(['Toeplitz','Toeplitz','DFT']).item()
    prepro = np.random.choice(['None', 'DFT']).item()

    return LD, conv_layer, total_layer, out_channel, k_size, cov_type, prepro