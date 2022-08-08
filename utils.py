import time
from functools import wraps
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
import scipy
import math
import matplotlib.pyplot as plt


def network_architecture_search():
    LD = np.random.choice([12,16,20]).item()
    conv_layer = np.random.choice([0,1,2,3]).item()
    total_layer = np.random.choice([3,4]).item()
    out_channels = np.random.choice([64,128,256]).item()
    k_size = np.random.choice([5,7]).item()

    return LD,conv_layer,total_layer,out_channels,k_size
