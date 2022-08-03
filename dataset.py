import numpy as np
import h5py
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
import matplotlib.pyplot as plt
from matplotlib import cm
import scipy
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import math
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt

class dataset(Dataset):
    def __init__(self,h,y):
        #complex numpy input
        super().__init__()
        self.h = torch.zeros(h.shape[0],1,64)
        self.h[:,:,:32] = torch.tensor(np.real(h))[:,None,:]
        self.h[:,:,32:] = torch.tensor(np.imag(h))[:,None,:]
        self.y = torch.zeros(y.shape[0],1,64)
        self.y[:,:,:32] = torch.tensor(np.real(y))[:,None,:]
        self.y[:,:,32:] = torch.tensor(np.imag(y))[:,None,:]

    def __len__(self):
        return self.h.size(0)

    def __getitem__(self,idx):
        return self.h[idx,:],self.y[idx,:]