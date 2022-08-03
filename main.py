import torch
import os
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import math
import datetime
import used_models as nw
import dataset as ds
import training as tr
import evaluation as ev


now = datetime.datetime.now()
date = str(now)[:10]
time = str(now)[11:16]
time = time[:2] + '_' + time[3:]
dir_path = '/home/ga42kab/lrz-nashome/Toeplitz_DFT_comparison/models/time_' + time
#dir_path = '../Simulations/Toeplitz_DFT_comparison/models/time_' + time
os.mkdir (dir_path)
glob_var_file = open(dir_path + '/glob_var_file.txt','w')
log_file = open(dir_path + '/log_file.txt','w')

device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')

#train_data = np.load('../Simulations/Toeplitz_DFT_comparison/data/scm3gpp_3-path-train.npy','r')
#val_data = np.load('../Simulations/Toeplitz_DFT_comparison/data/scm3gpp_3-path-eval.npy','r')
#test_data = np.load('../Simulations/Toeplitz_DFT_comparison/data/scm3gpp_3-path-test.npy','r')
train_data = np.load('/home/ga42kab/lrz-nashome/Toeplitz_DFT_comparison/data/scm3gpp_3-path-train.npy','r')
val_data = np.load('/home/ga42kab/lrz-nashome/Toeplitz_DFT_comparison/data/scm3gpp_3-path-eval.npy','r')
test_data = np.load('/home/ga42kab/lrz-nashome/Toeplitz_DFT_comparison/data/scm3gpp_3-path-test.npy','r')


N_ANT = 32
SNR_db = 5
BATCHSIZE = 64
cov_type = 'Toeplitz'
G_EPOCHS = 300
LEARNING_RATE = 5e-5
SNR_eff = 10**(SNR_db/10)
sig_n_val = math.sqrt(np.mean(np.linalg.norm(val_data,axis=1)**2)/(32 * SNR_eff))
sig_n_train = math.sqrt(np.mean(np.linalg.norm(train_data,axis=1)**2)/(32 * SNR_eff))
sig_n_test = math.sqrt(np.mean(np.linalg.norm(test_data,axis=1)**2)/(32 * SNR_eff))

if cov_type == 'DFT':
    F = nw.create_DFT(N_ANT)
    train_data = np.einsum('ij,kj->ki',F,train_data)
    test_data = np.einsum('ij,kj->ki', F, test_data)
    val_data = np.einsum('ij,kj->ki', F, val_data)

train_data_noisy = train_data + sig_n_train/np.sqrt(2) * (np.random.randn(*train_data.shape) + 1j * np.random.randn(*train_data.shape))
val_data_noisy = val_data + sig_n_val/np.sqrt(2) * (np.random.randn(*val_data.shape) + 1j * np.random.randn(*val_data.shape))
test_data_noisy = test_data + sig_n_test/np.sqrt(2) * (np.random.randn(*test_data.shape) + 1j * np.random.randn(*test_data.shape))

train_dataset = ds.dataset(train_data,train_data_noisy)
val_dataset = ds.dataset(val_data,val_data_noisy)
test_dataset = ds.dataset(test_data,test_data_noisy)

train_dataloader = DataLoader(train_dataset,batch_size=BATCHSIZE,shuffle=True)
test_dataloader = DataLoader(test_dataset,batch_size=8 * BATCHSIZE,shuffle=True)
val_dataloader = DataLoader(val_dataset,batch_size=8 * BATCHSIZE,shuffle=True)

# standardize data, such that E[||h||^2] = M
#train_data = train_data - np.mean(train_data,axis=0)
#train_data = train_data/(np.sqrt(np.mean(np.linalg.norm(train_data,axis=1)**2))) * np.sqrt(N_ANT)

#val_data = val_data - np.mean(val_data,axis=0)
#val_data = val_data/(np.sqrt(np.mean(np.linalg.norm(val_data,axis=1)**2))) * np.sqrt(N_ANT)
#val_data_noisy = val_data * sig_n * np.random.randn(*val_data.shape)

#test_data = test_data - np.mean(test_data,axis=0)
#test_data = test_data/(np.sqrt(np.mean(np.linalg.norm(test_data,axis=1)**2))) * np.sqrt(N_ANT)
if cov_type == 'DFT':
    VAE = nw.my_VAE_DFT(16).to(device)
if cov_type == 'Toeplitz':
    VAE = nw.my_VAE_Toeplitz(16,device).to(device)
#VAE = nw.my_VAE_Toeplitz(16,device)

risk_list,KL_list,RR_list,eval_risk, eval_NMSE_estimation = tr.training_gen_NN(LEARNING_RATE,cov_type, VAE, train_dataloader,val_dataloader, G_EPOCHS, torch.tensor(1).to(device),sig_n_val,device, log_file,dir_path)

VAE.eval()
ev.save_risk(risk_list,RR_list,KL_list,dir_path,'Risks')

ev.save_risk_single(eval_risk,dir_path,'Evaluation - ELBO')
ev.save_risk_single(eval_NMSE_estimation,dir_path,'Evaluation - NMSE estimation')

torch.save(VAE.state_dict(),dir_path + '/model_dict')
