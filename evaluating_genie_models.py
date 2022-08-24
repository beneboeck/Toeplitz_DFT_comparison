import os
import torch.nn as nn
import torch
import numpy as np
from utils import *
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import math
import datetime
import used_models as nw
import dataset as ds
import training as tr
import evaluation as ev
from scipy.linalg import toeplitz
import csv
from os.path import exists

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

path_DFT_VAE = '/home/ga42kab/lrz-nashome/Toeplitz_DFT_comparison/models/time_19_29/model_dict'
path_TN_VAE = '/home/ga42kab/lrz-nashome/Toeplitz_DFT_comparison/models/time_19_27/model_dict'
path_TD_VAE = '/home/ga42kab/lrz-nashome/Toeplitz_DFT_comparison/models/time_17_22/model_dict'

# CREATING FILES AND DIRECTORY
now = datetime.datetime.now()
date = str(now)[:10]
time = str(now)[11:16]
time = time[:2] + '_' + time[3:]
dir_path = '/home/ga42kab/lrz-nashome/trajectory_channel_prediction/evaluation/time_' + time
os.mkdir (dir_path)

train_data = np.load('/home/ga42kab/lrz-nashome/Toeplitz_DFT_comparison/data/scm3gpp_3-path-train.npy','r')[:40000,:]
val_data = np.load('/home/ga42kab/lrz-nashome/Toeplitz_DFT_comparison/data/scm3gpp_3-path-eval.npy','r')[:5000,:]
test_data = np.load('/home/ga42kab/lrz-nashome/Toeplitz_DFT_comparison/data/scm3gpp_3-path-test.npy','r')[:5000,:]

c_train_data = np.load('/home/ga42kab/lrz-nashome/Toeplitz_DFT_comparison/data/scm3gpp_3-path-cov-train.npy','r')[:40000,:]
c_val_data = np.load('/home/ga42kab/lrz-nashome/Toeplitz_DFT_comparison/data/scm3gpp_3-path-cov-eval.npy','r')[:5000,:]
c_test_data = np.load('/home/ga42kab/lrz-nashome/Toeplitz_DFT_comparison/data/scm3gpp_3-path-cov-test.npy','r')[:5000,:]
print(c_train_data.shape)
C_train_data = np.zeros((c_train_data.shape[0],32,32),dtype=complex)
C_val_data = np.zeros((c_val_data.shape[0],32,32),dtype=complex)
C_test_data = np.zeros((c_test_data.shape[0],32,32),dtype=complex)
for i in range(C_train_data.shape[0]):
    C_train_data[i,:,:] = toeplitz(np.conjugate(c_train_data[i,:]))
for i in range(C_val_data.shape[0]):
    C_val_data[i, :, :] = toeplitz(np.conjugate(c_val_data[i, :]))
    C_test_data[i, :, :] = toeplitz(np.conjugate(c_train_data[i, :]))

N_ANT = 32
SNR_db_list = [-10,-5,0,5,10,15,20]
BATCHSIZE = 50
G_EPOCHS = 700
LEARNING_RATE = 6e-5
LAMBDA = torch.tensor(1).to(device)

glob_file = open(dir_path + '/glob_var_file.txt','w') # only the important results and the framework
log_file = open(dir_path + '/log_file.txt','w') # log_file which keeps track of the training and such stuff
csv_file = open(dir_path + '/csv_file.txt','w')
csv_writer = csv.writer(csv_file)
glob_file.write('Date: ' +date +'\n')
glob_file.write('Time: ' + time + '\n\n')
glob_file.write('BATCHSIZE: ' + str(BATCHSIZE) +'\n')
glob_file.write('G_EPOCHS: ' +str(G_EPOCHS) +'\n')
glob_file.write(f'Learning Rate: {LEARNING_RATE}\n')
log_file.write('Date: ' +date +'\n')
log_file.write('Time: ' + time + '\n')
log_file.write('global variables successfully defined\n\n')
print('global var successful')

LD_VAE, conv_layer, total_layer, out_channel, k_size, cov_type, prepro = 56,1,3,64,7,'DFT','DFT'
model_DFT_VAE = nw.my_VAE(cov_type,LD_VAE,conv_layer,total_layer,out_channel,k_size,prepro,device).to(device)
model_DFT_VAE.load_state_dict(torch.load(path_DFT_VAE,map_location=device))

LD_VAE, conv_layer, total_layer, out_channel, k_size, cov_type, prepro = 56,2,3,128,7,'Toeplitz','None'
model_TN_VAE = nw.my_VAE(cov_type,LD_VAE,conv_layer,total_layer,out_channel,k_size,prepro,device).to(device)
model_TN_VAE.load_state_dict(torch.load(path_TN_VAE,map_location=device))

LD_VAE, conv_layer, total_layer, out_channel, k_size, cov_type, prepro = 48,2,3,128,7,'Toeplitz','DFT'
model_TD_VAE = nw.my_VAE(cov_type,LD_VAE,conv_layer,total_layer,out_channel,k_size,prepro,device).to(device)
model_TD_VAE.load_state_dict(torch.load(path_TD_VAE,map_location=device))

csv_writer.writerow(SNR_db_list)

NMSE_est_DFT = []
NMSE_est_TD = []
NMSE_est_TN = []


F = create_DFT(N_ANT)
train_data_DFT = np.einsum('ij,kj->ki',F,train_data)
test_data_DFT = np.einsum('ij,kj->ki', F, test_data)
val_data_DFT = np.einsum('ij,kj->ki', F, val_data)
C_train_data_DFT = F[None,:,:] @ C_train_data @ F.transpose().conjugate()[None,:,:]
C_test_data_DFT = F[None,:,:] @ C_test_data @ F.transpose().conjugate()[None,:,:]
C_val_data_DFT = F[None,:,:] @ C_val_data @ F.transpose().conjugate()[None,:,:]

for SNR_db in SNR_db_list:

    SNR_eff = 10**(SNR_db/10)
    sig_n_val = math.sqrt(np.mean(np.linalg.norm(val_data,axis=1)**2)/(32 * SNR_eff))
    sig_n_train = math.sqrt(np.mean(np.linalg.norm(train_data,axis=1)**2)/(32 * SNR_eff))
    sig_n_test = math.sqrt(np.mean(np.linalg.norm(test_data,axis=1)**2)/(32 * SNR_eff))

    train_data_noisy = train_data + sig_n_train/np.sqrt(2) * (np.random.randn(*train_data.shape) + 1j * np.random.randn(*train_data.shape))
    val_data_noisy = val_data + sig_n_val/np.sqrt(2) * (np.random.randn(*val_data.shape) + 1j * np.random.randn(*val_data.shape))
    test_data_noisy = test_data + sig_n_test/np.sqrt(2) * (np.random.randn(*test_data.shape) + 1j * np.random.randn(*test_data.shape))

    train_data_noisy_DFT = train_data_DFT + sig_n_train/np.sqrt(2) * (np.random.randn(*train_data.shape) + 1j * np.random.randn(*train_data.shape))
    val_data_noisy_DFT = val_data_DFT + sig_n_val/np.sqrt(2) * (np.random.randn(*val_data.shape) + 1j * np.random.randn(*val_data.shape))
    test_data_noisy_DFT = test_data_DFT + sig_n_test/np.sqrt(2) * (np.random.randn(*test_data.shape) + 1j * np.random.randn(*test_data.shape))

    train_dataset = ds.dataset(train_data,train_data_noisy,C_train_data)
    val_dataset = ds.dataset(val_data,val_data_noisy,C_val_data)
    test_dataset = ds.dataset(test_data,test_data_noisy,C_test_data)
    train_dataloader = DataLoader(train_dataset,batch_size=BATCHSIZE,shuffle=True)
    test_dataloader = DataLoader(test_dataset,batch_size=len(test_dataset),shuffle=True)
    val_dataloader = DataLoader(val_dataset,batch_size=len(val_dataset),shuffle=True)

    train_dataset_DFT = ds.dataset(train_data_DFT,train_data_noisy_DFT,C_train_data_DFT)
    val_dataset_DFT = ds.dataset(val_data_DFT,val_data_noisy_DFT,C_val_data_DFT)
    test_dataset_DFT = ds.dataset(test_data_DFT,test_data_noisy_DFT,C_test_data_DFT)
    train_dataloader_DFT = DataLoader(train_dataset_DFT,batch_size=BATCHSIZE,shuffle=True)
    test_dataloader_DFT = DataLoader(test_dataset_DFT,batch_size=len(test_dataset),shuffle=True)
    val_dataloader_DFT = DataLoader(val_dataset_DFT,batch_size=len(val_dataset),shuffle=True)

    NMSE_DFT = ev.channel_estimation(model_DFT_VAE, test_dataloader_DFT, sig_n_test, dir_path, device)
    NMSE_TD = ev.channel_estimation(model_TD_VAE, test_dataloader, sig_n_test, dir_path, device)
    NMSE_TN = ev.channel_estimation(model_TN_VAE, test_dataloader, sig_n_test, dir_path, device)

    NMSE_est_DFT.append(NMSE_DFT)
    NMSE_est_TD.append(NMSE_TD)
    NMSE_est_TN.append(NMSE_TN)

csv_writer.writerow(NMSE_est_DFT)
csv_writer.writerow(NMSE_est_TD)
csv_writer.writerow(NMSE_est_TN)

csv_file.close()