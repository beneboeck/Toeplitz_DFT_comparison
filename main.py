import torch
import os
import torch.nn as nn
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


now = datetime.datetime.now()
date = str(now)[:10]
time = str(now)[11:16]
time = time[:2] + '_' + time[3:]
dir_path = '/home/ga42kab/lrz-nashome/Toeplitz_DFT_comparison/models/time_' + time
overall_path = '/home/ga42kab/lrz-nashome/Toeplitz_DFT_comparison/'
#overall_path = '../Simulations/Toeplitz_DFT_comparison/'
#dir_path = '../Simulations/Toeplitz_DFT_comparison/models/time_' + time
os.mkdir (dir_path)
glob_var_file = open(dir_path + '/glob_var_file.txt','w')
log_file = open(dir_path + '/log_file.txt','w')

device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')

#train_data = np.load('../Simulations/Toeplitz_DFT_comparison/data/scm3gpp_3-path-train.npy','r')[:40000,:]
#val_data = np.load('../Simulations/Toeplitz_DFT_comparison/data/scm3gpp_3-path-eval.npy','r')[:5000,:]
#test_data = np.load('../Simulations/Toeplitz_DFT_comparison/data/scm3gpp_3-path-test.npy','r')[:5000,:]

#c_train_data = np.load('../Simulations/Toeplitz_DFT_comparison/data/scm3gpp_3-path-cov-train.npy','r')[:40000,:]
#c_val_data = np.load('../Simulations/Toeplitz_DFT_comparison/data/scm3gpp_3-path-cov-eval.npy','r')[:5000,:]
#c_test_data = np.load('../Simulations/Toeplitz_DFT_comparison/data/scm3gpp_3-path-cov-test.npy','r')[:5000,:]

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
SNR_db = 5
BATCHSIZE = 50
G_EPOCHS = 700
LEARNING_RATE = 6e-5
LAMBDA = torch.tensor(1).to(device)
SNR_eff = 10**(SNR_db/10)
sig_n_val = math.sqrt(np.mean(np.linalg.norm(val_data,axis=1)**2)/(32 * SNR_eff))
sig_n_train = math.sqrt(np.mean(np.linalg.norm(train_data,axis=1)**2)/(32 * SNR_eff))
sig_n_test = math.sqrt(np.mean(np.linalg.norm(test_data,axis=1)**2)/(32 * SNR_eff))

LD, conv_layer, total_layer, out_channel, k_size, cov_type, prepro = network_architecture_search_VAE()
if not(exists(overall_path + 'NAS_file.csv')):
    csvfile = open(overall_path + 'NAS_file.txt','w')
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(['Time','LD_VAE', 'conv_layer', 'total_layer', 'out_channel', 'k_size', 'cov_type','prepro','Est','NMSEcov'])
    csvfile.close()
glob_file = open(dir_path + '/glob_var_file.txt','w') # only the important results and the framework
log_file = open(dir_path + '/log_file.txt','w') # log_file which keeps track of the training and such stuff
glob_file.write('Date: ' +date +'\n')
glob_file.write('Time: ' + time + '\n\n')
glob_file.write('BATCHSIZE: ' + str(BATCHSIZE) +'\n')
glob_file.write('G_EPOCHS: ' +str(G_EPOCHS) +'\n')
glob_file.write(f'Learning Rate: {LEARNING_RATE}\n')
glob_file.write(f'SNR_db: {SNR_db}\n')
log_file.write('Date: ' +date +'\n')
log_file.write('Time: ' + time + '\n')
log_file.write('global variables successfully defined\n\n')
print('global var successful')
glob_file.write(f'\nlatent Dim VAE: {LD}\n')
glob_file.write(f'conv_layer: {conv_layer}\n')
glob_file.write(f'total_layer: {total_layer}\n')
glob_file.write(f'out_channel: {out_channel}\n')
glob_file.write(f'k_size: {k_size}\n')
glob_file.write(f'cov_type: {cov_type}\n')
glob_file.write(f'prepro: {prepro}\n')
print('SETUP')
print(LD, conv_layer, total_layer, out_channel, k_size, cov_type, prepro)
if cov_type == 'DFT':
    F = create_DFT(N_ANT)
    train_data = np.einsum('ij,kj->ki',F,train_data)
    test_data = np.einsum('ij,kj->ki', F, test_data)
    val_data = np.einsum('ij,kj->ki', F, val_data)
    C_train_data = F[None,:,:] @ C_train_data @ F.transpose().conjugate()[None,:,:]
    C_test_data = F[None,:,:] @ C_test_data @ F.transpose().conjugate()[None,:,:]
    C_val_data = F[None,:,:] @ C_val_data @ F.transpose().conjugate()[None,:,:]

train_data_noisy = train_data + sig_n_train/np.sqrt(2) * (np.random.randn(*train_data.shape) + 1j * np.random.randn(*train_data.shape))
val_data_noisy = val_data + sig_n_val/np.sqrt(2) * (np.random.randn(*val_data.shape) + 1j * np.random.randn(*val_data.shape))
test_data_noisy = test_data + sig_n_test/np.sqrt(2) * (np.random.randn(*test_data.shape) + 1j * np.random.randn(*test_data.shape))

train_dataset = ds.dataset(train_data,train_data_noisy,C_train_data)
val_dataset = ds.dataset(val_data,val_data_noisy,C_val_data)
test_dataset = ds.dataset(test_data,test_data_noisy,C_test_data)
train_dataloader = DataLoader(train_dataset,batch_size=BATCHSIZE,shuffle=True)
test_dataloader = DataLoader(test_dataset,batch_size=len(test_dataset),shuffle=True)
val_dataloader = DataLoader(val_dataset,batch_size=len(val_dataset),shuffle=True)

# standardize data, such that E[||h||^2] = M
#train_data = train_data - np.mean(train_data,axis=0)
#train_data = train_data/(np.sqrt(np.mean(np.linalg.norm(train_data,axis=1)**2))) * np.sqrt(N_ANT)

#val_data = val_data - np.mean(val_data,axis=0)
#val_data = val_data/(np.sqrt(np.mean(np.linalg.norm(val_data,axis=1)**2))) * np.sqrt(N_ANT)
#val_data_noisy = val_data * sig_n * np.random.randn(*val_data.shape)

#test_data = test_data - np.mean(test_data,axis=0)
#test_data = test_data/(np.sqrt(np.mean(np.linalg.norm(test_data,axis=1)**2))) * np.sqrt(N_ANT)
VAE = nw.my_VAE(cov_type,LD,conv_layer,total_layer,out_channel,k_size,prepro,device).to(device)

#VAE = nw.my_VAE_Toeplitz(16,device)

risk_list,KL_list,RR_list,eval_risk, eval_NMSE_estimation = tr.training_gen_NN(LEARNING_RATE,cov_type, VAE, train_dataloader,val_dataloader, G_EPOCHS, LAMBDA,sig_n_val,device, log_file,dir_path)

VAE.eval()
ev.save_risk(risk_list,RR_list,KL_list,dir_path,'Risks')

ev.save_risk_single(eval_risk,dir_path,'Evaluation - ELBO')
ev.save_risk_single(eval_NMSE_estimation,dir_path,'Evaluation - NMSE estimation')

torch.save(VAE.state_dict(),dir_path + '/model_dict')
NMSE_estimation = ev.channel_estimation(VAE, val_dataloader, sig_n_val, dir_path, device)
NMSE_cov = ev.NMSE_Cov(VAE,val_dataloader,device)
glob_file.write(f'NMSE_estimation: {NMSE_estimation}\n')
glob_file.write(f'NMSE_cov: {NMSE_cov}\n')
csv_file = open(overall_path + 'NAS_file.txt','a')
csv_writer = csv.writer(csv_file)
csv_writer.writerow([time,LD, conv_layer, total_layer, out_channel, k_size, cov_type,prepro,NMSE_estimation,NMSE_cov.item()])
glob_file.close()
csv_file.close()