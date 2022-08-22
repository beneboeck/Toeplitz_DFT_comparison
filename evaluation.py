import math
import torch
import training as tr
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

def save_risk(risk_list,RR_list,KL_list,model_path,title):
    risk = np.array(risk_list)
    np.save(model_path + '/risk_numpy',risk)
    plt.plot(risk,linewidth=1,label = 'Risk')
    plt.plot(np.array(RR_list), linewidth=1, label = 'RR')
    plt.plot(np.array(KL_list), linewidth=1, label = 'KL')
    plt.title(title)
    plt.legend()
    plt.savefig(model_path + '/' + title,dpi = 300)
    plt.close()

def save_risk_single(risk_list,model_path,title):
    risk = np.array(risk_list)
    np.save(model_path + '/risk_numpy',risk)
    plt.plot(risk,linewidth=1)
    plt.title(title)
    plt.savefig(model_path + '/' + title,dpi = 300)
    plt.close()

def eval_val(model,dataloader_val,lamba,device):

    iterator = iter(dataloader_val)
    sample = iterator.next()
    sample = sample[0].to(device)
    mu_out, Gamma, mu, log_var = model(sample)
    Risk, RR, KL = tr.risk_free_bits(lamba, sample, mu, log_var, mu_out, Gamma)

    return Risk

def NMSE_Cov(model,dataloader,device):
    iterator = iter(dataloader)
    samples = iterator.next()
    sample = samples[0].to(device)
    sCov = samples[2].to(device)
    mu_out, Gamma, mu, log_var = model(sample)
    L, Q = torch.linalg.eigh(Gamma)
    lCov = Q @ torch.diag_embed(1/L.cfloat()) @ Q.mH
    NMSE = torch.mean(torch.sum(torch.abs(lCov - sCov)**2,dim=(1,2))/torch.sum(torch.abs(sCov)**2,dim=(1,2)))

    return NMSE

def channel_estimation(model,dataloader_val,sig_n,dir_path,device):
    NMSE_list = []
    for ind, samples in enumerate(dataloader_val):
        sample = samples[0].to(device) # BS, 2, N_ANT, SNAPSHOTS
        received_signal = samples[1].to(device)
        sample_oi = torch.squeeze(sample[:,0,:] + 1j * sample[:,1,:]) # BS, N_ANT
        received_signal_oi = torch.squeeze(received_signal[:,0,:] + 1j * received_signal[:,1,:])
        mu_out,Cov_out = model.estimating(sample) # BS,N_ANT complex, BS, N_ANT, N_ANT complex
        L,U = torch.linalg.eigh(Cov_out)
        inv_matrix = U @ torch.diag_embed(1/(L + sig_n ** 2)).cfloat() @ U.mH
        h_hat = mu_out + torch.einsum('ijk,ik->ij', Cov_out @ inv_matrix, (received_signal_oi - mu_out))
        NMSE = torch.mean(torch.sum(torch.abs(sample_oi - h_hat) ** 2, dim=1) / torch.sum(torch.abs(sample_oi) ** 2,dim=1)).detach().to('cpu')
        NMSE_list.append(NMSE)

    NMSE = np.mean(np.array(NMSE_list))
    return NMSE

