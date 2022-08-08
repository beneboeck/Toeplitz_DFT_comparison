import torch
import os
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import math
import datetime
import used_models as nw
import evaluation as ev


def risk_free_bits(lamba,x,mu,log_std,mu_out,Gamma):
    x = torch.complex(torch.squeeze(x[:,:,:32]), torch.squeeze(x[:,:,32:]))
    Gamma[torch.abs(torch.imag(Gamma)) < 10 ** (-5)] = torch.real(Gamma[torch.abs(torch.imag(Gamma)) < 10 ** (-5)]) + 0j
    M, pivots = torch.lu(Gamma)
    P, L, U = torch.lu_unpack(M, pivots)
    diagU = torch.diagonal(U, dim1=1, dim2=2)
    log_detGamma = torch.sum(torch.log(torch.abs(diagU)), dim=1)
    argument = torch.einsum('ij,ij->i', torch.conj(x - mu_out), torch.einsum('ijk,ik->ij', Gamma, x - mu_out))
    Rec_err = torch.real(torch.mean(- log_detGamma + argument))
    KL =  torch.mean(torch.sum(torch.max(lamba,-0.5 * (1 + 2 * log_std - mu ** 2 - (2 * log_std).exp())),dim=1))
    return Rec_err + KL,Rec_err,KL



def training_gen_NN(lr, cov_type,model, loader,dataloader_val, epochs, lamba,sig_n, device, log_file,dir_path):

    risk_list= []
    KL_list = []
    RR_list = []
    eval_risk = []
    eval_NMSE_estimation = []
    slope = -1.
    lr_adaption = False

    optimizer = torch.optim.Adam(lr=lr, params=model.parameters())

    print('Start Training ')
    log_file.write('\n\nStart Training\n')

    for i in range(epochs):
        print('epoch')
        print(i)
        for ind, samples in enumerate(loader):
            sample = samples[0]
            sample = sample.to(device)

            if (cov_type == 'Toeplitz'):
                mu_out,B,C,Gamma, mu, log_std = model(sample)
                Risk, RR, KL = risk_free_bits(lamba, sample, mu, log_std, mu_out, Gamma)

            if (cov_type == 'DFT'):
                mu_out,Gamma, mu, log_std = model(sample)
                Risk, RR, KL = risk_free_bits(lamba, sample, mu, log_std, mu_out, Gamma)

            optimizer.zero_grad()
            Risk.backward()
            optimizer.step()

        print(f'Risk: {Risk:.4f}, epoch: {i}')
        log_file.write(f'Risk: {Risk}, epoch: {i}\n')
        risk_list.append(Risk.detach().to('cpu'))
        KL_list.append(KL.detach().to('cpu'))
        RR_list.append(RR.detach().to('cpu'))
        with torch.no_grad():
            if i%5 == 0:
                model.eval()
                Risk = ev.eval_val(model, dataloader_val,cov_type, lamba, device, dir_path)
                NMSE_estimation = ev.channel_estimation(model, dataloader_val, sig_n, dir_path, device)
                eval_risk.append(Risk.detach().to('cpu'))
                eval_NMSE_estimation.append(NMSE_estimation)
                model.train()
                print(f'Evaluation - NMSE_estimation: {NMSE_estimation:.4f}, Risk: {Risk:.4f}')
                log_file.write(f'Evaluation - NMSE_estimation: {NMSE_estimation:.4f}, Risk: {Risk:.4f}\n')
                if (i > 40) & (lr_adaption == False):
                    x_range_lr = torch.arange(5)
                    x_lr = torch.ones(5, 2)
                    x_lr[:, 0] = x_range_lr
                    beta_lr = torch.linalg.inv(x_lr.T @ x_lr) @ x_lr.T @ torch.tensor(eval_risk[-5:])[:, None]
                    slope_lr = beta_lr[0]
                    print('slope lr')
                    print(slope_lr)
                    log_file.write(f'slope of Evaluation ELBO (for learning rate): {slope_lr}\n')
                    if slope_lr > 0:
                        print('LEARNING RATE IS ADAPTED!')
                        log_file.write(f'LEARNING RATE IS ADAPTED!\n')
                        optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr']/5
                        lr_adaption = True

                if (i > 200) & (lr_adaption == True):
                    x_range = torch.arange(15)
                    x = torch.ones(15, 2)
                    x[:, 0] = x_range
                    beta = torch.linalg.inv(x.T @ x) @ x.T @ torch.tensor(eval_risk[-15:])[:, None]
                    slope = beta[0]
                    print('slope')
                    print(slope)
                    log_file.write(f'slope of Evaluation ELBO: {slope}\n')

            if slope > 0:
                log_file.write('BREAKING CONDITION, slope positive\n')
                log_file.write(f'number epochs: {i}')
                break

    return risk_list,KL_list,RR_list,eval_risk, eval_NMSE_estimation

