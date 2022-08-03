import math
import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
import torch.distributions as D
from math import floor
import math
from abc import abstractmethod
from typing import List, Callable, Union, Any, TypeVar, Tuple

Tensor = TypeVar('Tensor')
ndarray = TypeVar('ndarray')

def create_DFT(n_ant):
    F = np.zeros((n_ant,n_ant),dtype=np.complex)
    for i in range(n_ant):
        for j in range(n_ant):
            F[i,j] = 1/(np.sqrt(n_ant)) * np.exp( - 1j * 2 * math.pi * i * j/n_ant)
    return F

class my_VAE_DFT(nn.Module):
    def __init__(self,latent_dim):
        super().__init__()
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Conv1d(1,8,7,2,3),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.Conv1d(8,32,7,2,3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32,128,7,2,3),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )

        self.fc_mu = nn.Linear(8 * 128, self.latent_dim)
        self.fc_var = nn.Linear(8 * 128, self.latent_dim)


        self.decoder_input = nn.Linear(self.latent_dim, 8*128)

        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(128,32,7,2,3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.ConvTranspose1d(32, 8, 7, 2, 3),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.ConvTranspose1d(8, 1, 7, 2, 3),
            nn.BatchNorm1d(1),
            nn.ReLU(),
        )

        self.final_layer = nn.Linear(57, 96)

    def encode(self, x):
        out = self.encoder(x)
        out = nn.Flatten()(out)
        mu, log_std = self.fc_mu(out), self.fc_var(out)
        return mu, log_std

    def reparameterize(self, log_var, mu):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self,z):
        out = self.decoder_input(z)
        bs = out.size(0)
        out = out.view(bs,128,-1)
        out = self.decoder(out)
        out = torch.squeeze(out)
        out = self.final_layer(out)
        mu_real,mu_imag,log_pre = out.chunk(3,dim=1)
        return mu_real,mu_imag,log_pre

    def estimating(self,x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(log_var, mu)
        mu_real,mu_imag,log_pre = self.decode(z)
        mu_out = mu_real + 1j * mu_imag
        Cov_out = torch.diag_embed(1 / (torch.exp(log_pre))) + 0j
        return mu_out, Cov_out

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(log_var, mu)
        mu_real,mu_imag,log_pre = self.decode(z)
        mu_out = mu_real + 1j * mu_imag
        Gamma = torch.diag_embed(torch.exp(log_pre)) + 0j
        return mu_out,Gamma, mu, log_var


class my_VAE_Toeplitz(nn.Module):
    def __init__(self,latent_dim,device):
        super().__init__()
        self.latent_dim = latent_dim
        self.device = device
        rand_matrix = torch.randn(32,32)
        self.B_mask = torch.tril(rand_matrix)
        self.B_mask[self.B_mask != 0] = 1
        self.B_mask = self.B_mask[None,:,:].to(self.device)

        self.C_mask = torch.tril(rand_matrix,diagonal=-1)
        self.C_mask[self.C_mask != 0] = 1
        self.C_mask = self.C_mask[None,:,:].to(self.device)

        self.encoder = nn.Sequential(
            nn.Conv1d(1,8,7,2,3),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.Conv1d(8,32,7,2,3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32,128,7,2,3),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )

        self.fc_mu = nn.Linear(64 * 128, self.latent_dim)
        self.fc_var = nn.Linear(64 * 128, self.latent_dim)


        self.decoder_input = nn.Linear(self.latent_dim, 64*128)

        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(128,32,7,2,3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.ConvTranspose1d(32, 8, 7, 2, 3),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.ConvTranspose1d(8, 1, 7, 2, 3),
            nn.BatchNorm1d(1),
            nn.ReLU(),
        )

        self.final_layer = nn.Linear(64, 64 + 63)

    def encode(self, x):
        out = self.encoder(x)
        out = nn.Flatten()(out)
        mu, log_std = self.fc_mu(out), self.fc_var(out)
        return mu, log_std

    def reparameterize(self, log_var, mu):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self,z):
        bs = z.size(0)
        out = self.decoder_input(z)
        out = out.view(bs,128,-1)
        out = self.decoder(out)
        out = torch.squeeze(out)
        out = self.final_layer(out)
        mu_real,mu_imag,alpha = out[:,:32],out[:,32:64],out[:,64:]

        alpha_0 = alpha[:,0] # BS
        alpha_0 = torch.exp(alpha_0)[:,None]  # BS,1
        alpha_intermediate = alpha_0.clone()
        alpha_intermediate[alpha_0 > 3000] = 3000
        alpha_0 = alpha_intermediate.clone()
        alpha_rest = alpha[:,1:] # BS, 62
        alpha_rest = 0.022 * torch.abs(alpha_0) * nn.Tanh()(alpha_rest)
        alpha_rest = torch.complex(alpha_rest[:,:31], alpha_rest[:,31:])
        Alpha = torch.cat((alpha_0, alpha_rest), dim=1)
        Alpha_prime = torch.cat((torch.zeros(bs,1).to(self.device), Alpha[:, 1:].flip(1)),dim=1)
        values = torch.cat((Alpha, Alpha[:,1:].flip(1)), dim=1)
        i, j = torch.ones(32, 32).nonzero().T
        values = values[:,j - i].reshape(bs,32,32)
        B = values * self.B_mask
        values_prime = torch.cat((Alpha_prime, Alpha_prime[:,1:].flip(1)), dim=1)
        i, j = torch.ones(32, 32).nonzero().T
        values_prime2 = values_prime[:, :, j - i].reshape(bs,32,32)
        C = torch.conj(values_prime2 * self.C_mask)
        return mu_real,mu_imag,B,C

    def estimating(self,x):
        mu, log_var = self.encode(x)
        mu_real, mu_imag, B, C = self.decode(mu)
        mu_out = mu_real + 1j * mu_imag
        alpha_0 = B[:, 0, 0]
        Gamma = 1 / alpha_0[:, None, None] * (torch.matmul(B, torch.conj(B).permute(0, 2, 1)) - torch.matmul(C, torch.conj(C).permute(0, 2, 1)))
        L, U = torch.linalg.eigh(Gamma)
        Cov_out = U @ torch.diag_embed(1 / L).cfloat() @ U.mH
        return mu_out, Cov_out

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(log_var, mu)
        mu_real,mu_imag,B,C = self.decode(z)
        mu_out = mu_real + 1j * mu_imag
        alpha_0 = B[:,0,0]
        Gamma = 1 / alpha_0[:,None,None] * (torch.matmul(B, torch.conj(B).permute(0, 2, 1)) - torch.matmul(C,torch.conj(C).permute(0,2,1)))
        return mu_out,B,C,Gamma, mu, log_var

class VAECircCov(nn.Module):

    def __init__(self,
                 in_channels: int,
                 stride: int,
                 kernel_szs: List,
                 latent_dim: int,
                 hidden_dims: List = None,
                 input_size: int = 16,
                 input_dim: int = 1,
                 **kwargs) -> None:
        super(VAECircCov, self).__init__()

        self.latent_dim = latent_dim
        self.input_size = input_size  # this refers to the data size x
        self.input_dim = input_dim  # this refers to the actual input dimensionality of x or cond
        self.in_channels = in_channels
        self.pad = 1
        self.stride = stride
        self.use_iaf = kwargs['use_iaf']
        #self.act = get_activation(kwargs['act'])
        self.act = nn.ReLU()
        self.device = kwargs['device']
        self.cond_as_input = kwargs['cond_as_input']
        self.pi = torch.tensor(math.pi).to(self.device)
        self.lambda_z = torch.tensor(0.1, device=self.device)
        self.M = torch.tensor(self.input_size, device=self.device)

        self.embed_data = nn.Conv1d(in_channels, in_channels, kernel_size=1) if self.input_dim == 1 else nn.Conv2d(in_channels, in_channels, kernel_size=1)

        modules = []
        if hidden_dims is None:
            hidden_dims = [8, 16, 32, 64]
        self.hidden_dims = hidden_dims
        if kernel_szs is None:
            kernel_szs = [3 for _ in range(len(hidden_dims))]

        # in_channels += 1  # to account for the extra label channel

        # calculate encoder output dims
        tmp_size = self.input_size if self.input_dim == 1 else int(0.5*self.input_size)
        for i in range(len(hidden_dims)):
            tmp_size = floor((tmp_size + 2*self.pad - kernel_szs[i]) / self.stride + 1)
            if tmp_size < 1:
                hidden_dims = hidden_dims[:i]
                kernel_szs = kernel_szs[:i]
                break
            self.pre_latent = tmp_size

        # build encoder
        if self.input_dim == 1:
            for (i, h_dim) in enumerate(hidden_dims):
                modules.append(
                    nn.Sequential(
                        nn.Conv1d(in_channels,
                                  out_channels=h_dim,
                                  kernel_size=kernel_szs[i],
                                  stride=self.stride,
                                  padding=self.pad),
                        nn.BatchNorm1d(h_dim),
                        self.act)
                )
                in_channels = h_dim
            self.fc_mu = nn.Linear(hidden_dims[-1]*self.pre_latent, self.latent_dim)
            self.fc_var = nn.Linear(hidden_dims[-1]*self.pre_latent, self.latent_dim)

        elif self.input_dim == 2:
            for (i, h_dim) in enumerate(hidden_dims):
                modules.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels,
                                  out_channels=h_dim,
                                  kernel_size=kernel_szs[i],
                                  stride=self.stride,
                                  padding=self.pad),
                        nn.BatchNorm2d(h_dim),
                        self.act)
                    )
                in_channels = h_dim
            self.fc_mu = nn.Linear(hidden_dims[-1] * self.pre_latent**2, self.latent_dim)
            self.fc_var = nn.Linear(hidden_dims[-1] * self.pre_latent**2, self.latent_dim)

        else:
            raise ValueError('Only one- or two-dimensional input can be considered!')
        self.encoder = nn.Sequential(*modules)
        in_channels = h_dim

        # build decoder
        modules = []

        #if self.use_iaf:
        #    self.iaf = IAF(kwargs['n_blocks'], self.latent_dim, kwargs['hidden_iaf'], 1, cond_size=None)
        #else:
        self.decoder_embed_latent = nn.Linear(latent_dim, latent_dim)

        hidden_dims.reverse()
        kernel_szs.reverse()

        if self.input_dim == 1:
            self.decoder_input = nn.Linear(self.latent_dim, hidden_dims[0] * self.pre_latent)

            for i in range(len(hidden_dims)):
                modules.append(
                    nn.Sequential(
                        nn.ConvTranspose1d(hidden_dims[i],
                                           hidden_dims[i + 1] if i < len(hidden_dims)-1 else hidden_dims[-1],
                                           kernel_size=kernel_szs[i],
                                           stride=self.stride,
                                           padding=self.pad),
                        nn.BatchNorm1d(hidden_dims[i + 1] if i < len(hidden_dims)-1 else hidden_dims[-1]),
                        self.act)
                )

        elif self.input_dim == 2:
            self.decoder_input = nn.Linear(self.latent_dim, hidden_dims[0] * self.pre_latent**2)

            for i in range(len(hidden_dims)):
                modules.append(
                    nn.Sequential(
                        nn.ConvTranspose2d(hidden_dims[i],
                                           hidden_dims[i + 1] if i < len(hidden_dims)-1 else hidden_dims[-1],
                                           kernel_size=kernel_szs[i],
                                           stride=self.stride,
                                           padding=self.pad),
                        nn.BatchNorm2d(hidden_dims[i + 1] if i < len(hidden_dims)-1 else hidden_dims[-1]),
                        self.act)
                    )

        self.decoder = nn.Sequential(*modules)

        # calculate decoder output dims
        self.pre_out = self.pre_latent
        for i in range(len(hidden_dims)):
            self.pre_out = (self.pre_out - 1) * self.stride - 2 * self.pad + kernel_szs[i]
        self.pre_out = self.pre_out**2 if self.input_dim == 2 else self.pre_out

        self.final_layer = nn.Linear(hidden_dims[-1] * self.pre_out, int(1.5*self.input_size))

        #self.F = dft_matrix(int(self.input_size/2)).to(self.device)
        self.F = torch.tensor(create_DFT(input_dim)).to(self.device)
        #self.apply(WeightInitializer(get_initializer(kwargs['init'])).initialize_weights)


    def encode(self, data: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network and returns the latent codes.
        :param data: (Tensor) Input tensor to encoder [B x N]
        :return: (Tensor) List with mean and variance vector assuming a diagonal gaussian posterior
        """
        result = self.encoder(data)
        result = torch.flatten(result, start_dim=1)
        # Split the result into mu and var components of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_std = self.fc_var(result)

        return [mu, log_std]


    def decode(self, z: Tensor, **kwargs) -> Tensor:
        result = self.decoder_embed_latent(z)
        jacobians = 0
        result = self.decoder_input(result)  # self.decoder_embed_cond(kwargs['cond']))
        # shape result to fit dimensions after all conv layers of encoder and decode
        if self.input_dim == 1:
            result = result.view(len(result), -1, self.pre_latent)
        elif self.input_dim == 2:
            result = result.view(len(result), -1, self.pre_latent, self.pre_latent)
        result = self.decoder(result)
        # flatten result and put into final layer to predict [mu_real, mu_imag, c_diag]
        result = torch.flatten(result, start_dim=1)
        result = self.final_layer(result)
        return result, z, jacobians

    def reparameterize(self,mu,log_std,device):
        std = torch.exp(log_std)
        eps = torch.randn_like(std)
        return mu + eps * std, eps

    def forward(self, data: Tensor, **kwargs) -> List[Tensor]:
        # data = torch.flatten(data, start_dim=1)
        if self.cond_as_input:
            encoder_input = kwargs['cond']
        else:
            encoder_input = data
        encoder_input = self.embed_data(encoder_input.unsqueeze(1))
        data = data[:, :int(data.shape[1]/2)] + 1j * data[:, int(data.shape[1]/2):]

        mu_enc, log_std_enc = self.encode(encoder_input)

        if kwargs['train']:
            z_0, eps = self.reparameterize(mu_enc, log_std_enc, device=self.device)
        else:
            z_0, eps = mu_enc, torch.zeros_like(mu_enc).to(self.device)

        out, z, jacobians = self.decode(z_0)
        mu_out_real, mu_out_imag, log_prec = out.chunk(3, dim=1)
        mu_out = mu_out_real + 1j * mu_out_imag
        # mu_out[:] = 0

        # var_dec = (self.M / torch.sum(var_dec, dim=-1, keepdim=True)) * var_dec + 1e-6

        if not kwargs['train']:
            c_diag = torch.diag_embed(torch.exp(-log_prec)).type(torch.complex64).to(self.device)
            C = self.F.conj().T @ c_diag @ self.F
            mu = mu_out @ self.F.conj()
        else:
            C, mu = None, None

        return [mu_out, data, log_prec, mu_enc, log_std_enc, z_0, z, jacobians, mu, C]


    def loss_function(self, *args, **kwargs) -> dict:
        mu_out, data, log_prec, mu_enc, log_std_enc, z_0, z, jacobians, mu, C = args

        rec_loss = torch.sum(log_prec - (log_prec.exp() * ((data - mu_out).abs() ** 2)), dim=1) - self.M * torch.log(self.pi)

        kld_loss = kl_div_diag_gauss(mu_enc, log_std_enc)

        sp_loss = torch.sum(torch.exp(-log_prec), dim=-1)

        loss = rec_loss - kwargs['alpha'] * kld_loss
        loss_back = rec_loss.mean() - kwargs['alpha'] * torch.maximum(kld_loss.mean(), self.lambda_z) \
            - kwargs['beta'] * sp_loss.mean()

        return {'loss': loss, 'loss_back': loss_back, 'rec_loss': rec_loss, 'KLD': kld_loss}


    def sample(self, num_samples: int, **kwargs) -> Tensor:
        """
        Samples from the latent space and returns the corresponding image space map.
        :param num_samples: (Int) Number of samples
        :return: (Tensor)
        """

        # sample n_samples times in latent space
        z_0 = torch.randn(num_samples, self.latent_dim).to(self.device)

        # decode samples
        out, z, _ = self.decode(z_0)
        mu_out_real, mu_out_imag = out.chunk(3, dim=1)
        samples = mu_out_real + 1j * mu_out_imag

        return samples

    def generate_decoder(self, x: Tensor, num_samples: int, order: int, **kwargs) -> Tensor:
        """
        Samples from the decoder outputs for the specified model order given an input sample
        :param x: input sample
        :param num_samples: number of samples to create
        :param order: model order to create samples for
        :param kwargs:
        :return:
        """
        mu_out, _, log_prec, _, _, _, _, _ = self.forward(x, **kwargs)

        # gather complex normal distributed samples~(0,I) and reshape mean and std
        eps = torch.randn((len(x), num_samples, order, mu_out.shape[-1]), dtype=torch.cfloat).to(self.device)
        mu_out = mu_out.view(len(x), 1, 1, -1)
        std_out = torch.exp(-0.5 * log_prec).view(len(x), 1, 1, -1).type(torch.cfloat)

        # create gains such that there is a max. 9 dB gap
        rho = (1/8 + 7/8 * torch.rand((len(x), num_samples, order, 1))).to(self.device)
        rho[:, :, 0, 0] = 1
        rho = (rho / rho.sum(2, keepdim=True)).type(torch.cfloat)

        # create the actual samples~(mu,C) and sum up along model order
        x_gen = std_out * eps
        x_gen = torch.sum(torch.sqrt(rho) * x_gen, dim=2)

        return x_gen

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image.
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """
        return self.forward(x, **kwargs)[0]