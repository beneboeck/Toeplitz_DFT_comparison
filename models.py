import math
import torch
from utils import rel_mse, circulant, dft_matrix, WeightInitializer, compute_lmmse
import numpy as np
from torch import nn
from torch.nn import functional as F
import torch.distributions as D
from math import floor
from abc import abstractmethod
from typing import List, Callable, Union, Any, TypeVar, Tuple

Tensor = TypeVar('Tensor')
ndarray = TypeVar('ndarray')


class BaseVAE(nn.Module):

    def __init__(self) -> None:
        super(BaseVAE, self).__init__()

    def encode(self, data: Tensor) -> List[Tensor]:
        raise NotImplementedError

    def decode(self, latent_code: Tensor) -> Any:
        raise NotImplementedError

    def sample(self, batch_size: int, current_device: int, **kwargs) -> Tensor:
        raise RuntimeWarning()

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def forward(self, *inputs: Tensor) -> Tensor:
        pass

    @abstractmethod
    def loss_function(self, *inputs: Tensor, **kwargs) -> Tensor:
        pass



def reparameterize(mu: Tensor, log_std: Tensor, **kwargs) -> Tensor:
    """
    Sample from std. gaussian and reparameterize with found parameters.
    :param mu: (Tensor) Mean of the latent Gaussian
    :param log_std: (Tensor) Standard deviation of the latent Gaussian
    :return:
    """
    B, M = mu.shape
    try:
        eps = torch.randn((B, M)).to(kwargs['device'])
    except KeyError:
        eps = torch.randn((B, M))
    std = torch.exp(log_std)
    return eps * std + mu, eps



def calc_output_dim(input_size: int, hidden_dims: List, kernel_szs: List, pad: int = 1, stride: int = 1,
                    conv: str = 'forward'):
    tmp_size = input_size
    if conv == 'forward':
        for i in range(len(hidden_dims)):
            tmp_size = floor((tmp_size + 2 * pad - kernel_szs[i]) / stride + 1)
            if tmp_size < 1:
                hidden_dims = hidden_dims[:i]
                kernel_szs = kernel_szs[:i]
                break
    elif conv == 'backward':
        for i in range(len(hidden_dims)):
            tmp_size = (tmp_size - 1) * stride - 2 * pad + kernel_szs[i]
    else:
        raise ValueError('Forward or backward?')
    output_dim = tmp_size
    return output_dim, hidden_dims, kernel_szs


def kl_div_diag_gauss(mu_p: Tensor, log_std_p: Tensor, mu_q: Tensor = None, log_std_q: Tensor = None):
    """Calculates the KL divergence between the two diagonal Gaussians p and q, i.e., KL(p||q)"""
    var_p = torch.exp(2 * log_std_p)
    if mu_q is not None:
        var_q = torch.exp(2 * log_std_q)
        kl_div = 0.5 * (torch.sum(2 * (log_std_q - log_std_p) + (mu_p - mu_q) ** 2 / (var_q + 1e-8)
                                  + var_p / var_q, dim=1) - mu_p.shape[1])
    else:
        # KL divergence for q=N(0,I)
        kl_div = 0.5 * (torch.sum(-2 * log_std_p + mu_p ** 2 + var_p, dim=1) - mu_p.shape[1])
    return kl_div



def steering_vector(phi: Tensor, device, theta: Tensor = 0, array: str = 'ULA', N: int = 32):
    """
    Returns the steering vector corresponding to an antenna array geometry for batch-sized input of tensors. Antennas
    assumed to be placed with half-wavelength spacing.
    :param phi: azimuth angle [B x L]
    :param device [B x L]
    :param theta: elevation angle
    :param array: type of antenna array, e.g. ULA
    :param N: number of antennas in the array
    :return:
    """
    pi = torch.tensor(math.pi, device=device)
    if array == 'ULA':
        sin_phi = torch.sin(phi).unsqueeze(-1).repeat(1, 1, N).to(device)  # [B x L x N]
        i = torch.arange(N, device=device).view(1, 1, -1).repeat(phi.shape[0], phi.shape[1], 1)
        a = torch.exp(-1j * pi * i * sin_phi).to(device)
    else:
        raise ValueError('Only ULA implemented so far.')
    return a



class BetaVAE(BaseVAE):

    num_iter = 0  # Global static variable to keep track of iterations

    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 hidden_dims: List = None,
                 beta: int = 4,
                 gamma: float = 1000.,
                 max_capacity: int = 25,
                 Capacity_max_iter: int = 1e5,
                 loss_type:str = 'B',
                 **kwargs) -> None:
        super(BetaVAE, self).__init__()

        self.latent_dim = latent_dim
        self.beta = beta
        self.gamma = gamma
        self.loss_type = loss_type
        self.C_max = torch.Tensor([max_capacity])
        self.C_stop_iter = Capacity_max_iter

        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]

        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size= 3, stride= 2, padding  = 1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1]*4, latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1]*4, latent_dim)


        # Build Decoder
        modules = []

        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * 4)

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride = 2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )



        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
                            nn.ConvTranspose2d(hidden_dims[-1],
                                               hidden_dims[-1],
                                               kernel_size=3,
                                               stride=2,
                                               padding=1,
                                               output_padding=1),
                            nn.BatchNorm2d(hidden_dims[-1]),
                            nn.LeakyReLU(),
                            nn.Conv2d(hidden_dims[-1], out_channels= 3,
                                      kernel_size= 3, padding= 1),
                            nn.Tanh())

    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        result = self.decoder_input(z)
        result = result.view(-1, 512, 2, 2)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def forward(self, data: Tensor, **kwargs) -> Tensor:
        mu, log_var = self.encode(data)
        z = self.reparameterize(mu, log_var)
        return [self.decode(z), data, mu, log_var]

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        self.num_iter += 1
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]
        kld_weight = kwargs['M_N']  # Account for the minibatch samples from the dataset

        recons_loss =F.mse_loss(recons, input)

        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)

        if self.loss_type == 'H': # https://openreview.net/forum?id=Sy2fzU9gl
            loss = recons_loss + self.beta * kld_weight * kld_loss
        elif self.loss_type == 'B': # https://arxiv.org/pdf/1804.03599.pdf
            self.C_max = self.C_max.to(input.device)
            C = torch.clamp(self.C_max/self.C_stop_iter * self.num_iter, 0, self.C_max.data[0])
            loss = recons_loss + self.gamma * kld_weight* (kld_loss - C).abs()
        else:
            raise ValueError('Undefined loss type.')

        return {'loss': loss, 'Reconstruction_Loss':recons_loss, 'KLD':kld_loss}

    def sample(self,
               num_samples:int,
               current_device: int, **kwargs) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]



class VAEConv(BaseVAE):

    def __init__(self,
                 in_channels: int,
                 stride: int,
                 kernel_szs: List,
                 latent_dim: int,
                 hidden_dims: List = None,
                 input_size: int = 16,
                 **kwargs) -> None:
        super(VAEConv, self).__init__()

        self.latent_dim = latent_dim
        self.input_size = input_size
        self.in_channels = in_channels
        self.pad = 1
        self.stride = stride
        self.use_iaf = kwargs['use_iaf']
        self.act = get_activation(kwargs['act'])
        self.device = kwargs['device']
        self.pi = torch.tensor(math.pi).to(self.device)

        self.embed_data = nn.Conv1d(in_channels, in_channels, kernel_size=1)

        modules = []
        if hidden_dims is None:
            hidden_dims = [8, 16, 32, 64]
        self.hidden_dims = hidden_dims
        if kernel_szs is None:
            kernel_szs = [3 for _ in range(len(hidden_dims))]

        # in_channels += 1  # to account for the extra label channel

        # calculate encoder output dims
        tmp_size = self.input_size
        for i in range(len(hidden_dims)):
            tmp_size = floor((tmp_size + 2 * self.pad - kernel_szs[i]) / self.stride + 1)
            if tmp_size < 1:
                hidden_dims = hidden_dims[:i]
                kernel_szs = kernel_szs[:i]
                break
            self.pre_latent = tmp_size

        # build encoder
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

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1] * self.pre_latent, self.latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1] * self.pre_latent, self.latent_dim)

        # build decoder
        modules = []

        if self.use_iaf:
            self.iaf = IAF(kwargs['n_blocks'], self.latent_dim, kwargs['hidden_iaf'], 1, cond_size=None)
        else:
            self.decoder_embed_latent = nn.Linear(latent_dim, latent_dim)
        self.decoder_input = nn.Linear(self.latent_dim, hidden_dims[-1] * self.pre_latent)

        hidden_dims.reverse()
        kernel_szs.reverse()

        for i in range(len(hidden_dims)):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose1d(hidden_dims[i],
                                       hidden_dims[i + 1] if i < len(hidden_dims) - 1 else hidden_dims[-1],
                                       kernel_size=kernel_szs[i],
                                       stride=self.stride,
                                       padding=self.pad),
                    nn.BatchNorm1d(hidden_dims[i + 1] if i < len(hidden_dims) - 1 else hidden_dims[-1]),
                    self.act)
            )

        self.decoder = nn.Sequential(*modules)

        # calculate decoder output dims
        self.pre_out = self.pre_latent
        for i in range(len(hidden_dims)):
            self.pre_out = (self.pre_out - 1) * self.stride - 2 * self.pad + kernel_szs[i]

        self.final_layer = nn.Linear(hidden_dims[-1] * self.pre_out, self.input_size + 1)

        self.F = dft_matrix(int(self.input_size / 2)).to(self.device)

        self.apply(WeightInitializer(get_initializer(kwargs['init'])).initialize_weights)

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
        log_var = 2 * torch.tanh(self.fc_var(result))

        return [mu, log_var]

    def decode(self, z: Tensor, **kwargs) -> Tensor:
        if self.use_iaf:
            result, jacobians = self.iaf(z)
            z = result
        else:
            result = self.decoder_embed_latent(z)
            jacobians = 0
        result = self.decoder_input(result)
        # shape result to fit dimensions after all conv layers of encoder and decode
        result = result.view(len(result), -1, self.pre_latent)
        result = self.decoder(result)
        # flatten result and put into final layer to predict [mu_out]
        result = torch.flatten(result, start_dim=1)
        result = self.final_layer(result)
        return result, z, jacobians

    def forward(self, data: Tensor, **kwargs) -> List[Tensor]:
        data = torch.flatten(data, start_dim=1)
        embedded_data = self.embed_data(data.view(len(data), 1, -1))
        data = data[:, :int(data.shape[1] / 2)] + 1j * data[:, int(data.shape[1] / 2):]

        # encode embedded input to the encoder mean and variance
        mu_enc, log_var = self.encode(embedded_data)

        # draw a sample and reparameterize during training
        if kwargs['train']:
            z_0, eps = reparameterize(mu_enc, log_var, device=self.device)
        # directly forward mean value of encoder during testing
        else:
            z_0, eps = mu_enc, torch.zeros_like(mu_enc).to(self.device)

        # decode to get real and imaginary part of output mean value
        out, z, jacobians = self.decode(z_0)
        log_prec = out[:, -1]
        mu_out_real, mu_out_imag = out[:, :-1].chunk(2, dim=1)
        mu_out = mu_out_real + 1j * mu_out_imag

        return [mu_out, log_prec, data, eps, log_var, z_0, z, jacobians]

    def loss_function(self, *args, **kwargs) -> dict:
        mu_out, log_prec, data, eps, log_var, z_0, z, jacobians = args

        anneal = kwargs['anneal']
        N = mu_out.shape[-1]

        cross_loss = N * log_prec - torch.sum((data - mu_out).abs() ** 2, dim=1) * log_prec.exp()

        kld_loss = -0.5 * torch.sum((eps ** 2 + 2 * log_var - z ** 2), dim=1) - jacobians

        loss = cross_loss - kld_loss
        loss_back = cross_loss.mean() - torch.maximum(kld_loss.mean(), torch.tensor(0.5, device=self.device))
        return {'loss': loss, 'loss_back': loss_back, 'crossentropy_loss': cross_loss, 'KLD': kld_loss}

    def sample(self, num_samples: int, **kwargs) -> Tensor:
        """
        Samples from the latent space and returns the corresponding image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """

        # sample n_samples times in latent space
        z_0 = torch.randn(num_samples, self.latent_dim).to(self.device)

        # decode samples
        out, z, _ = self.decode(z_0)
        mu_out_real, mu_out_imag = out.chunk(2, dim=1)
        samples = mu_out_real + 1j * mu_out_imag

        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image.
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """
        return self.forward(x, **kwargs)[0]



class VAECircCov(BaseVAE):

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
        self.act = get_activation(kwargs['act'])
        self.device = kwargs['device']
        self.cond_as_input = kwargs['cond_as_input']
        self.pi = torch.tensor(math.pi).to(self.device)
        self.lambda_z = torch.tensor(0.1, device=self.device)
        self.M = torch.tensor(self.input_size, device=self.device)

        self.embed_data = nn.Conv1d(in_channels, in_channels, kernel_size=1) if self.input_dim == 1 else \
            nn.Conv2d(in_channels, in_channels, kernel_size=1)

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

        if self.use_iaf:
            self.iaf = IAF(kwargs['n_blocks'], self.latent_dim, kwargs['hidden_iaf'], 1, cond_size=None)
        else:
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

        self.F = dft_matrix(int(self.input_size/2)).to(self.device)

        self.apply(WeightInitializer(get_initializer(kwargs['init'])).initialize_weights)


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
        if self.use_iaf:
            result, jacobians = self.iaf(z)
            z = result
        else:
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
            z_0, eps = reparameterize(mu_enc, log_std_enc, device=self.device)
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

        rec_loss = torch.sum(log_prec - (log_prec.exp() * ((data - mu_out).abs() ** 2)), dim=1) \
            - self.M * torch.log(self.pi)

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



class VAEToeplitzCov(BaseVAE):

    def __init__(self,
                 in_channels: int,
                 stride: int,
                 kernel_szs: List,
                 latent_dim: int,
                 hidden_dims: List = None,
                 input_size: int = 16,
                 input_dim: int = 1,
                 **kwargs) -> None:
        super(VAEToeplitzCov, self).__init__()

        self.latent_dim = latent_dim
        self.input_size = input_size  # this refers to the data size x
        self.input_dim = input_dim  # this refers to the actual input dimensionality of x or cond
        self.in_channels = in_channels
        self.pad = 1
        self.stride = stride
        self.use_iaf = kwargs['use_iaf']
        self.act = get_activation(kwargs['act'])
        self.device = kwargs['device']
        self.cond_as_input = kwargs['cond_as_input']
        self.pi = torch.tensor(math.pi).to(self.device)
        self.lambda_z = torch.tensor(0.1, device=self.device)
        self.M = torch.tensor(self.input_size/2, device=self.device)

        self.embed_data = nn.Conv1d(in_channels, in_channels, kernel_size=1) if self.input_dim == 1 else \
            nn.Conv2d(in_channels, in_channels, kernel_size=1)

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

        if self.use_iaf:
            self.iaf = IAF(kwargs['n_blocks'], self.latent_dim, kwargs['hidden_iaf'], 1, cond_size=None)
        else:
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

        self.final_layer = nn.Linear(hidden_dims[-1] * self.pre_out, 2*self.input_size)

        self.F = dft_matrix(self.input_size)[:, :int(0.5*self.input_size)].to(self.device).to(torch.cdouble)

        self.apply(WeightInitializer(get_initializer(kwargs['init'])).initialize_weights)


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
        if self.use_iaf:
            result, jacobians = self.iaf(z)
            z = result
        else:
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
            z_0, eps = reparameterize(mu_enc, log_std_enc, device=self.device)
        else:
            z_0, eps = mu_enc, torch.zeros_like(mu_enc).to(self.device)

        out, z, jacobians = self.decode(z_0)
        mu_out_real_imag, log_var = out.chunk(2, dim=1)
        mu_out_real, mu_out_imag = mu_out_real_imag.chunk(2, dim=1)
        mu_out = mu_out_real + 1j * mu_out_imag
        # mu_out[:] = 0

        # var_dec = (self.M / torch.sum(var_dec, dim=-1, keepdim=True)) * var_dec + 1e-6

        c_diag = torch.diag_embed(torch.exp(log_var)).to(torch.cdouble)
        C = self.F.conj().T @ c_diag @ self.F  # + 1e-6*torch.eye(self.F.shape[1], device=self.device).unsqueeze(0)
        # C = torch.eye(self.F.shape[1], device=self.device).unsqueeze(0).repeat(len(data), 1, 1).type(torch.cfloat).to(self.device)
        mu = mu_out

        return [mu_out, data, log_var, mu_enc, log_std_enc, z_0, z, jacobians, mu, C]


    def loss_function(self, *args, **kwargs) -> dict:
        mu_out, data, log_var, mu_enc, log_std_enc, z_0, z, jacobians, mu, C = args

        # C_inv = torch.cholesky_inverse(torch.cholesky(C))
        # C_inv = torch.inverse(C)
        [e, U] = torch.linalg.eigh(C)
        # eps = 1e-6 * torch.ones_like(e, device=self.device)
        # e = torch.maximum(e, eps)
        w = (data - mu).unsqueeze(-1)
        # rec_loss = w.conj().transpose(-1, -2) @ C_inv @ w
        # rec_loss = torch.real(w.conj().transpose(-1, -2) @ torch.linalg.solve(C, w)).squeeze()
        # rec_loss = -rec_loss - torch.log(torch.det(C)) - self.M * torch.log(self.pi)
        rec_loss = torch.linalg.norm(torch.squeeze(U.conj().transpose(-1, -2) @ w) / torch.sqrt(e), dim=-1) ** 2
        rec_loss = -rec_loss - torch.sum(torch.log(e), dim=-1) - self.M * torch.log(self.pi)
        # eps = 3 * self.M * torch.ones_like(rec_loss, device=self.device)
        # rec_loss = torch.minimum(rec_loss, eps)
        # idx = torch.logical_not(torch.isinf(rec_loss))
        # rec_loss = rec_loss[idx]

        kld_loss = kl_div_diag_gauss(mu_enc, log_std_enc)

        loss = rec_loss - kwargs['alpha'] * kld_loss
        loss_back = rec_loss.mean() - kwargs['alpha'] * torch.maximum(kld_loss.mean(), self.lambda_z)

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



class VAEFullCov(BaseVAE):

    def __init__(self,
                 in_channels: int,
                 stride: int,
                 kernel_szs: List,
                 latent_dim: int,
                 hidden_dims: List = None,
                 input_size: int = 16,
                 input_dim: int = 1,
                 **kwargs) -> None:
        super(VAEFullCov, self).__init__()

        self.latent_dim = latent_dim
        self.input_size = input_size  # this refers to the data size x
        self.input_dim = input_dim  # this refers to the actual input dimensionality of x or cond
        self.in_channels = in_channels
        self.pad = 1
        self.stride = stride
        self.use_iaf = kwargs['use_iaf']
        self.act = get_activation(kwargs['act'])
        self.device = kwargs['device']
        self.cond_as_input = kwargs['cond_as_input']
        self.pi = torch.tensor(math.pi).to(self.device)
        self.lambda_z = torch.tensor(0.1, device=self.device)
        self.M = torch.tensor(self.input_size/2, device=self.device)

        self.embed_data = nn.Conv1d(in_channels, in_channels, kernel_size=1) if self.input_dim == 1 else \
            nn.Conv2d(in_channels, in_channels, kernel_size=1)

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

        if self.use_iaf:
            self.iaf = IAF(kwargs['n_blocks'], self.latent_dim, kwargs['hidden_iaf'], 1, cond_size=None)
        else:
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

        self.final_layer = nn.Linear(hidden_dims[-1] * self.pre_out, int(self.input_size/2)**2 + self.input_size)

        self.apply(WeightInitializer(get_initializer(kwargs['init'])).initialize_weights)


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
        if self.use_iaf:
            result, jacobians = self.iaf(z)
            z = result
        else:
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
            z_0, eps = reparameterize(mu_enc, log_std_enc, device=self.device)
        else:
            z_0, eps = mu_enc, torch.zeros_like(mu_enc).to(self.device)

        out, z, jacobians = self.decode(z_0)
        mu_out_real_imag = out[:, :self.input_size]  # real and imaginary part of decoder mean value
        l_diag = out[:, self.input_size:self.input_size+int(self.input_size/2)]  # real-valued diagonal of prec.
        l_low_real_imag = out[:, self.input_size+int(self.input_size/2):]  # complex-valued lower diagonal of prec.
        mu_out_real, mu_out_imag = mu_out_real_imag.chunk(2, dim=1)
        mu_out = mu_out_real + 1j * mu_out_imag
        l_low_real, l_low_imag = l_low_real_imag.chunk(2, dim=1)
        l_low = l_low_real + 1j * l_low_imag

        l_diag = self.M*torch.sigmoid(l_diag)
        L = torch.diag_embed(l_diag).to(torch.cfloat).to(self.device)
        idx = torch.tril_indices(L.shape[-2], L.shape[-1], -1)
        L[:, idx[0], idx[1]] = l_low

        mu = mu_out
        C = L

        return [mu_out, data, L, l_diag, mu_enc, log_std_enc, z_0, z, jacobians, mu, C]


    def loss_function(self, *args, **kwargs) -> dict:
        mu_out, data, L, l_diag, mu_enc, log_std_enc, z_0, z, jacobians, mu, C = args

        w = (data - mu).unsqueeze(-1)
        # P = L @ L.conj().transpose(-1, -2)
        rec_loss = -torch.linalg.norm(torch.squeeze(L.conj().transpose(-1, -2) @ w), dim=-1) ** 2 - \
                   + 2*torch.sum(torch.log(l_diag), dim=-1) - self.M * torch.log(self.pi)

        eps = 6*self.M * torch.ones_like(rec_loss, device=self.device)
        rec_loss = torch.minimum(rec_loss, eps)

        kld_loss = kl_div_diag_gauss(mu_enc, log_std_enc)

        loss = rec_loss - kwargs['alpha'] * kld_loss
        loss_back = rec_loss.mean() - kwargs['alpha'] * torch.maximum(kld_loss.mean(), self.lambda_z)

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



class VAECircCovNoisy(BaseVAE):

    def __init__(self,
                 in_channels: int,
                 stride: int,
                 kernel_szs: List,
                 latent_dim: int,
                 hidden_dims: List = None,
                 input_size: int = 16,
                 input_dim: int = 1,
                 **kwargs) -> None:
        super(VAECircCovNoisy, self).__init__()

        self.latent_dim = latent_dim
        self.input_size = input_size  # this refers to the data size x
        self.input_dim = input_dim  # this refers to the actual input dimensionality of x or cond
        self.in_channels = in_channels
        self.pad = 1
        self.stride = stride
        self.use_iaf = kwargs['use_iaf']
        self.act = get_activation(kwargs['act'])
        self.device = kwargs['device']
        self.cond_as_input = kwargs['cond_as_input']
        self.pi = torch.tensor(math.pi).to(self.device)
        self.lambda_z = torch.tensor(0.1, device=self.device)
        self.M = torch.tensor(self.input_size, device=self.device)

        self.embed_data = nn.Conv1d(in_channels, in_channels, kernel_size=1) if self.input_dim == 1 else \
            nn.Conv2d(in_channels, in_channels, kernel_size=1)

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

        if self.use_iaf:
            self.iaf = IAF(kwargs['n_blocks'], self.latent_dim, kwargs['hidden_iaf'], 1, cond_size=None)
        else:
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

        self.F = dft_matrix(int(self.input_size/2)).to(self.device)

        self.apply(WeightInitializer(get_initializer(kwargs['init'])).initialize_weights)


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
        if self.use_iaf:
            result, jacobians = self.iaf(z)
            z = result
        else:
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


    def forward(self, data: Tensor, **kwargs) -> List[Tensor]:
        # data = torch.flatten(data, start_dim=1)
        cond = kwargs['cond']
        encoder_input = self.embed_data(cond.unsqueeze(1))
        data = data[:, :int(data.shape[1]/2)] + 1j * data[:, int(data.shape[1]/2):]
        cond = cond[..., :int(cond.shape[-1]/2)] + 1j * cond[..., int(cond.shape[-1]/2):]

        mu_enc, log_std_enc = self.encode(encoder_input)

        if kwargs['train']:
            z_0, eps = reparameterize(mu_enc, log_std_enc, device=self.device)
        else:
            z_0, eps = mu_enc, torch.zeros_like(mu_enc).to(self.device)

        out, z, jacobians = self.decode(z_0)
        mu_out_real, mu_out_imag, log_var = out.chunk(3, dim=1)
        mu_out = mu_out_real + 1j * mu_out_imag
        # mu_out[:] = 0

        # var_dec = (self.M / torch.sum(var_dec, dim=-1, keepdim=True)) * var_dec + 1e-6

        if not kwargs['train']:
            sigma = kwargs['sigma'].unsqueeze(-1)
            var_h = torch.exp(log_var)
            var_y = var_h + (sigma ** 2)
            c_h_diag = torch.diag_embed(var_h).type(torch.complex64).to(self.device)
            C_h = self.F.conj().T @ c_h_diag @ self.F
            c_y_diag = torch.diag_embed(var_y).type(torch.complex64).to(self.device)
            C_y = self.F.conj().T @ c_y_diag @ self.F
            C = (C_h, C_y)
            mu = mu_out @ self.F.conj()
        else:
            C, mu = None, None

        return [mu_out, data, cond, log_var, mu_enc, log_std_enc, z_0, z, jacobians, mu, C]


    def loss_function(self, *args, **kwargs) -> dict:
        mu_out, data, cond, log_var, mu_enc, log_std_enc, z_0, z, jacobians, mu, C = args

        sigma = kwargs['sigma'].unsqueeze(-1)
        var = log_var.exp() + (sigma ** 2)

        rec_loss = torch.sum(-torch.log(var) - (((cond-mu_out).abs() ** 2) / var), dim=1) - self.M*torch.log(self.pi)

        kld_loss = kl_div_diag_gauss(mu_enc, log_std_enc)

        loss = rec_loss - kwargs['alpha'] * kld_loss
        loss_back = rec_loss.mean() - kwargs['alpha'] * torch.maximum(kld_loss.mean(), self.lambda_z)

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



class VAECircCovMIMO(BaseVAE):

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
        self.act = get_activation(kwargs['act'])
        self.device = kwargs['device']
        self.cond_as_input = kwargs['cond_as_input']
        self.pi = torch.tensor(math.pi).to(self.device)
        self.lambda_z = torch.tensor(0.1, device=self.device)
        self.M = torch.tensor(self.input_size, device=self.device)

        self.embed_data = nn.Conv1d(in_channels, in_channels, kernel_size=1) if self.input_dim == 1 else \
            nn.Conv2d(in_channels, in_channels, kernel_size=1)

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

        if self.use_iaf:
            self.iaf = IAF(kwargs['n_blocks'], self.latent_dim, kwargs['hidden_iaf'], 1, cond_size=None)
        else:
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

        self.F = dft_matrix(int(self.input_size/2)).to(self.device)

        self.apply(WeightInitializer(get_initializer(kwargs['init'])).initialize_weights)


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
        if self.use_iaf:
            result, jacobians = self.iaf(z)
            z = result
        else:
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
            z_0, eps = reparameterize(mu_enc, log_std_enc, device=self.device)
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

        rec_loss = torch.sum(log_prec - (log_prec.exp() * ((data - mu_out).abs() ** 2)), dim=1) \
            - self.M * torch.log(self.pi)

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



class VAEChannel(BaseVAE):

    def __init__(self,
                 in_channels: int,
                 stride: int,
                 kernel_szs: List,
                 latent_dim: int,
                 hidden_dims: List = None,
                 input_size: int = 16,
                 input_dim: int = 1,
                 n_paths: int = 1,
                 **kwargs) -> None:
        super(VAEChannel, self).__init__()

        self.latent_dim = latent_dim
        self.input_size = input_size  # this refers to the data size x
        self.input_dim = input_dim  # this refers to the actual input dimensionality of x or cond
        self.in_channels = in_channels
        self.pad = 1
        self.stride = stride
        self.use_iaf = kwargs['use_iaf']
        self.act = get_activation(kwargs['act'])
        self.device = kwargs['device']
        self.cond_as_input = kwargs['cond_as_input']
        self.pi = torch.tensor(math.pi).to(self.device)
        self.lambda_z = torch.tensor(0.1, device=self.device)
        self.M = torch.tensor(0.5*self.input_size, device=self.device, dtype=torch.int)
        self.L = n_paths
        self.tol = torch.tensor(1e-8, device=self.device)

        self.embed_data = nn.Conv1d(in_channels, in_channels, kernel_size=1) if self.input_dim == 1 else \
            nn.Conv2d(in_channels, in_channels, kernel_size=1)

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

        if self.use_iaf:
            self.iaf = IAF(kwargs['n_blocks'], self.latent_dim, kwargs['hidden_iaf'], 1, cond_size=None)
        else:
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

        self.final_layer_params = nn.Linear(hidden_dims[-1] * self.pre_out, 3*self.L)
        self.final_layer_prec = nn.Linear(hidden_dims[-1] * self.pre_out, int(self.input_size/2))

        self.F = dft_matrix(int(self.input_size/2)).to(self.device)

        self.apply(WeightInitializer(get_initializer(kwargs['init'])).initialize_weights)


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
        if self.use_iaf:
            result, jacobians = self.iaf(z)
            z = result
        else:
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
        params = self.final_layer_params(result)
        log_prec = self.final_layer_prec(result)
        # log_prec = torch.ones_like(log_prec, device=self.device)
        alpha, beta, phi = params.chunk(3, dim=1)  # path gain, phase, angle
        phi = (self.pi/2) * torch.tanh(phi)
        alpha = 10 * torch.sigmoid(alpha)
        beta = self.pi * torch.tanh(beta)
        return alpha, beta, phi, log_prec, z, jacobians


    def forward(self, data: Tensor, **kwargs) -> List[Tensor]:
        # data = torch.flatten(data, start_dim=1)
        if self.cond_as_input:
            encoder_input = kwargs['cond']
        else:
            encoder_input = data
        encoder_input = self.embed_data(encoder_input.unsqueeze(1))
        h_in = data[:, :int(data.shape[1]/2)] + 1j * data[:, int(data.shape[1]/2):]

        mu_enc, log_std_enc = self.encode(encoder_input)

        if kwargs['train']:
            z_0, eps = reparameterize(mu_enc, log_std_enc, device=self.device)
        else:
            z_0, eps = mu_enc, torch.zeros_like(mu_enc).to(self.device)

        alpha, beta, phi, log_prec, z, jacobians = self.decode(z_0)

        a = steering_vector(phi, device=self.device, array='ULA', N=int(0.5*self.input_size))
        alpha, beta = alpha.unsqueeze(-1), beta.unsqueeze(-1)
        h_out = torch.sum(alpha * torch.exp(1j * beta) * a, dim=1)
        h_out = h_out @ self.F.T

        C, mu = None, None

        return [h_out, h_in, log_prec, mu_enc, log_std_enc, z_0, alpha, z, mu, C]


    def loss_function(self, *args, **kwargs) -> dict:
        h_out, h_in, log_prec, mu_enc, log_std_enc, z_0, alpha, z, mu, C = args

        # rec_loss = torch.sum(log_prec - (torch.exp(log_prec) * (h_in - h_out).abs() ** 2), dim=1) \
        #     - self.M * torch.log(self.pi)
        rec_loss = torch.sum(h_in.conj() * h_out, dim=-1) \
                   / torch.maximum((torch.linalg.norm(h_in, dim=-1) * torch.linalg.norm(h_out, dim=-1)), self.tol)
        rec_loss = rec_loss.real - torch.abs(rec_loss.imag)

        kld_loss = kl_div_diag_gauss(mu_enc, log_std_enc)

        sp_loss = torch.sum(alpha.squeeze(), dim=-1)
        # log_barrier = torch.log(torch.sum(alpha.squeeze(), dim=-1))
        # log_barrier = torch.log(torch.mean(h_out.abs(), dim=-1))

        # loss = rec_loss - kwargs['alpha'] * kld_loss
        loss = rec_loss - kwargs['alpha'] * kld_loss - kwargs['beta'] * sp_loss
        # loss = rec_loss - kwargs['alpha'] + kwargs['tau'] * log_barrier
        # loss_back = rec_loss.mean() - kwargs['alpha'] * kld_loss.mean()
        loss_back = rec_loss.mean() - kwargs['alpha'] * torch.maximum(kld_loss.mean(), self.lambda_z) \
            - kwargs['beta'] * sp_loss.mean()
        # loss_back = rec_loss.mean() - kwargs['alpha'] * torch.maximum(kld_loss.mean(), self.lambda_z) \
        #     + kwargs['tau'] * log_barrier.mean()

        return {'loss': loss, 'loss_back': loss_back, 'rec_loss': rec_loss, 'KLD': kld_loss}



class VAECircCovMO(BaseVAE):

    def __init__(self,
                 in_channels: int,
                 stride: int,
                 kernel_szs: List,
                 latent_dim: int,
                 hidden_dims: List = None,
                 input_size: int = 16,
                 src: str = '1-5',
                 n_obs: int = 10,
                 **kwargs) -> None:
        super(VAECircCovMO, self).__init__()

        self.latent_dim = latent_dim
        self.input_size = input_size  # this refers to the data size x
        self.src = src.split('-')
        self.cond_size = len(self.src)
        self.n_obs = n_obs
        self.in_channels = in_channels
        self.pad = 1
        self.stride = [1, stride]
        self.act = get_activation(kwargs['act'])
        self.device = kwargs['device']
        self.pi = torch.tensor(math.pi).to(self.device)
        self.lambda_z = torch.tensor(0.1, device=self.device)
        self.M = torch.tensor(self.input_size/2, device=self.device)

        # encoder embedding for data
        self.encoder_embed_data = nn.Conv2d(in_channels, in_channels, kernel_size=1)

        # encoder embedding for condition
        modules = [nn.Linear(self.cond_size, self.input_size, bias=False), nn.BatchNorm1d(self.input_size), self.act]
        self.encoder_embed_cond = nn.Sequential(*modules)

        modules = []
        if hidden_dims is None:
            hidden_dims = [8, 16, 32, 64]
        self.hidden_dims = hidden_dims
        if kernel_szs is None:
            kernel_szs = [3 for _ in range(len(hidden_dims))]

        # in_channels += 1  # to account for the extra label channel

        # calculate encoder output dims
        tmp_size = self.input_size/2
        for i in range(len(hidden_dims)):
            tmp_size = np.floor((tmp_size + 2*self.pad - kernel_szs[i][1]) / self.stride[1] + 1)
            if tmp_size.any() < 1:
                hidden_dims = hidden_dims[:i]
                kernel_szs = kernel_szs[:i]
                break
            self.pre_latent = tmp_size

        # build encoder
        for (i, h_dim) in enumerate(hidden_dims):
            modules.append(nn.Sequential(nn.Conv2d(in_channels,
                                                   out_channels=h_dim,
                                                   kernel_size=kernel_szs[i],
                                                   stride=self.stride,
                                                   padding=self.pad),
                                         nn.BatchNorm2d(h_dim),
                                         self.act))
            in_channels = h_dim
        self.fc_mu = nn.Linear(hidden_dims[-1] * int(self.pre_latent), self.latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1] * int(self.pre_latent), self.latent_dim)

        self.encoder = nn.Sequential(*modules)
        in_channels = h_dim

        # build prior net
        self.prior_net = nn.Sequential(nn.Linear(self.cond_size, self.latent_dim, bias=False),
                                       nn.BatchNorm1d(self.latent_dim), self.act,
                                       nn.Linear(self.latent_dim, self.latent_dim, bias=False),
                                       nn.BatchNorm1d(self.latent_dim), self.act,
                                       nn.Linear(self.latent_dim, self.latent_dim, bias=False), self.act)
        self.fc_mu_prior = nn.Linear(self.latent_dim, self.latent_dim)
        self.fc_var_prior = nn.Linear(self.latent_dim, self.latent_dim)

        # decoder embedding for condition
        modules = [nn.Linear(self.cond_size, self.latent_dim, bias=False), nn.BatchNorm1d(self.latent_dim), self.act]
        self.decoder_embed_cond = nn.Sequential(*modules)

        # build decoder
        modules = []
        self.decoder_embed_latent = nn.Linear(self.latent_dim, self.latent_dim)
        self.decoder_input = nn.Conv1d(2, hidden_dims[0], kernel_size=1)
        # hidden_dims.reverse()
        # kernel_szs.reverse()

        for i in range(len(hidden_dims)):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose1d(hidden_dims[i],
                                       hidden_dims[i + 1] if i < len(hidden_dims)-1 else hidden_dims[-1],
                                       kernel_size=kernel_szs[i][1],
                                       stride=self.stride[0],
                                       padding=self.pad),
                    nn.BatchNorm1d(hidden_dims[i + 1] if i < len(hidden_dims)-1 else hidden_dims[-1]),
                    self.act)
            )

        self.decoder = nn.Sequential(*modules)

        # calculate decoder output dims
        self.pre_out = self.latent_dim
        for i in range(len(hidden_dims)):
            self.pre_out = (self.pre_out - 1) * self.stride[0] - 2 * self.pad + kernel_szs[i][1]

        self.final_layer = nn.Linear(hidden_dims[-1] * self.pre_out, int(1.5*self.input_size))

        self.F = dft_matrix(self.input_size).to(self.device).to(torch.cdouble)

        self.apply(WeightInitializer(get_initializer(kwargs['init'])).initialize_weights)


    def encode(self, data: Tensor, cond: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network and returns the latent codes.
        :param data: (Tensor) Input tensor to encoder [n_data x 2(real/imag) x n_obs x n_ant]
        :param cond: (Tensor) Input condition to encoder [n_data x n_src]
        :return: (Tensor) List with mean and variance vector assuming a diagonal gaussian posterior
        """
        # INFERNCE DISTRIBUTION
        # embed data and condition
        encoder_input_data = self.encoder_embed_data(data)
        encoder_input_cond = self.encoder_embed_cond(cond).reshape(len(cond), 2, 1, -1)
        # cat and put into encoder
        encoder_input = torch.cat([encoder_input_data, encoder_input_cond], dim=2)
        result = self.encoder(encoder_input)
        result = torch.flatten(torch.mean(result, dim=2), start_dim=1)
        # get mean and log_std of inference distribution
        mu_enc = self.fc_mu(result)
        log_std_enc = self.fc_var(result)
        # PRIOR DISTRIBUTION
        # get result of prior net
        result = self.prior_net(cond)
        # get mean and log_std of prior network
        mu_prior = self.fc_mu_prior(result)
        log_std_prior = self.fc_var_prior(result)
        return [mu_enc, log_std_enc, mu_prior, log_std_prior]


    def decode(self, z: Tensor, cond: Tensor) -> Tensor:
        # embed latent and cond
        decoder_input_z = self.decoder_embed_latent(z).unsqueeze(1)
        decoder_input_cond = self.decoder_embed_cond(cond).unsqueeze(1)
        # cat and put into decoder
        decoder_input = torch.cat([decoder_input_z, decoder_input_cond], dim=1)
        result = self.decoder_input(decoder_input)
        result = self.decoder(result)
        # flatten and put into final layer
        result = torch.flatten(result, start_dim=1)
        result = self.final_layer(result)
        # split output into mean
        mu_out_real, mu_out_imag, log_var = result.chunk(3, dim=1)
        mu_out = mu_out_real + 1j * mu_out_imag
        return mu_out, log_var


    def forward(self, data: Tensor, **kwargs) -> List[Tensor]:
        # data [n_data x n_obs x 2*n_ant]
        # cond [n_data x n_src]
        cond = kwargs['cond']
        mu_enc, log_std_enc, mu_prior, log_std_prior = self.encode(data, cond)
        data = torch.squeeze(data[:, 0, :, :] + 1j * data[:, 1, :, :])

        if kwargs['train']:
            z, eps = reparameterize(mu_enc, log_std_enc, device=self.device)
        else:
            z, eps = mu_enc, torch.zeros_like(mu_enc).to(self.device)

        mu_out, log_var = self.decode(z, cond)
        mu = mu_out

        return [mu_out, data, log_var, mu_enc, log_std_enc, mu_prior, log_std_prior, z, mu]


    def loss_function(self, *args, **kwargs) -> dict:
        mu_out, data, log_var, mu_enc, log_std_enc, mu_prior, log_std_prior, z, mu = args
        n_data, n_obs, n_ant = data.shape

        sigma = kwargs['sigma'].unsqueeze(-1)
        var = log_var.exp() + (sigma ** 2)

        rec_loss = -n_obs * (torch.sum(torch.log(var), dim=1) + self.M*torch.log(self.pi))
        for i in range(n_obs):
            w_i = data[:, i, :] - mu
            rec_loss += torch.sum(-(w_i.abs() ** 2) / var, dim=1)

        kld_loss = kl_div_diag_gauss(mu_enc, log_std_enc, mu_prior, log_std_prior)

        loss = rec_loss - kwargs['alpha'] * kld_loss
        loss_back = rec_loss.mean() - kwargs['alpha'] * torch.maximum(kld_loss.mean(), self.lambda_z)

        return {'loss': loss, 'loss_back': loss_back, 'rec_loss': rec_loss, 'KLD': kld_loss}


    def sample(self, num_samples: int, **kwargs) -> Tensor:
        """
        Samples from the latent space and returns the corresponding image space map.
        :param num_samples: (Int) Number of samples
        :return: (Tensor)
        """
        # sample n_samples times in latent space
        z = torch.randn(num_samples, self.latent_dim).to(self.device)
        # decode samples
        out = self.decode(z)
        mu_out_real_imag = out[:, :self.input_size]
        mu_out_real, mu_out_imag = mu_out_real_imag.chunk(2, dim=1)
        return mu_out_real + 1j * mu_out_imag


    def mo_inf(self, data, n_samples, sigma):
        """
        Determines the model order based on the inference distribution
        :param data: input data [n_data x n_obs x n_ant]
        :param n_samples: number of samples to draw for MC result
        :param sigma: the noise standard deviation
        :return:
        """
        log_like = torch.zeros([len(data), len(self.src), n_samples], device=self.device, dtype=torch.double)
        for i in range(len(self.src)):
            cond = torch.zeros([len(data), len(self.src)], device=self.device, dtype=torch.double)
            cond[:, i] = 1
            for j in range(n_samples):
                args_loss = self.forward(data, cond=cond, train=True, sigma=sigma)
                loss = self.loss_function(*args_loss, alpha=1, sigma=sigma)
                log_like[:, i, j] = loss['rec_loss']
        log_like = torch.mean(log_like, dim=-1)
        mo = torch.argmax(log_like, dim=1)
        return mo


    def mo_prior(self, data, n_samples, sigma):
        """
        Determines the model order based on the inference distribution
        :param data: input data [n_data x n_obs x n_ant]
        :param n_samples: number of samples to draw for MC result
        :param sigma: the noise standard deviation
        :return:
        """
        log_like = torch.zeros([len(data), len(self.src), n_samples], device=self.device, dtype=torch.double)
        data = torch.squeeze(data[:, 0, :, :] + 1j * data[:, 1, :, :])
        for i in range(len(self.src)):
            cond = torch.zeros([len(data), len(self.src)], device=self.device, dtype=torch.double)
            cond[:, i] = 1
            for j in range(n_samples):
                result = self.prior_net(cond)
                mu_prior = self.fc_mu_prior(result)
                log_std_prior = self.fc_var_prior(result)
                z, _ = reparameterize(mu_prior, log_std_prior, device=self.device)
                mu, log_var = self.decode(z, cond)
                args_loss = [mu, data, log_var, mu_prior, log_std_prior, mu_prior, log_std_prior, z, mu]
                loss = self.loss_function(*args_loss, alpha=1, sigma=sigma)
                log_like[:, i, j] = loss['rec_loss']
        log_like = torch.mean(log_like, dim=-1)
        mo = torch.argmax(log_like, dim=1)
        return mo



class VAEToeplitzCovMO(BaseVAE):

    def __init__(self,
                 in_channels: int,
                 stride: int,
                 kernel_szs: List,
                 latent_dim: int,
                 hidden_dims: List = None,
                 input_size: int = 16,
                 src: str = '1-5',
                 n_obs: int = 10,
                 **kwargs) -> None:
        super(VAEToeplitzCovMO, self).__init__()

        self.latent_dim = latent_dim
        self.input_size = input_size  # this refers to the data size x
        self.src = src.split('-')
        self.cond_size = len(self.src)
        self.n_obs = n_obs
        self.in_channels = in_channels
        self.pad = 1
        self.stride = [1, stride]
        self.act = get_activation(kwargs['act'])
        self.device = kwargs['device']
        self.pi = torch.tensor(math.pi).to(self.device)
        self.lambda_z = torch.tensor(0.1, device=self.device)
        self.M = torch.tensor(self.input_size/2, device=self.device)

        # encoder embedding for data
        self.encoder_embed_data = nn.Conv2d(in_channels, in_channels, kernel_size=1)

        # encoder embedding for condition
        modules = [nn.Linear(self.cond_size, self.input_size, bias=False), nn.BatchNorm1d(self.input_size), self.act]
        self.encoder_embed_cond = nn.Sequential(*modules)

        modules = []
        if hidden_dims is None:
            hidden_dims = [8, 16, 32, 64]
        self.hidden_dims = hidden_dims
        if kernel_szs is None:
            kernel_szs = [3 for _ in range(len(hidden_dims))]

        # in_channels += 1  # to account for the extra label channel

        # calculate encoder output dims
        tmp_size = self.input_size/2
        for i in range(len(hidden_dims)):
            tmp_size = np.floor((tmp_size + 2*self.pad - kernel_szs[i][1]) / self.stride[1] + 1)
            if tmp_size.any() < 1:
                hidden_dims = hidden_dims[:i]
                kernel_szs = kernel_szs[:i]
                break
            self.pre_latent = tmp_size

        # build encoder
        for (i, h_dim) in enumerate(hidden_dims):
            modules.append(nn.Sequential(nn.Conv2d(in_channels,
                                                   out_channels=h_dim,
                                                   kernel_size=kernel_szs[i],
                                                   stride=self.stride,
                                                   padding=self.pad),
                                         nn.BatchNorm2d(h_dim),
                                         self.act))
            in_channels = h_dim
        self.fc_mu = nn.Linear(hidden_dims[-1] * int(self.pre_latent), self.latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1] * int(self.pre_latent), self.latent_dim)

        self.encoder = nn.Sequential(*modules)
        in_channels = h_dim

        # build prior net
        self.prior_net = nn.Sequential(nn.Linear(self.cond_size, self.latent_dim, bias=False),
                                       nn.BatchNorm1d(self.latent_dim), self.act,
                                       nn.Linear(self.latent_dim, self.latent_dim, bias=False),
                                       nn.BatchNorm1d(self.latent_dim), self.act,
                                       nn.Linear(self.latent_dim, self.latent_dim, bias=False), self.act)
        self.fc_mu_prior = nn.Linear(self.latent_dim, self.latent_dim)
        self.fc_var_prior = nn.Linear(self.latent_dim, self.latent_dim)

        # decoder embedding for condition
        modules = [nn.Linear(self.cond_size, self.latent_dim, bias=False), nn.BatchNorm1d(self.latent_dim), self.act]
        self.decoder_embed_cond = nn.Sequential(*modules)

        # build decoder
        modules = []
        self.decoder_embed_latent = nn.Linear(self.latent_dim, self.latent_dim)
        self.decoder_input = nn.Conv1d(2, hidden_dims[0], kernel_size=1)
        # hidden_dims.reverse()
        # kernel_szs.reverse()

        for i in range(len(hidden_dims)):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose1d(hidden_dims[i],
                                       hidden_dims[i + 1] if i < len(hidden_dims)-1 else hidden_dims[-1],
                                       kernel_size=kernel_szs[i][1],
                                       stride=self.stride[0],
                                       padding=self.pad),
                    nn.BatchNorm1d(hidden_dims[i + 1] if i < len(hidden_dims)-1 else hidden_dims[-1]),
                    self.act)
            )

        self.decoder = nn.Sequential(*modules)

        # calculate decoder output dims
        self.pre_out = self.latent_dim
        for i in range(len(hidden_dims)):
            self.pre_out = (self.pre_out - 1) * self.stride[0] - 2 * self.pad + kernel_szs[i][1]

        self.final_layer = nn.Linear(hidden_dims[-1] * self.pre_out, 2*self.input_size)

        self.F = dft_matrix(self.input_size)[:, :int(0.5 * self.input_size)].to(self.device).to(torch.cdouble)

        self.apply(WeightInitializer(get_initializer(kwargs['init'])).initialize_weights)


    def encode(self, data: Tensor, cond: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network and returns the latent codes.
        :param data: (Tensor) Input tensor to encoder [n_data x 2(real/imag) x n_obs x n_ant]
        :param cond: (Tensor) Input condition to encoder [n_data x n_src]
        :return: (Tensor) List with mean and variance vector assuming a diagonal gaussian posterior
        """
        # INFERNCE DISTRIBUTION
        # embed data and condition
        encoder_input_data = self.encoder_embed_data(data)
        encoder_input_cond = self.encoder_embed_cond(cond).reshape(len(cond), 2, 1, -1)
        # cat and put into encoder
        encoder_input = torch.cat([encoder_input_data, encoder_input_cond], dim=2)
        result = self.encoder(encoder_input)
        result = torch.flatten(torch.mean(result, dim=2), start_dim=1)
        # get mean and log_std of inference distribution
        mu_enc = self.fc_mu(result)
        log_std_enc = self.fc_var(result)
        # PRIOR DISTRIBUTION
        # get result of prior net
        result = self.prior_net(cond)
        # get mean and log_std of prior network
        mu_prior = self.fc_mu_prior(result)
        log_std_prior = self.fc_var_prior(result)
        return [mu_enc, log_std_enc, mu_prior, log_std_prior]


    def decode(self, z: Tensor, cond: Tensor, sigma) -> Tensor:
        # embed latent and cond
        decoder_input_z = self.decoder_embed_latent(z).unsqueeze(1)
        decoder_input_cond = self.decoder_embed_cond(cond).unsqueeze(1)
        # cat and put into decoder
        decoder_input = torch.cat([decoder_input_z, decoder_input_cond], dim=1)
        result = self.decoder_input(decoder_input)
        result = self.decoder(result)
        # flatten and put into final layer
        result = torch.flatten(result, start_dim=1)
        result = self.final_layer(result)
        # split output into mean
        mu_out_real_imag, log_var = result.chunk(2, dim=1)
        mu_out_real, mu_out_imag = mu_out_real_imag.chunk(2, dim=1)
        mu_out = mu_out_real + 1j * mu_out_imag
        # form Toeplitz covariance from output
        c_diag = torch.diag_embed(torch.exp(log_var)).to(torch.cdouble)
        C = self.F.conj().T @ c_diag @ self.F
        # add noise covariance
        sigma = sigma.reshape((-1, 1, 1))
        C_n = sigma * torch.eye(C.shape[-1], dtype=torch.cdouble, device=self.device).unsqueeze(0).repeat(len(C), 1, 1)
        C += C_n
        return mu_out, C, log_var


    def forward(self, data: Tensor, **kwargs) -> List[Tensor]:
        # data [n_data x n_obs x 2*n_ant]
        # cond [n_data x n_src]
        cond = kwargs['cond']
        mu_enc, log_std_enc, mu_prior, log_std_prior = self.encode(data, cond)
        data = torch.squeeze(data[:, 0, :, :] + 1j * data[:, 1, :, :])

        if kwargs['train']:
            z, eps = reparameterize(mu_enc, log_std_enc, device=self.device)
        else:
            z, eps = mu_enc, torch.zeros_like(mu_enc).to(self.device)

        mu_out, C, log_var = self.decode(z, cond, kwargs['sigma'])
        mu = mu_out

        return [mu_out, data, log_var, mu_enc, log_std_enc, mu_prior, log_std_prior, z, mu, C]


    def loss_function(self, *args, **kwargs) -> dict:
        mu_out, data, log_var, mu_enc, log_std_enc, mu_prior, log_std_prior, z, mu, C = args
        n_data, n_obs, n_ant = data.shape

        [e, U] = torch.linalg.eigh(C)
        w = (data - mu.unsqueeze(1)).unsqueeze(-1)
        rec_loss = -n_obs * (torch.sum(torch.log(e), dim=-1) + self.M * torch.log(self.pi))
        for i in range(n_obs):
            w_i = w[:, i, :, :]
            rec_loss -= torch.linalg.norm(torch.squeeze(U.conj().transpose(-1, -2) @ w_i) / torch.sqrt(e), dim=-1) ** 2

        kld_loss = kl_div_diag_gauss(mu_enc, log_std_enc, mu_prior, log_std_prior)

        loss = rec_loss - kwargs['alpha'] * kld_loss
        loss_back = rec_loss.mean() - kwargs['alpha'] * torch.maximum(kld_loss.mean(), self.lambda_z)

        return {'loss': loss, 'loss_back': loss_back, 'rec_loss': rec_loss, 'KLD': kld_loss}


    def sample(self, num_samples: int, **kwargs) -> Tensor:
        """
        Samples from the latent space and returns the corresponding image space map.
        :param num_samples: (Int) Number of samples
        :return: (Tensor)
        """
        # sample n_samples times in latent space
        z = torch.randn(num_samples, self.latent_dim).to(self.device)
        # decode samples
        out = self.decode(z)
        mu_out_real_imag = out[:, :self.input_size]
        mu_out_real, mu_out_imag = mu_out_real_imag.chunk(2, dim=1)
        return mu_out_real + 1j * mu_out_imag


    def mo_inf(self, data, n_samples, sigma):
        """
        Determines the model order based on the inference distribution
        :param data: input data [n_data x n_obs x n_ant]
        :param n_samples: number of samples to draw for MC result
        :param sigma: the noise standard deviation
        :return:
        """
        log_like = torch.zeros([len(data), len(self.src), n_samples], device=self.device, dtype=torch.double)
        for i in range(len(self.src)):
            cond = torch.zeros([len(data), len(self.src)], device=self.device, dtype=torch.double)
            cond[:, i] = 1
            for j in range(n_samples):
                args_loss = self.forward(data, cond=cond, train=True, sigma=sigma)
                loss = self.loss_function(*args_loss, alpha=1, sigma=sigma)
                log_like[:, i, j] = loss['rec_loss']
        log_like = torch.mean(log_like, dim=-1)
        mo = torch.argmax(log_like, dim=1)
        return mo


    def mo_prior(self, data, n_samples, sigma):
        """
        Determines the model order based on the inference distribution
        :param data: input data [n_data x n_obs x n_ant]
        :param n_samples: number of samples to draw for MC result
        :param sigma: the noise standard deviation
        :return:
        """
        log_like = torch.zeros([len(data), len(self.src), n_samples], device=self.device, dtype=torch.double)
        data = torch.squeeze(data[:, 0, :, :] + 1j * data[:, 1, :, :])
        for i in range(len(self.src)):
            cond = torch.zeros([len(data), len(self.src)], device=self.device, dtype=torch.double)
            cond[:, i] = 1
            for j in range(n_samples):
                result = self.prior_net(cond)
                mu_prior = self.fc_mu_prior(result)
                log_std_prior = self.fc_var_prior(result)
                z, _ = reparameterize(mu_prior, log_std_prior, device=self.device)
                mu, C, log_var = self.decode(z, cond, sigma)
                args_loss = [mu, data, log_var, mu_prior, log_std_prior, mu_prior, log_std_prior, z, mu, C]
                loss = self.loss_function(*args_loss, alpha=1, sigma=sigma)
                log_like[:, i, j] = loss['rec_loss']
        log_like = torch.mean(log_like, dim=-1)
        mo = torch.argmax(log_like, dim=1)
        return mo



class VAEFullCovMO(BaseVAE):

    def __init__(self,
                 in_channels: int,
                 stride: int,
                 kernel_szs: List,
                 latent_dim: int,
                 hidden_dims: List = None,
                 input_size: int = 16,
                 src: str = '1-5',
                 n_obs: int = 10,
                 **kwargs) -> None:
        super(VAEFullCovMO, self).__init__()

        self.latent_dim = latent_dim
        self.input_size = input_size  # this refers to the data size x
        self.src = src.split('-')
        self.cond_size = len(self.src)
        self.n_obs = n_obs
        self.in_channels = in_channels
        self.pad = 1
        self.stride = stride
        self.act = get_activation(kwargs['act'])
        self.device = kwargs['device']
        self.pi = torch.tensor(math.pi).to(self.device)
        self.lambda_z = torch.tensor(0.1, device=self.device)
        self.M = torch.tensor(self.input_size/2, device=self.device)

        # encoder embedding for data
        self.encoder_embed_data = nn.Conv2d(in_channels, in_channels, kernel_size=1)

        # encoder embedding for condition
        modules = [nn.Linear(self.cond_size, self.input_size, bias=False), nn.BatchNorm1d(self.input_size), self.act]
        self.encoder_embed_cond = nn.Sequential(*modules)

        modules = []
        if hidden_dims is None:
            hidden_dims = [8, 16, 32, 64]
        self.hidden_dims = hidden_dims
        if kernel_szs is None:
            kernel_szs = [3 for _ in range(len(hidden_dims))]

        # in_channels += 1  # to account for the extra label channel

        # calculate encoder output dims
        tmp_size = np.array([self.n_obs + 1, self.input_size/2])
        for i in range(len(hidden_dims)):
            tmp_size = np.floor((tmp_size + 2*self.pad - kernel_szs[i]) / self.stride + 1)
            if tmp_size.any() < 1:
                hidden_dims = hidden_dims[:i]
                kernel_szs = kernel_szs[:i]
                break
            self.pre_latent = tmp_size

        # build encoder
        for (i, h_dim) in enumerate(hidden_dims):
            modules.append(nn.Sequential(nn.Conv2d(in_channels,
                                                   out_channels=h_dim,
                                                   kernel_size=kernel_szs[i],
                                                   stride=self.stride,
                                                   padding=self.pad),
                                         nn.BatchNorm2d(h_dim),
                                         self.act))
            in_channels = h_dim
        self.fc_mu = nn.Linear(hidden_dims[-1] * int(np.prod(self.pre_latent)), self.latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1] * int(np.prod(self.pre_latent)), self.latent_dim)

        self.encoder = nn.Sequential(*modules)
        in_channels = h_dim

        # build prior net
        self.prior_net = nn.Sequential(nn.Linear(self.cond_size, self.latent_dim, bias=False),
                                       nn.BatchNorm1d(self.latent_dim), self.act,
                                       nn.Linear(self.latent_dim, self.latent_dim, bias=False),
                                       nn.BatchNorm1d(self.latent_dim), self.act,
                                       nn.Linear(self.latent_dim, self.latent_dim, bias=False), self.act)
        self.fc_mu_prior = nn.Linear(self.latent_dim, self.latent_dim)
        self.fc_var_prior = nn.Linear(self.latent_dim, self.latent_dim)

        # decoder embedding for condition
        modules = [nn.Linear(self.cond_size, self.latent_dim, bias=False), nn.BatchNorm1d(self.latent_dim), self.act]
        self.decoder_embed_cond = nn.Sequential(*modules)

        # build decoder
        modules = []
        self.decoder_embed_latent = nn.Linear(self.latent_dim, self.latent_dim)
        self.decoder_input = nn.Conv1d(2, hidden_dims[0], kernel_size=1)
        # hidden_dims.reverse()
        # kernel_szs.reverse()

        for i in range(len(hidden_dims)):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose1d(hidden_dims[i],
                                       hidden_dims[i + 1] if i < len(hidden_dims)-1 else hidden_dims[-1],
                                       kernel_size=kernel_szs[i][1],
                                       stride=self.stride,
                                       padding=self.pad),
                    nn.BatchNorm1d(hidden_dims[i + 1] if i < len(hidden_dims)-1 else hidden_dims[-1]),
                    self.act)
            )

        self.decoder = nn.Sequential(*modules)

        # calculate decoder output dims
        self.pre_out = self.latent_dim
        for i in range(len(hidden_dims)):
            self.pre_out = (self.pre_out - 1) * self.stride - 2 * self.pad + kernel_szs[i][1]

        self.final_layer = nn.Linear(hidden_dims[-1] * self.pre_out, int(self.input_size/2)**2 + self.input_size)

        self.apply(WeightInitializer(get_initializer(kwargs['init'])).initialize_weights)


    def encode(self, data: Tensor, cond: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network and returns the latent codes.
        :param data: (Tensor) Input tensor to encoder [n_data x 2(real/imag) x n_obs x n_ant]
        :param cond: (Tensor) Input condition to encoder [n_data x n_src]
        :return: (Tensor) List with mean and variance vector assuming a diagonal gaussian posterior
        """
        # INFERNCE DISTRIBUTION
        # embed data and condition
        encoder_input_data = self.encoder_embed_data(data)
        encoder_input_cond = self.encoder_embed_cond(cond).reshape(len(cond), 2, 1, -1)
        # cat and put into encoder
        encoder_input = torch.cat([encoder_input_data, encoder_input_cond], dim=2)
        result = self.encoder(encoder_input)
        result = torch.flatten(result, start_dim=1)
        # get mean and log_std of inference distribution
        mu_enc = self.fc_mu(result)
        log_std_enc = self.fc_var(result)
        # PRIOR DISTRIBUTION
        # get result of prior net
        result = self.prior_net(cond)
        # get mean and log_std of prior network
        mu_prior = self.fc_mu_prior(result)
        log_std_prior = self.fc_var_prior(result)
        return [mu_enc, log_std_enc, mu_prior, log_std_prior]


    def decode(self, z: Tensor, cond: Tensor) -> Tensor:
        # embed latent and cond
        decoder_input_z = self.decoder_embed_latent(z).unsqueeze(1)
        decoder_input_cond = self.decoder_embed_cond(cond).unsqueeze(1)
        # cat and put into decoder
        decoder_input = torch.cat([decoder_input_z, decoder_input_cond], dim=1)
        result = self.decoder_input(decoder_input)
        result = self.decoder(result)
        # flatten and put into final layer
        result = torch.flatten(result, start_dim=1)
        result = self.final_layer(result)
        return result


    def forward(self, data: Tensor, **kwargs) -> List[Tensor]:
        # data [n_data x n_obs x 2*n_ant]
        # cond [n_data x n_src]
        cond = kwargs['cond']
        mu_enc, log_std_enc, mu_prior, log_std_prior = self.encode(data, cond)
        data = torch.squeeze(data[:, 0, :, :] + 1j * data[:, 1, :, :])

        if kwargs['train']:
            z, eps = reparameterize(mu_enc, log_std_enc, device=self.device)
        else:
            z, eps = mu_enc, torch.zeros_like(mu_enc).to(self.device)

        out = self.decode(z, cond)

        mu_out_real_imag = out[:, :self.input_size]  # real and imaginary part of decoder mean value
        l_diag = out[:, self.input_size:self.input_size+int(self.input_size/2)]  # real-valued diagonal of prec.
        l_low_real_imag = out[:, self.input_size+int(self.input_size/2):]  # complex-valued lower diagonal of prec.
        mu_out_real, mu_out_imag = mu_out_real_imag.chunk(2, dim=1)
        mu_out = mu_out_real + 1j * mu_out_imag
        l_low_real, l_low_imag = l_low_real_imag.chunk(2, dim=1)
        l_low = l_low_real + 1j * l_low_imag

        l_diag = torch.exp(l_diag)
        L = torch.diag_embed(l_diag).to(torch.cdouble).to(self.device)
        idx = torch.tril_indices(L.shape[-2], L.shape[-1], -1)
        L[:, idx[0], idx[1]] = l_low

        mu = mu_out
        C = L

        return [mu_out, data, L, l_diag, mu_enc, log_std_enc, mu_prior, log_std_prior, z, mu, C]


    def loss_function(self, *args, **kwargs) -> dict:
        [mu_out, data, L, l_diag, mu_enc, log_std_enc, mu_prior, log_std_prior, z, mu, C] = args
        [n_data, n_obs, n_ant] = data.shape

        w = (data - mu.unsqueeze(1)).unsqueeze(-1)
        rec_loss = 2*n_obs*torch.sum(torch.log(l_diag), dim=-1) - n_ant*n_obs*torch.log(self.pi)
        for i in range(n_obs):
            w_i = w[:, i, :, :]
            rec_loss -= torch.linalg.norm(torch.squeeze(L.conj().transpose(-1, -2) @ w_i), dim=-1) ** 2

        kld_loss = kl_div_diag_gauss(mu_enc, log_std_enc, mu_prior, log_std_prior)

        loss = rec_loss - kwargs['alpha'] * kld_loss
        loss_back = rec_loss.mean() - kwargs['alpha'] * torch.maximum(kld_loss.mean(), self.lambda_z)

        return {'loss': loss, 'loss_back': loss_back, 'rec_loss': rec_loss, 'KLD': kld_loss}


    def sample(self, num_samples: int, **kwargs) -> Tensor:
        """
        Samples from the latent space and returns the corresponding image space map.
        :param num_samples: (Int) Number of samples
        :return: (Tensor)
        """
        # sample n_samples times in latent space
        z = torch.randn(num_samples, self.latent_dim).to(self.device)
        # decode samples
        out = self.decode(z)
        mu_out_real_imag = out[:, :self.input_size]
        mu_out_real, mu_out_imag = mu_out_real_imag.chunk(2, dim=1)
        return mu_out_real + 1j * mu_out_imag


    def mo_inf(self, data):
        """
        Determines the model order based on the inference distribution
        :param data: input data [n_data x n_obs x n_ant]
        :return:
        """
        log_like = torch.zeros([len(data), len(self.src)], device=self.device, dtype=torch.double)
        for i in range(len(self.src)):
            cond = torch.zeros([len(data), len(self.src)], device=self.device, dtype=torch.double)
            cond[:, i] = 1
            args_loss = self.forward(data, cond=cond, train=False)
            loss = self.loss_function(*args_loss, alpha=1)
            log_like[:, i] = loss['rec_loss']
        mo = torch.argmax(log_like, dim=1)
        return mo



class ConditionalVAEConv(BaseVAE):

    def __init__(self,
                 in_channels: int,
                 cond_size: int,
                 latent_dim: int,
                 hidden_dims: List = None,
                 kernel_szs: List = None,
                 input_size: List = [16, 16],
                 **kwargs) -> None:
        super(ConditionalVAEConv, self).__init__()

        self.latent_dim = latent_dim
        self.input_size = input_size
        self.in_channels = in_channels
        pad = (1, 0)
        st = (2, 1)
        self.input_prod = input_size[0] * input_size[1]
        self.use_iaf = kwargs['use_iaf']

        self.embed_cond = nn.Linear(cond_size, self.input_prod, bias=False)
        self.embed_data = nn.Conv2d(in_channels, in_channels, kernel_size=(1, 1))

        modules = []
        if hidden_dims is None:
            hidden_dims = [8, 16, 32, 64]
        self.hidden_dims = hidden_dims
        if kernel_szs is None:
            kernel_szs = [(3, 1) for i in range(len(hidden_dims))]

        in_channels += 1  # to account for the extra label channel

        # build encoder
        for (i, h_dim) in enumerate(hidden_dims):
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim, kernel_size=kernel_szs[i], stride=st, padding=pad),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        # calculate encoder output dims
        self.pre_hidden_x1, self.pre_hidden_x2 = self.input_size
        for i in range(len(hidden_dims)):
            self.pre_hidden_x1 = floor((self.pre_hidden_x1 + 2*pad[0] - kernel_szs[i][0]) / 2 + 1)
            self.pre_hidden_x2 = floor((self.pre_hidden_x2 + 2*pad[1] - kernel_szs[i][1]) / 2 + 1)
        pre_hidden = self.pre_hidden_x1 * self.pre_hidden_x2

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1]*pre_hidden, latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1]*pre_hidden, latent_dim)

        # build decoder
        modules = []

        if self.use_iaf:
            self.iaf = IAF(kwargs['n_blocks'], latent_dim, 2*latent_dim, 2, cond_size=cond_size)
        else:
            self.decoder_embed_latent = nn.Linear(latent_dim, latent_dim)
            self.decoder_embed_cond = nn.Linear(cond_size, latent_dim, bias=False)
        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * pre_hidden)

        hidden_dims.reverse()
        kernel_szs.reverse()

        for i in range(len(hidden_dims)):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1] if i < len(hidden_dims)-1 else hidden_dims[-1],
                                       kernel_size=kernel_szs[i],
                                       stride=st,
                                       padding=pad),
                    nn.BatchNorm2d(hidden_dims[i + 1] if i < len(hidden_dims)-1 else hidden_dims[-1]),
                    nn.LeakyReLU())
            )

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
                            # nn.Linear(in_channels*input_size[0]*input_size[1], in_channels*input_size[0]*input_size[1])
                            nn.Conv2d(hidden_dims[-1], out_channels=self.in_channels, kernel_size=(3, 3), padding=(1, 1))
                            # nn.Flatten(start_dim=1),
                            # nn.Linear(self.in_channels * self.input_prod, self.in_channels * self.input_prod)
        )

    def encode(self, data: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network and returns the latent codes.
        :param data: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List with mean and variance vector assuming a diagonal gaussian posterior
        """
        result = self.encoder(data)
        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]


    def decode(self, z: Tensor, **kwargs) -> Tensor:
        if self.use_iaf:
            result, jacobians = self.iaf(z, cond=kwargs['cond'])
            z = result
        else:
            result = self.decoder_embed_latent(z) + self.decoder_embed_cond(kwargs['cond'])
            jacobians = torch.zeros(len(z))
        result = self.decoder_input(result)
        result = result.view(-1, self.hidden_dims[0], self.pre_hidden_x1, self.pre_hidden_x2)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result, z, jacobians


    def forward(self, data: Tensor, **kwargs) -> List[Tensor]:
        cond = kwargs['cond']
        sigma = kwargs['sigma']
        n_samples = kwargs['n_samples']
        embedded_cond = self.embed_cond(cond)
        embedded_cond = embedded_cond.view(-1, self.input_size[0], self.input_size[1]).unsqueeze(1)
        embedded_input = self.embed_data(data)

        mu, log_var = self.encode(torch.cat([embedded_input, embedded_cond], dim=1))

        z_0, eps = reparameterize(mu, log_var, n=n_samples)

        mu_out = torch.zeros((len(z_0), n_samples, 2, self.input_size[0], self.input_size[1]))
        jacobians = torch.zeros((len(z_0), n_samples))
        z = torch.zeros((len(z_0), n_samples, self.latent_dim))
        for i in range(n_samples):  # decode all reparameterized samples separately
            mu_out[:, i, ...], z[:, i, :], jacobians[:, i] = self.decode(z_0[:, i, :], cond=cond)

        data = torch.unsqueeze(data, 1)

        return [mu_out, data, eps, log_var, z, jacobians]


    def loss_function(self, *args, **kwargs) -> dict:
        mu_out = args[0]
        data = args[1]
        eps = args[2]
        log_var = args[3].unsqueeze(1)
        z = args[4]
        jacobians = args[5]

        var_prior = kwargs['var_prior']
        beta = kwargs['beta']

        # cross_loss = -0.5 * (rel_mse(data, mu_out) / var_prior) / mu_out.shape[1]
        cross_loss = -0.5 * (torch.sum((data - mu_out) ** 2, dim=(1, 2, 3, 4)) / var_prior) / mu_out.shape[1]

        kld_loss = 0.5 * torch.sum(eps ** 2 + log_var - z ** 2, dim=(1, 2)) + jacobians.sum(1)

        loss = cross_loss + beta * kld_loss
        return {'loss': loss, 'crossentropy_Loss': cross_loss, 'KLD': -kld_loss}


    def sample(self, num_samples: int, current_device: int, **kwargs) -> Tensor:
        """
        Samples from the latent space and returns the corresponding image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        cond = kwargs['cond']
        sigma = kwargs['sigma']
        n_samples = kwargs['n_samples']
        z = torch.randn(num_samples, n_samples, self.latent_dim)
        # z = torch.zeros(num_samples, n_samples, self.latent_dim)

        z = z.to(current_device)

        samples = torch.zeros(len(z), n_samples, 2, self.input_size[0], self.input_size[1])
        for i in range(n_samples):
            samples[:, i, ...], _, _ = self.decode(z[:, i, :], cond=cond)

        return samples.sum(1, keepdim=True)


    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image.
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """
        return self.forward(x, **kwargs)[0]



class ConditionalVAEFully(BaseVAE):

    def __init__(self,
                 cond_size: int,
                 latent_dim: int,
                 hidden_dims: List = None,
                 input_size: int = 16,
                 **kwargs) -> None:
        super(ConditionalVAEFully, self).__init__()

        self.latent_dim = latent_dim
        self.input_size = 2*input_size
        self.use_iaf = kwargs['use_iaf']

        self.embed_cond = nn.Linear(cond_size, self.input_size, bias=False)
        self.embed_data = nn.Linear(self.input_size, self.input_size)

        modules = []
        if hidden_dims is None:
            hidden_dims = [64, 64, 64]
        self.hidden_dims = hidden_dims

        # build encoder
        for i in range(len(hidden_dims)):
            modules.append(
                nn.Sequential(
                    nn.Linear(self.input_size if i == 0 else hidden_dims[i - 1], hidden_dims[i]),
                    nn.BatchNorm1d(hidden_dims[i]),
                    nn.LeakyReLU())
            )

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1], latent_dim)

        # build decoder
        modules = []


        if self.use_iaf:
            self.iaf = IAF(kwargs['n_blocks'], latent_dim, 2*latent_dim, 2, cond_size=cond_size)
        else:
            self.decoder_embed_latent = nn.Linear(latent_dim, latent_dim)
            self.decoder_embed_cond = nn.Linear(cond_size, latent_dim, bias=False)
            self.decoder_input = nn.Linear(latent_dim, latent_dim)

        hidden_dims.reverse()

        for i in range(len(hidden_dims)):
            modules.append(
                nn.Sequential(
                    nn.Linear(latent_dim if i == 0 else hidden_dims[i - 1], hidden_dims[i]),
                    nn.BatchNorm1d(hidden_dims[i]),
                    nn.LeakyReLU())
            )

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Linear(hidden_dims[-1], self.input_size)


    def encode(self, data: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network and returns the latent codes.
        :param data: (Tensor) Input tensor to encoder [B x N]
        :return: (Tensor) List with mean and variance vector assuming a diagonal gaussian posterior
        """
        result = self.encoder(data)
        # Split the result into mu and var components of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]


    def decode(self, z: Tensor, **kwargs) -> Tensor:
        if self.use_iaf:
            result, jacobians = self.iaf(z, cond=kwargs['cond'])
            z = result
        else:
            result = self.decoder_input(self.decoder_embed_latent(z) + self.decoder_embed_cond(kwargs['cond']))
            jacobians = torch.zeros(len(z))
        result = self.decoder(result)
        result = self.final_layer(result)
        return result, z, jacobians


    def forward(self, data: Tensor, **kwargs) -> List[Tensor]:
        cond = torch.flatten(kwargs['cond'], start_dim=1)
        sigma = kwargs['sigma']
        n_samples = kwargs['n_samples']
        data = torch.flatten(data, start_dim=1)
        embedded_cond = self.embed_cond(cond)
        embedded_data = self.embed_data(data)

        mu, log_var = self.encode(embedded_data + embedded_cond)

        z_0, eps = reparameterize(mu, log_var, n=n_samples)

        mu_out = torch.zeros((len(z_0), n_samples, self.input_size))
        jacobians = torch.zeros((len(z_0), n_samples))
        z = torch.zeros((len(z_0), n_samples, self.latent_dim))
        for i in range(n_samples):  # decode all reparameterized samples separately
            mu_out[:, i, :], z[:, i, :], jacobians[:, i] = self.decode(z_0[:, i, :], cond=cond)

        data = torch.unsqueeze(data, 1)

        return [mu_out, data, eps, log_var, z, jacobians]


    def loss_function(self, *args, **kwargs) -> dict:
        mu_out = args[0]
        data = args[1]
        eps = args[2]
        log_var = args[3].unsqueeze(1)
        z = args[4]
        jacobians = args[5]

        var_prior = kwargs['var_prior']
        beta = kwargs['beta']

        # cross_loss = -0.5 * (rel_mse(data, mu_out) / var_prior) / mu_out.shape[1]
        cross_loss = -0.5 * (torch.sum((data - mu_out) ** 2, dim=(1, 2)) / var_prior) / mu_out.shape[1]

        kld_loss = 0.5 * torch.sum((eps ** 2 + log_var - z ** 2), dim=(1, 2)) + jacobians.sum(1)

        loss = cross_loss + beta * kld_loss
        return {'loss': loss, 'crossentropy_loss': cross_loss, 'KLD': -kld_loss}


    def sample(self, num_samples: int, current_device: int, **kwargs) -> Tensor:
        """
        Samples from the latent space and returns the corresponding image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        cond = kwargs['cond']
        sigma = kwargs['sigma']
        n_samples = kwargs['n_samples']
        z = torch.randn(num_samples, n_samples, self.latent_dim)
        # z = torch.zeros(num_samples, self.latent_dim)

        z = z.to(current_device)

        samples = torch.zeros(len(z), n_samples, self.input_size)
        for i in range(n_samples):
            samples[:, i, :], _, _ = self.decode(z[:, i, :], cond=cond)

        return samples.sum(1, keepdim=True)

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image.
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """
        return self.forward(x, **kwargs)[0]



class CondVAECircCov(BaseVAE):

    def __init__(self,
                 in_channels: int,
                 stride: int,
                 kernel_szs: List,
                 cond_size: int,
                 latent_dim: int,
                 hidden_dims: List = None,
                 input_size: int = 16,
                 cond_dims: int = 1,
                 **kwargs) -> None:
        super(CondVAECircCov, self).__init__()

        self.latent_dim = latent_dim
        self.input_size = input_size
        self.in_channels = in_channels
        self.pad = 1
        self.stride = stride
        self.use_iaf = kwargs['use_iaf']
        self.act = get_activation(kwargs['act'])
        self.device = kwargs['device']
        self.pi = torch.tensor(math.pi).to(self.device)
        self.lambda_z = torch.tensor(kwargs['lambda_z'], device=self.device)
        self.cond_dims = cond_dims

        modules = []
        if hidden_dims is None:
            hidden_dims = [8, 16, 32, 64]
        self.hidden_dims = hidden_dims
        if kernel_szs is None:
            kernel_szs = [3 for _ in range(len(hidden_dims))]

        # in_channels += 1  # to account for the extra label channel

        # calculate encoder output dims
        tmp_size = self.input_size
        for i in range(len(hidden_dims)):
            tmp_size = floor((tmp_size + 2*self.pad - kernel_szs[i]) / self.stride + 1)
            if tmp_size < 1:
                hidden_dims = hidden_dims[:i]
                kernel_szs = kernel_szs[:i]
                break
            self.pre_latent = tmp_size

        # build encoder
        self.embed_data = nn.Conv1d(in_channels, in_channels, kernel_size=1)

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

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1]*self.pre_latent, self.latent_dim)
        self.fc_std = nn.Linear(hidden_dims[-1]*self.pre_latent, self.latent_dim)

        # build cond input net for encoder
        if self.cond_dims == 1:
            modules = [nn.Conv1d(1, 4, kernel_size=kernel_szs[0], padding='same'),
                       self.act,
                       nn.Flatten(start_dim=1),
                       nn.Linear(4 * self.input_size, self.input_size)]
        elif self.cond_dims == 2:
            modules = [nn.Conv2d(1, 4, kernel_size=kernel_szs[0], padding='same'),
                       self.act,
                       nn.Flatten(start_dim=1),
                       nn.Linear(int(4 * (0.5 * self.input_size) ** 2), self.input_size)]
        self.encoder_cond_input = nn.Sequential(*modules)

        # calculate prior net output dims
        tmp_size = int(0.5 * self.input_size) if self.cond_dims == 2 else self.input_size
        for i in range(len(hidden_dims)):
            tmp_size = floor((tmp_size + 2*self.pad - kernel_szs[i]) / self.stride + 1)
            if tmp_size < 1:
                hidden_dims = hidden_dims[:i]
                kernel_szs = kernel_szs[:i]
                break
            self.pre_latent = tmp_size

        # build prior net
        modules = []
        in_channels = 1

        for (i, h_dim) in enumerate(hidden_dims):
            if self.cond_dims == 1:
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
            elif self.cond_dims == 2:
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

        self.prior_net = nn.Sequential(*modules)
        self.fc_mu_prior = nn.Linear(hidden_dims[-1]*(self.pre_latent**self.cond_dims), self.latent_dim)
        self.fc_std_prior = nn.Linear(hidden_dims[-1]*(self.pre_latent**self.cond_dims), self.latent_dim)

        hidden_dims.reverse()
        kernel_szs.reverse()

        # build decoder
        modules = []

        if self.use_iaf:
            self.iaf = IAF(kwargs['n_blocks'], self.latent_dim, kwargs['hidden_iaf'], 1, cond_size=cond_size)
        else:
            self.decoder_embed_latent = nn.Linear(latent_dim, latent_dim)
            # self.decoder_embed_cond = nn.Linear(cond_size, latent_dim, bias=False)
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

        self.decoder = nn.Sequential(*modules)

        # build cond input net for decoder
        if self.cond_dims == 1:
            modules = [nn.Conv1d(1, 4, kernel_size=kernel_szs[0], padding='same'),
                       self.act,
                       nn.Flatten(start_dim=1),
                       nn.Linear(4 * self.input_size, self.latent_dim)]
        elif self.cond_dims == 2:
            modules = [nn.Conv2d(1, 4, kernel_size=kernel_szs[0], padding='same'),
                       self.act,
                       nn.Flatten(start_dim=1),
                       nn.Linear(int(4 * (0.5*self.input_size) ** 2), self.latent_dim)]
        self.decoder_cond_input = nn.Sequential(*modules)

        # calculate decoder output dims
        self.pre_out = self.pre_latent
        for i in range(len(hidden_dims)):
            self.pre_out = (self.pre_out - 1) * self.stride - 2 * self.pad + kernel_szs[i]

        self.final_layer = nn.Linear(hidden_dims[-1] * self.pre_out, int(1.5*self.input_size))

        self.F = dft_matrix(int(self.input_size/2)).to(self.device)

        self.apply(WeightInitializer(get_initializer(kwargs['init'])).initialize_weights)


    def encode(self, data: Tensor, cond: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network and returns the latent codes.
        :param data: (Tensor) Input tensor to encoder [B x C x N]
        :param cond: (Tensor) Condition to prior network [B x C x M]
        :return: (Tensor) List with mean and variance vector assuming a diagonal gaussian posterior
        """
        # Get result of encoding x
        embedded_cond = self.encoder_cond_input(cond).unsqueeze(1)
        result = self.encoder(data + embedded_cond)
        # result = self.encoder(data)
        result = torch.flatten(result, start_dim=1)
        # Split the result into mu and std of the posterior distribution
        mu = self.fc_mu(result)
        log_std = self.fc_std(result)
        # Get result of prior network
        result = self.prior_net(cond)
        result = torch.flatten(result, start_dim=1)
        # Split the result into mu and std of the prior distribution
        mu_prior = self.fc_mu_prior(result)
        log_std_prior = self.fc_std_prior(result)
        return [mu, log_std, mu_prior, log_std_prior]


    def decode(self, z: Tensor, **kwargs) -> Tensor:
        # embed latent code and condition
        result = self.decoder_embed_latent(z)
        embedded_cond = self.decoder_cond_input(kwargs['cond'].unsqueeze(1))
        result = self.decoder_input(result + embedded_cond)
        # result = self.decoder_input(result)
        # shape result to fit dimensions after all conv layers of encoder and decode
        result = result.view(len(result), -1, self.pre_latent)
        result = self.decoder(result)
        # flatten result and put into final layer to predict [mu_real, mu_imag, c_diag]
        result = torch.flatten(result, start_dim=1)
        result = self.final_layer(result)
        return result, z, 0


    def forward(self, data: Tensor, **kwargs) -> List[Tensor]:
        cond = kwargs['cond']
        data = torch.flatten(data, start_dim=1)
        embedded_data = self.embed_data(data.unsqueeze(1))
        data = data[:, :int(data.shape[1]/2)] + 1j * data[:, int(data.shape[1]/2):]

        mu_enc, log_std, mu_prior, log_std_prior = self.encode(embedded_data, cond.unsqueeze(1))

        if kwargs['train']:
            z_0, eps = reparameterize(mu_enc, log_std, device=self.device)
        else:
            z_0, eps = mu_enc, torch.zeros_like(mu_enc).to(self.device)

        out, z, jacobians = self.decode(z_0, cond=cond)
        mu_out_fft_real, mu_out_fft_imag, log_prec = out.chunk(3, dim=1)
        mu_out_fft = mu_out_fft_real + 1j * mu_out_fft_imag

        if not kwargs['train']:
            c_diag = torch.diag_embed(torch.exp(-log_prec)).type(torch.complex64).to(self.device)
            C = self.F.conj().T @ c_diag @ self.F
            mu_out = mu_out_fft @ self.F.conj()
        else:
            C, mu_out = None, None

        return [mu_out_fft, data, log_prec, mu_enc, log_std, mu_prior, log_std_prior, z_0, z, jacobians, mu_out, C]


    def loss_function(self, *args, **kwargs) -> dict:
        mu_out_fft, data, log_prec, mu_enc, log_std, mu_prior, log_std_prior, z_0, z, jacobians, mu_out, C = args

        w = data - mu_out_fft
        rec_loss = torch.sum(log_prec, dim=1) - torch.sum(log_prec.exp() * w.abs() ** 2, dim=1)
        rec_loss -= mu_out_fft.shape[1] * torch.log(self.pi)

        kld_loss = kl_div_diag_gauss(mu_enc, log_std, mu_prior, log_std_prior)
        # rec_loss = torch.zeros_like(kld_loss, device=self.device)
        # mse_loss = torch.sum((mu_enc - mu_prior) ** 2, dim=1)

        loss = rec_loss - kwargs['alpha'] * kld_loss  # - kwargs['beta'] * mse_loss

        loss_back = rec_loss.mean() - kwargs['alpha'] * torch.maximum(kld_loss.mean(), self.lambda_z)
                    # - kwargs['beta'] * mse_loss.mean()

        return {'loss': loss, 'loss_back': loss_back, 'rec_loss': rec_loss, 'KLD': kld_loss}


    def sample(self, num_samples: int, **kwargs) -> Tensor:
        """
        Samples from the latent space and returns the corresponding image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        cond = kwargs['cond']
        sigma = kwargs['sigma']

        # sample n_samples times in latent space
        z_0 = torch.randn(len(cond), num_samples, self.latent_dim).to(self.device)
        z_0 = z_0.to(self.device)

        # init complex valued output vector
        x_est_all = torch.zeros(len(cond), num_samples, int(self.input_size/2)).type(torch.complex64).to(self.device)
        log_p_z = torch.zeros(len(cond), num_samples, 1).to(self.device)
        z = torch.randn(len(cond), num_samples, self.latent_dim).to(self.device)

        # loop over every latent sample and compute LMMSE estimate
        for i in range(num_samples):

            # get decoder output (mean and precision)
            # out, z[:, i, :], jacobians = self.decode(z_0[:, i, :], cond=cond)
            z[:, i, :] = self.flow.inverse(z_0[:, i, :], cond)[0]
            log_p_z[:, i, 0] = self.flow.log_prob(z[:, i, :], cond)[0]

            out, _, jacobians = self.decode(z[:, i, :])
            mu_real_fft, mu_imag_fft, log_prec = out.chunk(3, dim=1)
            mu_fft = mu_real_fft + 1j * mu_imag_fft

            # construct cov and mean value in non fft domain
            c_diag = torch.diag_embed(torch.exp(-log_prec)).type(torch.complex64).to(self.device)
            C = self.F.conj().T @ c_diag @ self.F
            mu = mu_fft @ self.F.conj()

            # calculate log_p(z|y) for every z sample
            # log_p_z_0 = -0.5 * (self.latent_dim * torch.log(2*self.pi) + torch.sum(z_0[:, i, :] ** 2, dim=1))
            # log_p_z[:, i, :] = torch.unsqueeze(log_p_z_0 - jacobians, 1)

            # calculate LMMSE estimate for every z
            x_est_all[:, i, :] = compute_lmmse(C, mu, cond, sigma)

        # compute weighted sum weighted of all estimates
        log_p_z -= torch.logsumexp(log_p_z, dim=1, keepdim=True)
        x_est = torch.sum(log_p_z.exp() * x_est_all, dim=1)
        # x_est = torch.mean(x_est_all, dim=1)

        return x_est

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image.
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """
        return self.forward(x, **kwargs)[0]

    def decode_prior(self, cond):
        # Get result of prior network
        result = self.prior_net(cond.unsqueeze(1))
        result = torch.flatten(result, start_dim=1)
        # get mean value of prior network
        mu_prior = self.fc_mu_prior(result)
        # decode mean value to get output of decoder
        out, z, jacobians = self.decode(mu_prior, cond=cond)
        # split output into mean and log_prec
        mu_out_fft_real, mu_out_fft_imag, log_prec = out.chunk(3, dim=1)
        mu_out_fft = mu_out_fft_real + 1j * mu_out_fft_imag
        # get real cov and mean in non-fourier domain
        c_diag = torch.diag_embed(torch.exp(-log_prec)).type(torch.complex64).to(self.device)
        C = self.F.conj().T @ c_diag @ self.F
        mu_out = mu_out_fft @ self.F.conj()
        return mu_out, C



class GMVAE(BaseVAE):

    def __init__(self,
                 in_channels: int,
                 stride: int,
                 kernel_szs: List,
                 latent_dim: int,
                 hidden_dims: List = None,
                 input_size: int = 16,
                 n_clusters: int = 2,
                 **kwargs) -> None:
        super(GMVAE, self).__init__()

        self.latent_dim = latent_dim
        self.input_size = input_size
        self.in_channels = in_channels
        self.n_clusters = n_clusters
        self.pad = 1
        self.stride = stride
        self.act = get_activation(kwargs['act'])
        self.device = kwargs['device']
        self.pi = torch.tensor(math.pi).to(self.device)
        self.log_K = torch.log(torch.tensor(self.n_clusters).to(self.device))
        self.lambda_y = torch.tensor(0.05, device=self.device)
        self.lambda_z = torch.tensor(0.5, device=self.device)

        if hidden_dims is None:
            hidden_dims = [8, 16, 32, 64]
        self.hidden_dims = hidden_dims
        if kernel_szs is None:
            kernel_szs = [3 for _ in range(len(hidden_dims))]


        # embedding of input to get a hidden version
        self.embed_x = nn.Sequential(nn.Conv1d(in_channels, in_channels, kernel_size=1),
                                     nn.BatchNorm1d(in_channels),
                                     self.act)


        # build encoder from x to y that outputs the logits of q(y|x)
        self.encoder_x_to_y = nn.Sequential(nn.Linear(self.input_size, 2*self.n_clusters),
                                            nn.BatchNorm1d(2*self.n_clusters), self.act,
                                            nn.BatchNorm1d(2*self.n_clusters), self.act,
                                            nn.Linear(2*self.n_clusters, self.n_clusters))


        # build encoder from x and y (which is from encoder above) to z which outputs the mean and diag cov of q(z|x,y)
        self.embed_y = nn.Linear(self.n_clusters, self.input_size, bias=False)
        self.pre_latent, hidden_dims, kernel_szs = calc_output_dim(self.input_size, hidden_dims, kernel_szs,
                                                                   self.pad, self.stride, conv='forward')
        in_channels += 1  # account for stacked x_embed and y
        modules = []
        for (i, h_dim) in enumerate(hidden_dims):
            modules.append(
                nn.Sequential(
                    nn.Conv1d(in_channels, out_channels=h_dim, kernel_size=kernel_szs[i],
                              stride=self.stride, padding=self.pad),
                    nn.BatchNorm1d(h_dim),
                    self.act)
            )
            in_channels = h_dim

        self.encoder_xy_to_z = nn.Sequential(*modules,
                                             nn.Flatten(start_dim=1),
                                             nn.Linear(hidden_dims[-1]*self.pre_latent, 2*self.latent_dim))


        # build (prior) decoder from y to z that outputs mean and diag cov of p(z|y) (for generative process)
        self.decoder_y_to_z = nn.Sequential(nn.Linear(self.n_clusters, 2*self.latent_dim, bias=False), self.act,
                                            nn.Linear(2*self.latent_dim, 2*self.latent_dim, bias=False), self.act,
                                            nn.Linear(2*self.latent_dim, 2*self.latent_dim, bias=False))


        # build decoder from z to x that outputs mean and diag cov of p(x|z) (for generative process)
        modules = []
        self.decoder_z_to_x_input = nn.Linear(self.latent_dim, hidden_dims[-1] * self.pre_latent)

        hidden_dims.reverse()
        kernel_szs.reverse()

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

        # calculate decoder output dims and build blocks
        self.pre_out, _, _ = calc_output_dim(self.pre_latent, hidden_dims, kernel_szs, self.pad, self.stride,
                                             conv='backward')
        self.decoder_z_to_x = nn.Sequential(*modules,
                                            nn.Flatten(start_dim=1),
                                            nn.Linear(hidden_dims[-1] * self.pre_out, int(1.5*self.input_size)))


        self.F = dft_matrix(int(self.input_size/2)).to(self.device)
        self.apply(WeightInitializer(get_initializer(kwargs['init'])).initialize_weights)


    def encode(self, data: Tensor, **kwargs) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder networks and returns the mean and log_std of q(z|x,y), the
        logits of p(y|x), and the mean and log_std of p(z|y) (z-prior).
        :param data: (Tensor) Input tensor to encoder [B x N]
        :return: (Tensor) List with mu_z_xy, log_std_z_xy, logp_y, y
        """
        # get logits of q(y|x)
        result = torch.flatten(data, start_dim=1)
        result = self.encoder_x_to_y(result)
        logp_y = result - torch.logsumexp(result, dim=1, keepdim=True)

        # reparameterize with Gumbel-softmax
        g = -torch.log(-torch.log(torch.rand_like(logp_y)))
        y = torch.softmax((logp_y + g) / kwargs['tau'], dim=1)

        # get statistics of q(z|x,y)
        y_embed = self.embed_y(y).view(len(result), 1, self.input_size)
        xy = torch.cat([data, y_embed], dim=1)
        result = self.encoder_xy_to_z(xy)
        mu_z_xy, log_std_z_xy = result.chunk(2, dim=1)

        return [mu_z_xy, log_std_z_xy, logp_y, y]


    def decode_z(self, z: Tensor) -> Tensor:
        result = self.decoder_z_to_x_input(z)

        # shape result to fit dimensions after all conv layers of encoder and decode
        result = result.view(len(result), -1, self.pre_latent)
        result = self.decoder_z_to_x(result)

        # shape into [mu_real, mu_imag, -log c_diag]
        mu_x_z_real, mu_x_z_imag, log_prec_x_z = result.chunk(3, dim=1)
        mu_x_z = mu_x_z_real + 1j * mu_x_z_imag
        return mu_x_z, log_prec_x_z


    def decode_y(self, y: Tensor) -> Tensor:
        result = self.decoder_y_to_z(y)
        mu_z_y, log_std_z_y = result.chunk(2, dim=1)
        return mu_z_y, log_std_z_y


    def forward(self, data: Tensor, **kwargs) -> List[Tensor]:
        data = torch.flatten(data, start_dim=1)
        embedded_data = self.embed_x(data.view(len(data), 1, -1))
        data = data[:, :int(data.shape[1]/2)] + 1j * data[:, int(data.shape[1]/2):]

        # encode x to y for q(y|x) & encode x and y to z for q(z|x,y)
        mu_z_xy, log_std_z_xy, logp_y, y = self.encode(embedded_data, **kwargs)

        # decode y to z for p(z|y)
        mu_z_y, log_std_z_y = self.decode_y(y)

        if kwargs['train']:
            z_xy, eps = reparameterize(mu_z_xy, log_std_z_xy, device=self.device)
        else:
            z_xy, eps = mu_z_xy, torch.zeros_like(mu_z_xy).to(self.device)

        mu_x_z, log_prec_x_z = self.decode_z(z_xy)

        return [mu_x_z, data, log_prec_x_z, mu_z_xy, log_std_z_xy, logp_y, y, eps, mu_z_y, log_std_z_y, z_xy, []]


    def loss_function(self, *args, **kwargs) -> dict:
        [mu_x_z, data, log_prec_x_z, mu_z_xy, log_std_z_xy, logp_y, y, eps, mu_z_y, log_std_z_y, z_xy, _] = args

        # reconstruction loss log p(x|z)
        n_log_pi = mu_x_z.shape[-1] * torch.log(self.pi)
        rec_loss = -n_log_pi + torch.sum(log_prec_x_z - log_prec_x_z.exp() * (data - mu_x_z).abs() ** 2, dim=1)

        # KL divergence between q(z|y,x) and p(z|y)
        kld_loss_z = kl_div_diag_gauss(mu_z_xy, log_std_z_xy, mu_z_y, log_std_z_y)

        # KL divergence between q(y|x) and p(y)
        kld_loss_y = torch.sum(logp_y.exp() * (self.log_K + logp_y), dim=1)

        loss = rec_loss - kwargs['alpha'] * kld_loss_y - kwargs['beta'] * kld_loss_z
        loss_back = rec_loss.mean() - kwargs['alpha'] * torch.maximum(kld_loss_z.mean(), self.lambda_z)\
                    - kwargs['beta'] * torch.maximum(kld_loss_y.mean(), self.lambda_y)

        return {'loss': loss, 'loss_back': loss_back, 'rec_loss': rec_loss, 'KLD': kld_loss_y + kld_loss_z,
                'kld_y': kld_loss_y, 'kld_z': kld_loss_z}


    def sample(self, num_samples: int, label: int, **kwargs) -> Tensor:
        """
        Samples from the latent space and returns the corresponding image space map.
        :param num_samples: (Int) Number of samples
        :param label: (int) label to sample from
        :return: (Tensor)
        """
        # create one-hot vectors of label
        y_idx = label * torch.ones(num_samples).type(torch.LongTensor)
        y = torch.nn.functional.one_hot(y_idx, num_classes=self.n_clusters).to(self.device).type(torch.FloatTensor)

        # decode the one-hot vectors to z and reparameterize
        mu_z_y, log_prec_z_y = self.decode_y(y)
        z_y, _ = reparameterize(mu_z_y, -log_prec_z_y, device=self.device)

        # decode z to x
        mu_x_z, log_prec_x_z = self.decode_z(z_y)

        return mu_x_z

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input x, returns the reconstructed image.
        :param x: (Tensor) [B x C x M]
        :return: (Tensor) [B x C x M]
        """
        return self.forward(x, **kwargs)[0]

    def infer_label(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input x, returns the inferred labels.
        :param x:
        :param kwargs:
        :return:
        """
        logp_y = self.forward(x, **kwargs)[5]
        p_y = logp_y.exp()
        y = torch.argmax(p_y, dim=1).type(torch.LongTensor)
        return y, p_y

    def sample_mean_latent_clusters(self, num_samples: int):
        """
        Given a number of samples, creates num_samples for every latent cluster by sampling from p(z|y)
        :param num_samples:
        :return:
        """
        z = []
        for i in range(self.n_clusters):
            # create one-hot vector of current label
            y_idx = i * torch.ones(num_samples).type(torch.LongTensor)
            y = torch.nn.functional.one_hot(y_idx, num_classes=self.n_clusters).to(self.device).type(torch.FloatTensor)

            # decode y to z and reparameterize
            mu_z_y, log_std_z_y = self.decode_y(y)
            z_i, _ = reparameterize(mu_z_y, log_std_z_y)
            z.append(z_i)

        return z



def compute_lmmse_est(t: Tensor, y: Tensor, sigma: ndarray) -> Tensor:
    """
    Given a batch of complex valued toeplitz covariances and observations, returns the corresponding LMMSE estimate
    :param t:
    :param y:
    :param sigma:
    :return:
    """
    L, N, M = t.shape[0], t.shape[2], y.shape[1]
    # t = torch.squeeze(t).detach().numpy()
    t = torch.squeeze(t)
    t[:, 0] = 1

    # preprocess real and imaginary parts of components
    y = torch.complex(y[:, :int(M/2)], y[:, int(M/2):])
    h_est = torch.zeros((L, N), dtype=torch.complex64)
    # mu = torch.from_numpy(t[:, 2, ...] + 1j * t[:, 3, ...])
    mu = t[:, 2, ...] + 1j * t[:, 3, ...]
    t = t[:, 0, ...] + 1j * t[:, 1, ...]
    # W = torch.zeros((L, N, N), dtype=torch.complex64)

    for i in range(len(t)):

        # compute LMMSE estimate for given observation and delta
        # C = torch.from_numpy(scipy.linalg.toeplitz(np.conj(t[i])))
        C = circulant(t[i])
        W = torch.matmul(C, torch.inverse(C + sigma[i]**2 * torch.eye(N)))
        h_est[i] = mu[i] + torch.matmul(y[i] - mu[i], W)

    # reshape output
    h_est = h_est.view(L, 1, N, 1)
    h_est = torch.cat([h_est.real, h_est.imag], dim=1)

    # L, N, M = t.shape[0], t.shape[2], y.shape[1]
    # t = torch.squeeze(t).detach().numpy()
    # # t[:, 0] = 1
    #
    # # preprocess real and imaginary parts of components
    # h_est = torch.zeros((L, 2, N), dtype=torch.float)
    # mu_real = torch.from_numpy(t[:, 2, ...])
    # mu_imag = torch.from_numpy(t[:, 3, ...])
    #
    # for i in range(len(t)):
    #
    #     # compute LMMSE estimate for given observation and delta
    #     C = torch.from_numpy(scipy.linalg.toeplitz(t[i, 0, :]))
    #     W = torch.matmul(torch.inverse(C + sigma[i] ** 2 * torch.eye(N)), C)
    #     h_est[i, 0, :] = mu_real[i] + torch.matmul(W, y[i, :int(M/2)] - mu_real[i])
    #
    #     C = torch.from_numpy(scipy.linalg.toeplitz(t[i, 1, :]))
    #     W = torch.matmul(torch.inverse(C + sigma[i] ** 2 * torch.eye(N)), C)
    #     h_est[i, 1, :] = mu_imag[i] + torch.matmul(W, y[i, int(M/2):] - mu_imag[i])
    #
    # # reshape output
    # h_est = h_est.view(L, 2, N, 1)

    return h_est



class BatchNorm1d(nn.Module):
    def __init__(self, input_size, momentum=0.9, eps=1e-5):
        """
        BatchNorm layer for vector-like input
        Input:
        input_size:     size of input vector
        momentum:       momentum for running mean calculation
        eps:            robustness term for stability of division
        """
        super().__init__()
        self.momentum = momentum
        self.eps = eps

        self.batch_mean = None
        self.batch_var = None

        self.weight = nn.Parameter(torch.zeros(input_size), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(input_size), requires_grad=True)

        self.register_buffer('running_mean', torch.zeros(input_size))
        self.register_buffer('running_var', torch.ones(input_size))

    def forward(self, x, cond=None):
        if self.training:
            self.batch_mean = torch.mean(x, dim=0)
            self.batch_var = torch.var(x, dim=0)

            # update running mean
            self.running_mean.mul_(self.momentum).add_(self.batch_mean.data * (1 - self.momentum))
            self.running_var.mul_(self.momentum).add_(self.batch_var.data * (1 - self.momentum))

            mean = self.batch_mean
            var = self.batch_var
        else:
            mean = self.running_mean
            var = self.running_var

        x_hat = (x - mean) / torch.sqrt(var + self.eps)
        y = torch.exp(self.weight) * x_hat + self.bias

        # compute jacobian
        log_abs_det_jacobian = (self.weight - 0.5 * torch.log(var + self.eps)).expand_as(x)

        return y, log_abs_det_jacobian

    def inverse(self, y, cond=None):
        if self.training:
            mean = self.batch_mean
            var = self.batch_var
        else:
            mean = self.running_mean
            var = self.running_var

        x_hat = (y - self.bias) * torch.exp(-self.weight)
        x = x_hat * torch.sqrt(var + self.eps) + mean

        log_abs_det_jacobian = (0.5 * torch.log(var + self.eps) - self.weight).expand_as(x)

        return x, log_abs_det_jacobian



def create_masks(input_size, hidden_size, n_hidden, input_order='sequential', input_degrees=None):
    # MADE paper sec 4:
    # degrees of connections between layers -- ensure at most in_degree - 1 connections
    degrees = []

    # set input degrees to what is provided in args (the flipped order of the previous layer in a stack of mades);
    # else init input degrees based on strategy in input_order (sequential or random)
    if input_order == 'sequential':
        degrees += [torch.arange(input_size)] if input_degrees is None else [input_degrees]
        for _ in range(n_hidden + 1):
            degrees += [torch.arange(hidden_size) % (input_size - 1)]
        # degrees += [torch.arange(input_size) % input_size - 1] if input_degrees is None else [
        #     input_degrees % input_size - 1]

    elif input_order == 'random':
        degrees += [torch.randperm(input_size)] if input_degrees is None else [input_degrees]
        for _ in range(n_hidden + 1):
            min_prev_degree = min(degrees[-1].min().item(), input_size - 1)
            degrees += [torch.randint(min_prev_degree, input_size, (hidden_size,))]
        # min_prev_degree = min(degrees[-1].min().item(), input_size - 1)
        # degrees += [torch.randint(min_prev_degree, input_size, (input_size,)) - 1] if input_degrees is None else [
        #     input_degrees - 1]

    # construct masks of hidden layers
    masks = []
    for (d0, d1) in zip(degrees[:-1], degrees[1:]):
        masks += [(d1.unsqueeze(-1) >= d0.unsqueeze(0)).float()]

    # construct mask of output layer
    masks += [(degrees[0].unsqueeze(-1) > degrees[-1].unsqueeze(0)).float()]

    return masks, degrees[0]



class MaskedLinear(nn.Linear):
    """
    MADE building block layer
    zeros out the connections of a linear layer according to an input mask
    """

    def __init__(self, input_size, n_outputs, mask, cond_size=None):
        super().__init__(input_size, n_outputs)

        self.register_buffer('mask', mask)

        self.cond_size = cond_size
        if cond_size is not None:
            self.cond_weight = nn.Parameter(torch.rand(n_outputs, cond_size) / math.sqrt(cond_size))

    def forward(self, x, cond=None):
        out = F.linear(x, self.weight * self.mask, self.bias)
        if cond is not None:
            out += F.linear(cond, self.cond_weight)
        return out

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None)



class FlowListExecution(nn.Sequential):
    """Executes all flows provided as module list"""

    def forward(self, x, cond):
        sum_log_abs_det_jacobians = 0
        for flow in self:
            x, log_abs_det_jacobian = flow(x, cond)
            sum_log_abs_det_jacobians += log_abs_det_jacobian
        return x, sum_log_abs_det_jacobians.sum(1)

    def inverse(self, u, cond):
        sum_log_abs_det_jacobians = 0
        for flow in reversed(self):
            u, log_abs_det_jacobian = flow.inverse(u, cond)
            sum_log_abs_det_jacobians += log_abs_det_jacobian
        return u, sum_log_abs_det_jacobians.sum(1)



class MADE(nn.Module):
    """ MADE with standard Gaussian as base distribution """

    def __init__(self, input_size, hidden_size, n_hidden, activation='relu', input_order='sequential',
                 input_degrees=None, cond_size=None):
        """
        Args:
            input_size      --   int      --   dim of inputs
            hidden_size     --   int      --   dim of hidden layers
            n_hidden        --   int      --   number of hidden layers
            activation      --   str      --   activation function to use
            input_order     --   str      --   variable order for creating the autoregressive masks (sequential|random)
            input_degrees   --   tensor   --   flipped input degrees of previous MADE in MAF (optional)
            cond_size       --   int      --   size of conditional vector (emtpy makes model unconditional)
        """
        super().__init__()

        # base distribution for calculation of log prob under the model
        self.register_buffer('base_dist_mean', torch.zeros(input_size))
        self.register_buffer('base_dist_var', torch.ones(input_size))

        # create masks
        masks, self.input_degrees = create_masks(input_size, hidden_size, n_hidden, input_order, input_degrees)

        # setup activation
        if activation == 'relu':
            activation_fn = nn.ReLU()
        elif activation == 'tanh':
            activation_fn = nn.Tanh()
        else:
            raise ValueError('Check activation function.')

        # construct model
        self.net_input = MaskedLinear(input_size, hidden_size, masks[0], cond_size)
        self.net = []
        for m in masks[1:-1]:
            self.net += [activation_fn, MaskedLinear(hidden_size, hidden_size, m)]
        self.net += [activation_fn, MaskedLinear(hidden_size, 2 * input_size, masks[-1].repeat(2, 1))]
        self.net = nn.Sequential(*self.net)

    @property
    def base_dist(self):
        return D.Normal(self.base_dist_mean, self.base_dist_var)

    def forward(self, x, cond=None):

        # MAF eq 4 -- return mean and log std
        m, log_std = self.net(self.net_input(x, cond)).chunk(chunks=2, dim=-1)
        u = x * torch.exp(log_std) + m

        # MAF eq 5
        log_abs_det_jacobian = log_std

        return u, log_abs_det_jacobian

    def inverse(self, u, cond=None):

        # MAF eq 3
        x = torch.zeros_like(u)

        # run through reverse model
        for i in self.input_degrees:
            m, log_std = self.net(self.net_input(x, cond)).chunk(chunks=2, dim=-1)
            x[:, i] = (u[:, i] - m[:, i]) * torch.exp(-log_std[:, i])
        log_abs_det_jacobian = - log_std

        return x, log_abs_det_jacobian

    def log_prob(self, x, cond=None):
        u, log_abs_det_jacobian = self.forward(x, cond)
        return torch.sum(self.base_dist.log_prob(u), dim=1) + log_abs_det_jacobian, u



class IAF(nn.Module):
    """ MAF which stacks MADEs """

    def __init__(self, n_blocks, input_size, hidden_size, n_hidden, activation='relu',
                 input_order='sequential', cond_size=None, batch_norm=True):
        """
        Args:
            n_blocks       --   int    --   number of MADE building blocks
            input_size     --   int    --   dim of inputs
            hidden_size    --   int    --   dim of hidden layers in every MADE
            n_hidden       --   int    --   number of hidden layers in every MADE
            activation     --   str    --   activation function to use
            input_order    --   str    --   variable order for creating the autoregressive masks (sequential|random)
            cond_size      --   int    --   size of conditional vector (emtpy makes model unconditional)
            batch_norm     --   bool   --   indicator to use batch normalization
        """
        super().__init__()

        # base distribution for calculation of log prob under the model
        self.register_buffer('base_dist_mean', torch.zeros(input_size))
        self.register_buffer('base_dist_var', torch.ones(input_size))

        # construct model
        modules = []
        self.input_degrees = None
        for i in range(n_blocks):
            modules += [MADE(input_size, hidden_size, n_hidden, activation, input_order, self.input_degrees, cond_size)]
            self.input_degrees = modules[-1].input_degrees.flip(0)
            modules += batch_norm * [BatchNorm1d(input_size)]
        self.net = FlowListExecution(*modules)

    @property
    def base_dist(self):
        return D.Normal(self.base_dist_mean, self.base_dist_var)

    def forward(self, x, cond=None):
        return self.net(x, cond)

    def inverse(self, u, cond=None):
        return self.net.inverse(u, cond)

    def log_prob(self, x, cond=None):
        u, sum_log_abs_det_jacobians = self.forward(x, cond)
        return torch.sum(self.base_dist.log_prob(u), dim=1) + sum_log_abs_det_jacobians, u



def get_activation(act_str):
    return{
        'relu': nn.ReLU(),
        'lrelu': nn.LeakyReLU(),
        'tanh': nn.Tanh()
    }[act_str]



def get_initializer(initializer):
    return {
        'n': torch.nn.init.normal_,
        'u': torch.nn.init.uniform_,
        'k_n': torch.nn.init.kaiming_normal_,
        'k_u': torch.nn.init.kaiming_uniform_,
        'x_n': torch.nn.init.xavier_normal_,
        'x_u': torch.nn.init.xavier_uniform_,
    }[initializer]



def weights_init(m):
    if isinstance(m, nn.Conv2d):
        m.weight.data.normal_(0.0, 0.02)
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
