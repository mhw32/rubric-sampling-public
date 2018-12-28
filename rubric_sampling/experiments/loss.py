from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import math
import numpy as np

import torch
import torch.nn.functional as F

LOG2PI = float(np.log(2.0 * math.pi))


def p_program_label_melbo(  seq, length, label,
                            seq_logits_xy, label_out_xy, z_xy, z_mu_xy, z_logvar_xy,
                            seq_logits_x, label_out_x, z_x, z_mu_x, z_logvar_x,
                            seq_logits_y, label_out_y, z_y, z_mu_y, z_logvar_y,
                            annealing_factor=1., alpha_program=1., alpha_label=1.,
                            lambda_program=1., lambda_label=1.):
    
    p_x_y = p_program_label_elbo(   seq, seq_logits_xy, label, label_out_xy, z_xy, z_mu_xy, z_logvar_xy,
                                    alpha_program=alpha_program, alpha_label=alpha_label,
                                    annealing_factor=annealing_factor)
    p_x = p_program_elbo(   seq, seq_logits_x, z_x, z_mu_x, z_logvar_x,
                            annealing_factor=annealing_factor)
    p_y = p_label_elbo(label, label_out_y, z_y, z_mu_y, z_logvar_y,
                       annealing_factor=annealing_factor)

    MELBO = p_x_y + lambda_program * p_x + lambda_label * p_y

    return MELBO


def p_program_label_elbo(   seq, seq_logits, label, label_out, z, z_mu, z_logvar, 
                            annealing_factor=1., alpha_program=1., alpha_label=1.):
    r"""Lower bound on the joint distribution over program strings (x) and labels (y).
    Critically, the decoder is auto-regressive.

    log p(x,y) >= E_q(z|x,y)[alpha_program * log p(x|z,x') + alpha_label * log p(y|z) + 
                             log p(z) - log q(z|x,y)]
    """
    log_p_x_given_z = -categorical_program_log_pdf(seq[:, 1:], seq_logits[:, :-1])
    log_p_y_given_z = -bernoulli_log_pdf(label, label_out)

    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    kl_q_z_given_x_y_and_p_z = -0.5 * (1 + z_logvar - z_mu.pow(2) - z_logvar.exp())
    kl_q_z_given_x_y_and_p_z = torch.sum(kl_q_z_given_x_y_and_p_z, dim=1)

    # lower bound on marginal likelihood
    ELBO =  alpha_program * log_p_x_given_z + alpha_label * log_p_y_given_z + \
            annealing_factor * kl_q_z_given_x_y_and_p_z
    ELBO = torch.mean(ELBO)

    return ELBO


def p_program_elbo(seq, seq_logits, z, z_mu, z_logvar, annealing_factor=1.):
    r"""Lower bound on a program string.

    log p(x) >= E_q(z|x)[log p(x|z,x') + log p(z) - log q(z|x)]

    Notably, the generative model is autoregressive.
    """
    log_p_x_given_z = -categorical_program_log_pdf(seq[:, 1:], seq_logits[:, :-1])
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    kl_q_z_given_x_and_p_z = -0.5 * (1 + z_logvar - z_mu.pow(2) - z_logvar.exp())
    kl_q_z_given_x_and_p_z = torch.sum(kl_q_z_given_x_and_p_z, dim=1)

    # lower bound on marginal likelihood
    elbo = log_p_x_given_z + annealing_factor * kl_q_z_given_x_and_p_z
    elbo = torch.mean(elbo)

    return elbo


def p_label_elbo(label, label_out, z, z_mu, z_logvar, annealing_factor=1.):
    r"""Lower bound on label evidence.

    log p(y) >= E_q(z|y)[log p(y,z) - log q(z|y)]

    @param label: torch.Tensor
                  observation of a label
    @param label_out: torch.Tensor
                      tensor of outs (post-sigmoid) for bernoulli label
    @param z: torch.Tensor
              latent sample
    @param z_mu: torch.Tensor
                 mean of variational distribution
    @param z_logvar: torch.Tensor
                     log-variance of variational distribution
    """
    log_p_y_given_z = -bernoulli_log_pdf(label, label_out)
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    kl_q_z_given_y_and_p_z = -0.5 * (1 + z_logvar - z_mu.pow(2) - z_logvar.exp())
    kl_q_z_given_y_and_p_z = torch.sum(kl_q_z_given_y_and_p_z, dim=1)

    # lower bound on marginal likelihood
    elbo = log_p_y_given_z + annealing_factor * kl_q_z_given_y_and_p_z
    elbo = torch.mean(elbo)

    return elbo


def categorical_program_log_pdf(seq, seq_logits):
    r"""Log-likelihood of each token of program parameterized
    as a categorical distribution over a vocabulary.
    """
    n, s, v = seq_logits.size()
    seq_logits_2d = seq_logits.contiguous().view(n * s, v)
    seq_2d = seq[:, :s].contiguous().view(n * s)
    loss = -F.cross_entropy(seq_logits_2d, seq_2d, reduction='none')
    loss = loss.view(n, s)
    loss = torch.sum(loss, dim=1)

    return loss


def bernoulli_log_pdf(x, mu):
    r"""Log-likelihood of data given ~Bernoulli(mu)

    @param x: PyTorch.Tensor
              ground truth input
    @param mu: PyTorch.Tensor
               Bernoulli distribution parameters
    @return log_pdf: PyTorch.Tensor
                     log-likelihood
    """
    mu = torch.clamp(mu, 1e-7, 1.-1e-7)
    return torch.sum(x * torch.log(mu) + (1. - x) * torch.log(1. - mu), dim=1)


def gaussian_log_pdf(x, mu, logvar):
    r"""Log-likelihood of data given ~N(mu, exp(logvar))
    
    log f(x) = log(1/sqrt(2*pi*var) * e^(-(x - mu)^2 / var))
             = -1/2 log(2*pi*var) - 1/2 * ((x-mu)/sigma)^2
             = -1/2 log(2pi) - 1/2log(var) - 1/2((x-mu)/sigma)^2
             = -1/2 log(2pi) - 1/2[((x-mu)/sigma)^2 + log var]
    
    @param x: samples from gaussian
    @param mu: mean of distribution
    @param logvar: log variance of distribution
    @return log_pdf: PyTorch.Tensor
                     log-likelihood
    """
    global LOG2PI
    log_pdf = -LOG2PI * x.size(1) / 2. - \
        torch.sum(logvar + torch.pow(x - mu, 2) / (torch.exp(logvar) + 1e-7), dim=1) / 2.

    return log_pdf


def unit_gaussian_log_pdf(x):
    r"""Log-likelihood of data given ~N(0, 1)
    
    @param x: PyTorch.Tensor
              samples from gaussian
    @return log_pdf: PyTorch.Tensor
                     log-likelihood
    """
    global LOG2PI
    return -LOG2PI * x.size(1) / 2. - torch.sum(torch.pow(x, 2), dim=1) / 2.
