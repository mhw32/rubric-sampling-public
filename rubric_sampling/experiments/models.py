from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import sys
import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.nn.utils.rnn as rnn_utils

from .loss import (
    gaussian_log_pdf,
    unit_gaussian_log_pdf,
    bernoulli_log_pdf,
    categorical_program_log_pdf,
    log_mean_exp,
)


class ProgramRNN(nn.Module):
    r"""Supervised recurrent neural network for predicting feedback labels.
    Parameterizes p(label|program). 

    @param z_dim: integer
                  size of latent vector
    @param label_dim: integer
                      number of (binary) labels
    @param vocab_size: integer
                       size of vocabulary
    @param embedding_dim: integer [default: 300]
                          size of learned embedding
    @param hidden_dim: integer [default: 256]
                       size of hidden layer
    @param word_dropout: float [default: 0.5]
                         probability to dropout input sequences to decoder
    @param num_layers: integer [default: 2]
                       number of hidden layers in GRU
    """
    def __init__(   self, z_dim, label_dim, vocab_size, embedding_dim=300, 
                    hidden_dim=256, num_layers=2):
        super(ProgramRNN, self).__init__()

        self.z_dim = z_dim
        self.label_dim = label_dim
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.embedding_module = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.encoder = ProgramEncoder(  self.embedding_module, self.z_dim, 
                                        hidden_dim=self.hidden_dim, 
                                        num_layers=self.num_layers)
        self.decoder = LabelDecoder(self.z_dim, self.label_dim, 
                                    hidden_dim=self.hidden_dim)

    def forward(self, seq, length):
        z_mu, _  = self.encoder(seq, length)
        return self.decoder(z_mu)


class ProgramMVAE(nn.Module):
    r"""Multimodal Variational Autoencoder (MVAE) with Expert Supervision.
    The MVAE is the equivalent of a soft transformation of the expert pCFG graph.

    @param z_dim: integer
                  size of latent vector
    @param label_dim: integer
                      number of (binary) labels
    @param vocab_size: integer
                       size of vocabulary
    @param sutilos_idx: integer
                    index for start-of-sentence
    @param eos_idx: integer
                    index for end-of-sentence
    @param pad_idx: integer
                    index for padding
    @param unk_idx: integer
                    index for unknown tokens
    @param embedding_dim: integer [default: 300]
                          size of learned embedding
    @param hidden_dim: integer [default: 256]
                       size of hidden layer
    @param word_dropout: float [default: 0.5]
                         probability to dropout input sequences to decoder
    @param num_layers: integer [default: 2]
                       number of hidden layers in GRU
    """
    def __init__(self, z_dim, label_dim, vocab_size, sos_idx, eos_idx, pad_idx, unk_idx,
                 embedding_dim=300, hidden_dim=256, word_dropout=0.5, num_layers=2):
        super(ProgramMVAE, self).__init__()

        self.z_dim = z_dim
        self.label_dim = label_dim
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx
        self.pad_idx = pad_idx
        self.word_dropout = word_dropout
        self.num_layers = num_layers

        self.product_experts = ProductOfExperts()
        self.embedding_module = nn.Embedding(self.vocab_size, self.embedding_dim)

        self.program_encoder = ProgramEncoder(  self.embedding_module, self.z_dim, 
                                                hidden_dim=self.hidden_dim, 
                                                num_layers=self.num_layers)
        self.program_decoder = ProgramDecoder(
            self.embedding_module, self.z_dim, self.sos_idx, self.eos_idx, self.pad_idx, self.unk_idx,
            hidden_dim=self.hidden_dim, word_dropout=self.word_dropout, num_layers=self.num_layers)
        
        self.label_encoder = LabelEncoder(
            self.z_dim, self.label_dim, hidden_dim=self.hidden_dim)
        self.label_decoder = LabelDecoder(
            self.z_dim, self.label_dim, hidden_dim=self.hidden_dim)
        
    def reparametrize(self, z_mu, z_logvar):
        std = torch.exp(0.5 * z_logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(z_mu)

    def forward(self, seq, length, label, hide_seq=False):
        if hide_seq:
            z_mu, z_logvar = self.inference(None, None, label)
        else:
            z_mu, z_logvar = self.inference(seq, length, label)

        z = self.reparameterize(z_mu, z_logvar)
        seq_logits = self.program_decoder(z, seq, length)
        label_out = self.label_decoder(z)

        return seq_logits, label_out, z, z_mu, z_logvar

    def prior_expert(self, mu):
        p_mu = torch.zeros_like(mu)
        p_logvar = torch.zeros_like(mu)
        return p_mu, p_logvar

    def inference(self, seq, length, label):
        z_mu, z_logvar = [], []
        
        if seq is not None:
            program_mu, program_logvar = self.program_encoder(seq, length)
            z_mu.append(program_mu.unsqueeze(0))
            z_logvar.append(program_logvar.unsqueeze(0))

        if label is not None:
            label_mu, label_logvar = self.label_encoder(label)
            z_mu.append(label_mu.unsqueeze(0))
            z_logvar.append(label_logvar.unsqueeze(0))

        prior_mu, prior_logvar = self.prior_expert(z_mu[0])
        z_mu.append(prior_mu)
        z_logvar.append(prior_logvar)

        z_mu = torch.cat(z_mu, dim=0)
        z_logvar = torch.cat(z_logvar, dim=0)

        z_mu, z_logvar = self.product_experts(z_mu, z_logvar)

        return z_mu, z_logvar

    def get_joint_marginal(self, seq, length, label, n_samples=100):
        z_mu, z_logvar = self.inference(seq, length, label)

        log_w = []
        for i in xrange(n_samples):
            z_i = self.reparameterize(z_mu, z_logvar)
            x_logits_i = self.decode_text(z_i, seq, length)
            y_out_i = self.decode_label(z_i)

            log_p_x_given_z_i = categorical_program_log_pdf(seq[:, 1:], x_logits_i[:, :-1])
            log_p_y_given_z_i = bernoulli_log_pdf(label, y_out_i)
            log_q_z_given_x_y_i = gaussian_log_pdf(z_i, z_mu, z_logvar)
            log_p_z_i = unit_gaussian_log_pdf(z_i)

            log_w_i = log_p_x_given_z_i + log_p_y_given_z_i + log_p_z_i - log_q_z_given_x_y_i
            log_w.append(log_w_i.unsqueeze(1))

        log_w = torch.cat(log_w, dim=1)
        log_p_x_y = log_mean_exp(log_w, dim=1)
        log_p_x_y = -torch.mean(log_p_x_y)

        return log_p_x_y

    def get_text_marginal(self, seq, length, n_samples=100):
        z_mu, z_logvar = self.inference(seq, length, None)

        log_w = []
        for i in xrange(n_samples):
            z_i = self.reparameterize(z_mu, z_logvar)
            seq_logits_i = self.decode_text(z_i, seq, length)

            # probability of text is product of probabilities of each word
            log_p_x_given_z_i = categorical_program_log_pdf(seq[:, 1:], seq_logits_i[:, :-1])
            log_q_z_given_x_i = gaussian_log_pdf(z_i, z_mu, z_logvar)
            log_p_z_i = unit_gaussian_log_pdf(z_i)

            log_w_i = log_p_x_given_z_i + log_p_z_i - log_q_z_given_x_i
            log_w.append(log_w_i.unsqueeze(1))

        log_w = torch.cat(log_w, dim=1)
        log_p_x = log_mean_exp(log_w, dim=1)
        log_p_x = -torch.mean(log_p_x)

        return log_p_x

    def get_label_marginal(self, y, n_samples=100):
        z_mu, z_logvar = self.inference(None, None, y)

        log_w = []
        for i in xrange(n_samples):
            z_i = self.reparameterize(z_mu, z_logvar)
            y_out_i = self.decode_label(z_i)

            log_p_y_given_z_i = bernoulli_log_pdf(y, y_out_i)
            log_q_z_given_y_i = gaussian_log_pdf(z_i, z_mu, z_logvar)
            log_p_z_i = unit_gaussian_log_pdf(z_i)

            log_w_i = log_p_y_given_z_i + log_p_z_i - log_q_z_given_y_i
            log_w.append(log_w_i.unsqueeze(1))

        log_w = torch.cat(log_w, dim=1)
        log_p_y = log_mean_exp(log_w, dim=1)
        log_p_y = -torch.mean(log_p_y)

        return log_p_y


class ProgramEncoder(nn.Module):
    r"""Parameterizes q(z|program) with RNN.

    Inspired by Bowman et. al. (https://arxiv.org/abs/1511.06349).

    @param embedding_module: nn.Embedding
                             we initialize this separately from the encoder
    @param z_dim: integer
                  size of latent vector
    @param hidden_dim: integer [default: 256]
                       size of hidden layer
    @param num_layers: integer [default: 2]
                       number of hidden layers in GRU
    """
    def __init__(self, embedding_module, z_dim, hidden_dim=256, num_layers=2):
        super(ProgramEncoder, self).__init__()

        self.embedding = embedding_module
        self.embedding_dim = self.embedding.embedding_dim
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.gru = nn.GRU(self.embedding_dim, self.hidden_dim, num_layers=num_layers)
        self.h2mu = nn.Linear(self.hidden_dim * self.num_layers, self.z_dim)
        self.h2logvar = nn.Linear(self.hidden_dim * self.num_layers, self.z_dim)

    def forward(self, seq, length):
        batch_size = seq.size(0)

        if batch_size > 1:
            # sort in decreasing order of length in order to pack
            # sequence; if only 1 element in batch, nothing to do.
            sorted_lengths, sorted_idx = torch.sort(length, descending=True)
            seq = seq[sorted_idx]

        embed_seq = self.embedding(seq)
        # reorder from (B,L,D) to (L,B,D)
        embed_seq = embed_seq.transpose(0, 1)

        packed = rnn_utils.pack_padded_sequence(
            embed_seq,
            sorted_lengths.data.tolist() if batch_size > 1 else length.data.tolist())

        _, hidden = self.gru(packed)
        hidden = hidden.permute(1, 0, 2).contiguous()
        hidden = hidden.view(batch_size, self.hidden_dim * self.num_layers)

        if batch_size > 1:
            _, reversed_idx = torch.sort(sorted_idx)
            hidden = hidden[reversed_idx]

        z_mu = self.h2mu(hidden)
        z_logvar = self.h2logvar(hidden)

        return z_mu, z_logvar


class ProgramDecoder(nn.Module):
    r"""Parameterizes p(program|z) with RNN to generate a distribution of a 
    sequence of tokens. Assumes a maximum sequence length and a fixed vocabulary.

    We return logits to a categorical so please use
        nn.CrossEntropy
    instead of
        nn.NLLLoss

    Inspired by Bowman et. al. (https://arxiv.org/abs/1511.06349).

    @param embedding_module: nn.Embedding
                             pass the embedding module (share with encoder)
    @param z_dim: integer
                  size of latent vector
    @param sos_idx: integer
                    index for start-of-sentence
    @param eos_idx: integer
                    index for end-of-sentence
    @param pad_idx: integer
                    index for padding
    @param unk_idx: integer
                    index for unknown tokens
    @param hidden_dim: integer [default: 256]
                       size of hidden layer
    @param word_dropout: float [default: 0.5]
                         probability to dropout input sequences to decoder
    @param num_layers: integer [default: 2]
                       number of hidden layers in GRU
    """
    def __init__(   self, embedding_module, z_dim, sos_idx, eos_idx, pad_idx, unk_idx,
                    hidden_dim=256, word_dropout=0.5, num_layers=2):
        super(ProgramDecoder, self).__init__()

        self.embedding = embedding_module
        self.embedding_dim = embedding_module.embedding_dim
        self.vocab_size = embedding_module.num_embeddings
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx
        self.pad_idx = pad_idx
        self.unk_idx = unk_idx
        self.word_dropout = word_dropout
        self.num_layers = num_layers
        self.gru = nn.GRU(  self.embedding_dim, self.hidden_dim,
                            num_layers=self.num_layers)
        self.z2h = nn.Linear(self.z_dim, self.hidden_dim * self.num_layers)
        self.outputs2vocab = nn.Linear(self.hidden_dim, self.vocab_size)

    def forward(self, z, seq, length):
        batch_size = z.size(0)

        if batch_size > 1:
            sorted_lengths, sorted_idx = torch.sort(length, descending=True)

            z = z[sorted_idx]
            seq = seq[sorted_idx]

        if self.word_dropout > 0:
            # randomly replace with unknown tokens
            prob = torch.rand(seq.size())
            prob[(seq.cpu().data - self.sos_index) & \
                 (seq.cpu().data - self.pad_index) == 0] = 1
            mask_seq = seq.clone()
            mask_seq[(prob < self.word_dropout).to(z.device)] = self.unk_idx
            seq = mask_seq

        embed_seq = self.embedding(seq)
        # reorder from (B,L,D) to (L,B,D)
        embed_seq = embed_seq.transpose(0, 1)

        packed = rnn_utils.pack_padded_sequence(
            embed_seq,
            sorted_lengths.data.tolist() if batch_size > 1 else length.data.tolist())

        hidden = self.z2h(z)
        hidden = hidden.view(batch_size, self.num_layers, self.hidden_dim)
        hidden = hidden.permute(1, 0, 2).contiguous()

        outputs, _ = self.gru(packed, hidden)
        outputs = rnn_utils.pad_packed_sequence(outputs, batch_first=True)[0]
        outputs = outputs.contiguous()

        # reorder from (L,B,D) to (B,L,D)
        outputs = outputs.transpose(0, 1)

        if batch_size > 1:
            _, reversed_idx = torch.sort(sorted_idx)
            outputs = outputs[reversed_idx]

        max_length = outputs.size(1)
        outputs_2d = outputs.view(batch_size * max_length, self.hidden_dim)
        outputs_2d = self.outputs2vocab(outputs_2d)
        outputs = outputs_2d.view(batch_size, max_length, self.vocab_size)

        return outputs.contiguous()

    def sample(self, z, max_seq_len, greedy=False):
        r"""Sample tokens in an auto-regressive framework.
        
        @param z: torch.Tensor
                  sample of latent variables
        @param max_seq_len: integer
                            maximum size of sequence
        @param greedy: boolean [default: False]
                       pick most likely token or sample token?
        """
        with torch.no_grad():
            batch_size = z.size(0) 

            # initialize hidden state
            hidden = self.z2h(z)
            hidden = hidden.view(batch_size, self.num_layers, self.hidden_dim)
            hidden = hidden.permute(1, 0, 2).contiguous()

            # first input is SOS token
            inputs = np.array([self.sos_idx for _ in xrange(batch_size)])
            inputs = torch.from_numpy(inputs)
            inputs = inputs.unsqueeze(1)
            inputs = inputs.to(z.device)
            
            # save SOS as first generated token
            inputs_npy = inputs.squeeze(1).cpu().numpy()
            sampled_ids = [[w] for w in inputs_npy]

            # (B,L,D) to (L,B,D)
            inputs = inputs.transpose(0, 1)

            # compute embeddings
            inputs = self.embedding(inputs)

            for i in xrange(max_seq_len):
                outputs, hidden = self.gru(inputs, hidden)  # outputs: (L=1,B,H)
                outputs = outputs.squeeze(0)                # outputs: (B,H)
                outputs = self.outputs2vocab(outputs)       # outputs: (B,V)

                if greedy:
                    predicted = outputs.max(1)[1]
                    predicted = predicted.unsqueeze(1)
                else:
                    outputs = F.softmax(outputs, dim=1)
                    predicted = torch.multinomial(outputs, 1)

                predicted_npy = predicted.squeeze(1).cpu().numpy()
                predicted_lst = predicted_npy.tolist()

                for w, so_far in zip(predicted_lst, sampled_ids):
                    if so_far[-1] != self.eos_index:
                        so_far.append(w)

                inputs = predicted.transpose(0, 1)          # inputs: (L=1,B)
                inputs = self.embedding(inputs)             # inputs: (L=1,B,E)

            sampled_lengths = [len(text) for text in sampled_ids]
            sampled_lengths = np.array(sampled_lengths)

            max_length = max(sampled_lengths)
            padded_ids = np.ones((batch_size, max_length)) * self.pad_idx

            for i in xrange(batch_size):
                padded_ids[i, :sampled_lengths[i]] = sampled_ids[i]

            sampled_lengths = torch.from_numpy(sampled_lengths).long()
            sampled_ids = torch.from_numpy(padded_ids).long()

        return sampled_ids, sampled_lengths


class LabelEncoder(nn.Module):
    r"""Parameterizes q(z|label).

    @param z_dim: integer
                  number of latent dimensions
    @param label_dim: integer
                      number of label dimensions
    @param hidden_dim: integer [default: 256]
                       number of hidden dimensions
    """
    def __init__(self, z_dim, label_dim, hidden_dim=256):
        super(LabelEncoder, self).__init__()
        self.z_dim = z_dim
        self.label_dim = label_dim
        self.hidden_dim = hidden_dim

        self.net = nn.Sequential(
            nn.Linear(self.label_dim, self.hidden_dim),
            Swish(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            Swish(),
            # here we don't explicitly separate into (mu, logvar)
            # but equivalent thing
            nn.Linear(self.hidden_dim, self.z_dim * 2))

    def forward(self, x):
        h = self.net(x)
        z_mu, z_logvar = torch.chunk(h, 2, dim=1)

        return z_mu, z_logvar


class LabelDecoder(nn.Module):
    r"""Parameterizes p(label|z).
    
    @param z_dim: integer
                  number of latent dimensions
    @param label_dim: integer
                      number of label dimensions
    @param hidden_dim: integer [default: 256]
                       number of hidden dimensions
    """
    def __init__(self, z_dim, label_dim, hidden_dim=256):
        super(LabelDecoder, self).__init__()

        self.z_dim = z_dim
        self.label_dim = label_dim
        self.hidden_dim = hidden_dim

        self.net = nn.Sequential(
            nn.Linear(self.z_dim, self.hidden_dim),
            Swish(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            Swish(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            Swish(),
            nn.Linear(self.hidden_dim, self.label_dim))

    def forward(self, z):
        # we assume binary labels
        return F.sigmoid(self.net(z))


class ProductOfExperts(nn.Module):
    r"""Return parameters for product of independent experts.
    See https://arxiv.org/pdf/1410.7827.pdf for equations.

    @param mu: M x D for M experts
    @param logvar: M x D for M experts
    """
    def forward(self, mu, logvar, eps=1e-8):
        var = torch.exp(logvar) + eps
        T = 1 / var  # precision of i-th Gaussian expert at point x
        pd_mu = torch.sum(mu * T, dim=0) / torch.sum(T, dim=0)
        pd_var = 1 / torch.sum(T, dim=0)
        pd_logvar = torch.log(pd_var)

        return pd_mu, pd_logvar


class Swish(nn.Module):
    def forward(self, x):
        return x * F.sigmoid(x)


def swish(x):
    return x * F.sigmoid(x)
