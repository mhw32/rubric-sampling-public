r"""
Generate synthetic programs with feedback labels
using the variational autoencoder.
"""

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import sys
import math
import json
import cPickle
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from .models import ProgramMVAE
from .utils import (
    idx2word, 
    tensor_to_labels, 
    SOS_TOKEN,
    EOS_TOKEN,
    PAD_TOKEN,
    UNK_TOKEN,
)
from .rubric_utils.load_params import get_label_params


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint_path', type=str, help='path to trained model file')
    parser.add_argument('out_dir', type=str, help='where to save model samples')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for dataset [default: 64]')
    parser.add_argument('--num-samples', type=int, default=1e6, metavar='N',
                        help='input batch size for dataset [default: 1000000]')
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='enables CUDA training [default: False]')
    args = parser.parse_args()
    args.cuda = args.cuda and torch.cuda.is_available()
    device = torch.device('cuda' if args.cuda else 'cpu')

    if not os.path.isdir(args.out_dir):
        os.makedirs(args.out_dir)

    checkpoint = torch.load(args.checkpoint_path)
    train_args = checkpoint['cmd_line_args']
    state_dict = checkpoint['state_dict']
    vocab = checkpoint['vocab']
    w2i, i2w = vocab['w2i'], vocab['i2w']

    sos_idx = vocab['w2i'][SOS_TOKEN]
    eos_idx = vocab['w2i'][EOS_TOKEN]
    pad_idx = vocab['w2i'][PAD_TOKEN]
    unk_idx = vocab['w2i'][UNK_TOKEN]

    label_dim, ix_to_label, label_to_ix, _, _ = get_label_params(train_args.problem_id)

    model = ProgramMVAE(train_args.z_dim, label_dim, len(vocab['w2i']), sos_idx, eos_idx, pad_idx, unk_idx, 
                        embedding_dim=train_args.embedding_dim, hidden_dim=train_args.hidden_dim,
                        word_dropout=train_args.word_dropout, num_layers=train_args.num_layers)
    model = model.to(device)
    model.load_state_dict(state_dict)
    model.eval()

    N_minibatches = (args.num_samples // args.batch_size)
    N_leftover = args.num_samples - N_minibatches * args.batch_size
    assert N_leftover >= 0

    programs, labels = [], []

    pbar_size = N_minibatches + int(N_leftover > 0)
    pbar = tqdm(total=pbar_size)

    with torch.no_grad():
        for i in xrange(N_minibatches):
            # sample from z ~ N(0,1)
            z = torch.randn(args.batch_size, train_args.z_dim)
            z = z.to(device)
            
            seq, length = model.program_decoder.sample(z, train_args.max_seq_len, greedy=False)
            label = model.label_decoder(z)

            # stop storing these on GPU
            seq, label = seq.cpu(), label.cpu()

            # convert programs to strings
            seq = idx2word(seq, i2w=i2w, pad_idx=w2i[PAD_TOKEN])

            # convert labels to strings
            label = [tensor_to_labels(t, label_dim, ix_to_label) for t in label]

            programs.extend(seq)
            labels.extend(label)

            pbar.update()

        if N_leftover > 0:
            z = torch.randn(N_leftover, train_args.z_dim)
            z = z.to(device)

            # stop storing these on GPU
            programs_ = programs_.cpu()
            labels_ = labels_.cpu().data

            seq, length = model.program_decoder.sample(z, train_args.max_seq_len, greedy=False)
            label = model.label_decoder(z)
            seq, label = seq.cpu(), label.cpu()

            seq = idx2word(seq, i2w=i2w, pad_idx=w2i[PAD_TOKEN])
            label = [tensor_to_labels(t, label_dim, ix_to_label) for t in label]

            programs.extend(seq)
            labels.extend(label)

            pbar.update()

    pbar.close()

    with open(os.path.join(args.out_dir, 'samples_mvae.pickle'), 'wb') as fp:
        cPickle.dump({'programs': programs, 'labels': labels}, fp)
