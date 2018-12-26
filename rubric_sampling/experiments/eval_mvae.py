r"""The paper reports accuracies for feedback related to geometry 
and loop concepts, and do so split by different parts of the zipf.
"""

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import sys
import random
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score

import torch
import torch.utils.data as data
import torch.nn.functional as F

from .models import ProgramMVAE
from .datasets import load_dataset
from .config import default_hyperparams
from .utils import (
    AverageMeter, 
    merge_args_with_dict, 
    ZIPF_CLASS,
    SOS_TOKEN,
    EOS_TOKEN,
    PAD_TOKEN,
    UNK_TOKEN,
)
from .rubric_utils.load_params import get_label_params


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint_path', type=str, help='where model checkpoint is saved')
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='enables CUDA training [default: False]')
    args = parser.parse_args()
    args.cuda = args.cuda and torch.cuda.is_available()
    device = torch.device('cuda' if args.cuda else 'cpu')

    # reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    checkpoint = torch.load(args.checkpoint_path)
    train_args = checkpoint['cmd_line_args']
    state_dict = checkpoint['state_dict']
    vocab = checkpoint['vocab']
    
    sos_idx = vocab['w2i'][SOS_TOKEN]
    eos_idx = vocab['w2i'][EOS_TOKEN]
    pad_idx = vocab['w2i'][PAD_TOKEN]
    unk_idx = vocab['w2i'][UNK_TOKEN]

    label_dim, _, _, loop_ix, geometry_ix = get_label_params(train_args.problem_id)

    # we use the annotated test set; note: use the vocab from training
    dataset = load_dataset( 'annotated', train_args.problem_id, 'test', vocab=vocab,
                            max_seq_len=train_args.max_seq_len, min_occ=train_args.min_occ)

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=train_args.batch_size, shuffle=False)

    model = ProgramMVAE(train_args.z_dim, label_dim, len(vocab['w2i']), sos_idx, eos_idx, pad_idx, unk_idx, 
                        embedding_dim=train_args.embedding_dim, hidden_dim=train_args.hidden_dim,
                        word_dropout=train_args.word_dropout, num_layers=train_args.num_layers)
    model = model.to(device)
    model.load_state_dict(state_dict)

    # store true and predictions here
    y_true = [[] for _ in xrange(len(ZIPF_CLASS))]
    y_pred = [[] for _ in xrange(len(ZIPF_CLASS))]

    with torch.no_grad():
        with tqdm(total=len(loader)) as pbar:
            for (seq, length, label, zipf) in loader:
                batch_size = len(seq)
                seq = seq.to(device)
                length = length.to(device)
                label = label.to(device)

                z_mu, _ = model.inference(seq, length, None)
                label_out = model.label_decoder(z_mu)
                label_pred = torch.round(label_out).cpu().numpy()

                label = label.cpu().numpy()
                zipf = zipf.numpy()

                for j in xrange(len(ZIPF_CLASS)):
                    # store the actual predictions but do so
                    # by splitting this up by zipf position
                    y_true[j].append(label[zipf == j])
                    y_pred[j].append(label_pred[zipf == j])

                pbar.update()

    for i in xrange(len(ZIPF_CLASS)):
        y_true[i] = np.concatenate(y_true[i], axis=0)
        y_pred[i] = np.concatenate(y_pred[i], axis=0)

    f1 = np.zeros((len(ZIPF_CLASS), label_dim))
    acc = np.zeros((len(ZIPF_CLASS), label_dim))

    for i in xrange(len(ZIPF_CLASS)):  # we don't care about the HEAD
        for j in xrange(label_dim):
            f1[i, j] = f1_score(y_true[i][:, j], y_pred[i][:, j])
            acc[i, j] = accuracy_score(y_true[i][:, j], y_pred[i][:, j])

    f1_loop = np.mean(f1[:, loop_ix], axis=1)
    f1_geom = np.mean(f1[:, geometry_ix], axis=1)
    acc_loop = np.mean(acc[:, loop_ix], axis=1)
    acc_geom = np.mean(acc[:, geometry_ix], axis=1)

    inv_zipf_map = {v: k for k, v in ZIPF_CLASS.iteritems()}

    for i in xrange(len(ZIPF_CLASS)):
        print('zipf %s | loop f1=%.4f, acc=%.4f | geometry f1=%.4f, acc=%.4f' % (
            inv_zipf_map[i], f1_loop[i], acc_loop[i], f1_geom[i], acc_geom[i]))
