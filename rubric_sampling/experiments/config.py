from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

default_hyperparams = {
    'z_dim': 20,
    'embedding_dim': 300,
    'hidden_dim': 256,
    'num_layers': 1,
    'min_occ': 1,
    'batch_size': 64,
    'epochs': 200,
    'lr': 1e-3,
    'log_interval': 10,
    'alpha_program': 1,
    'alpha_label': 10,
    'lambda_program': 1,
    'lambda_label': 10,
    'lambda_unlabeled': 1,
    'word_dropout': 0.5,
    'seed': 1,
}
