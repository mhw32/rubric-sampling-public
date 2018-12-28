from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import sys
import random
import numpy as np
from tqdm import tqdm

import torch
import torch.optim as optim
import torch.utils.data as data
import torch.nn.functional as F

from .models import ProgramRNN
from .datasets import load_dataset
from .config import default_hyperparams
from .loss import p_program_label_melbo, p_program_elbo
from .utils import (
    AverageMeter, 
    save_checkpoint, 
    merge_args_with_dict,
    SOS_TOKEN,
    EOS_TOKEN,
    PAD_TOKEN,
    UNK_TOKEN,
)
from .rubric_utils.load_params import get_label_params, get_max_seq_len


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=str, help='annotated|synthetic')
    parser.add_argument('problem_id', type=int, help='1|2|3|4|5|6|7|8')
    parser.add_argument('out_dir', type=str, help='where to save outputs')
    parser.add_argument('--add-unlabeled-data', store='true', default=False,
                        help='learn with unlabeled data [default: False]')
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='enables CUDA training [default: False]')
    args = parser.parse_args()
    args.cuda = args.cuda and torch.cuda.is_available()
    merge_args_with_dict(args, default_hyperparams)
    device = torch.device('cuda' if args.cuda else 'cpu')
    args.max_seq_len = get_max_seq_len(args.problem_id)

    label_dim, _, _, _, _ = get_label_params(args.problem_id)

    # reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if not os.path.isdir(args.out_dir):
        os.makedirs(args.out_dir)

    train_dataset = load_dataset(   args.dataset, args.problem_id, 'train', vocab=None, 
                                    max_seq_len=args.max_seq_len, min_occ=args.min_occ)
    val_dataset = load_dataset( args.dataset, args.problem_id, 'val', vocab=train_dataset.vocab, 
                                max_seq_len=args.max_seq_len, min_occ=args.min_occ)
    test_dataset = load_dataset(args.dataset, args.problem_id, 'test', vocab=train_dataset.vocab, 
                                max_seq_len=args.max_seq_len, min_occ=args.min_occ)

    w2i = train_dataset.vocab['w2i']
    vocab_size = len(w2i)
    print('vocab_size: %d...' % vocab_size)
    sos_idx = w2i[SOS_TOKEN]
    eos_idx = w2i[EOS_TOKEN]
    pad_idx = w2i[PAD_TOKEN]
    unk_idx = w2i[UNK_TOKEN]

    train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    if args.add_unlabeled_data:
        unlabeled dataset = load_dataset(   'unlabeled', args.problem_id, 'train', vocab=train_dataset.vocab, 
                                            max_seq_len=args.max_seq_len, min_occ=args.min_occ)
        unlabeled_loader = data.DataLoader( unlabeled_dataset, batch_size=args.batch_size, 
                                            # drop last bc we want to have the right size
                                            shuffle=True, drop_last=True)
        unlabeled_iterator = unlabeled_loader.__iter__()

    model = ProgramMVAE(args.z_dim, label_dim, vocab_size, sos_idx, eos_idx, pad_idx, unk_idx, 
                        embedding_dim=args.embedding_dim, hidden_dim=args.hidden_dim,
                        word_dropout=args.word_dropout, num_layers=args.num_layers)
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)


    def get_unlabeled_minibatch():
        r"""Fail-safe way of sampling unlabeled programs (since we have 
        access to a lot of unlabeled data).
        """
        try:
            seq, length = unlabeled_iterator.__next__()
        except:
            unlabeled_iterator = unlabeled_loader.__iter__()
            seq, length = unlabeled_iterator.__next__()
        
        return seq, length


    def train(epoch):
        model.train()
        loss_meter = AverageMeter()
        acc_meter = AverageMeter()

        for batch_idx, (seq, length, label, _) in enumerate(train_loader):
            assert label is not None
            batch_size = len(seq)
            seq = seq.to(device)
            length = length.to(device)
            label = label.to(device)

            optimizer.zero_grad()
            seq_logits_xy, label_out_xy, z_xy, z_mu_xy, z_logvar_xy = model(seq, length, label)
            seq_logits_x, label_out_x, z_x, z_mu_x, z_logvar_x = model(seq, length, None)
            seq_logits_y, label_out_y, z_y, z_mu_y, z_logvar_y = model(seq, length, label, hide_text=True)

            elbo = p_program_label_melbo(   seq, length, label,
                                            seq_logits_xy, label_out_xy, z_xy, z_mu_xy, z_logvar_xy,
                                            seq_logits_x, label_out_x, z_x, z_mu_x, z_logvar_x,
                                            seq_logits_y, label_out_y, z_y, z_mu_y, z_logvar_y,
                                            alpha_program=args.alpha_program,
                                            alpha_label=args.alpha_label,
                                            lambda_program=args.lambda_program,
                                            lambda_label=args.lambda_label)

            if args.add_unlabeled_data:
                seq_u, length_u = get_unlabeled_minibatch()
                seq_u = seq_u.to(device)
                length_u = length_u.to(device)

                seq_logits_u, _, z_u, z_mu_u, z_logvar_u = model(seq_u, length_u, None)
                elbo_u = p_program_elbo(seq, seq_logits_u, z_u, z_mu_u, z_logvar_u)
                elbo = elbo + args.lambda_unlabeled * elbo_u

            elbo.backward()
            optimizer.step()

            loss_meter.update(-elbo.item(), batch_size)

            # greedily sample from the conditional distribution to make a prediction
            label_pred = model.label_decoder(z_mu_x)
            acc = torch.mean(torch.round(label_pred).detach() == label.detach())
            acc_meter.update(acc.item(), batch_size)

            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tELBO: {:.6f}\tAccuracy: {:.4f}'.format(
                    epoch, batch_idx * batch_size, len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss_meter.avg, 
                    acc_meter.avg))

        print('====> Train Epoch: {}\tELBO: {:.4f}\tAccuracy: {:.4f}'.format(
            epoch, loss_meter.avg, acc_meter.avg))
        
        return loss_meter.avg, acc_meter.avg


    def test(epoch, loader, name='Test'):
        model.eval()

        log_p_x_y_meter = AverageMeter()
        log_p_x_meter = AverageMeter()
        log_p_y_meter = AverageMeter()
        acc_meter = AverageMeter()

        with torch.no_grad():
            pbar = tqdm(total=len(loader))
            for batch_idx, (seq, length, label, _) in enumerate(loader):
                batch_size = len(seq)
                seq = seq.to(device)
                length = length.to(device)
                label = label.to(device).float()

                log_p_x_y = model.get_joint_marginal(seq, length, label, n_samples=10)
                log_p_x = model.get_program_marginal(seq, length, n_samples=10)
                log_p_y = model.get_label_marginal(label, n_samples=10)

                log_p_x_meter.update(log_p_x.item(), batch_size)
                log_p_y_meter.update(log_p_y.item(), batch_size)
                log_p_x_y_meter.update(log_p_x_y.item(), batch_size)

                z_mu_x, _ = model.inference(seq, length, None)
                label_pred = model.label_decoder(z_mu_x)
                acc = torch.mean(torch.round(label_pred).detach() == label.detach())
                acc_meter.update(acc.item(), batch_size)

                pbar.update()
            pbar.close()

        print('====> {} Epoch: {}\tlog p(x,y): {:.4f}\tlog p(x): {:.4f}\tlog p(y): {:.4f}\tAccuracy: {:.4f}'.format(
            name, epoch, -log_p_x_y_meter.avg, -log_p_x_meter.avg, -log_p_y_meter.avg, acc_meter.avg))

        return log_p_x_y_meter.avg, log_p_x_meter.avg, log_p_y_meter.avg, acc_meter.avg


    best_loss = sys.maxint
    track_train_elbo = np.zeros(args.epochs)
    track_val_log_p_x_y = np.zeros(args.epochs)
    track_val_log_p_x = np.zeros(args.epochs)
    track_val_log_p_y = np.zeros(args.epochs)
    track_test_log_p_x_y = np.zeros(args.epochs)
    track_test_log_p_x = np.zeros(args.epochs)
    track_test_log_p_y = np.zeros(args.epochs)
    track_train_acc = np.zeros(args.epochs)
    track_val_acc = np.zeros(args.epochs)
    track_test_acc = np.zeros(args.epochs)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train(epoch)
        val_log_p_x_y, val_log_p_x, val_log_p_y, val_acc = test(epoch, val_loader, name='Val')
        test_log_p_x_y, test_log_p_x, test_log_p_y, test_acc = test(epoch, test_loader, name='Test')

        is_best = val_log_p_x_y < best_loss
        best_loss = min(val_log_p_x_y, best_loss)

        track_train_elbo[epoch - 1] = train_loss
        track_val_log_p_x[epoch - 1] = val_log_p_x
        track_val_log_p_y[epoch - 1] = val_log_p_y
        track_val_log_p_x_y[epoch - 1] = val_log_p_x_y
        track_test_log_p_x[epoch - 1] = test_log_p_x
        track_test_log_p_y[epoch - 1] = test_log_p_y
        track_test_log_p_x_y[epoch - 1] = test_log_p_x_y
        track_train_acc[epoch - 1] = train_acc
        track_val_acc[epoch - 1] = val_acc
        track_test_acc[epoch - 1] = test_acc

        save_checkpoint({
            'state_dict': model.state_dict(),
            'cmd_line_args': args,
            'vocab': train_dataset.vocab,
            'max_seq_length': max_seq_length,
            'min_occ': min_occ,
        }, is_best, folder=args.out_dir)

        np.save(os.path.join(args.out_dir, 'train_elbo.npy'), track_train_elbo)
        np.save(os.path.join(args.out_dir, 'val_log_p_x.npy'), track_val_log_p_x)
        np.save(os.path.join(args.out_dir, 'val_log_p_y.npy'), track_val_log_p_y)
        np.save(os.path.join(args.out_dir, 'val_log_p_x_y.npy'), track_val_log_p_x_y)
        np.save(os.path.join(args.out_dir, 'test_log_p_x.npy'), track_test_log_p_x)
        np.save(os.path.join(args.out_dir, 'test_log_p_y.npy'), track_test_log_p_y)
        np.save(os.path.join(args.out_dir, 'test_log_p_x_y.npy'), track_test_log_p_x_y)
        np.save(os.path.join(args.out_dir, 'train_acc.npy'), track_train_acc)
        np.save(os.path.join(args.out_dir, 'val_acc.npy'), track_val_acc)
        np.save(os.path.join(args.out_dir, 'test_acc.npy'), track_test_acc)
