r"""Train a neural network to predict feedback for a program string."""

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
from .utils import AverageMeter, save_checkpoint, merge_args_with_dict
from .datasets import load_dataset
from .config import default_hyperparams
from .rubric_utils.load_params import get_label_params, get_max_seq_len


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=str, help='annotated|synthetic')
    parser.add_argument('problem_id', type=int, help='1|2|3|4|5|6|7|8')
    parser.add_argument('out_dir', type=str, help='where to save outputs')
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

    train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    model = ProgramRNN( args.z_dim, label_dim, train_dataset.vocab_size, embedding_dim=args.embedding_dim, 
                        hidden_dim=args.hidden_dim, num_layers=args.num_layers)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)


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
            label_out = model(seq, length)
            loss = F.binary_cross_entropy(label_out, label)

            loss.backward()
            loss_meter.update(loss.item(), batch_size)

            optimizer.step()
            acc = np.mean(torch.round(label_out).detach().numpy() == label.detach().numpy())
            acc_meter.update(acc, batch_size)

            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAccuracy: {:.4f}'.format(
                    epoch, batch_idx * batch_size, len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss_meter.avg,
                    acc_meter.avg))

        print('====> Epoch: {}\tLoss: {:.4f}\tAccuracy: {:.4f}'.format(
            epoch, loss_meter.avg, acc_meter.avg))
        
        return loss_meter.avg, acc_meter.avg


    def test(epoch, loader, name='Test'):
        model.eval()
        loss_meter = AverageMeter()
        acc_meter = AverageMeter()

        with torch.no_grad():
            with tqdm(total=len(loader)) as pbar:
                for (seq, length, label, _) in loader:
                    assert label is not None
                    batch_size = len(seq)
                    seq = seq.to(device)
                    length = length.to(device)
                    label = label.to(device)

                    label_out = model(seq, length)
                    loss = F.binary_cross_entropy(label_out, label)
                    loss_meter.update(loss.item(), batch_size)

                    acc = np.mean(torch.round(label_out.cpu()).numpy() == label.cpu().numpy())
                    acc_meter.update(acc, batch_size)
                    pbar.update()

        print('====> {} Epoch: {}\tLoss: {:.4f}\tAccuracy: {:.4f}'.format(
            name, epoch, loss_meter.avg, acc_meter.avg))
        
        return loss_meter.avg, acc_meter.avg


    best_loss = sys.maxint
    track_train_loss = np.zeros(args.epochs)
    track_val_loss = np.zeros(args.epochs)
    track_test_loss = np.zeros(args.epochs)
    track_train_acc = np.zeros(args.epochs)
    track_val_acc = np.zeros(args.epochs)
    track_test_acc = np.zeros(args.epochs)

    for epoch in xrange(1, args.epochs + 1):
        train_loss, train_acc = train(epoch)
        val_loss, val_acc = test(epoch, val_loader, name='Val')
        test_loss, test_acc = test(epoch, test_loader, name='Test')
        
        track_train_loss[epoch - 1] = train_loss
        track_val_loss[epoch - 1] = val_loss
        track_test_loss[epoch - 1] = test_loss
        track_train_acc[epoch - 1] = train_acc
        track_val_acc[epoch - 1] = val_acc
        track_test_acc[epoch - 1] = test_acc

        is_best = val_loss < best_loss
        best_loss = min(val_loss, best_loss)

        save_checkpoint({
            'state_dict': model.state_dict(),
            'cmd_line_args': args,
            'vocab': train_dataset.vocab,
        }, is_best, folder=args.out_dir)
        
        np.save(os.path.join(args.out_dir, 'train_loss.npy'), track_train_loss)
        np.save(os.path.join(args.out_dir, 'val_loss.npy'), track_val_loss)
        np.save(os.path.join(args.out_dir, 'test_loss.npy'), track_test_loss)
        np.save(os.path.join(args.out_dir, 'train_acc.npy'), track_train_acc)
        np.save(os.path.join(args.out_dir, 'val_acc.npy'), track_val_acc)
        np.save(os.path.join(args.out_dir, 'test_acc.npy'), track_test_acc)
