from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import json
import cPickle
import numpy as np
from tqdm import tqdm

from .utils import (
    train_test_split, 
    flatten_ast, 
    removeColors,
    convert_V2_to_V3,
    labels_to_numpy,
)
from .rubric_utils.load_params import get_label_params


def preprocess_source_pickle(source_pickle):
    r"""The data is provided in abstract tree syntax (AST) form.
    For each tree, flatten via depth-first traversal and dump
    to a single output file.

    If a pickle of counts is provided then we will duplicate any
    program by the correct number of repetitions.

    Run this function on the data prior to loading it for training
    (one time operation -- saves outputs to disk).

    NOTE: this ONLY contains unique solutions.

    @param source_pickle: cPickle file
                          contains unique AST by index.
    """
    with open(source_pickle, 'rb') as fp:
        data = cPickle.load(fp)
        n_rows = len(data)

    programs = []

    for ix in tqdm(range(n_rows)):
        ast = data[ix]
        removeColors(ast)  # remove colors
        ast = flatten_ast(ast)
        program = ' '.join(ast)
        programs.append(program)

    # important, make sure there are no duplicates here
    # otherwise we could get programs bleeding into the training set
    # from the test set.
    programs = list(set(programs))

    return programs


def preprocess_synthetic_samples(problem_id, sample_pickle):
    r"""Similar to <preprocess_source_pickle> but cleans up synthetic
    samples from a probabilistic model.

    @param problem_id: integer
                       1|2|3|4|5|6|7|8
    @param sample_pickle: cPickle file
                          contains raw string program and list of labels.
    """
    label_dim, _, label_to_ix, _, _ = get_label_params(problem_id)

    with open(sample_pickle, 'rb') as fp:
        data = cPickle.load(fp)
        n_rows = len(data)

    programs = []
    program2label = {}

    pbar = tqdm(total=n_rows)
    for program, values in data.iteritems():
        # don't use newline separation since that will print out
        # as several lines and we lose structure.
        label = values.keys()[0].strip().split('\n')
        label = labels_to_numpy(label, label_dim, label_to_ix).tolist()

        programs.append(program)
        program2label[program] = label

        pbar.update()
    pbar.close()

    # important, make sure there are no duplicates here
    programs = list(set(programs))
    labels = [program2label[prog] for prog in programs]

    return programs, labels


def preprocess_human_annotations(problem_id, source_pickle, annotation_pickle):
    r"""Similar to <preprocess_source_pickle> but for programs where
    humans annotated each program.

    @param problem_id: integer
                       1|2|3|4|5|6|7|8
    @param source_pickle: cPickle file
                          contains unique AST by index.
    @param annotation_pickle: cPickle file
                              contains annotations by index.
    """
    label_dim, _, label_to_ix, _, _ = get_label_params(problem_id)

    with open(annotation_pickle) as fp:
        annotations = cPickle.load(fp)
        n_annotations = len(annotations)

    with open(source_pickle) as fp:
        ast_programs = cPickle.load(fp)

    programs = []
    program2label = {}

    for i in tqdm(range(n_annotations)):
        annotation = annotations[i]
        program_id = int(annotation['answerId'])
        targets = json.loads(annotation['gradeData'])
        targets = convert_V2_to_V3(targets)
        targets = labels_to_numpy(targets, label_dim, label_to_ix).tolist()

        ast = ast_programs[program_id]
        removeColors(ast)
        ast = flatten_ast(ast)
        program = ' '.join(ast)

        programs.append(program)
        program2label[program] = targets

    # important, make sure there are no duplicates here
    programs = list(set(programs))
    labels = [program2label[prog] for prog in programs]

    return programs, labels


if __name__ == "__main__":
    # run this file to get things set up for training!
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=str, help='unlabeled|annotated|synthetic')
    parser.add_argument('problem_id', type=int, help='1|2|3|4|5|6|7|8')
    parser.add_argument('out_dir', type=str, help='where to dump outputs')
    parser.add_argument('--source-pickle', type=str, help='path to the source pickle file')
    parser.add_argument('--annotation-pickle', type=str, help='path to the annotation pickle file')
    parser.add_argument('--sample-pickle', type=str, help='path to the sample pickle file')
    args = parser.parse_args()

    if not os.path.isdir(args.out_dir):
        os.makedirs(args.out_dir)

    if args.dataset == 'unlabeled':
        programs = preprocess_source_pickle(args.source_pickle)
        train_lst, val_lst, test_lst = train_test_split([programs], train_frac=0.8, val_frac=0.1, test_frac=0.1)
        train_programs, val_programs, test_programs = train_lst[0], val_lst[0], test_lst[0]
        with open(os.path.join(args.out_dir, 'all.pickle'), 'wb') as fp:
            cPickle.dump({'programs': programs}, fp)
        with open(os.path.join(args.out_dir, 'train.pickle'), 'wb') as fp:
            cPickle.dump({'programs': train_programs}, fp)
        with open(os.path.join(args.out_dir, 'val.pickle'), 'wb') as fp:
            cPickle.dump({'programs': val_programs}, fp)
        with open(os.path.join(args.out_dir, 'test.pickle'), 'wb') as fp:
            cPickle.dump({'programs': test_programs}, fp)
    elif args.dataset == 'annotated':
        programs, labels = preprocess_human_annotations(args.problem_id, args.source_pickle, args.annotation_pickle)
        train_lst, val_lst, test_lst = train_test_split([programs, labels], train_frac=0.8, val_frac=0.1, test_frac=0.1)
        train_programs, val_programs, test_programs = train_lst[0], val_lst[0], test_lst[0]
        train_labels, val_labels, test_labels = train_lst[1], val_lst[1], test_lst[1]
        with open(os.path.join(args.out_dir, 'all.pickle'), 'wb') as fp:
            cPickle.dump({'programs': programs, 'labels': labels}, fp)
        with open(os.path.join(args.out_dir, 'train.pickle'), 'wb') as fp:
            cPickle.dump({'programs': train_programs, 'labels': train_labels}, fp)
        with open(os.path.join(args.out_dir, 'val.pickle'), 'wb') as fp:
            cPickle.dump({'programs': val_programs, 'labels': val_labels}, fp)
        with open(os.path.join(args.out_dir, 'test.pickle'), 'wb') as fp:
            cPickle.dump({'programs': test_programs, 'labels': test_labels}, fp)
    elif args.dataset == 'synthetic':
        programs, labels = preprocess_synthetic_samples(args.problem_id, args.sample_pickle)
        train_lst, val_lst, test_lst = train_test_split([programs, labels], train_frac=0.8, val_frac=0.1, test_frac=0.1)
        train_programs, val_programs, test_programs = train_lst[0], val_lst[0], test_lst[0]
        train_labels, val_labels, test_labels = train_lst[1], val_lst[1], test_lst[1]
        with open(os.path.join(args.out_dir, 'all.pickle'), 'wb') as fp:
            cPickle.dump({'programs': programs, 'labels': labels}, fp)
        with open(os.path.join(args.out_dir, 'train.pickle'), 'wb') as fp:
            cPickle.dump({'programs': train_programs, 'labels': train_labels}, fp)
        with open(os.path.join(args.out_dir, 'val.pickle'), 'wb') as fp:
            cPickle.dump({'programs': val_programs, 'labels': val_labels}, fp)
        with open(os.path.join(args.out_dir, 'test.pickle'), 'wb') as fp:
            cPickle.dump({'programs': test_programs, 'labels': test_labels}, fp)
    else:
        raise Exception('dataset-type %s not supported.' % args.dataset)
