r"""For a number of programs in the dataset, predict its semantic
labels and also highlight sections of code via PCFG.
"""

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import random
import cPickle
import numpy as np

from .viterbi import ViterbiParser
from ..rubric_utils.load_params import (
    get_pcfg_params, 
    get_pcfg_path,
    get_codeorg_data_root,
)
from .chartparser import load_indexed_asts
from .chartparser import get_integer_domain
from .chartparser import build_pcfg
from ..pcfg_utils.models.simulate import loadCfg, reset_prodIndex


def extract_regions_from_tree(tree, domain):
    r"""Create a dictionary from semantic label to (start, end) character index
    on a (most likely) parsed probabilistic tree. 
    
    @param tree: PCFG
    @param domain: list of strings
                   space of labels
    @return np.array # labels x # characters
    """
    mapping = tree.pos()
    n_tokens = len(mapping)
    regions = np.zeros((n_tokens, len(domain)))
    for i, (token, node, tags) in enumerate(mapping):
        for tag in tags.split(','):
            if len(tag) > 0:
                j = domain.index(tag)
                regions[i, j] = 1
    regions = regions.T
    # sum 1s from start to end for each label
    for i in xrange(len(domain)):
        ix = np.where(regions[i] == 1)[0]
        if len(ix) > 0:
            regions[i, ix[0]:ix[-1]+1] = 1
    return regions


def pcfg_tag_domain(theta, cfg_path):
    r"""Return an ordering on all the different semantic tags in PCFG.

    @param theta: np.array
                  parameters of pcfg
    @param cfg_path: string
                     path to pCFG file.
    """
    pcfg = loadCfg(theta, cfg_path)
    reset_prodIndex()

    domain = []
    values = pcfg.values()
    for lst in values:
        for item in lst:
            domain.extend(item['tags'])

    domain = list(set(domain))
    return domain


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('problem_id', type=int, help='1|2|3|4|5|6|7|8')
    parser.add_argument('out_dir', type=str, help='where to dump files')
    parser.add_argument('--max-number', default=10, help='max number of programs to highlight')
    parser.add_argument('--student-pcfg', action='store_true', default=False,
                        help='student author for PCFG [default: teacher]')
    parser.add_argument('--random-theta', action='store_true', default=False,
                        help='use random parameters for PCFG [default: expert parameters]')
    args = parser.parse_args()

    data_root = get_codeorg_data_root(1, 'raw')
    author = 'student' if args.student_pcfg else 'teacher'
    theta = get_pcfg_params(args.problem_id, author=author, random=args.random_theta)
    cfg_path = get_pcfg_path(args.problem_id, author=author)

    domain = pcfg_tag_domain(theta, cfg_path)

    # load dataset statistics to construct grammar
    data, data_ids = load_indexed_asts(data_root, args.problem_id, True)

    combined = list(zip(data, data_ids))
    random.shuffle(combined)
    data, data_ids = zip(*combined)

    integer_domain = get_integer_domain([d for d in data if d is not None])
    
    # build the PCFG and construct viterbi parser
    pcfg = build_pcfg(cfg_path, theta, integer_domain, False)
    parser = ViterbiParser(pcfg)

    # get maximum length of a token
    max_length = 0
    for tokens in data:
        if tokens is not None:
            max_length = max(max_length, len(tokens))

    outputs = {}  # id to matrix
    outputs_ = {} # string to matrix
    n_success = 0

    for j, (tokens, index) in enumerate(zip(data, data_ids)):
        print('%d / %d successfully parsed. %d to go.' % (n_success, j + 1, len(data) - j - 1))

        if tokens is None:
            # this program never compiled 
            matrix = np.ones((len(domain), max_length)) * -1
            outputs[index] = matrix
            continue
        
        n_tokens = len(tokens)
        generator = parser.parse(tokens)
        try:
            tree = next(generator)
        except StopIteration:
            # pcfg doesn't support this program
            matrix = np.ones((len(domain), max_length)) * -1
            outputs[index] = matrix
            continue
        except ValueError:
            # pcfg failed to parse this program
            matrix = np.ones((len(domain), max_length)) * -1
            outputs[index] = matrix
            continue

        # pull out a map from label to tree
        matrix = np.ones((len(domain), max_length)) * -1
        sub_matrix = extract_regions_from_tree(tree, domain)
        matrix[:, :n_tokens] = sub_matrix

        outputs[index] = matrix
        outputs_[' '.join(tokens)] = matrix
        n_success += 1

        if args.max_number is not None:
            if n_success >= int(args.max_number):
                break

    with open(os.path.join(args.out_dir, 'highlightIdMap-p%d.pickle' % args.problem_id), 'wb') as fp:
        cPickle.dump(outputs, fp)

    with open(os.path.join(args.out_dir, 'highlightProgramMap-p%d.pickle' % args.problem_id), 'wb') as fp:
        cPickle.dump(outputs_, fp)
