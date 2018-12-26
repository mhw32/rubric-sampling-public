r"""Generate labeled programs from PCFG."""

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import sys
import cPickle

from .pcfg_utils.simulate import generateTrajectories
from .rubric_utils.load_params import get_pcfg_params, get_pcfg_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('problem_id', type=int, help='1|2|3|4|5|6|7|8')
    parser.add_argument('out_dir', type=str, help='where to dump files')
    parser.add_argument('--student-pcfg', action='store_true', default=False,
                        help='student author for PCFG [default: teacher]')
    parser.add_argument('--random-theta', action='store_true', default=False,
                        help='use random parameters for PCFG [default: expert parameters]')
    # generate a lot b/c we only keep unique
    parser.add_argument('--num-samples', type=int, default=1000000,
                        help='number of data points to sample [default: 1e6')
    args = parser.parse_args()

    theta = get_pcfg_params(args.problem_id, author= 'student' if args.student else 'teacher', 
                            random=args.random_theta)
    cfg_path = get_pcfg_path(args.problem_id, author= 'student' if args.student else 'teacher')
    program2count, program2label = generateTrajectories(theta, args.num_samples, cfg_path, verbose=True)

    label_map_path = os.path.join(args.out_dir, 'labels-%d.pickle' % args.problem_id)
    count_map_path = os.path.join(args.out_dir, 'counts-%d.pickle' % args.problem_id)

    if not os.path.isdir(args.out_dir):
        os.makedirs(args.out_dir)

     with open(label_map_path, 'wb') as fp:
        cPickle.dump(program2label, fp)

    with open(count_map_path, 'wb') as fp:
        cPickle.dump(program2count, fp)
