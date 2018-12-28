r"""Generic utilities."""

# everything loads from this file so do not make relative imports
import os
import shutil
import random
import numpy as np

from collections import Counter
from collections import defaultdict, OrderedDict

import torch

PAD_TOKEN = '<pad>'
UNK_TOKEN = '<unk>'
SOS_TOKEN = '<sos>'
EOS_TOKEN = '<eos>'

OPEN_BRACKET = '('
END_BRACKET = ')'

ZIPF_CLASS = {'head': 0, 'body': 1, 'tail': 2}

EXPERIMENTS_ROOT = os.path.realpath(os.path.dirname(__file__))
RUBRICS_ROOT = os.path.realpath(os.path.join(EXPERIMENTS_ROOT, '../../rubrics'))
DATASETS_ROOT = os.path.realpath(os.path.join(EXPERIMENTS_ROOT, '../../datasets'))
PCFG_UTILS_ROOT = os.path.realpath(os.path.join(EXPERIMENTS_ROOT, 'pcfg_utils'))


# --- utilities for reading and cleaning abstract syntax trees ---


def removeColors(ast):
    r"""Many Code.org problems come with the ability to color
    particular elements in the environment. These are not crucial
    to the logic of our curriculum so we should remove them as to
    reduce the size the of solution space.

    NOTE: this function makes edit in-place.

    @param ast: abstract syntax tree
    """
    for child in ast.children:
        removeColors(child)

        newChildren = []
	for child in ast.children:
		if child.rootName != 'SetColor':
			newChildren.append(child)

	ast.children = newChildren


def flatten_ast(ast):
    r"""Neural nets cannot take trees as input. For simplicity, we
    can flatten the tree into a string. 

    @param ast: abstract syntax tree
    """
    flat = [OPEN_BRACKET, ast.rootName]
    for child in ast.children:
        if child:
            flat += flatten_ast(child)
    flat.append(END_BRACKET)
    return flat


def unflatten_ast(data):
    r"""Convert our flattened string representation back into a
    tree (this may not work 100% of the time). We will use this to
    convert generated synthetic examples into AST.

    @param data: string
    @return ast: abstract syntax string
    """
    try:
        assert _check_if_valid_program(data)
    except AssertionError, e:
        raise BadProgramString(e.args)
    token, start_pos, end_pos = _find_first_token(data)

    # make tree grounded at token
    root = Tree(token)

    if start_pos is None and end_pos is None:
        # if theres no more string then return
        return root

    # forget about the token and its parentheses
    data = data[start_pos:end_pos + 1].strip()

    # split this into a list of child strings
    children = _split_into_children_strings(data)

    for child in children:
        # recursion to the rescue
        subtree = unflatten_ast(child)
        root.addChild(subtree)

    return root


class BadProgramString(Exception):
    pass


def _check_if_valid_program(data):
    # make sure number of ( and ) match
    return data.count('(') == data.count(')')


def _find_first_token(data):
    assert data[0] == '('
    end_pos = _find_matching_bracket(data, 0)
    substr = data[1:end_pos]  # ignore the wrap
    if '(' in substr:
        # find next open bracket
        pos = substr.index('(')
        # assumes tokens are single char and no white space between them
        token = substr[1:pos].strip()
        return token, pos + 1, end_pos - 1
    else:
        token = substr.strip()
        return token, None, None


def _find_matching_bracket(data, pos):
    """Find matching by looping through sentence O(n).

    @param data: list of strings
                 list of tokens
    @param pos: integer
                index of the ( in the data for which we are looking
                for the matching ).
    """
    assert data[pos] == '('
    depth = 0
    substr = data[pos:]
    for i in xrange(len(substr)):
        if substr[i] == '(':
            depth += 1
        elif substr[i] == ')':
            depth -= 1
            if depth == 0:
                break
    assert depth == 0
    return pos + i  # index of matching )


def _split_into_children_strings(data):
    i = 0
    children = []
    while i < len(data):
        if data[i] == '(':
            j = _find_matching_bracket(data, i)
            children.append(data[i:j+1])
            i = j + 1
        else:
            assert data[i] != ')'  # we should never hit this...
            i += 1
    return children


def flatten_trace(trace):
    trace = ' '.join(trace)
    trace = trace.replace('(', ' ( ')
    trace = trace.replace(')', ' ) ')
    trace = re.sub('\s+', ' ', trace).strip()
    return trace


# --- utilities for processing labels ---


def convert_V2_to_V3(v2_label_dict):
    r"""Unfortunately, when we labeled data, different people used
    different terminology. So we need to standardize them manually.

    Not pretty code but it works.

    @param v2_label_dict: dict
                          map from key to boolean/int
    """
    v3_labels = []
    if v2_label_dict.get('turn-perfect', False):
        v3_labels.append('Correct turn direction')
        v3_labels.append('Correct invert angle')
        v3_labels.append('Correct equalateral amount')
    if v2_label_dict['turning'] == 0:
        v3_labels.append('Standard Strategy')
    if v2_label_dict['turning'] == 1:
        v3_labels.append('Clockwise Strategy')
    if v2_label_dict['turning'] == 2:
        v3_labels.append('Random Strategy')
    if v2_label_dict.get('move-confusion', False):
        v3_labels.append('Random move amount')
    if v2_label_dict.get('angle-invert', False):
        v3_labels.append('Does not invert angle')
    if v2_label_dict.get('turn-none', False):
        v3_labels.append('No Turn')
    if v2_label_dict.get('turn-lr-confusion', False):
        v3_labels.append('Left/Right confusion')
    if v2_label_dict.get('loop-order', False):
        v3_labels.append('Body order is incorrect (turn/move)')
    if v2_label_dict.get('loop-three-times', False):
        v3_labels.append("Doesn't loop three times")
    if v2_label_dict.get('looping-copy-body', False):
        v3_labels.append('Repetition of bodies')
    if v2_label_dict.get('loop-nesting', False):
        v3_labels.append('Does not get nesting')
    if v2_label_dict.get('move-bf-confusion', False):
        v3_labels.append('Backwards/Forwards confusion')
    if v2_label_dict.get('loop-pre-post', False):
        v3_labels.append('Does not get pre/post condition')
    if v2_label_dict.get('looping-no-repeat', False):
        v3_labels.append("Doesn't use a repeat")
    if v2_label_dict.get('move-default', False):
        v3_labels.append('Default move')
    if v2_label_dict.get('angle-default', False):
        v3_labels.append('Default turn')
    if v2_label_dict.get('loop-missing', False):
        v3_labels.append('Body is missing statements')
    if v2_label_dict.get('moving-none', False):
        v3_labels.append('No Move')
    if v2_label_dict.get('looping-perfect', False):
        v3_labels.append('Correct body order')
        v3_labels.append('Looped')
        v3_labels.append('Correct repeat num')
    if v2_label_dict.get('angle-equilateral', False):
        v3_labels.append('Does not know equilateral is 60')
    if v2_label_dict.get('moving-perfect', False):
        v3_labels.append('Correct move direction')
        v3_labels.append('Correct move amount')
    if v2_label_dict.get('loop-extra', False):
        v3_labels.append('Body has extra statements')

    # sanity checks
    if 'Does not invert angle' in v3_labels:
        assert 'Correct invert angle' not in v3_labels
    if 'Correct invert angle' in v3_labels:
        assert 'Does not invert angle' not in v3_labels
    if 'Body order is incorrect (turn/move)' in v3_labels:
        assert 'Correct body order' not in v3_labels
    if 'Correct body order' in v3_labels:
        assert 'Body order is incorrect (turn/move)' not in v3_labels
    if 'Does not know equilateral is 60' in v3_labels:
        assert 'Correct equalateral amount' not in v3_labels
    if 'Correct equalateral amount' in v3_labels:
        assert 'Does not know equilateral is 60' not in v3_labels
    if 'Standard Strategy' in v3_labels:
        assert 'Clockwise Strategy' not in v3_labels
        assert 'Random Strategy' not in v3_labels
    if 'Clockwise Strategy' in v3_labels:
        assert 'Standard Strategy' not in v3_labels
        assert 'Random Strategy' not in v3_labels
    if 'Random Strategy' in v3_labels:
        assert 'Standard Strategy' not in v3_labels
        assert 'Clockwise Strategy' not in v3_labels

    return v3_labels


def labels_to_numpy(labels, label_dim, label_to_ix_dict):
    tensor_np = np.zeros(label_dim)
    for label in labels:
        if label in label_to_ix_dict:
            tensor_np[label_to_ix_dict[label]] = 1
    tensor_np = tensor_np.astype(np.int)
    
    return tensor_np


def labels_to_tensor(labels, label_dim, label_to_ix_dict):
    tensor_np = labels_to_numpy(label_dim, label_to_ix_dict)
    tensor = torch.from_numpy(tensor_np).long()
    
    return tensor


def tensor_to_labels(tensor, label_dim, ix_to_label_dict):
    assert tensor.size(0) == label_dim
    labels = []
    for ix in xrange(tensor.size(0)):
        if tensor[ix] >= 0.5:
            label = ix_to_label_dict[ix]
            labels.append(label)
    
    return ','.join(labels)


# --- miscellanous utilities ---


def train_test_split(array_list, train_frac=0.8, val_frac=0.1, test_frac=0.1):
    r"""Split data into three subsets (train, validation, and test).

    @param: array_list 
            list of np.arrays/torch.Tensors
            we will split each entry accordingly
    @param train_frac: float [default: 0.8]
                       must be within (0.0, 1.0)
    @param val_frac: float [default: 0.8]
                     must be within [0.0, 1.0)     
    @param train_frac: float [default: 0.8]
                       must be within (0.0, 1.0)
    """
    assert (train_frac + val_frac + test_frac) == 1.0

    train_list, test_list = [], []
    if val_frac > 0.0:
        val_list = []

    for array in array_list:
        size = len(array)

        train_array = array[:int(train_frac * size)]
        if val_frac > 0.0:
            val_array = array[
                int(train_frac * size):
                int((train_frac + val_frac) * size)
            ]
        test_array = array[int((train_frac + val_frac) * size):]

        train_list.append(train_array)
        test_list.append(test_array)

        if val_frac > 0.0:
            val_list.append(val_array)

    if val_frac > 0.0:
        return train_list, val_list, test_list
    else:
        return train_list, test_list


def log_mean_exp(x, dim=1):
    r"""log(1/k * sum(exp(x))): this normalizes x.

    @param x: PyTorch.Tensor
              samples from gaussian
    @param dim: integer (default: 1)
                which dimension to take the mean over
    @return: PyTorch.Tensor
             mean of x
    """
    m = torch.max(x, dim=dim, keepdim=True)[0]
    return m + torch.log(torch.mean(torch.exp(x - m),
                         dim=dim, keepdim=True))


def merge_dicts(*dict_args):
    r"""Given any number of dicts, shallow copy and merge into a new dict,
    precedence goes to key value pairs in latter dicts.
    """
    result = {}
    for dictionary in dict_args:
        result.update(dictionary)
    return result


class OrderedCounter(Counter, OrderedDict):
    r"""Counter that remembers the order elements are first encountered"""

    def __repr__(self):
        return '%s(%r)' % (self.__class__.__name__, OrderedDict(self))

    def __reduce__(self):
        return self.__class__, (OrderedDict(self),)


def idx2word(idx, i2w, pad_idx):
    r"""Used to convert samples back to text."""
    sent_str = [str()]*len(idx)
    for i, sent in enumerate(idx):
        for word_id in sent:
            if word_id == pad_idx:
                break
            sent_str[i] += i2w[str(word_id)] + " "
        sent_str[i] = sent_str[i].strip()

    return sent_str


def interpolate(start, end, steps):
    interpolation = np.zeros((start.shape[0], steps + 2))
    for dim, (s,e) in enumerate(zip(start,end)):
        interpolation[dim] = np.linspace(s,e,steps+2)

    return interpolation.T


def merge_args_with_dict(args, dic):
    for k, v in dic.iteritems():
        setattr(args, k, v)


def normalize_dictionary(dictionary):
    Z = float(sum(dictionary.values()))
    return {k: v / Z for k, v in dictionary.iteritems()}


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(state, is_best, folder='./', filename='checkpoint.pth.tar'):
    # saves a copy of the model (+ properties) to filesystem
    if not os.path.isdir(folder):
        os.makedirs(folder)
    torch.save(state, os.path.join(folder, filename))
    if is_best:
        shutil.copyfile(os.path.join(folder, filename),
                        os.path.join(folder, 'model_best.pth.tar'))
