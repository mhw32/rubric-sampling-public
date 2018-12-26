from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import re
import sys
import copy
import cPickle
import itertools
import numpy as np
from tqdm import tqdm

from .grammar import PCFG
from nltk.parse.viterbi import ViterbiParser

from ..pcfg_utils.simulate import loadCfg
from ..pcfg_utils.treeToCode.astToBlocky import toBlocky
from ..utils import flatten_ast


def pcfg_data_likelihood(cfg_path, weights, data, counts, epsilon=1e-10):
    """Compute the log-likelihood of the real programs dataset 
    using PCFG with user-specified weights.

    @param cfg_path: string
                     path to PCFG dump
    @param weights: np.array
                    parameters of CFG.
    @param data: list of code segments
                 each code segment is a list of strings (space-sep)
    @param counts: each data point is not weighted equally
                   we weight by occurrence
    @param epsilon: default to use for empty trees [default: 1e-10]
    @return log_lik: float
                     log likelihood of dataset.
    """
    # space of possible integers (some of the language
    # requires a countably infinite number of possiblilities.
    # we only care about encoding the real program-space so
    # we only explicitly model the integers in the real set.
    integer_domain = get_integer_domain(data)
    pcfg = build_pcfg(cfg_path, weights, integer_domain, True)
    parser = ViterbiParser(pcfg)
    log_like = 0
    missing = 0
    for i, (code, cnt) in enumerate(zip(data, counts)):
        generator = parser.parse(code)
        if generator is not None:
            tree = next(generator)
            ll = tree.logprob()
        else:  # this program is not covered by the pCFG
            ll = np.log(epsilon)
            log_like += -ll * cnt
            missing += 1
    return log_like


def build_pcfg(cfg_path, weights, integer_domain, make_cnf=False):
    """Use Chris' ZeroShotLearning code to construct the probabilistic
    CFG from a set of parameters. I imagine we will have to call this 
    program once for each parameter setting.

    @param cfg_path: string
                     path to PCFG dump
    @param weights: np.array
                    parameters of the CFG.
    @param integer_domain: np.array
                           space of possible integers
    @param make_cnf: boolean [default: False]
                     explicitly make grammar CNF
    @return pcfg: nltk PCFG object    
                  CFG for Code.org problem 1
    """
    pcfg = loadCfg(weights, cfg_path)
    BAD_KEYS = [
        "// Context Free Grammar for:\n// Codestudio Lesson 10, problem 1\n\n// Syntax:\n// * conditional probability weight\n// # tags associated with production\n// {{NonTerminal}}\n//",
        "prod", 
        "",
    ]
    for key in BAD_KEYS:
        if key in pcfg:
            del pcfg[key]
    # unroll each randInt production aka randInt(0, 3) ==> 0, 1, ...
    pcfg = unroll_randInt_calls(pcfg, integer_domain)
    pcfg = pcfg_to_list(pcfg)
    if make_cnf:
        pcfg = make_chomsky_normal_form(pcfg)
    pcfg = pcfg_to_string(pcfg)
    pcfg = PCFG.fromstring(pcfg, has_tags=True)
    return pcfg


def unroll_randInt_calls(pcfg, integer_domain):
    """Replace all randInt calls with relevant integers from the 
    real dataset. This is a hacky way of DISCRETIZING a CONTINUOUS
    function call (that cannot be represented by a CFG).
    @param pcfg: dictionary object
    @integer_domain: np.array
                     space of possible integers
    @return: pcfg with no randInt calls
    """
    new_pcfg = {}
    regex = re.compile(r'randInt\(((\w+|,)+)?\)')
    for rule, maps in pcfg.iteritems():
        new_maps = []
        for m in maps:
            if 'randInt' in m['text']:
                proba = m['p']
                for match in findall(regex, m['text']):
                    ch_s = match.index('(')
                    ch_e = match.index(')')
                    int_args = match[ch_s+1:ch_e]
                    if len(int_args) > 0 or ',' not in int_args:
                        n_opt = len(integer_domain)
                        proba_i = float(proba) / n_opt
                        for opt in integer_domain:
                            m_i = copy.deepcopy(m)
                            m_i['p'] = proba_i
                            m_i['text'] = str(opt)
                            new_maps.append(m_i)
                    else:
                        int_s, int_e = int_args.split(',')
                        int_s = int(int_s)
                        int_e = int(int_e)
                        sub_domain = integer_domain[
                            np.logical_and(integer_domain >= int_s,
                                           integer_domain <= int_e)]
                        for opt in sub_domain:
                            m_i = copy.deepcopy(m)
                            m_i['p'] = proba_i
                            m_i['text'] = str(opt)
                            new_maps.append(m_i)
            else:  # just stick with the old version
                new_maps.append(m)
        new_pcfg[rule] = new_maps
    return new_pcfg


def load_real_asts(data_root, problem_id, ignore=False):
    """Load all AST-trees from real programs.
    
    @param data_root: string
                      path to raw data
    @param problem_id: integer
                       1|2|3|4|5|6|7|8
    @param ignore: boolean [default: False]
                   if True, ignore programs that do not compile
                   there are about 515 programs that do not compile
    """
    with open('%s/sources-%d.pickle' % (data_root, problem_id)) as fp:
        data = cPickle.load(fp)
    with open('%s/countMap-%d.pickle' % (data_root, problem_id)) as fp:
        count = cPickle.load(fp)
    domain = data.keys()
    code_segments = []
    code_counts = []
    for k in domain:
        ast = data[k]
        try:
            text = toBlocky(ast)
        except:
            if ignore:
                continue
            else:
                raise
        text = clean_text(text)
        text = hack_text(text)
        text = separate_delimiters(text)
        text = text.replace('MoveForward', 'Move')
        text = text.replace('MoveBackwards', 'MoveBackward')
        text = text.split(' ')
        text = [t for t in text if len(t) > 0]
        cnt = count[k]
        code_segments.append(text)
        code_counts.append(cnt)
    return code_segments, code_counts


def load_indexed_asts(data_root, problem_id, ignore=False):
    """Load all AST-trees from real programs.

    @param ignore: boolean [default: False]
                   if True, ignore programs that do not compile
                   there are about 515 programs that do not compile
    @return code_segments: list of strings
                           cleaned up ASTs
    @return code_indexes: list of integers
                          the position of this code segment
    """
    with open('%s/sources-%d.pickle' % (data_root, problem_id)) as fp:
        data = cPickle.load(fp)

    domain = data.keys()
    code_segments = []
    code_indexes = []

    for k in domain:
        ast = data[k]
        try:
            text = toBlocky(ast)
        except:
            if ignore:
                code_segments.append(None)
                code_indexes.append(k)
                continue
            else:
                raise
        text = clean_text(text)
        text = hack_text(text)
        text = separate_delimiters(text)
        text = text.replace('MoveForward', 'Move')
        text = text.replace('MoveBackwards', 'MoveBackward')
        text = text.split(' ')
        text = [t for t in text if len(t) > 0]
        code_segments.append(text)
        code_indexes.append(k)

    return code_segments, code_indexes


def get_integer_domain(codes):
    """Get a set of all unique integers from the real programs. We 
    will use this as a placeholder for randInt(). 
    @param codes: loaded dataset of programs (unique)
    @return integers: np.array
                      domain of string programs
    """
    tokens = list(set(list(itertools.chain.from_iterable(codes))))
    integers = []
    for token in tokens:
        try:
            int(token)
            integers.append(int(token))
        except ValueError:
            continue
    integers = np.array(integers)
    return integers


def clean_text(s):
    """Replace //n and extra spaces.
    
    @param s: string
              piece of text to be cleaned
    """
    s = s.replace('\\n', '')
    s = s.replace('\n', '')
    s = re.sub('\s+', ' ', s)
    s = s.strip()
    return s


def findall(regex, seq):
    # like regex.findall but works with overlapping
    # sequences of characters
    resultlist = []
    pos = 0

    while True:
       result = regex.search(seq, pos)
       if result is None:
          break
       resultlist.append(seq[result.start():result.end()])
       pos = result.start()+1
    return resultlist


def hack_text(s):
    """Do some hacks to prevent some harder parsing problems.
    NOTE: these are not very SAFE hacks.
    @param s: string
              piece of text to add hacks
    """
    regex1 = re.compile(r'{{\w+}}{{\w+}}')
    regex2 = re.compile(r'{{{\w+}}')
    regex3 = re.compile(r'{{\w+}}}')

    for match in findall(regex1, s):
        s = s.replace(match, match.replace('}}{{', '}} {{'))
    for match in findall(regex2, s):
        s = s.replace(match, match[0] + ' ' + match[1:])
    for match in findall(regex3, s):
        s = s.replace(match, match[:-1] + ' ' + match[-1])
    return s


def separate_delimiters(s):
    """\\w(\\w ==> \\w ( \\w. Sample for {, ), }.
    @param s: string
              piece of text to add hacks
    """
    regex1 = re.compile(r'\(((?:\w|\s|\*|\,)+)?\)')
    regex2 = re.compile(r'\{((?:\w|\s|\*|\,)+)?\}')
    for match in findall(regex1, s):
        # add border spacing to parenthesis + add border spacing for comma if arguments present
        s = s.replace(match, match.replace('(', ' ( ').replace(')', ' ) ').replace(',', ' , '))
    for match in findall(regex2, s):
        s = s.replace(match, match.replace('{', ' { ').replace('}', ' } ').replace(',', ' , '))
    return s.strip()


def replace_special_token(s):
    """Convert {{a}} ==> **a**."""
    regex = re.compile(r'{{\w+}}')
    for match in findall(regex, s):
        s = s.replace(match, match.replace('{{', '**').replace('}}', '**'))
    return s


def pcfg_to_list(pcfg):
    """Convert a dictionary pCFG form to a list form of relations.
    @param pcfg: dictionary 
                 context-free grammar from Chris' code.
    @return: list of tuples
             each tuple is a pattern to a set of probabilistic rules
    """
    new_rules = []
    
    # important to have the root be the first (Program)
    key_order = list(set(pcfg.keys()) - set(['Program']))
    key_order = ['Program'] + key_order

    for key in key_order:
        values = pcfg[key]
        new_values = []
        for value in values:
            text = value['text']
            new_parts = []  # store components here
            # process the text
            text = clean_text(text)
            text = hack_text(text)
            text = replace_special_token(text)
            text = separate_delimiters(text)

            # edge case: only include the empty string
            if len(text) == 0:
                new_value = ([text], value['p'], value['tags'])
                new_values.append(new_value)
                continue

            # split into components by space
            parts = text.split()
            
            for part in parts:
                # if variable, remove {{ }}
                if '**' in part:
                    part = part.replace('**', '')
                else:  # else, add strings to denote output
                    part = "'" + part + "'"
                new_parts.append(part)

            # remove empty indices
            new_parts = [p for p in new_parts if len(p) > 0]
            new_value = (new_parts, value['p'], value['tags'])
            new_values.append(new_value)
        new_rules.append((key, new_values))
    
    return new_rules


def pcfg_to_string(pcfg):
    """To be called after <pcfg_to_list>. Convert convert the 
    structure into a string for NLTK.
    @param pcfg: list of tuples
    @return: string
    """
    string = ''
    for pattern, rules in pcfg:
        parts = []
        for tokens, proba, tags in rules:
            parts.append('{output} [{proba}] {{{tags}}}'.format(
                output=' '.join(tokens), 
                proba='{0:f}'.format(proba),
                tags=','.join(tags)))
        string += '\t' + '{token} -> '.format(token=pattern) + ' | '.join(parts) + '\n'
    return '\n' + string


def make_chomsky_normal_form(pcfg):
    pcfg = _make_chomsky_normal_form_1(pcfg)
    count = 0
    while True:
        pcfg, max_size, count = _make_chomsky_normal_form_2(pcfg, count)
        if max_size == 2:
            break
    return pcfg


def _make_chomsky_normal_form_1(pcfg):
    """Convert to Chomsky Normal Form (CNF): Step 1.
    
    Eliminate terminals if they exist with other non-terminals.
       X --> xY ==> X --> ZY and Z --> x
    """
    # is this a terminal token?
    is_non_terminal = lambda x: not ("'" in x)

    # store any new terminal rules
    terminal_pattern_to_token = {}
    terminal_token_to_pattern = {}
    terminal_count = 0

    # first do one pass and fix stage 1.
    new_patterns = [] 
    for pattern, rules in pcfg:
        new_rules = []
        for tokens, proba in rules:
            is_non_term = [is_non_terminal(token) for token in tokens]
            if len(set(is_non_term)) > 1:
                new_tokens = []
                # break apart into two rules
                for i, token in enumerate(tokens):
                    if is_non_term[i] == False:  # is_terminal node                        
                        if token in terminal_token_to_pattern:
                            new_token = terminal_token_to_pattern[token]
                        else:
                            new_token = 'Constant%d' % terminal_count
                            terminal_pattern_to_token[new_token] = token
                            terminal_token_to_pattern[token] = new_token
                            terminal_count += 1
                        new_tokens.append(new_token)
                    else:
                        new_tokens.append(token)
                new_rules.append((new_tokens, proba))
            else:  # do nothing
                new_rules.append((tokens, proba))
        new_patterns.append((pattern, new_rules))

    # add any terminal rules as new patterns (at the end)
    for pattern, token in terminal_pattern_to_token.iteritems():
        new_patterns.append((pattern, [([token], 1.0)]))

    return new_patterns


def _make_chomsky_normal_form_2(pcfg, count=0):
    """Convert to Chomsky Normal Form (CNF): Step 2.
    Eliminate RHS with greater than 2 non-terminals.
       X --> XYZ ==> X --> PZ and P --> XY
    Also returns the max size to know if we have to 
    recursively call this function.
    """
    # is this a terminal token?
    is_non_terminal = lambda x: not ("'" in x)

    # store new rules here
    max_size = 0
    merged_count = count
    merged_token_to_pattern = {}
    merged_pattern_to_token = {}

    # first do one pass and fix stage 1.
    new_patterns = [] 
    for pattern, rules in pcfg:
        new_rules = []
        for tokens, proba in rules:
            is_non_term = [is_non_terminal(token) for token in tokens]
            max_size = max(max_size, len(is_non_term))
            if len(is_non_term) > 2:
                # split into two sets
                tokens1, tokens2 = tokens[:2], tokens[2:]
                tokens1key = '-'.join(tokens1)
                if tokens1key in merged_token_to_pattern:
                    new_token = merged_token_to_pattern[tokens1key]
                else:
                    new_token = 'Merge%d' % merged_count
                    merged_token_to_pattern[tokens1key] = new_token
                    merged_pattern_to_token[new_token] = tokens1
                    merged_count += 1
                new_rules.append(([new_token] + tokens2, proba))
            else:  # do nothing
                new_rules.append((tokens, proba))
        new_patterns.append((pattern, new_rules))

    for pattern, token in merged_pattern_to_token.iteritems():
        new_patterns.append((pattern, [(token, 1.0)]))

    return new_patterns, max_size, merged_count


def find_substr(s, ch):
    return [m.start() for m in re.finditer(ch, s)]


def complement_interval(intervals, domain_min, domain_max):
    """Compute the union of intervals: domain - union(intervals).
    We assume that intervals is sorted in increasing order.
    """
    if len(intervals) == 0:
        return [(domain_min, domain_max)]

    complement = []
    if intervals[0][0] > domain_min:
        complement.append((domain_min, intervals[0][0] - 1))

    if len(intervals) > 1:
        for i1, i2 in zip(intervals[0:-1], intervals[1:]):
            if i2[0] - i1[1] > 0:
                complement.append((i1[1] + 1, i2[0] - 1))

    if intervals[-1][1] < domain_max - 1:
        complement.append((intervals[-1][1] + 1, domain_max))

    return complement


def sort_two_lists(list1, list2):
    """Sort both lists based on list1"""
    list1, list2 = (list(t) for t in zip(*sorted(zip(list1, list2))))
    return list1, list2


def sanity_test():
    """Unit Test to make sure this stuff is working.
    This function should NOT break.
    """
    from ..rubric_utils.load_params import (
        get_pcfg_params, 
        get_pcfg_path,
        get_codeorg_data_root,
    )

    data_root = get_codeorg_data_root(1, 'unlabeled')
    theta = get_pcfg_params(1, author='teacher', random=False)
    cfg_path = get_pcfg_path(1, author='teacher')

    data, counts = load_real_asts(data_root, 1, True)
    integer_domain = get_integer_domain(data)
    # CKY parser for p-cfgs...
    pcfg = build_pcfg(cfg_path, theta, integer_domain, False)
    parser = ViterbiParser(pcfg)
    generator = parser.parse(['Move', '(', '50', ')'])
    tree = next(generator)
    # print(tree.logprob())
    print(tree)


if __name__ == "__main__":
    sanity_test()
