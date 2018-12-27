from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import numpy as np
from ..utils import DATASETS_ROOT


def get_label_params(problem_id):
    r"""Load constants for a particular problem.
    
    @param problem_id: integer 
                       1|2|3|4|5|6|7|8
    """
    if problem_id == 1:
        from .p1_utils import IX_TO_LABEL, LOOP_LABELS_IX, GEOMETRY_LABELS_IX
    elif problem_id == 2:
        from .p2_utils import IX_TO_LABEL, LOOP_LABELS_IX, GEOMETRY_LABELS_IX
    elif problem_id == 3:
        from .p3_utils import IX_TO_LABEL, LOOP_LABELS_IX, GEOMETRY_LABELS_IX
    elif problem_id == 4:
        from .p4_utils import IX_TO_LABEL, LOOP_LABELS_IX, GEOMETRY_LABELS_IX
    elif problem_id == 5:
        from .p5_utils import IX_TO_LABEL, LOOP_LABELS_IX, GEOMETRY_LABELS_IX
    elif problem_id == 6:
        from .p6_utils import IX_TO_LABEL, LOOP_LABELS_IX, GEOMETRY_LABELS_IX
    elif problem_id == 7:
        from .p7_utils import IX_TO_LABEL, LOOP_LABELS_IX, GEOMETRY_LABELS_IX
    elif problem_id == 8:
        from .p8_utils import IX_TO_LABEL, LOOP_LABELS_IX, GEOMETRY_LABELS_IX

    LABEL_TO_IX = {v: k for k, v in IX_TO_LABEL.iteritems()}
    N_LABELS = len(IX_TO_LABEL.keys())

    return N_LABELS, IX_TO_LABEL, LABEL_TO_IX, LOOP_LABELS_IX, GEOMETRY_LABELS_IX


def get_pcfg_params(problem_id, author='teacher', random=False):
    r"""Return parameters for a PCFG for any problem.

    @param problem_id: integer 
                       1|2|3|4|5|6|7|8
    @param author: string [default: teacher]
                   teacher|student
                   use PCFG written by a professor in computer science (teacher) 
                   or an undergraduate teaching assistant (student)
    @param random: boolean [default: False]
                   use random parameters for PCFG; otherwise use the expert
                   parameters chosen by the author
    """
    if problem_id == 1:
        from .p1_utils import STUDENT_PCFG_PARAMS, TEACHER_PCFG_PARAMS
    elif problem_id == 2:
        from .p2_utils import STUDENT_PCFG_PARAMS, TEACHER_PCFG_PARAMS
    elif problem_id == 3:
        from .p3_utils import STUDENT_PCFG_PARAMS, TEACHER_PCFG_PARAMS
    elif problem_id == 4:
        from .p4_utils import STUDENT_PCFG_PARAMS, TEACHER_PCFG_PARAMS
    elif problem_id == 5:
        from .p5_utils import STUDENT_PCFG_PARAMS, TEACHER_PCFG_PARAMS
    elif problem_id == 6:
        from .p6_utils import STUDENT_PCFG_PARAMS, TEACHER_PCFG_PARAMS
    elif problem_id == 7:
        from .p7_utils import STUDENT_PCFG_PARAMS, TEACHER_PCFG_PARAMS
    elif problem_id == 8:
        from .p8_utils import STUDENT_PCFG_PARAMS, TEACHER_PCFG_PARAMS

    if author == 'teacher':
        return np.random.rand(len(TEACHER_PCFG_PARAMS)) if random else TEACHER_PCFG_PARAMS
    elif author == 'student':
        return np.random.rand(len(STUDENT_PCFG_PARAMS)) if random else STUDENT_PCFG_PARAMS
    else:
        raise Exception('author %s not supported.' % author)


def get_pcfg_path(problem_id, author='teacher'):
    r"""Return path to  PCFg for any problem.

    @param problem_id: integer 
                       1|2|3|4|5|6|7|8
    @param author: string [default: teacher]
                   teacher|student
                   use PCFG written by a professor in computer science (teacher) 
                   or an undergraduate teaching assistant (student)
    """
    if problem_id == 1:
        from .p1_utils import STUDENT_PCFG_PATH, TEACHER_PCFG_PATH
    elif problem_id == 2:
        from .p2_utils import STUDENT_PCFG_PATH, TEACHER_PCFG_PATH
    elif problem_id == 3:
        from .p3_utils import STUDENT_PCFG_PATH, TEACHER_PCFG_PATH
    elif problem_id == 4:
        from .p4_utils import STUDENT_PCFG_PATH, TEACHER_PCFG_PATH
    elif problem_id == 5:
        from .p5_utils import STUDENT_PCFG_PATH, TEACHER_PCFG_PATH
    elif problem_id == 6:
        from .p6_utils import STUDENT_PCFG_PATH, TEACHER_PCFG_PATH
    elif problem_id == 7:
        from .p7_utils import STUDENT_PCFG_PATH, TEACHER_PCFG_PATH
    elif problem_id == 8:
        from .p8_utils import STUDENT_PCFG_PATH, TEACHER_PCFG_PATH

    return TEACHER_PCFG_PATH if author == 'teacher' else STUDENT_PCFG_PATH


def get_codeorg_data_root(problem_id, dataset='unlabeled'):
    r"""Return path to folder containing data.

    @param problem_id: integer 
                       1|2|3|4|5|6|7|8
    @param dataset: string
                    unlabeled|annotated|synthetic|raw
    """
    return os.path.join(DATASETS_ROOT, 'codeorg', dataset, 'p%d' % problem_id)

