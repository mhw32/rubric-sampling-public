r"""Utilities specific to problem 1 (out of 8)."""

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
from ..utils import DATASETS_ROOT

# map from integers to feedback labels
IX_TO_LABEL = {
    0: u'Standard Strategy',
    1: u'Does not get nesting',
    2: u'Does not get pre/post condition',
    3: u"Doesn't use a repeat",
    4: u'Repetition of bodies',
    5: u"Doesn't loop three times",
    6: u'Left/Right confusion',
    7: u'Does not know equilateral is 60',
    8: u'Does not invert angle',
    9: u'Default turn',
    10: u'Random move amount',
    11: u'Default move',
    12: u'Body order is incorrect (turn/move)',
}

# which labels account for looping
LOOP_LABELS_IX = [1, 2, 3, 4, 5, 12]

# which labels account for geometry
GEOMETRY_LABELS_IX = [6, 7, 8, 9, 10, 11]

# each PCFG has a number of hyperparameters dictating the 
# probability of each decision branch. Here are the hyperparameters
# chosen by the grammar author.
TEACHER_PCFG_PARAMS = [100, 2, 3, 4, 20, 70, 10, 20, 80, 100, 50, 30, 30, 30, 15, 5, 90, 10, 5, 100, 100, 50, 50, 30, 30, 10, 10, 50, 100, 10, 100, 20, 20, 20, 20, 70, 10, 50, 20, 100, 10, 100, 50, 50, 100, 40, 15, 30, 30, 5, 5, 0.05]
STUDENT_PCFG_PARAMS = [37, 37, 22, 4, 90, 10, 10, 20, 75, 75, 75, 20, 30, 30, 5, 5, 80, 10, 10, 80, 5, 5, 80, 20, 40, 15, 15, 15, 15, 50, 50, 90, 5, 5, 96, 2, 2, 5, 5, 5, 5, 70, 5, 5, 90, 10, 90, 10, 80, 10, 10, 10, 10, 5, 5, 5, 45, 5, 15, 10, 5, 30, 10, 10, 100, 25, 0.05]

RUBRIC_DIR = os.path.join(os.path.dirname(__file__), '../../../rubrics')
STUDENT_PCFG_PATH = os.path.join(RUBRIC_DIR, 'p1', 'student.cfg')
TEACHER_PCFG_PATH = os.path.join(RUBRIC_DIR, 'p1', 'teacher.cfg')

DATA_ROOT = os.path.join(DATASETS_ROOT, 'codeorg', 'p1')
