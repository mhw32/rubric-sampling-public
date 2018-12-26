r"""Utilities specific to problem 4 (out of 8)."""

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
from ..utils import DATASETS_ROOT


IX_TO_LABEL = {
    0: 'Repeated body ',
    1: 'For loop: no loop',
    2: 'For loop: armslength',
    3: 'Single shape: body incorrect',
    4: 'For loop: repeat instead of for',
    5: 'Move: forward/backward confusion',
    6: 'For loop: correct end',
    7: 'For loop: wrong end',
    8: 'Move: constant',
    9: 'Correct loop structure',
    10: 'Turn: wrong constant',
    11: 'Move: no move',
    12: 'Correct body: no loop',
    13: 'For loop: wrong delta',
    14: 'For loop: clockwise strategy is incorrect',
    15: 'Precondition mismatch',
    16: 'Correct repeat num',
    17: 'Turn: no turn',
    18: 'Single shape: wrong MT order',
    19: 'Single body (correct)',
    20: 'For loop: wrong start',
    21: 'Correct body order',
    22: 'Single shape: adds inner loop',
    23: 'CWTurnStart: incorrect',
    24: 'Turn: left/Right confusion',
    25: 'Single shape: wrong iter #'
}
LOOP_LABELS_IX = [0, 1, 2, 3, 4, 6, 7, 13, 14, 15, 18, 20, 22, 25]
GEOMETRY_LABELS_IX = [5, 8, 10, 11, 17, 24]

STUDENT_PCFG_PARAMS = [ 95, 4, 80, 5, 5, 2, 2, 10, 5, 100, 2, 2, 2, 100, 2, 6, 1, 6, 2, 2, 93, 7, 100, 100, 100, 10, 100, 5, 100, 10, 90, 90, 10, 100, 3, 5, 80, 5, 1, 1, 1, 1, 5, 100, 5, 1, 1, 2, 90, 1, 1, 100, 100, 3, 100, 3, 100, 1, 25, 20, 20, 10, 50, 0.05]
TEACHER_PCFG_PARAMS = None

RUBRIC_DIR = os.path.join(os.path.dirname(__file__), '../../../rubrics')
STUDENT_PCFG_PATH = os.path.join(RUBRIC_DIR, 'p4', 'student.cfg')
TEACHER_PCFG_PATH = None

DATA_ROOT = os.path.join(DATASETS_ROOT, 'codeorg', 'p4')
