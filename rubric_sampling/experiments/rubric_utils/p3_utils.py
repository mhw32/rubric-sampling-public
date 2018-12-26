r"""Utilities specific to problem 3 (out of 8)."""

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
from ..utils import DATASETS_ROOT


IX_TO_LABEL  = {
    0: 'For loop: no loop',
    1: 'For loop: armslength',
    2: 'Single shape: body incorrect',
    3: 'For loop: repeat instead of for',
    4: 'Move: forward/backward confusion',
    5: 'For loop: correct end',
    6: 'Correct body order (CW)',
    7: 'Single shape: for instead of repeat',
    8: 'For loop: wrong end',
    9: 'Move: constant',
    10: 'Correct loop structure',
    11: 'Turn: wrong constant',
    12: 'Move: no move',
    13: 'Single shape: armslength',
    14: 'For loop: wrong delta',
    15: 'For loop: clockwise strategy is incorrect',
    16: 'Precondition mismatch',
    17: 'Turn: left/Right confusion',
    18: 'Turn: no turn',
    19: 'Single body (correct)',
    20: 'For loop: wrong start',
    21: 'Correct inner loop structure',
    22: 'Single shape: missing repeat',
    23: 'CWTurnStart: incorrect',
    24: 'Correct repeat num',
    25: 'Single shape: wrong iter #',
    26: 'Correct body order (CCW)',
}
LOOP_LABELS_IX = [0, 1, 2, 3, 7, 8, 13, 14, 15, 16, 20, 22, 25]
GEOMETRY_LABELS_IX =[4, 9, 11, 12, 17, 18]

STUDENT_PCFG_PARAMS = [95, 5, 80, 5, 5, 2, 10, 10, 100, 10, 10, 2, 10, 100, 2, 2, 2, 2, 2, 2, 93, 7, 100, 100, 100, 10, 100, 5, 100, 10, 90, 90, 10, 100, 3, 5, 80, 5, 1, 1, 1, 1, 5, 100, 5, 1, 1, 2, 90, 1, 1, 100, 100, 3, 100, 3, 100, 1, 10, 7, 7, 4, 72, 0.05]
TEACHER_PCFG_PARAMS = None

RUBRIC_DIR = os.path.join(os.path.dirname(__file__), '../../../rubrics')
STUDENT_PCFG_PATH = os.path.join(RUBRIC_DIR, 'p3', 'student.cfg')
TEACHER_PCFG_PATH = None

DATA_ROOT = os.path.join(DATASETS_ROOT, 'codeorg', 'p3')
