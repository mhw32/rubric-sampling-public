r"""Utilities specific to problem 7 (out of 8)."""

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
from ..utils import DATASETS_ROOT


IX_TO_LABEL = {
    0: 'For loop: no loop',
    1: 'For loop: armslength',
    2: 'Single shape: body incorrect',
    3: 'For loop: repeat instead of for',
    4: 'Move: forward/backward confusion',
    5: 'For loop: correct end',
    6: 'Move: should not reference counter',
    7: 'Single shape: nesting issue',
    8: 'Turn: wrong opp',
    9: 'For loop: wrong end   ',
    10: 'Single shape: for instead of repeat',
    11: 'Correct loop structure',
    12: 'Move: no move',
    13: 'Turn: constant',
    14: 'Turn: missing opp',
    15: 'Single shape: armslength',
    16: 'Turn: wrong multiple',
    17: 'Unnecessary loop structure',
    18: 'For loop: wrong delta',
    19: 'For loop: wrong end',
    20: 'Turn: no turn',
    21: 'Single shape: wrong MT order',
    22: 'Single body (correct)',
    23: 'For loop: wrong start',
    24: 'Correct body order',
    25: 'Turn: left/right confusion',
    26: 'Correct inner loop structure',
    27: 'Single shape: missing repeat',
    28: 'Correct repeat num',
    29: 'Single shape: wrong iter #',
    30: 'Move: wrong constant'
}
LOOP_LABELS_IX = [0, 1, 2, 3, 7, 9, 10, 15, 17, 18, 19, 21, 23, 27, 29]
GEOMETRY_LABELS_IX = [4, 6, 8, 12, 13, 14, 16, 20, 25, 30]

STUDENT_PCFG_PARAMS = [100, 70, 5, 5, 2, 10, 5, 100, 10, 10, 10, 10, 2, 10, 10, 18, 72, 3, 3, 3, 3, 3, 3, 3, 3, 100, 5, 5, 5, 5, 5, 100, 15, 100, 3, 100, 100, 100, 100, 10, 20, 10, 3, 100, 30, 20, 20, 5, 10, 100, 10, 100, 20, 80, 100, 3, 3, 3, 3, 3, 10, 3, 3, 3, 3, 3, 3, 3, 5, 3, 3, 3, 3, 5, 2, 100, 5, 5, 1, 100, 1, 100, 100, 1, 1, 0.05]
TEACHER_PCFG_PARAMS = None

RUBRIC_DIR = os.path.join(os.path.dirname(__file__), '../../../rubrics')
STUDENT_PCFG_PATH = os.path.join(RUBRIC_DIR, 'p7', 'student.cfg')
TEACHER_PCFG_PATH = None

