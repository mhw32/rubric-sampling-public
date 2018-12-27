r"""Utilities specific to problem 6 (out of 8)."""

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
from ..utils import DATASETS_ROOT


IX_TO_LABEL = {
    0: 'Repeated body ',
    1: 'For loop: no loop',
    2: 'For loop: armslength',
    3: 'Turn: no turn ',
    4: 'Single shape: body incorrect',
    5: 'For loop: repeat instead of for',
    6: 'Move: forward/backward confusion',
    7: 'For loop: correct end',
    8: 'For loop: wrong end',
    9: 'Incorrect: no viable clockwise strategy',
    10: 'Correct loop structure',
    11: 'Turn: wrong constant',
    12: 'Correct structure: no loop',
    13: 'Move: no move',
    14: 'Incorrectly ordered commands; no viable CW strategy',
    15: 'For loop: wrong delta',
    16: 'Turns beforehand for clockwise strategy',
    17: 'Move: constant',
    18: 'Correct CW turn start',
    19: 'Correct repeat num',
    20: 'Single body (correct)',
    21: 'Correctly ordered commands for CCW strategy',
    22: 'For loop: wrong start',
    23: 'Single shape: adds inner loop',
    24: 'Turn: left/Right confusion',
    25: 'Single shape: wrong iter #',
}
LOOP_LABELS_IX = [0, 1, 2, 4, 5, 8, 15, 22, 23, 25]
GEOMETRY_LABELS_IX = [3, 6, 11, 13, 17, 24]

STUDENT_PCFG_PARAMS = [100, 80, 5, 5, 2, 2, 10, 5, 100, 2, 2, 2, 100, 2, 6, 1, 6, 2, 2, 93, 7, 100, 100, 100, 10, 100, 5, 100, 10, 90, 90, 10, 100, 3, 90, 3, 3, 5, 100, 5, 1, 1, 2, 90, 1, 1, 100, 100, 3, 100, 5, 3, 3, 100, 3, 1, 25, 10, 10, 5, 50, 0.05]
TEACHER_PCFG_PARAMS = None

RUBRIC_DIR = os.path.join(os.path.dirname(__file__), '../../../rubrics')
STUDENT_PCFG_PATH = os.path.join(RUBRIC_DIR, 'p6', 'student.cfg')
TEACHER_PCFG_PATH = None

