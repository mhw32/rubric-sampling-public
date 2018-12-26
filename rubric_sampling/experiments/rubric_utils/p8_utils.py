r"""Utilities specific to problem 8 (out of 8)."""

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
from ..utils import DATASETS_ROOT


IX_TO_LABEL = {
	0 : 'Move: wrong multiple',
	1 : 'Move: constant',
	2 : 'Move: wrong opp',
	3 : 'Move: missing opp',
	4 : 'Move: correct',
	5 : 'Move: forward/backward confusion',
	6 : 'Move: no move',
	7 : 'Turn: constant',
	8 : 'Turn: wrong multiple',
	9 : 'Turn: wrong opp',
	10 : 'Turn: missing opp',
	11 : 'Turn: no turn',
	12 : 'Turn: left/right confusion',
	13 : 'Single shape: wrong iter #',
	14 : 'Single shape: body incorrect',
	15 : 'Single shape: wrong MT order',
	16 : 'Single shape: missing repeat',
	17 : 'Single shape: nesting issue',
	18 : 'Single shape: armslength',
	19 : 'For loop: wrong start',
	20 : 'For loop: wrong end',
	21 : 'For loop: wrong delta',
	22 : 'For loop: not looping by sides',
	23 : 'For loop: no loop',
	24 : 'For loop: armslength',
	25 : 'For loop: repeat instead of for',
}
LOOP_LABELS_IX = [13, 14, 15, 16, 17, 18,19, 20, 21, 22, 23, 24, 25]
GEOMETRY_LABELS_IX = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

STUDENT_PCFG_PARAMS = [100, 70, 5, 5, 5, 5, 2, 10, 5, 100, 10, 10, 10, 10, 10, 2, 100, 3, 3, 10, 10, 3, 40, 100, 5, 5, 5, 5, 5, 100, 15, 100, 100, 100, 10, 100, 10, 20, 10, 3, 100, 30, 20, 20, 5, 10, 100, 10, 100, 10, 10, 10, 30, 10, 40, 40, 5, 20, 80, 100, 3, 3, 3, 3, 3, 10, 3, 3, 3, 3, 3, 3, 3, 5, 3, 3, 3, 3, 5, 100, 3, 100, 10, 20, 10, 3, 3, 20, 2, 50, 50, 50, 20, 1, 100, 1, 100, 100, 1, 1, 0.05]
TEACHER_PCFG_PARAMS = None

RUBRIC_DIR = os.path.join(os.path.dirname(__file__), '../../../rubrics')
STUDENT_PCFG_PATH = os.path.join(RUBRIC_DIR, 'p8', 'student.cfg')
TEACHER_PCFG_PATH = None

DATA_ROOT = os.path.join(DATASETS_ROOT, 'codeorg', 'p8')
