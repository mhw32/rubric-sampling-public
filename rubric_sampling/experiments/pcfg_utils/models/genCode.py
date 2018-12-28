from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import random
import json
import numpy as np
import operator
import math
import copy

from .tree import Tree
import .blocky as blocky

PGM_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), 'pgm'))

	
def loadDecisionTrees():
	trees = []
	treeNames = json.load(open(os.path.join(PGM_DIR, 'decisions.json')))
	for name in treeNames:
		filePath = os.path.join(PGM_DIR, '%s.json' % name)
		tree = json.load(open(filePath))
		trees.append(tree)
	return trees


def collectDecisions(decisionTrees):
	decisions = []
	for tree in decisionTrees:
		decisions.extend(chose(tree))
		if 'NoPlan' in decisions: 
			return set(decisions)
	return set(decisions)


def generateCode(decisions):
	program = Tree('Program')
	program.addChild(Tree('WhenRun'))
	if 'AddColor' in decisions:
		colorBlock = getColorBlock(decisions)
		program.addChild(colorBlock)
	if 'ClockwisePlan' in decisions:
		if 'ClockwiseExtraTurn' in decisions:
			program.addChild(generateFirstTurn(decisions))
		program.addChild(generateRepeat(decisions))
	if 'CounterClockwisePlan' in decisions:
		program.addChild(generateRepeat(decisions))
	if 'NoPlan' in decisions:
		program.addChild(generateRandomCode(decisions, 0))
	blocky.normalize(program)

	return program


def generateCompletednessState():
	return random.random()


def generateTrajectory(goalTree):
	asts = []

	curr = copy.deepcopy(goalTree)
	asts.append(curr)
	while blocky.countUserBlocks(curr) > 0:
		curr = copy.deepcopy(curr)
		userBlocks = getPostOrderBlocks(curr)
		index = choseBlockToRemove(userBlocks)
		blocky.removeBlock(curr, userBlocks[index])
		asts.insert(0, curr)
	return asts


def generatePartialWork(goalTree):
	ast = copy.deepcopy(goalTree)

	numBlocksToKeep = choseNumBlocksToKeep(goalTree)
	
	while blocky.countUserBlocks(ast) > numBlocksToKeep:
		userBlocks = getPostOrderBlocks(ast)

		'''blockStr = ''
		for block in userBlocks:
			blockStr += block.rootName + ','
		print '[' + blockStr + ']'''

		if len(userBlocks) == 0: raise Exception('no blocks left to remove')
		index = choseBlockToRemove(userBlocks)
		blocky.removeBlock(ast, userBlocks[index])


	nBlocks = blocky.countUserBlocks(ast)
	nGoalBlocks = blocky.countUserBlocks(goalTree)
	if nGoalBlocks == 0: 
		completedness = 0
	else:
		completedness = 1.0 * nBlocks / nGoalBlocks
		
	return ast, completedness


def choseBlockToRemove(userBlocks):
	'''if len(userBlocks) == 1: return 0
	if random.random() < 0.95: return 0
	return random.randint(1, len(userBlocks) - 1)'''
	return 0


def choseNumBlocksToKeep(goalTree):
	numBlocks = blocky.countUserBlocks(goalTree)
	while numBlocks > 0:
		if random.random() < 0.85:
			return numBlocks
		numBlocks -= 1
	return numBlocks


def getPostOrderBlocks(goalTree):
	blocks = []

	for child in reversed(goalTree.getChildren()):
		childList = getPostOrderBlocks(child)
		blocks.extend(childList)


	if blocky.isUserBlock(goalTree):
		blocks.append(goalTree)

	return blocks


def getColorBlock(decisions):
	block = Tree('SetColor')
	value = Tree('Value')
	block.addChild(value)

	if 'AddRandomColor' in decisions:
		value.addChild(Tree('RandomColor'))
	else:
		value.addChild(Tree('Red'))
	return block


def generateFirstTurn(decisions):
	left = not 'LeftRightConfusion' in decisions
	angle = 90
	if 'ThinkTriangle' in decisions:
		if 'GetEqualateral' in decisions:
			angle = 60
		else:
			if random.random() < 0.6:
				angle = 45
			else:
				angle = 30

	if 'AngleInvariance' in decisions:
		left = not left
		angle = 360 - angle
 
 	code = Tree('Turn')
 	code.addChild(Tree(getTurnString(left)))
	value = code.addChild(Tree('Value'))
	number = value.addChild(Tree('Number'))
	number.addChild(Tree(str(angle)))
	return code


def generateRepeat(decisions):
	if 'GetRepeat' in decisions:
		n = getRepeatN(decisions)
		repeat = Tree('Repeat')
		value = repeat.addChild(Tree('Value'))
		number = value.addChild(Tree('Number'))
		number.addChild(Tree(str(n)))
		body = repeat.addChild(Tree('Body'))
		body.addChild(generateBody(decisions))

		if doesntUnderstandNesting(decisions):
			bodyBlock = body.children[0]
				
			assert len(bodyBlock.children) == 2

			# Some fancy machinery to make a poorly nested block
			newBlock = Tree('Block')
			newBlock.addChild(repeat)
			newBlock.addChild(bodyBlock.children[1])
			del bodyBlock.children[1]

			return newBlock
		else:
			return repeat
		return repeat
	else:
		code = Tree('Block')
		nBodies = getBodiesN(decisions)
		bodyCode = generateBody(decisions)
		for i in range(nBodies):
			code.addChild(copy.deepcopy(bodyCode))
		if nBodies == 1:
			if random.random() > 0.5:
				code.addChild(copy.deepcopy(code.children[0]))
		return code


def doesntUnderstandNesting(decisions):
	if 'DontGetNesting' in decisions:
		if 'GetBodyCombo' in decisions:
			return True
	return False


def generateBody(decisions):
	body = Tree('Block')
	if 'GetBodyCombo' in decisions:
		
		if 'GetBodyComboOrder' in decisions:
			body.addChild(generateMove(decisions))
			body.addChild(generateBodyTurn(decisions))
		else:
			body.addChild(generateBodyTurn(decisions))
			body.addChild(generateMove(decisions))
	if 'OneBlockBody' in decisions:
		if random.random() < 0.5:
			body.addChild(generateMove(decisions))
		else:
			body.addChild(generateBodyTurn(decisions))
	if 'BodyConfusion' in decisions:
		body.addChild(generateRandomCode(decisions, 0))
	return body


def generateMove(decisions):
	code = Tree('Move')
	code.addChild(Tree('Forward'))
	value = code.addChild(Tree('Value'))
	number = value.addChild(Tree('Number'))
	if 'Move50' in decisions:
		number.addChild(Tree(str(50)))
	elif 'MoveDefault' in decisions:
		number.addChild(Tree(str(100)))
	else:
		commonDistractors = range(10, 105, 5)
		if random.random() > 0.8: 
			n = random.choice(commonDistractors)
		else:
			n = random.randint(0, 200)
		number.addChild(Tree(str(n)))
	return code


def getBodiesN(decisions):
	if 'NoRepeat' in decisions:
		return 1
	if 'GetThreeSides' in decisions:
		return 3
	return random.choice([2, 4])


def getRepeatN(decisions):
	if 'GetThreeSides' in decisions:
		return 3
	if 'NoRepeatCounterAttempt' in decisions:
		return 0
	if 'DontGetThreeSides' in decisions:
		return random.choice([1, 2, 4])


def generateBodyTurn(decisions):
	code = Tree('Turn')

	isLeft = 'CounterClockwisePlan' in decisions
	angle = getBodyAngle(decisions)
	if 'LeftRightConfusion' in decisions:
		isLeft = not isLeft
	if 'Angle360Invariance' in decisions:
		isLeft = not isLeft
		angle = 360 - angle
	code.addChild(Tree(getTurnString(isLeft)))
	
	value = code.addChild(Tree('Value'))
	number = value.addChild(Tree('Number'))
	number.addChild(Tree(str(angle)))
	return code


def getBodyAngle(decisions):
	if 'ThinkSquare' in decisions:
		return 90

	if 'GetEqualateral' in decisions:
		baseAngle = 60
	else:
		commonDistractors = [30,60,90,120,160,180,45,90]
		if random.random() > 0.5: return random.choice(commonDistractors)
		if random.random() > 0.5: return random.choice(range(0, 180, 5))
		return random.randint(0, 360)

	if 'ThinksToInvertAngle' in decisions:
		return 180 - baseAngle
	return baseAngle


def getTurnString(isLeft):
	if isLeft:
		return 'Left'
	return 'Right'


def generateRandomCode(decisions, depth):
	c = random.random()
	p = 0.8 + depth * 0.1
	if c < p:
		return generateRandomTerminal(decisions, depth)
	else:
		code = Tree('Block')
		code.addChild(generateRandomTerminal(decisions, depth))
		code.addChild(generateRandomCode(decisions, depth+1))
		return code


def generateRandomTerminal(decisions, depth):
	if random.random() < 0.9:
		if random.random() < 0.5:
			code = Tree('Move')
			if random.random() < 0.9:
				code.addChild(Tree('Forward'))
			else:
				code.addChild(Tree('Backward'))
			value = code.addChild(Tree('Value'))
			number = value.addChild(Tree('Number'))
			amount = random.choice([0, 50, 100, 150])
			number.addChild(Tree(str(amount)))
			return code
		else:
			left = random.random()
			angle = random.choice(range(0, 390, 30))
			code = Tree('Turn')
			code.addChild(Tree(getTurnString(left)))
			value = code.addChild(Tree('Value'))
			number = value.addChild(Tree('Number'))
			number.addChild(Tree(str(angle)))
			return code
	else:
		repeat = Tree('Repeat')
		value = repeat.addChild(Tree('Value'))
		number = value.addChild(Tree('Number'))
		n = random.randint(1, 4)
		number.addChild(Tree(str(n)))
		body = repeat.addChild(Tree('Body'))
		body.addChild(generateRandomCode(decisions, depth+1))
		return repeat


def chose(decisionTree):
	d = []

	# Base Case
	if not 'children' in decisionTree: 
		return []

	# Recursive Case
	children = decisionTree['children']
	child = choseChild(children)
	d.append(child['name'])
	d.extend(chose(child))
	return d


def choseChild(children):
	weights = []
	for child in children:
		weights.append(float(child['weight']))
	p = np.array(weights) / sum(weights)
	return np.random.choice(children, p=p)
