from __future__ import division
from __future__ import print_function
from __future__ import absolute_import


def isValid(ast):
	if ast.rootName == 'Move':
		return isMoveValid(ast)
	if ast.rootName == 'Turn':
		return isTurnValid(ast)

	for child in ast.children:
		if not isValid(child): return False

	return True


def countUserBlocks(ast):
	nBlocks = 0
	if isUserBlock(ast):
		nBlocks += 1

	for child in ast.children:
		nBlocks += countUserBlocks(child)

	return nBlocks


def removeColors(ast):
	for child in ast.children:
		removeColors(child)

	newChildren = []
	for child in ast.children:
		if child.rootName != 'SetColor':
			newChildren.append(child)

	ast.children = newChildren


def normalize(ast):
	for child in ast.children:
		normalize(child)

	newChildren = []

	for child in ast.children:
		if child.rootName == 'Block':
			for grandchild in child.children:
				newChildren.append(grandchild)
		else:
			newChildren.append(child)

	ast.children = newChildren
	

def removeBlock(root, block):
	parent = root.getParent(block)
	childIndex = parent.getChildIndex(block)
	parent.removeChild(block)

	# special case the repeat block
	if block.rootName == 'Repeat':
		body = block.children[1]
		for bodyChild in reversed(body.getChildren()):
			parent.addChildAt(bodyChild, childIndex)
	

def isUserBlock(ast):
	nodeType = ast.rootName
	if unicode(nodeType).isnumeric(): return False
	if nodeType[0] == '#': return False
	userBlockMap = {
		'Program': False,
		'WhenRun': False,
		'Repeat': True,
		'Number': True,
		'Variable':True,
		'Counter':False,
		'Move': True,
		'Turn':True,
		'Arithmetic':True,
		'Multiplication':False,
		'Division':False,
		'Addition':False,
		'Subtraction':False,
		'SetColor':True,
		'Color':True,
		'RandomColor':True,
		'Forward':False,
		'Backward':False,
		'Left':False,
		'Right':False,
		'Body':False,
		'Red':False,
		'Value':False,
		'???':False,
		'For':True,
		'Arithmetic':True
	}
	if not nodeType in userBlockMap:
		raise Exception('unkown block: '+ nodeType)
	return userBlockMap[nodeType]


def isRepeatValid(ast):
	body = ast.children[1]
	return len(body.children) != 0


def isMoveValid(ast):
	if len(ast.children) != 2:
		return False
	return isValidValue(ast.children[1])


def isTurnValid(ast):
	if len(ast.children) != 2:
		return False
	return isValidValue(ast.children[1])


def isValidValue(valueNode):
	if not valueNode.rootName == 'Value': return False
	if len(valueNode.children) != 1: return False
	numberNode = valueNode.children[0]
	if not numberNode.rootName == 'Number': return False
	if len(numberNode.children) != 1: return False
	num = numberNode.children[0].rootName
	return unicode(num).isnumeric()
