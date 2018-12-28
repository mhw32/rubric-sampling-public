from __future__ import division
from __future__ import print_function
from __future__ import absolute_import


# a public static method
def toBlocky(ast):
    t = PseudoCodeTranslator()
    return t.toPseudoCode(ast)

# Class: Psuedo Code Translator
# ------------------------
# Can turn an AST into psuedo code.
class PseudoCodeTranslator():

    # Function: To Pseudo Code
    # -----------------
    # The public method of pseudoCodeTranslator.
    def toPseudoCode(self, ast):
        pseudoCodeStr = self.expandCodeBlock(0, ast)
        pseudoCodeStr += '\n'
        return pseudoCodeStr

    # Function: Expand Code Block
    # -----------------
    # Turns a codeblock AST (eg the body of a method or the body of a loop)
    # into pseudoCode.
    def expandCodeBlock(self, indent, codeBlock):
        codestr = ''
        for child in codeBlock.children:
            blockType = child.rootName

            # Ignore the when run
            if blockType == 'WhenRun': continue

            # Basic commands
            elif blockType == 'Turn' or blockType == 'Move':
                param1 = child.children[0].rootName
                param2 = self.getValue(child.children[1])
                codestr += self.getIndent(indent) + blockType
                codestr += param1 + "("
                codestr += param2 + ') \n'

            elif blockType in ['SetColor', 'DrawWidth', 'Alpha']:
                color = child.children[0].children[0].rootName
                codestr += self.getIndent(indent) + blockType  + '('
                codestr += color + ') \n'

            # For loops
            elif blockType == 'Repeat':
                numTimes = self.getValue(child.children[0])
                codestr += self.getIndent(indent)
                codestr += 'Repeat(' + str(numTimes) + ') { \n'
                if len(child.children) >= 2:
                    body = child.children[1]
                    codestr += self.expandCodeBlock(indent + 1, body)
                codestr += '} '

            # For loops
            elif blockType == 'For':
                startValue = self.getValue(child.children[0])
                endValue = self.getValue(child.children[1])
                deltaValue = self.getValue(child.children[2])
    
                codestr += self.getIndent(indent)
                line = 'For('
                line += 'x='+ str(startValue) + ', '
                line += 'x<=' + str(endValue) + ', '
                line += 'x+=' + str(deltaValue)
                line += '):'
                codestr += line + '\n'
                if len(child.children) < 4:
                    raise Exception('for loop missing body')

                if len(child.children) >= 4:
                    body = child.children[3]
                    codestr += self.expandCodeBlock(indent + 1, body)

            # Might be useful later
            # While loops
            elif blockType == 'while':
                condition = self.expandConditionBlock(block['condition'])
                body = block['body']
                codestr += self.getIndent(indent)
                # If crashProtection is on, check if you are crashed before
                # entering a while loop
                if self.crashProtection:
                    codestr += 'while not isCrashed() and ' + condition + ':\n'
                else:
                    codestr += 'while ' + condition + ':\n'
                codestr += self.expandCodeBlock(indent + 1, body)

            # If statements
            elif blockType == 'if':
                condition = self.expandConditionBlock(block['condition'])
                body = block['body']
                codestr += self.getIndent(indent)
                codestr += 'if ' + condition + ':\n'
                codestr += self.expandCodeBlock(indent + 1, body)

            # If/else statements
            elif blockType == 'ifElse':
                condition = self.expandConditionBlock(block['condition'])
                ifBody = block['ifBody']
                elseBody = block['elseBody']
                codestr += self.getIndent(indent) + 'if ' + condition + ':\n'
                codestr += self.expandCodeBlock(indent + 1, ifBody)
                codestr += self.getIndent(indent) + 'else:\n'
                codestr += self.expandCodeBlock(indent + 1, elseBody) + '\n'

            # Invoking user defined methods
            elif blockType == 'invoke':
                methodName = block['method']
                codestr += self.getIndent(indent) + methodName + '()\n'

            # Somehow these can end up as root types...
            elif blockType == 'Number': return ''
            elif blockType == 'Variable': return ''
            elif blockType == 'Arithmetic': return ''

            # Opps! There must have been a parse error.
            else:
                raise Exception("unknown type: \"" + blockType + "\"")

        return codestr

    def getValue(self, valueBlock):
        if len(valueBlock.children) == 0: return '???'

        # the child says if the block is a number or math or a variable
        typeBlock = valueBlock.children[0]
        valueType = typeBlock.rootName

        if valueType == 'Number':
            return typeBlock.children[0].rootName

        if valueType == 'Variable':
            return 'x'

        if valueType == 'Arithmetic':
            opp = self.getOpp(typeBlock.children[0].rootName)
            lhs = self.getValue(typeBlock.children[1])
            rhs = self.getValue(typeBlock.children[2])
            return lhs + opp + rhs

        raise Exception('unknown type ' + valueType)

    def getOpp(self, oppStr):
        if oppStr == 'Multiply':
            return '*'
        if oppStr == 'Divide':
            return '/'
        if oppStr == 'Add':
            return '+'
        if oppStr == 'Minus':
            return '-'
        return oppStr

    # Function: Expand Condition Block
    # -----------------
    # Condition blocks can be complex (eg using the not expression).
    # We currently do not support other boolean expressions
    # (eg and / or). If they are supported in the future, relevant code
    # should go here.
    def expandConditionBlock(self, block):
        codestr = ''
        if not "type" in block:
            raise Exception("block has no type: \"" + block + "\"")
        blockType = block['type']
        if self.isConditionTest(blockType):
            codestr += blockType + '()'
        elif blockType == 'not':
            conditionStr = self.expandConditionBlock(block['condition'])
            codestr += 'not ' + conditionStr
        else:
            raise Exception("unknown type: \"" + blockType + "\"")
        return codestr

    # Function: Get Indent
    # -----------------
    # Turns a code block depth into an indent string.
    def getIndent(self, depth):
        indentStr = ''
        for i in range(depth):
            indentStr += ' '
        return indentStr

    # Function: Is Condition Test
    # -----------------
    # There are five predefined conditions
    def isConditionTest(self, blockType):
        if blockType == 'markersPresent': return True
        if blockType == 'noMarkersPresent': return True
        if blockType == 'leftIsClear': return True
        if blockType == 'rightIsClear': return True
        if blockType == 'frontIsClear': return True
        return False

    # Function: Is Command
    # -----------------
    # There are five predefined commands
    def isCommand(self, blockType):
        if blockType == 'Move': return True
        if blockType == 'Turn': return True
        if blockType == 'SetColor': return True
        return False

