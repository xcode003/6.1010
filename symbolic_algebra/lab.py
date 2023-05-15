"""
6.1010 Spring '23 Lab 10: Symbolic Algebra
"""

import doctest

# NO ADDITIONAL IMPORTS ALLOWED!
# You are welcome to modify the classes below, as well as to implement new
# classes and helper functions as necessary.


class Symbol:
    '''
    Represents a symbol (number, variable, operation, etc.),
    and implements any behavior shared by symbols
    '''
    precedence = float('inf')
    right_parens = False # -, /
    left_parens = False # **

    # x + y, x is Symbol and y is or isn't
    def __add__(self, y):
        return Add(self, y)
    
    # x + y, x is not Symbol and y is
    def __radd__(self, x):
        return Add(x, self)
    
    def __sub__(self, y):
        return Sub(self, y)
    
    def __rsub__(self, x):
        return Sub(x, self)
    
    def __mul__(self, y):
        return Mul(self, y)
    
    def __rmul__(self, x):
        return Mul(x, self)
    
    def __truediv__(self, y):
        return Div(self, y)
    
    def __rtruediv__(self, x):
        return Div(x, self)
    
    def __pow__(self, y):
        return Pow(self, y)
    
    def __rpow__(self, x):
        return Pow(x, self)

    def simplify(self):
        return self.__class__(self.get_value())

class Var(Symbol):
    '''
    Represents a variable, with a string name, as a symbol
    '''
    def __init__(self, n):
        """
        Initializer.  Store an instance variable called `name`, containing the
        value passed in to the initializer.
        """
        self.name = n

    def get_value(self):
        return self.name

    def __eq__(self, other):
        if type(other) == Var:
            return other.name == self.name
        return False

    def eval(self, mapping):
        if self.name not in mapping:
            raise NameError(f'the variable {self.name} was not assigned')
        return mapping[self.name]
    
    def deriv(self, diff_var):
        if self.name == diff_var:
            return Num(1)
        return Num(0)

    def __str__(self):
        return self.name

    def __repr__(self):
        return f"Var('{self.name}')"


class Num(Symbol):
    '''
    Represents a number (int or float) as a symbol
    '''
    def __init__(self, n):
        """
        Initializer.  Store an instance variable called `n`, containing the
        value passed in to the initializer.
        """
        self.n = n

    def get_value(self):
        return self.n

    def __eq__(self, other):
        if type(other) == Num:
            return other.n == self.n
        elif isinstance(other, (float, int)):
            return self.n == other
        return False

    def eval(self, _):
        return self.n
    
    def deriv(self, _):
        return Num(0)

    def __str__(self):
        return str(self.n)

    def __repr__(self):
        return f'Num({self.n})'


class BinOp(Symbol):
    '''
    Represents a symbolic expression, defied by 
    '''
    operand = None
    def __init__(self, left, right):
        left = self.convert_to_symbol(left)
        right = self.convert_to_symbol(right)

        self.left = left
        self.right = right

    def convert_to_symbol(self, _input):
        if isinstance(_input, str):
            return Var(_input)
        elif isinstance(_input, float) or isinstance(_input, int):
            return Num(_input)
        return _input
    
    def __eq__(self, other):
        if type(self) == type(other):
            return self.left == other.left \
                and self.right == other.right
        return False

    def eval(self, mapping):
        return self.operate(self.left.eval(mapping), self.right.eval(mapping))

    # works, so long as the base cases var and num work
    # creates and returns a new instance
    def simplify(self):
        left_expr = self.left
        right_expr = self.right

        #print(f'here1 {left_expr}\nhere2 {right_expr}')
        if isinstance(left_expr, Num) and isinstance(right_expr, Num): #both numbers
            # prevents trying to simplify the 0 ** -1 case
            #if self.operand != '**' or left_expr.n != 0 or right_expr.n >= 0:
                #print(f'and* -> {Num(self.operate(left_expr.n, right_expr.n))}')
            return Num(self.operate(left_expr.n, right_expr.n))
        return self

    def __repr__(self):
        return f'{self.__class__.__name__}({repr(self.left)}, {repr(self.right)})'

    def __str__(self):
        return_str = ''
        if self.left.precedence < self.precedence or \
            self.left.precedence == self.precedence and self.left_parens:
            return_str += f'({str(self.left)}) {self.operand} '
        else:
            return_str += f'{str(self.left)} {self.operand} '

        if self.right.precedence < self.precedence or \
            self.right.precedence == self.precedence and self.right_parens:
            return_str += f'({str(self.right)})'
        else:
            return_str += f'{str(self.right)}'

        return return_str

class Add(BinOp):
    """
    Represent a symbolic expression for addition
    """
    operand = '+'
    precedence = 1
    
    # careful with concatenation error; isinstance?
    def operate(self, x, y):
        return x + y
    
    def deriv(self, diff_var):
        return self.left.deriv(diff_var) + self.right.deriv(diff_var)
    
    def simplify(self):
        def special_simplify(left_expr, right_expr):
            '''
            Implements special simplification
            rules pertaining to Add
            
            one or the other of the input
            is guaranteed to be a Num
            '''
            if left_expr == 0 or right_expr == 0: # either == 0
                if left_expr == 0:
                    return right_expr
                return left_expr
            return Add(left_expr, right_expr)

        result = special_simplify(self.left.simplify(), self.right.simplify())
        if isinstance(result, (Num, Var)):
            return result
        return BinOp.simplify(result)

class Sub(BinOp):
    """
    Represent a symbolic expression for subtraction
    """
    operand = '-'
    precedence = 1
    right_parens = True
    
    def operate(self, x, y):
        return x - y
    
    def deriv(self, diff_var):
        return self.left.deriv(diff_var) - self.right.deriv(diff_var)
    
    def simplify(self):
        def special_simplify(left_expr, right_expr):
            '''
            Implements special simplification
            rules pertaining to Sub
            
            one or the other of the input
            is guaranteed to be a Num
            '''
            if right_expr == 0: # r_val == 0
                return left_expr
            return Sub(left_expr, right_expr)

        result = special_simplify(self.left.simplify(), self.right.simplify())
        if isinstance(result, (Num, Var)):
            return result
        return BinOp.simplify(result)

class Mul(BinOp):
    """
    Represent a symbolic expression for multiplication
    """
    operand = '*'
    precedence = 2
    
    def operate(self, x, y):
        return x * y
    
    def deriv(self, diff_var):
        return self.left * self.right.deriv(diff_var) \
            + self.right * self.left.deriv(diff_var)
    
    def simplify(self):
        def special_simplify(left_expr, right_expr):
            '''
            Implements special simplification
            rules pertaining to Mul
            
            one or the other of the input
            is guaranteed to be a Num
            '''
            if left_expr == 0 or right_expr == 0: # either == 0
                return Num(0)
            elif left_expr == 1:
                return right_expr
            elif right_expr == 1:
                return left_expr
            return Mul(left_expr, right_expr)

        result = special_simplify(self.left.simplify(), self.right.simplify())
        if isinstance(result, (Num, Var)):
            return result
        return BinOp.simplify(result)

class Div(BinOp):
    """
    Represent a symbolic expression for division
    """
    operand = '/'
    precedence = 2
    right_parens = True

    def operate(self, x, y):
        return x / y
    
    def deriv(self, diff_var):
        return (
                self.right * self.left.deriv(diff_var) \
                - self.left * self.right.deriv(diff_var)
                ) / ((self.right) * (self.right))
    
    def simplify(self):
        def special_simplify(left_expr, right_expr):
            '''
            Implements special simplification
            rules pertaining to Div
            
            one or the other of the input
            is guaranteed to be a Num
            '''
            if left_expr == 0: #l_val == 0
                return Num(0)
            elif right_expr == 1:
                return left_expr
            return Div(left_expr, right_expr)

        result = special_simplify(self.left.simplify(), self.right.simplify())
        if isinstance(result, (Num, Var)):
            return result
        return BinOp.simplify(result)
    
class Pow(BinOp):
    """
    Represent a symbolic expression for exponentiation
    """
    operand = '**'
    precedence = 3
    left_parens = True

    def operate(self, x, y):
        return x ** y

    def deriv(self, diff_var):
        if not isinstance(self.right, Num):
            raise TypeError('derivative cannot be taken of \
                            expression rasied to a power other than Num')
        return self.right * (self.left ** (self.right - 1)) * self.left.deriv(diff_var)

    def simplify(self):
        def special_simplify(left_expr, right_expr):
            '''
            Implements special simplification
            rules pertaining to Pow

            one or the other of the input
            is guaranteed to be a Num
            '''
            if right_expr == 0:
                return Num(1)
            elif right_expr == 1:
                return left_expr
            elif left_expr == 0:
                return Num(0)
            return Pow(left_expr, right_expr)
    
        result = special_simplify(self.left.simplify(), self.right.simplify())
        if isinstance(result, (Num, Var)):
            return result
        return BinOp.simplify(result)

def expression(input_str):
    tokens = tokenize(input_str)
    expr = parse(tokens)
    return expr

def tokenize(input_str):
    '''
    Takes in a string rep of a binary
    expression (arbitrarily nested)
    and returns a list of tokens; a
    token can be a variable name, a
    number, an operation or a parentheses

    Expression occurs as (9 + x), x, or 9 only
    '''
    rough_tokens = input_str.split(' ')
    new_tokens = []
    for token in rough_tokens:
        while token[0] == '(':
            new_tokens.append('(')
            token = token[1:]
        right_tokens = []
        while token[-1] == ')':
            right_tokens.append(')')
            token = token[:-1]
        
        new_tokens.append(token)
        new_tokens += right_tokens
    return new_tokens

def parse(tokens):
    '''
    Takes in a list of tokens and
    recursively returns the proper
    recursive strucutre of Symbol
    subclasses representing the
    tokenized expression
    '''
    def parse_expression(index):
        '''
        Starts at the index in the
        list of tokens, and returns
        the next expression
        '''
        if tokens[index] != '(': # must be var or num
            if tokens[index] in 'abcdefghijklmnopqrstuvwxyz\
                                ABCDEFGHIJKLMNOPQRSTUVWXYZ':
                return Var(tokens[index])
            else:
                return Num(float(tokens[index]))
        else:
            op = index
            l_paren = 1
            r_paren = 0
            first = True
            # explores (3 + ...) or ((3 + ...))
            # stops when finds operator for outermost expression
            # second term is the next index after that
            while l_paren > r_paren + 1 or first:
                op += 1
                if tokens[op] == '(':
                    l_paren += 1
                elif tokens[op] == ')':
                    r_paren += 1
                first = False
            op += 1 # operator index
            operations = {'+':Add,'-':Sub,'*':Mul,'/':Div,'**':Pow}
            # Beautiful
            return operations[tokens[op]](parse_expression(index+1), \
                                          parse_expression(op+1))
    return parse_expression(0)

if __name__ == '__main__':
    doctest.testmod()
    #x = Var('x')
    #y = Var('y')
    # z = 2*x - x*y + 3*y
    # print(z.deriv('y'))  # unsimplified, but the following gives us (2 - y)
    # #2 * 1 + x * 0 - (x * 0 + y * 1) + 3 * 0 + y * 0
    # print(z.deriv('y').simplify())

    # expr = x+y*90.9+(2*(y-3)*((y-3)*2-3)*(y-2))-2.01*x/(y+(-3)*y/(x-3))
    # print(expr)
    # print(tokenize(str(expr)))

    print(repr(2 ** Var('x')))
    #Pow(Num(2), Var('x'))
    ex = expression('(x ** 2)')
    print(ex.deriv('x'))
    #Mul(Mul(Num(2), Pow(Var('x'), Sub(Num(2), Num(1)))), Num(1))
    print(ex.deriv('x').simplify())
    #2 * x
    print(Pow(Add(Var('x'), Var('y')), Num(1)))
    #(x + y) ** 1
    print(Pow(Add(Var('x'), Var('y')), Num(1)).simplify())
    #x + y
