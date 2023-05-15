"""
6.1010 Spring '23 Lab 11: LISP Interpreter Part 1
"""
#!/usr/bin/env python3

import sys
import doctest

sys.setrecursionlimit(20_000)

# NO ADDITIONAL IMPORTS!

#############################
# Scheme-related Exceptions #
#############################


class SchemeError(Exception):
    """
    A type of exception to be raised if there is an error with a Scheme
    program.  Should never be raised directly; rather, subclasses should be
    raised.
    """

    pass


class SchemeSyntaxError(SchemeError):
    """
    Exception to be raised when trying to evaluate a malformed expression.
    """

    pass


class SchemeNameError(SchemeError):
    """
    Exception to be raised when looking up a name that has not been defined.
    """

    pass


class SchemeEvaluationError(SchemeError):
    """
    Exception to be raised if there is an error during evaluation other than a
    SchemeNameError.
    """

    pass


############################
# Tokenization and Parsing #
############################


def number_or_symbol(value):
    """
    Helper function: given a string, convert it to an integer or a float if
    possible; otherwise, return the string itself

    >>> number_or_symbol('8')
    8
    >>> number_or_symbol('-5.32')
    -5.32
    >>> number_or_symbol('1.2.3.4')
    '1.2.3.4'
    >>> number_or_symbol('x')
    'x'
    """
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def tokenize(source):
    """
    Splits an input string into meaningful tokens (left parens, right parens,
    other whitespace-separated values).  Returns a list of strings.

    Arguments:
        source (str): a string containing the source code of a Scheme
                      expression
    """
    # newline split
    rough_tokens = source.split("\n")
    # trim tokens from comments
    # replace () with space separated value
    # split on remaining whitespace
    tokens = []
    for tok in rough_tokens:
        if ";" in tok:
            tok = tok[: tok.find(";")]
        if "(" in tok:
            tok = tok.replace("(", " ( ")
        if ")" in tok:
            tok = tok.replace(")", " ) ")
        tokens += tok.split()
    while "" in tokens:
        tokens.remove("")
    return tokens


def parse(tokens):
    """
    Parses a list of tokens, constructing a representation where:
        * symbols are represented as Python strings
        * numbers are represented as Python ints or floats
        * S-expressions are represented as Python lists

    Arguments:
        tokens (list): a list of strings representing tokens
    """

    # think about:
    # (a
    # a(
    # a
    # (
    def find_expr_end(elements):
        """
        Takes in a list of tokens,
        assuming the first is an open
        parentheses, and returns the
        index of the corresponding
        closing parentheses

        Will always find a corresponding
        parentheses, since there is an
        initial check for parentheses
        matching in number
        """
        index = 0
        l_paren = 1
        r_paren = 0
        while l_paren != r_paren:
            index += 1
            if elements[index] == "(":
                l_paren += 1
            elif elements[index] == ")":
                r_paren += 1
        return index

    def recursive_parse(elements, wrapped=False):
        """
        Assumes equal parentheses
        """
        parsed = []  # wrapping parentheses
        if len(elements) > 0:
            if len(elements) == 1:
                # wrapped word
                if wrapped:
                    return [number_or_symbol(elements[0])]
                # unwrapped word
                return number_or_symbol(elements[0])
            else:
                sub_skip = -1
                for index, tok_ in enumerate(elements):
                    if index > sub_skip:
                        if tok_ == "(":
                            sub_skip = index + find_expr_end(elements[index:])
                            if sub_skip != len(elements) - 1 and not wrapped:
                                # raise error
                                # situation like this, not fully wrapped:
                                # (a, b), c, d
                                # )a(
                                raise SchemeSyntaxError("Invalid expression")
                            result = recursive_parse(
                                elements[index + 1 : sub_skip], True
                            )
                            if result != []:
                                # unwrapped list
                                if (
                                    isinstance(result, list) and not wrapped
                                ):  # takes care of wrapping inside
                                    return result
                                # wrapped list, self wrapped/unwrapped word
                                parsed.append(result)
                            elif (
                                wrapped
                            ):  # empty parentheses in code (not single symbol)
                                parsed.append(result)
                        elif wrapped:
                            parsed.append(number_or_symbol(tok_))
                        else:
                            # a, b, c
                            raise SchemeSyntaxError("Invalid expression")
        return parsed
        # else: returns None and does not add anything to list from parent call

    if tokens == []:
        raise SchemeSyntaxError("No tokens")
    if tokens.count("(") == tokens.count(")"):
        return recursive_parse(tokens)
    else:
        raise SchemeSyntaxError
        # error with missing parentheses


##########################
# Frame and Lambda Funcs #
##########################


class Frame:
    '''
    A representation of a frame for
    organizing data access, including
    for function calls
    '''
    def __init__(self, parent=None, objects=None):
        self.parent = parent
        self.objects = objects if objects is not None else {}
        self.name = "built-in"

    def set_name(self, string):
        self.name = string

    def __getitem__(self, item):
        if item in self.objects:
            return self.objects[item]
        elif self.parent is not None:
            return self.parent[item]
        else:
            raise SchemeNameError(f"Symbol '{item}' is not in {self.name} frame")

    def __setitem__(self, item, value):
        self.objects[item] = value


class Function:
    """
    Defines a new function object
    parameters should be a list
    """

    def __init__(self, op_frame, parameters, expression):
        self.op_frame = op_frame
        self.parameters = parameters
        self.expression = expression

    def __call__(self, inputs):
        if len(inputs) != len(self.parameters):
            # error in num vars
            raise SchemeEvaluationError("number of vars do not aling with parameters")
        else:
            vars_ = {param: val for param, val in zip(self.parameters, inputs)}
            func_frame = Frame(parent=self.op_frame, objects=vars_)
            return evaluate(self.expression, frame=func_frame)


######################
# Built-in Functions #
######################


def mult(args):
    product = args[0]
    for term in args[1:]:
        product *= term
    return product


def div(args):
    result = args[0]
    for term in args[1:]:
        result /= term
    return result


scheme_builtins = {
    "+": sum,
    "-": lambda args: -args[0] if len(args) == 1 else (args[0] - sum(args[1:])),
    "*": mult,
    "/": div,
}


##############
# Evaluation #
##############


def evaluate(tree, frame=None):
    """
    Evaluate the given syntax tree according to the rules of the Scheme
    language.

    Arguments:
        tree (type varies): a fully parsed expression, as the output from the
                            parse function
    """

    # support for parent frames
    if frame is None:
        par = Frame(objects=scheme_builtins)
        frame = Frame(parent=par)

    if isinstance(tree, (int, float)):
        return tree
    elif isinstance(tree, str):  # gets value from frame (var, func)
        return frame[tree]
    # list
    elif tree[0] == "define":
        if len(tree[1:]) == 2:
            if isinstance(tree[1], list):
                # create function (short definition)
                result = Function(frame, tree[1][1:], tree[2])
                frame[tree[1][0]] = result
            else:
                result = evaluate(tree[2], frame)
                frame[tree[1]] = result
            return result
        else:
            pass
            # error, not correct num args for var definition
    elif tree[0] == "lambda":
        # create function
        try:
            return Function(frame, tree[1], tree[2])
        except:
            raise SchemeEvaluationError("not enough args")
    else:
        # call function
        try:
            func = evaluate(tree[0], frame)
            return func([evaluate(param, frame) for param in tree[1:]])
        except TypeError:
            raise SchemeEvaluationError(f"Function is not callable")


def result_and_frame(tree, frame=None):
    if frame is None:
        par = Frame(objects=scheme_builtins)
        frame = Frame(parent=par)
    return evaluate(tree, frame), frame


def repl(verbose=False):
    """
    Read in a single line of user input, evaluate the expression, and print
    out the result. Repeat until user inputs "QUIT"

    Arguments:
        verbose: optional argument, if True will display tokens and parsed
            expression in addition to more detailed error output.
    """
    import traceback

    _, frame = result_and_frame(["+"])  # make a global frame
    while True:
        input_str = input("in> ")
        if input_str == "QUIT":
            return
        try:
            token_list = tokenize(input_str)
            if verbose:
                print("tokens>", token_list)
            expression = parse(token_list)
            if verbose:
                print("expression>", expression)
            output, frame = result_and_frame(expression, frame)
            print("  out>", output)
        except SchemeError as e:
            if verbose:
                traceback.print_tb(e.__traceback__)
            print("Error>", repr(e))


if __name__ == "__main__":
    # code in this block will only be executed if lab.py is the main file being
    # run (not when this module is imported)

    # uncommenting the following line will run doctests from above
    # doctest.testmod()
    repl(True)
    # tokens = tokenize('(expr 1) (expr 2)')
    # print(f'parsing: {tokens}')
    # print(parse(tokens))
