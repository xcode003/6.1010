"""
6.1010 Spring '23 Lab 12: LISP Interpreter Part 2
"""
#!/usr/bin/env python3
import sys

sys.setrecursionlimit(20_000)

# KEEP THE ABOVE LINES INTACT, BUT REPLACE THIS COMMENT WITH YOUR lab.py FROM
# THE PREVIOUS LAB, WHICH SHOULD BE THE STARTING POINT FOR THIS LAB.

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
    """
    A representation of a frame for
    organizing data access, including
    for function calls
    """

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

    def set_item_in_chain(self, item, value, original_name):
        if item in self.objects:
            self.objects[item] = value
        elif self.parent is not None:
            return self.parent.set_item_in_chain(item, value, original_name)
        else:
            raise SchemeNameError(
                f"Symbol '{item}' is not in the frame chain \
                    starting with '{original_name}' frame"
            )


class Function:
    """
    Defines a new function object
    parameters should be a list
    """

    def __init__(self, op_frame, parameters, expression, name="unnamed"):
        self.op_frame = op_frame
        self.parameters = parameters
        self.expression = expression
        self.name = name

    def __call__(self, inputs):
        if len(inputs) != len(self.parameters):
            # error in num vars
            raise SchemeEvaluationError(
                f"number of vars ({len(inputs)}) do not align with \
                    parameters ({self.parameters}) for function ({self.name})"
            )
        else:
            vars_ = dict(zip(self.parameters, inputs))
            func_frame = Frame(parent=self.op_frame, objects=vars_)
            return evaluate(self.expression, frame=func_frame)


class Pair:
    """
    Defines a two element pair cell
    -> to be used in linked lists
    """

    def __init__(self, car, cdr):
        self.car = car
        self.cdr = cdr


######################
# Built-in Functions #
######################


def mult(args):
    """
    Returns the successive
    multiplication of args
    """
    if len(args) == 0:
        return 0
    product = args[0]
    for term in args[1:]:
        product *= term
    return product


def div(args):
    """
    Returns the successive
    division of args
    """
    if len(args) == 0:
        return 0
    result = args[0]
    for term in args[1:]:
        result /= term
    return result


def make_not(arg):
    """
    Returns the negation
    of the passed argument
    """
    if len(arg) != 1:
        raise SchemeEvaluationError("wrong number of arguments for not")
    return not arg[0]

def make_cons(args):
    """
    Constructs and returns
    a cons cell given the
    car and cdr input (args)
    """
    if len(args) != 2:
        raise SchemeEvaluationError("wrong number of arguments for cons")
    return Pair(*args)

def make_car(arg):
    """
    Returns the car of a cons cell
    """
    if len(arg) != 1 or not isinstance(arg[0], Pair):
        raise SchemeEvaluationError(
            "wrong number of arguments for car, or argument is not Pair cell"
        )
    return arg[0].car


def make_cdr(arg):
    """
    Returns the cdr of a cons cell
    """
    if len(arg) != 1 or not isinstance(arg[0], Pair):
        raise SchemeEvaluationError(
            "wrong number of arguments for cdr, or argument is not Pair cell"
        )
    return arg[0].cdr


def make_list(args):
    """
    Constructs a linked list
    using cons cells given a
    list of arguments
    """
    if len(args) == 0:
        return []
    else:
        return Pair(args[0], make_list(args[1:]))


def find_nil(pair):
    """
    Recursively goes down a pair object,
    returning the length and whether or not
    the pair is a linked list
    """
    if isinstance(pair, Pair):
        found, sub_length = find_nil(pair.cdr)
        return found, 1 + sub_length
    elif pair == []:
        return True, 0
    return False, 1  # not cons cell


def is_list(arg):
    """
    Determines whether or not
    an input is a cons cell
    of the proper form to be
    considered a list (chain of
    cons cells with last entry
    being nil == [])
    """
    if len(arg) != 1:
        raise SchemeEvaluationError("wrong number of arguments, expected 1")
    if arg[0] == []:
        return True  # empty linked list
    elif not isinstance(arg[0], Pair):
        return False
    return find_nil(arg[0])[0]


def len_list(arg):
    """
    Returns the length of
    a linked list; may not be
    a properly formed linked list
    """
    if len(arg) != 1:
        raise SchemeEvaluationError("wrong number of arguments, expected 1")
    found, length = find_nil(arg[0])
    if not found:
        raise SchemeEvaluationError("input is not a list")
    return length


def get_list_index(args):
    """
    Returns element at index in
    linked list, or cons cell

    -> Args may not be proper linked list
    """
    def find_element(pair, index):
        """
        Recursively searches for element
        at index in pair object,
        assuming index >= 0
        """
        if pair == []:
            raise SchemeEvaluationError("index out of bounds for list")
        elif index == 0:
            return pair.car
        return find_element(pair.cdr, index - 1)

    if len(args) != 2:
        raise SchemeEvaluationError("wrong number of arguments, expected 2")
    if not isinstance(args[1], int) or args[1] < 0:
        raise SchemeEvaluationError("index must be an integer >= 0")
    if is_list([args[0]]):
        return find_element(args[0], args[1])
    # not list, but cons cell (even if nested cons cell?)
    elif isinstance(args[0], Pair):  # index is good at this point
        if args[1] == 0:
            return args[0].car
        raise SchemeEvaluationError("cannot access index > 0 for a non-list cons cell")
    raise SchemeEvaluationError("error, first arg must be a list or cons cell")


# append helper
def shallow_copy(arg, end_ref):
    """
    Takes in a argument, may
    not be a proper linked list,
    returns shallow copy

    end_reference refers to what the
    last item in the shallow
    copy should point to (another
    list for append)
    """
    if arg == []:
        return end_ref
    if not isinstance(arg, Pair) or (
        not isinstance(arg.cdr, Pair) and not arg.cdr == []
    ):
        raise SchemeEvaluationError("part of input was not a proper list")
    return Pair(arg.car, shallow_copy(arg.cdr, end_ref))


def append_lists(args):
    """
    Takes in a list of objects,
    not necessarily all proper
    linked lists, and appends
    a shallow copy of each,
    in order, to return
    """
    if len(args) == 0:
        return []
    new_list = []
    for index in range(len(args) - 1, -1, -1):
        new_list = shallow_copy(args[index], new_list)
    return new_list


def modify_with_func(args, modification_logic, params=2):
    """
    Takes in a function and a list through args,
    returns new list modified by function
    according to the logic provided by
    modification_logic
    """
    if len(args) != params:
        raise SchemeEvaluationError("error, incorrect number of args")

    func = args[0]
    list_ = args[1]
    init_val = None

    if not callable(func):
        raise SchemeEvaluationError("first argument should be function")

    if params == 3:
        init_val = args[2]

    def recursive_result(sub_list, sofar_val=init_val):
        """
        Returns new value modified in
        some way by func above, determined
        by modification_logic function
        """
        if sub_list == []:
            if not sofar_val:
                return []
            else:
                # returns initial value if
                # empty list to reduce with function
                return sofar_val

        # check for properly formed linked list while exploring list
        if not isinstance(sub_list, Pair) or (
            not isinstance(sub_list.cdr, Pair) and sub_list.cdr != []
        ):
            raise SchemeEvaluationError("part of input was not a proper list")

        # is this passed function maintaining the original closure?
        return modification_logic(func, sub_list, recursive_result, sofar_val)

    return recursive_result(list_)


def map_logic(func, sub_list, recursive_result, _):
    """
    Logic to use within recursive result
    linked list exploration method,
    in order to build a list where
    each element is transformed using
    a function
    """
    return Pair(func([sub_list.car]), recursive_result(sub_list.cdr))


def filter_logic(func, sub_list, recursive_result, _):
    """
    Logic to use within recursive result
    linked list exploration method,
    in order to build a list using
    a function as filter (for True)
    """
    if func([sub_list.car]):
        return Pair(sub_list.car, recursive_result(sub_list.cdr))
    # skips element
    return recursive_result(sub_list.cdr)


# sofar_val here is updated tracking value
def reduce_logic(func, sub_list, recursive_result, sofar_val):
    """
    Logic to use within recursive result
    linked list exploration method,
    in order to successively apply a
    function to a list, starting with an initial value
    """
    new_base_val = func([sofar_val, sub_list.car])
    return recursive_result(sub_list.cdr, new_base_val)


# debugging helper
def print_list(arg):
    """
    Prints the elements
    in a linked list, in order
    """
    if arg == []:
        print("nil")
        return
    elif isinstance(arg, (float, int)):
        print(f"{arg} ")
        return
    else:
        print(f"{arg.car} ", end="")
        print_list(arg.cdr)
        return


# function inputs will always be lists, even if size 1
scheme_builtins = {
    "+": sum,
    "-": lambda args: -args[0] if len(args) == 1 else (args[0] - sum(args[1:])),
    "*": mult,
    "/": div,
    "not": make_not,
    "cons": make_cons,
    "car": make_car,
    "cdr": make_cdr,
    "list": make_list,
    "list?": is_list,
    "length": len_list,
    "list-ref": get_list_index,
    "append": append_lists,
    "map": lambda args: modify_with_func(args, map_logic),
    "filter": lambda args: modify_with_func(args, filter_logic),
    "reduce": lambda args: modify_with_func(args, reduce_logic, 3),

    #evaluates all arguments, inherently through
    # function call, and returns last expression result
    "begin": lambda args: args[-1],
    "print": lambda arg: print_list(arg[0]),
    "#t": True,
    "#f": False,
    "nil": [],
}


##############
# Evaluation #
##############


def evaluate_file(file_name, run_frame=None):
    """
    Takes in the name of a file, optional
    argument for frame, and evaluates the
    single expression in the file
    """
    with open(file_name) as f_:
        expr = f_.read()
    expr = parse(tokenize(expr))
    return evaluate(expr, run_frame)


def evaluate(tree, frame=None):
    """
    Evaluate the given syntax tree according to the rules of the Scheme
    language.

    Arguments:
        tree (type varies): a fully parsed expression, as the output from the
                            parse function
    """
    # creates new frame if no frame passed in
    if frame is None:
        par = Frame(objects=scheme_builtins)
        frame = Frame(parent=par)

    #handles special forms with a tailored set of functions
    comparisons = ("equal?", ">", ">=", "<", "<=", "and", "or")
    special_forms = (
        #conditionals -> [:7]
        "equal?",
        ">",
        ">=",
        "<",
        "<=",
        "and",
        "or",
        #other special forms
        "define",
        "lambda",
        "if",
        "del",
        "let",
        "set!"
    )
    other_special_form_funcs = {
        "define":handle_define,
        "lambda":handle_lambda,
        "if":handle_if,
        "del":handle_del,
        "let":handle_let,
        "set!":handle_set
    }

    if isinstance(tree, (int, float)):
        return tree
    elif isinstance(tree, str):  # gets value from frame (var, func)
        return frame[tree]
    # list input
    else:
        if len(tree) == 0:
            raise SchemeEvaluationError("empty expression")
        elif tree[0] in special_forms:
            if tree[0] in special_forms[:7]:
                return handle_comparisons(tree, frame)
            else:
                return other_special_form_funcs[tree[0]](tree, frame)
        else:
            # function call
            func = evaluate(tree[0], frame)

            if callable(func):
                return func([evaluate(param, frame) for param in tree[1:]])
            else:
                raise SchemeEvaluationError("Function is not callable")

def handle_comparisons(tree, run_frame):
    """
    Logic for evaluating
    comparison special forms
    """
    if tree[0] != "or":
        if tree[0] != "and":
            if tree[0] == "equal?":
                operation = lambda x, y: x == y
            elif tree[0] == ">":
                operation = lambda x, y: x > y
            elif tree[0] == ">=":
                operation = lambda x, y: x >= y
            elif tree[0] == "<":
                operation = lambda x, y: x < y
            elif tree[0] == "<=":
                operation = lambda x, y: x <= y
            return short_circuit(tree, run_frame, operation)
        else:  # and
            # accounts for first value in list
            return short_circuit(tree, run_frame, style="and")
    elif tree[0] == "or":
        # accounts for first value in list
        return short_circuit(tree, run_frame, style="or")

def handle_define(tree, run_frame):
    """
    Logic for evaluating
    define special form
    """
    # define variable
    if len(tree[1:]) == 2:
            if isinstance(tree[1], list):
                # create function (short definition)
                # passes arguments and expression for function body
                result = Function(run_frame, tree[1][1:], tree[2], tree[1][0])
                run_frame[tree[1][0]] = result
            else:
                result = evaluate(tree[2], run_frame)
                run_frame[tree[1]] = result
            return result
    else:
        # error, not correct num args for var definition
        raise SchemeEvaluationError(
            f"not correct number of args for \
                variable definition (should be 2, but received {len(tree[1:])})"
        )

def handle_lambda(tree, run_frame):
    """
    Logic for evaluating
    lambda special form
    """
    # create function
    #return Function(run_frame, tree[1], tree[2])
    try:
        return Function(run_frame, tree[1], tree[2])
    except:
        raise SchemeEvaluationError("not enough args provided")

def handle_if(tree, run_frame):
    """
    Logic for evaluating
    conditional special form
    """
    # conditional
    if len(tree[1:]) != 3:
        raise SchemeEvaluationError(
            "wrong number of expressions for if, expected 3"
        )
    predicate = evaluate(tree[1], run_frame)
    if predicate:
        return evaluate(tree[2], run_frame)
    else:
        return evaluate(tree[3], run_frame)

def handle_del(tree, run_frame):
    """
    Logic for evaluating
    del special form (local
    scope)
    """
    if len(tree[1:]) != 1:
        raise SchemeEvaluationError("expected one argument for del")
    if tree[1] in run_frame.objects:
        return run_frame.objects.pop(tree[1])
    raise SchemeNameError("variable is not bound in local frame")

def handle_let(tree, run_frame):
    """
    Logic for evaluating
    let special form (local
    scope define + expression eval)
    """
    if len(tree[1:]) != 2:
        raise SchemeEvaluationError("expected two arguments, vars and expression")
    bindings = {}
    for var in tree[1]:
        if len(var) != 2:
            raise SchemeEvaluationError(
                "variable definitions should consist of the variable \
                    name followed by the value to bind"
            )
        bindings[var[0]] = evaluate(var[1], run_frame)
    new_frame = Frame(parent=run_frame, objects=bindings)
    return evaluate(
        tree[2], new_frame
    )  # evaluate expression in newly created local frame

def handle_set(tree, run_frame):
    """
    Logic for evaluating
    set! special form
    (unrestricted scope redined)
    """
    if len(tree[1:]) != 2:
        raise SchemeEvaluationError(
            "expected two arguments for set!, variable name and new binding"
        )
    new_value = evaluate(tree[2], run_frame)
    run_frame.set_item_in_chain(tree[1], new_value, run_frame.name)
    return new_value

def short_circuit(tree, run_frame, operation=None, style="and"):
    """
    Evaluates a short-circuited operation
    on tree[1:], based on style (short-circuit type).

    'and' style -> returns False if one call returns False;
    > otherwise follows 'and'

    'or' -> returns True if one call returns True;
    > otherwise follows 'or'
    """
    if len(tree[1:]) < 2:
        raise SchemeEvaluationError("not enough args")
    # compares adjacent pairs of expressions
    if style == "and":
        if operation:
            # breaks out to return with one False comparison
            # skips first tree element in zip
            for item1, item2 in zip(tree[1:-1], tree[2:]):
                if not operation(
                    evaluate(item1, run_frame), evaluate(item2, run_frame)
                ):
                    return False
            return True
        else:  # traditional and
            # breaks out to return with one False comparison
            # skips first tree element
            for item in tree[1:]:
                if not evaluate(item, run_frame):
                    return False
            return True
    else:  # style = 'or'
        if operation:
            # breaks out to return with one True comparison
            # skips first tree element in zip
            for item1, item2 in zip(tree[1:-1], tree[2:]):
                if operation(evaluate(item1, run_frame), evaluate(item2, run_frame)):
                    return True
            return False
        else:  # traditional or
            # breaks out to return with one True comparison
            # skips first tree element
            for item in tree[1:]:
                if evaluate(item, run_frame):
                    return True
            return False


def result_and_frame(tree, frame=None):
    """
    Evaluates a tree in the given frame,
    child of global frame if none provided,
    and returns evaluation + frame
    used in evalaution
    """
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

    # run files passed in command line before starting repl!
    files = sys.argv[1:]
    for f_ in files:
        evaluate_file(f_, frame)

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
