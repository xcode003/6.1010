"""
6.1010 Spring '23 Lab 8: SAT Solver
"""

#!/usr/bin/env python3

import sys
import typing
import doctest

sys.setrecursionlimit(10_000)
# NO ADDITIONAL IMPORTS


def satisfying_assignment(formula):
    """
    Find a satisfying assignment for a given CNF formula.
    Returns that assignment if one exists, or None otherwise.

    # >>> satisfying_assignment([])
    # {}
    # >>> x = satisfying_assignment([[('a', True),
    ('b', False), ('c', True)]])
    # >>> x.get('a', None) is True or x.get('b', None)
    is False or x.get('c', None) is True
    # True
    # >>> satisfying_assignment([[('a', True)], [('a', False)]])
    """
    # base cases:
    # - empty list CNF -> always satisfied
    #   -> soltion is good
    # - nested empty list CNF -> never satisfied
    #   -> solution not possible
    # backtracking chooses value for one variable, and calls
    # recursively with simplified formula (takes into account
    # how variable is assigned)
    # -----------------
    # efficiency modifications:
    # handles unit cases faster
    # -----------------
    # print(f'Level: {i}')

    # reduction
    # repeatedly finds unit cases and
    # propagates them through formula
    assignment = None
    partial_result = {}
    while assignment != {}:
        assignment = {}
        for clause in formula:
            if len(clause) == 1:
                assignment[clause[0][0]] = clause[0][1]
        if assignment:
            formula = new_formula(formula, assignment)
            partial_result = partial_result | assignment

    # base case
    if formula == [[]]:
        return None
    # if formula is [] from True/False backtracking below,
    # partial result will be {} and variable data will be
    # added in the upper recursive call. If partial_result
    # is not {}, then there was some reduction that resulted in {}
    if formula == []:
        return {} | partial_result

    # selects first variable in formula
    assignment[formula[0][0][0]] = True
    true_result = satisfies_helper(formula, assignment, partial_result)
    if true_result:
        return true_result

    # otherwise switches choice
    assignment[formula[0][0][0]] = False
    false_result = satisfies_helper(formula, assignment, partial_result)
    if false_result:
        return false_result
    return None


def satisfies_helper(formula, modifications, initial_data=None):
    """
    Takes in a CNF formula, a dictionary of
    variable assignments, and returns a
    solution mapping; None if no solution exists
    """
    initial_data = {} if initial_data is None else initial_data
    # all vars in modifications should be removed from formula
    new_conditions = new_formula(formula, modifications)
    partial_solution = satisfying_assignment(new_conditions)
    return (
        initial_data.copy() | modifications.copy() | partial_solution
        if partial_solution is not None
        else None
    )


def new_formula(formula, vars_update):
    """
    Takes in a CNF formula, and a dict of
    boolean variable assignments

    Returns a new CNF formula that uses
    the variable assignment to reduce
    the given formula
    -> [] indicates always fulfilled CNF
    -> list  indicates never fulfilled CNF
    # (a OR b OR !c) AND (c or d)
    """
    # form = [[('a', True), ('b', True), ('c', False)], [('c', True), ('d', True)]]
    # ('c', False)
    updated_formula = []
    for clause in formula:
        new_clause = []
        literal_count = 0
        zero_clause = False
        for literal in clause:
            if literal[0] in vars_update.keys():
                # if any var_update aligns with literal,
                if literal[1] == vars_update[literal[0]]:
                    # clause -> 1, so clause disappears
                    # not added to updated_formula
                    new_clause = []
                    break
                else:  # if var_update does not align with literal,
                    literal_count += 1
                    # removing all literals in clause -> clause == 0
                    if literal_count == len(clause):
                        zero_clause = True
                    continue
            else:
                new_clause.append(literal)  # -> tuple, no aliasing
        # -> does not append clause if clause cancels out
        if new_clause:
            updated_formula.append(new_clause)
        elif zero_clause:
            return [[]]
    return updated_formula


def sudoku_board_to_sat_formula(sudoku_board):
    """
    Generates a SAT formula that, when solved, represents a solution to the
    given sudoku board.  The result should be a formula of the right form to be
    passed to the satisfying_assignment function above.
    """
    size = len(sudoku_board)
    # list of numbers that have been already placed on the board
    placed_numbers = []

    tile_exclusivity = []
    row_exclusivity = []
    col_exclusivity = []

    square_locs = {}
    square_exclusivity = []
    # although we are iterating row-col,
    # because row = col (guarantee),
    # the col and row rules can
    # be processed at the same time
    for n1 in range(size):
        for n2 in range(size):
            # n1, n2 treated as row, col
            # recording all tiles with a filled
            # number for later reduction
            num = sudoku_board[n1][n2]
            if num != 0:
                placed_numbers.append(create_board_var(n1, n2, num))
            # one num per tile rule:
            tile_exclusivity += create_single_tile_rules(n1, n2, size)
            # ---------------
            # n2 treated as 'n', n1 treated as row
            # one num per row rule:
            row_exclusivity += create_row_col_rules(n2 + 1, size, row=n1)
            # ---------------
            # n2 treated as 'n', n1 treated as col
            # one num per col rule:
            col_exclusivity += create_row_col_rules(n2 + 1, size, col=n1)

            # assigns locations to corresponding sub-square
            square_locs.setdefault(
                (n1 // (size**0.5), n2 // (size**0.5)), []
            ).append((n1, n2))

    # construct square rules
    for _, locs in square_locs.items():
        square_exclusivity += create_single_square_rules(locs, size)

    fixed_num_rule = [[(var_, True)] for var_ in placed_numbers]

    # 'and' all rules
    total_rules = (
        tile_exclusivity
        + row_exclusivity
        + col_exclusivity
        + square_exclusivity
        + fixed_num_rule
    )

    return total_rules


def create_single_square_rules(locs, n):
    """
    Given a list of locations in an nxn
    playing square, and the size n,
    returns the ruleset requiring number
    exclusivity in playing squares
    """
    rules = []
    for i in range(n):
        square_num_vars = [
            create_board_var(coord[0], coord[1], i + 1) for coord in locs
        ]
        rules += singular_presence_from_list(square_num_vars)
    return rules


def create_row_col_rules(n, size, row=None, col=None):
    """
    Given either the static row or col, the
    size of the nxn board, and a number n
    within that size, returns the rules
    for the correpsonding row/col
    (if row is defined, rules are for row)
    (if col is defined, rules are for col)
    """
    # both shouldn't be None
    assert row is not None or col is not None
    vars_ = None
    if row is not None:
        vars_ = [create_board_var(row, c, n) for c in range(size)]
    else:
        vars_ = [create_board_var(r, col, n) for r in range(size)]
    return singular_presence_from_list(vars_)


def create_single_tile_rules(row, col, n):
    """
    Given the coordinates of a tile,
    and the size of the nxn board,
    returns the number exclusivity rule
    """
    variables = [create_board_var(row, col, num + 1) for num in range(n)]
    return singular_presence_from_list(variables)


def singular_presence_from_list(variables):
    """
    Given a list of variables, returns a
    CNF format rule that restricts only
    one of those variables to be True
    """
    or_chain = [[(var_, True) for var_ in variables]]
    and_chain = and_of_all_or_combinations(variables, False)
    return or_chain + and_chain


def and_of_all_or_combinations(variables, value):
    """
    Given a list of variables, returns
    a list of all 'or' expressions
    linking two 'not'-ted variables
    """
    combos = []
    for i_1 in range(len(variables)):
        for i_2 in range(i_1 + 1, len(variables)):
            combos.append([(variables[i_1], value), (variables[i_2], value)])
    return combos


def create_board_var(row, col, num):
    """
    Given the location of a tile, and a
    number, returns a string variable name
    to represent that number on that tile
    """
    return f"x{row}_{col}_{num}"


def assignments_to_sudoku_board(assignments, n):
    """
    Given a variable assignment as given by satisfying_assignment, as well as a
    size n, construct an n-by-n 2-d array (list-of-lists) representing the
    solution given by the provided assignment of variables.

    If the given assignments correspond to an unsolvable board, return None
    instead.
    """
    board = []
    if assignments is not None:
        for r in range(n):
            row = []
            for c in range(n):
                number = [
                    num
                    for num in range(1, n + 1)
                    if assignments[create_board_var(r, c, num)] is True
                ]
                row.append(number[0])
            board.append(row)
        return board
    return None


if __name__ == "__main__":
    _doctest_flags = doctest.NORMALIZE_WHITESPACE | doctest.ELLIPSIS
    doctest.testmod(optionflags=_doctest_flags)

    grid = [
        [0, 8, 0, 0, 0, 0, 0, 9, 0],  # https://sudoku.com/expert/ # works
        [0, 1, 0, 0, 8, 6, 3, 0, 2],
        [0, 0, 0, 3, 1, 0, 0, 0, 0],
        [0, 0, 4, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 5],
        [0, 0, 0, 2, 6, 1, 0, 0, 4],
        [0, 0, 0, 5, 4, 0, 0, 0, 6],
        [3, 0, 9, 0, 0, 0, 8, 0, 0],
        [2, 0, 0, 0, 0, 0, 0, 0, 0],
    ]
    conditions = sudoku_board_to_sat_formula(grid)
    solution = satisfying_assignment(conditions)

    # debugging
    # locs = {}
    # for n1 in range(len(grid)):
    #     for n2 in range(len(grid)):
    #         locs[f'x{n1}_{n2}_'] = []
    # for key, val in solution.items():
    #     locs[key[:-1]].append((key, val))
    #     if key[-1] == '0':
    #         print('I found a zero')
    # for key in locs.keys():
    #     if len(locs[key]) != 4:
    #         print(locs[key])
    # print([len(locs[key]) for key in locs.keys()])

    new_board = assignments_to_sudoku_board(solution, len(grid))

    if new_board is not None:
        for row_ in new_board:
            print(row_)
    else:
        print("No solution")

    # ----------------------

    # rules = [
    #     [('a', True), ('b', True), ('c', True)],
    #     [('a', False)],
    #     [('d', False), ('e', True), ('a', True), ('g', True)],
    #     [('h', False), ('c', True), ('a', False), ('f', True)]
    # ]

    # print(satisfying_assignment(rules))

    # ----------------------
