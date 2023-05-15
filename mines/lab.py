"""
6.1010 Spring '23 Lab 7: Mines
"""

#!/usr/bin/env python3

import typing
import doctest

# NO ADDITIONAL IMPORTS ALLOWED!


def dump(game):
    """
    Prints a human-readable version of a game (provided as a dictionary)
    """
    for key, val in sorted(game.items()):
        if isinstance(val, list) and val and isinstance(val[0], list):
            print(f"{key}:")
            for inner in val:
                print(f"    {inner}")
        else:
            print(f"{key}:", val)


# 2-D IMPLEMENTATION


def new_game_2d(num_rows, num_cols, bombs):
    """
    Start a new game.

    Return a game state dictionary, with the 'dimensions', 'state', 'board' and
    'hidden' fields adequately initialized.

    Parameters:
       num_rows (int): Number of rows
       num_cols (int): Number of columns
       bombs (list): List of bombs, given in (row, column) pairs, which are
                     tuples

    Returns:
       A game state dictionary

    >>> dump(new_game_2d(2, 4, [(0, 0), (1, 0), (1, 1)]))
    board:
        ['.', 3, 1, 0]
        ['.', '.', 1, 0]
    bomb_locs: [(0, 0), (1, 0), (1, 1)]
    dimensions: (2, 4)
    hidden:
        [True, True, True, True]
        [True, True, True, True]
    state: ongoing
    """
    # board = []
    # hidden = []
    # for row in range(num_rows):
    #     board_row = []
    #     for col in range(num_cols):
    #         if (row, col) in bombs:
    #             board_row.append(".")
    #         else:
    #             board_row.append(count_adjacent_bombs(bombs, row, col))
    #     board.append(board_row)
    #     hidden.append([True]*num_cols)
    # return {
    #     "dimensions": (num_rows, num_cols),
    #     "board": board,
    #     "hidden": hidden,
    #     "state": "ongoing"
    # }
    return new_game_nd((num_rows, num_cols), bombs)


# helper function from before refactoring
# def count_adjacent_bombs(bombs, row, col):
#     '''
#     Returns the number of adjacent bombs
#     to a given location
#     '''
#     count = 0
#     for row_offset in [-1, 0, 1]:
#         for col_offset in [-1, 0, 1]:
#             if (row + row_offset, col + col_offset) in bombs:
#                 count += 1
#     return count


def dig_2d(game, row, col):
    """
    Reveal the cell at (row, col), and, in some cases, recursively reveal its
    neighboring squares.

    Update game['hidden'] to reveal (row, col).  Then, if (row, col) has no
    adjacent bombs (including diagonally), then recursively reveal (dig up) its
    eight neighbors.  Return an integer indicating how many new squares were
    revealed in total, including neighbors, and neighbors of neighbors, and so
    on.

    The state of the game should be changed to 'defeat' when at least one bomb
    is revealed on the board after digging (i.e. game['hidden'][bomb_location]
    == False), 'victory' when all safe squares (squares that do not contain a
    bomb) and no bombs are revealed, and 'ongoing' otherwise.

    Parameters:
       game (dict): Game state
       row (int): Where to start digging (row)
       col (int): Where to start digging (col)

    Returns:
       int: the number of new squares revealed

    >>> game = {'dimensions': (2, 4),
    ...         'board': [['.', 3, 1, 0],
    ...                   ['.', '.', 1, 0]],
    ...         'bomb_locs': [(0, 0), (1, 0), (1, 1)],
    ...         'hidden': [[True, False, True, True],
    ...                  [True, True, True, True]],
    ...         'state': 'ongoing'}
    >>> dig_2d(game, 0, 3)
    4
    >>> dump(game)
    board:
        ['.', 3, 1, 0]
        ['.', '.', 1, 0]
    bomb_locs: [(0, 0), (1, 0), (1, 1)]
    dimensions: (2, 4)
    hidden:
        [True, False, False, False]
        [True, True, False, False]
    state: victory

    >>> game = {'dimensions': [2, 4],
    ...         'board': [['.', 3, 1, 0],
    ...                   ['.', '.', 1, 0]],
    ...         'bomb_locs': [(0, 0), (1, 0), (1, 1)],
    ...         'hidden': [[True, False, True, True],
    ...                  [True, True, True, True]],
    ...         'state': 'ongoing'}
    >>> dig_2d(game, 0, 0)
    1
    >>> dump(game)
    board:
        ['.', 3, 1, 0]
        ['.', '.', 1, 0]
    bomb_locs: [(0, 0), (1, 0), (1, 1)]
    dimensions: [2, 4]
    hidden:
        [False, False, True, True]
        [True, True, True, True]
    state: defeat
    """
    # revealed = 0
    # if game["state"] != "defeat" and game["state"] != "victory":
    #     # else, ongoing
    #     # count a revealed tile only if it is initially hidden
    #     # recursive reveal here
    #     revealed = recursive_reveal(game, row, col)
    #     if game["board"][row][col] == ".":
    #         game["state"] = "defeat"
    #     else:
    #         hidden_squares = 0
    #         for r in range(game["dimensions"][0]):
    #             for c in range(game["dimensions"][1]):
    #                 # hidden tiles which are not bombs
    #                 if game["board"][r][c] != "." and game["hidden"][r][c]:
    #                     hidden_squares += 1
    #         if hidden_squares == 0:
    #             game["state"] = "victory"
    # return revealed
    return dig_nd(game, (row, col))


# helper function from before refactoring


# def recursive_reveal(game, row, col):
#     '''
#     Recursively reveals tiles that are marked as 0
#     Assumes that initial location is not bomb
#     Modifies game, and returns number of tiles revealed
#     '''
#     count = 0
#     newly_revealed = False
#     if game['hidden'][row][col] is True:
#         game['hidden'][row][col] = False
#         newly_revealed = True
#         count += 1

#     if newly_revealed and game['board'][row][col] == 0:
#         for r_offset in [-1, 0, 1]:
#             for c_offset in [-1, 0, 1]:
#                 if r_offset != 0 or c_offset != 0:
#                     new_row = row + r_offset
#                     new_col = col + c_offset
#                     if in_range(game['dimensions'], new_row, new_col):
#                         count += recursive_reveal(game, new_row, new_col)
#     return count

#helper function from before refactoring
# def in_range(dimensions, row, col):
#     '''
#     Returns whether or not row col is in range of board
#     '''
#     if 0 <= row < dimensions[0] and 0 <= col < dimensions[1]:
#         return True
#     return False


def render_2d_locations(game, xray=False):
    """
    Prepare a game for display.

    Returns a two-dimensional array (list of lists) of '_' (hidden squares),
    '.' (bombs), ' ' (empty squares), or '1', '2', etc. (squares neighboring
    bombs).  game['hidden'] indicates which squares should be hidden.  If
    xray is True (the default is False), game['hidden'] is ignored and all
    cells are shown.

    Parameters:
       game (dict): Game state
       xray (bool): Whether to reveal all tiles or just the that are not
                    game['hidden']

    Returns:
       A 2D array (list of lists)

    >>> render_2d_locations({'dimensions': (2, 4),
    ...         'state': 'ongoing',
    ...         'board': [['.', 3, 1, 0],
    ...                   ['.', '.', 1, 0]],
    ...         'hidden':  [[True, False, False, True],
    ...                   [True, True, False, True]]}, False)
    [['_', '3', '1', '_'], ['_', '_', '1', '_']]

    >>> render_2d_locations({'dimensions': (2, 4),
    ...         'state': 'ongoing',
    ...         'board': [['.', 3, 1, 0],
    ...                   ['.', '.', 1, 0]],
    ...         'hidden':  [[True, False, True, False],
    ...                   [True, True, True, False]]}, True)
    [['.', '3', '1', ' '], ['.', '.', '1', ' ']]
    """
    # representation = []
    # for row in range(game['dimensions'][0]):
    #     entries = game['board'][row][:]
    #     for col in range(game['dimensions'][1]):
    #         if entries[col] == 0:
    #             entries[col] = ' '
    #         else:
    #             entries[col] = str(entries[col])
    #         if game['hidden'][row][col] and not xray:
    #             entries[col] = '_'
    #     representation.append(entries)
    # return representation
    return render_nd(game, xray)


def render_2d_board(game, xray=False):
    """
    Render a game as ASCII art.

    Returns a string-based representation of argument 'game'.  Each tile of the
    game board should be rendered as in the function
        render_2d_locations(game)

    Parameters:
       game (dict): Game state
       xray (bool): Whether to reveal all tiles or just the ones allowed by
                    game['hidden']

    Returns:
       A string-based representation of game

    >>> render_2d_board({'dimensions': (2, 4),
    ...                  'state': 'ongoing',
    ...                  'board': [['.', 3, 1, 0],
    ...                            ['.', '.', 1, 0]],
    ...                  'hidden':  [[False, False, False, True],
    ...                            [True, True, False, True]]})
    '.31_\\n__1_'
    """
    board = render_2d_locations(game, xray)
    str_board = ""
    for r_index, row in enumerate(board):
        for elem in row:
            str_board += elem
        if r_index < len(board) - 1:
            str_board += "\n"
    return str_board


# N-D IMPLEMENTATION


def new_game_nd(dimensions, bombs):
    """
    Start a new game.

    Return a game state dictionary, with the
    'dimensions', 'state', 'bomb_locs', 'board'
    and 'hidden' fields adequately initialized.


    Args:
       dimensions (tuple): Dimensions of the board
       bombs (list): Bomb locations as a list of tuples, each an
                     N-dimensional coordinate

    Returns:
       A game state dictionary

    >>> g = new_game_nd((2, 4, 2), [(0, 0, 1), (1, 0, 0), (1, 1, 1)])
    >>> dump(g)
    board:
        [[3, '.'], [3, 3], [1, 1], [0, 0]]
        [['.', 3], [3, '.'], [1, 1], [0, 0]]
    bomb_locs: [(0, 0, 1), (1, 0, 0), (1, 1, 1)]
    dimensions: (2, 4, 2)
    hidden:
        [[True, True], [True, True], [True, True], [True, True]]
        [[True, True], [True, True], [True, True], [True, True]]
    state: ongoing
    """
    board = create_nd_board(dimensions, 0)
    set_nd_board_values(board, bombs, dimensions) ##could be slow
    hidden = create_nd_board(dimensions, True)

    return {
        "dimensions": dimensions,
        "state": "ongoing",
        "board": board,
        "hidden": hidden,
        "bomb_locs": bombs,
    }


def create_nd_board(dimensions, value):
    """
    Returns a nested list representing an nd
    board, all values initialized to 0
    >>> a = create_nd_board((2,2,2), 0)
    >>> print(a)
    [[[0, 0], [0, 0]], [[0, 0], [0, 0]]]
    >>> a[0][0][0] = 1
    >>> print(a)
    [[[1, 0], [0, 0]], [[0, 0], [0, 0]]]
    """
    board = [value] * dimensions[0]
    if len(dimensions) == 1:
        return board
    new_dim = tuple(list(dimensions)[1:])
    for _ in range(len(board)):
        board[_] = create_nd_board(new_dim, value)
    return board


def set_nd_board_values(board, bombs, dimensions):
    """
    Modifies nested list (game board)
    so that each element is either a bomb
    or a number representing the correct
    number of adjacent bombs
    """
    set_value(board, bombs, ".")
    bomb_adjacent = set()
    bombs = set(bombs)

    for bomb_loc in bombs:
        # all neighbors not including bomb locations
        for neighbor in get_nd_neighbors(dimensions, bomb_loc) - bombs:
            bomb_adjacent.add(neighbor)

    for mine_neighbor in bomb_adjacent:
        nearby = get_nd_neighbors(dimensions, mine_neighbor)
        num_bombs = len(nearby & bombs)
        set_value(board, mine_neighbor, num_bombs)


# -------------


def set_value(board, location_s, value):
    """
    Recursive helper to go through game board
    and set location(s) to given value
    """
    if isinstance(location_s, list):
        for loc in location_s:
            set_value(board, loc, value)
    else:
        if len(location_s) == 1:  # 1d
            board[location_s[0]] = value
        else:
            set_value(board[location_s[0]], tuple(list(location_s)[1:]), value)


def get_value(board, location):
    """
    Recursive helper to go through game board
    and get value at location
    """
    if len(location) == 1:  # 1d
        return board[location[0]]
    else:
        return get_value(board[location[0]], tuple(list(location)[1:]))


def get_all_coords(dimensions):
    """
    Returns a set of all tuple locations
    in board, given dimensions
    >>> print(get_all_coords((2,2)))
    {(1, 0), (0, 1), (1, 1), (0, 0)}
    """
    all_coords = {(index,) for index in range(dimensions[0])}
    if len(dimensions) == 1:
        return all_coords
    base_coords = get_all_coords(tuple(list(dimensions)[1:]))
    return appended_set_combinations(all_coords, base_coords)


def get_nd_neighbors(dimensions, loc):
    """
    Given a board dimensions and a location,
    returns a set of in-range neighbor
    locations (including the original loc)
    """
    local_neighbors = set()
    for offset in range(-1,2):
        new_loc = loc[0] + offset
        if 0 <= new_loc < dimensions[0]:
            local_neighbors.add((new_loc,))
    if len(loc) == 1:
        return local_neighbors

    new_dim = tuple(list(dimensions)[1:])
    new_loc = tuple(list(loc[1:]))
    sub_neighbors = get_nd_neighbors(new_dim, new_loc)

    return appended_set_combinations(local_neighbors, sub_neighbors)


def appended_set_combinations(set1, set2):
    """
    Takes in a set of tuples, set1
    and set2 which can have n
    dimensionality each

    Returns a set with each tuple
    in set1 appended to the front of
    each tuple in set2 (all appended combinations)
    """
    new_set = set()
    for base in set2:
        for element in set1:
            new_set.add(tuple(list(element) + list(base)))
    return new_set


def get_state(game): ##slow?
    """
    Returns the state of the given game
    'ongoing', 'victory', 'defeat'
    """
    for bomb in game["bomb_locs"]:
        if get_value(game["hidden"], bomb) is False:
            return "defeat"
    all_free_tiles = get_all_coords(game["dimensions"]) - set(game["bomb_locs"])
    for tile in all_free_tiles:
        if get_value(game["hidden"], tile) is True:
            return "ongoing"
    return "victory"

#--------------------------
def dig_nd(game, coordinates):
    """
    Recursively dig up square at coords and neighboring squares.

    Update the hidden to reveal square at coords; then recursively reveal its
    neighbors, as long as coords does not contain and is not adjacent to a
    bomb.  Return a number indicating how many squares were revealed.  No
    action should be taken and 0 returned if the incoming state of the game
    is not 'ongoing'.

    The updated state is 'defeat' when at least one bomb is revealed on the
    board after digging, 'victory' when all safe squares (squares that do
    not contain a bomb) and no bombs are revealed, and 'ongoing' otherwise.

    Args:
       coordinates (tuple): Where to start digging

    Returns:
       int: number of squares revealed

    >>> g = {'dimensions': (2, 4, 2),
    ...      'board': [[[3, '.'], [3, 3], [1, 1], [0, 0]],
    ...                [['.', 3], [3, '.'], [1, 1], [0, 0]]],
    ...      'bomb_locs': [(0, 0, 1), (1, 0, 0), (1, 1, 1)],
    ...      'hidden': [[[True, True], [True, False], [True, True],
    ...                [True, True]],
    ...               [[True, True], [True, True], [True, True],
    ...                [True, True]]],
    ...      'state': 'ongoing'}
    >>> dig_nd(g, (0, 3, 0))
    8
    >>> dump(g)
    board:
        [[3, '.'], [3, 3], [1, 1], [0, 0]]
        [['.', 3], [3, '.'], [1, 1], [0, 0]]
    bomb_locs: [(0, 0, 1), (1, 0, 0), (1, 1, 1)]
    dimensions: (2, 4, 2)
    hidden:
        [[True, True], [True, False], [False, False], [False, False]]
        [[True, True], [True, True], [False, False], [False, False]]
    state: ongoing
    >>> g = {'dimensions': (2, 4, 2),
    ...      'board': [[[3, '.'], [3, 3], [1, 1], [0, 0]],
    ...                [['.', 3], [3, '.'], [1, 1], [0, 0]]],
    ...      'bomb_locs': [(0, 0, 1), (1, 0, 0), (1, 1, 1)],
    ...      'hidden': [[[True, True], [True, False], [True, True],
    ...                [True, True]],
    ...               [[True, True], [True, True], [True, True],
    ...                [True, True]]],
    ...      'state': 'ongoing'}
    >>> dig_nd(g, (0, 0, 1))
    1
    >>> dump(g)
    board:
        [[3, '.'], [3, 3], [1, 1], [0, 0]]
        [['.', 3], [3, '.'], [1, 1], [0, 0]]
    bomb_locs: [(0, 0, 1), (1, 0, 0), (1, 1, 1)]
    dimensions: (2, 4, 2)
    hidden:
        [[True, False], [True, False], [True, True], [True, True]]
        [[True, True], [True, True], [True, True], [True, True]]
    state: defeat
    """
    tiles_revealed = 0
    visited = set()
    def recursive_reveal(tile_loc):
        set_value(game["hidden"], tile_loc, False)
        count = 1
        visited.add(tile_loc)
        if get_value(game["board"], tile_loc) == 0:
            for neighbor in get_nd_neighbors(game["dimensions"], tile_loc):
                # checks to see if neighbor has been visited in this set,
                # or if neighbor has already been dug (nonzero, non-mine tile)
                if neighbor not in visited and get_value(game['hidden'], neighbor) is True:
                    count += recursive_reveal(neighbor)
        return count

    if (
        game["state"] != "victory"
        and game["state"] != "defeat"
        and get_value(game["hidden"], coordinates) is True
    ):
        tiles_revealed = recursive_reveal(coordinates)
        game["state"] = get_state(game)
    return tiles_revealed
#--------------------------

#dig 2d
# revealed = 0
# if game["state"] != "defeat" and game["state"] != "victory":
#     # else, ongoing
#     # count a revealed tile only if it is initially hidden
#     # recursive reveal here
#     revealed = recursive_reveal(game, row, col)
#     if game["board"][row][col] == ".":
#         game["state"] = "defeat"
#     else:
#         hidden_squares = 0
#         for r in range(game["dimensions"][0]):
#             for c in range(game["dimensions"][1]):
#                 # hidden tiles which are not bombs
#                 if game["board"][r][c] != "." and game["hidden"][r][c]:
#                     hidden_squares += 1
#         if hidden_squares == 0:
#             game["state"] = "victory"
# return revealed

#def recursive_reveal(game, row, col):
#     '''
#     Recursively reveals tiles that are marked as 0
#     Assumes that initial location is not bomb
#     Modifies game, and returns number of tiles revealed
#     '''
#     count = 0
#     newly_revealed = False
#     if game['hidden'][row][col] is True:
#         game['hidden'][row][col] = False
#         newly_revealed = True
#         count += 1

#     if newly_revealed and game['board'][row][col] == 0:
#         for r_offset in [-1, 0, 1]:
#             for c_offset in [-1, 0, 1]:
#                 if r_offset != 0 or c_offset != 0:
#                     new_row = row + r_offset
#                     new_col = col + c_offset
#                     if in_range(game['dimensions'], new_row, new_col):
#                         count += recursive_reveal(game, new_row, new_col)
#     return count








def render_nd(game, xray=False):
    """
    Prepare the game for display.

    Returns an N-dimensional array (nested lists) of '_' (hidden squares), '.'
    (bombs), ' ' (empty squares), or '1', '2', etc. (squares neighboring
    bombs).  The game['hidden'] array indicates which squares should be
    hidden.  If xray is True (the default is False), the game['hidden'] array
    is ignored and all cells are shown.

    Args:
       xray (bool): Whether to reveal all tiles or just the ones allowed by
                    game['hidden']

    Returns:
       An n-dimensional array of strings (nested lists)

    >>> g = {'dimensions': (2, 4, 2),
    ...      'board': [[[3, '.'], [3, 3], [1, 1], [0, 0]],
    ...                [['.', 3], [3, '.'], [1, 1], [0, 0]]],
    ...      'hidden': [[[True, True], [True, False], [False, False],
    ...                [False, False]],
    ...               [[True, True], [True, True], [False, False],
    ...                [False, False]]],
    ...      'state': 'ongoing'}
    >>> render_nd(g, False)
    [[['_', '_'], ['_', '3'], ['1', '1'], [' ', ' ']],
     [['_', '_'], ['_', '_'], ['1', '1'], [' ', ' ']]]

    >>> render_nd(g, True)
    [[['3', '.'], ['3', '3'], ['1', '1'], [' ', ' ']],
     [['.', '3'], ['3', '.'], ['1', '1'], [' ', ' ']]]
    """
    printed_board = create_nd_board(game["dimensions"], "_")
    all_locations = get_all_coords(game["dimensions"])
    for loc in all_locations:
        if xray or get_value(game["hidden"], loc) is False:
            value = str(get_value(game["board"], loc))
            if value == "0":
                value = " "
            set_value(printed_board, loc, value)
    return printed_board


if __name__ == "__main__":
    # Test with doctests. Helpful to debug individual lab.py functions.
    _doctest_flags = doctest.NORMALIZE_WHITESPACE | doctest.ELLIPSIS
    doctest.testmod(optionflags=_doctest_flags)  # runs ALL doctests

    # Alternatively, can run the doctests JUST for specified function/methods,
    # e.g., for render_2d_locations or any other function you might want.  To
    # do so, comment out the above line, and uncomment the below line of code.
    # This may be useful as you write/debug individual doctests or functions.
    # Also, the verbose flag can be set to True to see all test results,
    # including those that pass.
    #
    # doctest.run_docstring_examples(
    #    render_2d_locations,
    #    globals(),
    #    optionflags=_doctest_flags,
    #    verbose=False
    # )
