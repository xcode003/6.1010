'''
6.1010 Spring '23 Lab 4: Snekoban Game
'''

import os
import sys
import copy
import json
import pickle

import json
import typing

# NO ADDITIONAL IMPORTS!


direction_vector = {
    'up': (-1, 0),
    'down': (+1, 0),
    'left': (0, -1),
    'right': (0, +1),
}

def new_game(level_description):
    '''
    Given a description of a game state, create and return a game
    representation of your choice.

    The given description is a list of lists of lists of strs, representing the
    locations of the objects on the board (as described in the lab writeup).

    For example, a valid level_description is:

    [
        [[], ['wall'], ['computer']],
        [['target', 'player'], ['computer'], ['target']],
    ]

    The exact choice of representation is up to you; but note that what you
    return will be used as input to the other functions.
    '''
    # iteration 1 -> dict connecting locations to sets of objects
    # level = {}

    # for row in range(len(level_description)):
    #     for col, tile in enumerate(level_description[row]):
    #         level[(row, col)] = set(tile)
    # return level

    # iteration 2 -> dict connecting object types to sets of location tuples
    level = {
        'height': len(level_description),
        'width': len(level_description[0])
    }

    for row in range(len(level_description)):
        for col, tile in enumerate(level_description[row]):
            for element in tile:
                level.setdefault(element, set()).add((row, col))
    return level


def victory_check(game):
    '''
    Given a game representation (of the form returned from new_game), return
    a Boolean: True if the given game satisfies the victory condition, and
    False otherwise.
    '''
    if len(game.get('computer', set())) == 0 or \
        len(game.get('target', set())) == 0:
        return False
    return game['computer'] == game['target']

def step_game(game, direction):
    '''
    Given a game representation (of the form returned from new_game), return a
    new game representation (of that same form), representing the updated game
    after running one step of the game.  The user's input is given by
    direction, which is one of the following: {'up', 'down', 'left', 'right'}.

    This function should not mutate its input.
    '''
    # first, player moves
    move = direction_vector[direction]

    def legal_move_check(board, loc, player=False):
        '''
        Takes in a board and a row, col tuple
        and checks to see if the player/computer can
        move to that location given the
        current game board

        Conditions:
        - moves must be within board
        - players and computers cannot be in walls
        - players and computers cannot be in other computers

        Returns 0 if move is illegal, 1 if the move is legal,
        and 2 if the player is pushing a computer
        '''
        height = board['height']
        width = board['width']
        row = loc[0]
        col = loc[1]

        # in-bounds check
        if 0 <= row < height and 0 <= col < width:
            # checks if location is shared by wall
            if loc not in board.get('wall', set()):
                if loc not in board.get('computer', set()): # legal move
                    return 1
                elif player is True: # pushing computer
                    new_computer_loc = transform_loc(loc, move)
                    # computer can be pushed
                    if legal_move_check(board, new_computer_loc):
                        return 2
        # computer is pushed into computer,
        # or player/computer is pushed into
        # wall, or move is out of bounds
        return 0

    # no aliasing, makes deep copy
    new_level = copy_game_board(game)

    new_player_loc = transform_loc(list(new_level['player'])[0], move)

    status = legal_move_check(new_level, new_player_loc, player=True)
    if status != 0: # legal move
        if status == 2: # legal computer push
            new_computer_loc = transform_loc(new_player_loc, move)
            new_level['computer'].discard(new_player_loc)
            new_level['computer'].add(new_computer_loc)
        # either way, player is moved
        new_level['player'] = {new_player_loc}
    # if illegal move, board is unchanged
    return new_level

def transform_loc(loc, direction):
    '''
    Takes in a row, col tuple and
    a direction tuple, returns a new
    location transformed by that direction
    '''
    return (loc[0] + direction[0], loc[1] + direction[1])

def dump_game(game):
    '''
    Given a game representation (of the form returned from new_game), convert
    it back into a level description that would be a suitable input to new_game
    (a list of lists of lists of strings).

    This function is used by the GUI and the tests to see what your game
    implementation has done, and it can also serve as a rudimentary way to
    print out the current state of your game for testing and debugging on your
    own.
    '''
    # iteration 1 -> dict connecting locations to sets of objects
    # rows, cols = sorted(list(game.keys()), key=lambda x:x[0] + x[1], reverse=True)[0]
    # level_description = [ [0]*(cols+1) ]*(rows+1)

    # for row_col in game:
    #     level_description[row_col[0]][row_col[1]] = list(game[row_col])
    # return level_description

    # iteration 2 -> dict connecting object types to sets of location tuples
    # build list of lists
    height = game['height']
    width = game['width']
    new_board = [ [0] * width for _ in range(height)]

    # game is a dictionary mapping game objects to sets of locations
    # creates a temporary dictionary mapping locations to lists of game objects
    dimensions = ['height', 'width']
    complement_map = {}
    for key in game:
        if key not in dimensions:
            for location in game[key]:
                complement_map.setdefault(location, []).append(key)

    for row in range(height):
        for col in range(width):
            new_board[row][col] = complement_map.setdefault((row, col), [])
    return new_board


def solve_puzzle(game):
    '''
    Given a game representation (of the form returned from new game), find a
    solution.

    Return a list of strings representing the shortest sequence of moves ('up',
    'down', 'left', and 'right') needed to reach the victory condition.

    If the given level cannot be solved, return None.
    '''    
    def make_state(level):
        '''
        Takes in a game as returned by
        new_game, and returns a compressed,
        hashable version of it
        '''
        return (list(level['player'])[0], frozenset(level['computer']))

    def find_neighbors(game_instance):
        '''
        Given a game as represented in its compressed,
        hashable form, returns a dictionary mapping
        hashable neighor states to the associated moves
        '''
        full_board = copy_game_board(game)
        full_board['player'] = {game_instance[0]}
        full_board['computer'] = set(game_instance[1])

        neighbors = {}

        for move in direction_vector:
            updated_state = make_state(step_game(full_board, move))
            if updated_state != full_board:
                neighbors[updated_state] = move
        return neighbors

    def impossible_to_win(game_instance):
        '''
        Takes in an hashable state,
        and returns whether it is impossible
        to win as a result of:
        - a computer being in a corner
        - either the number of computers or targets being 0
        - the number of computer != the number of targets
        '''
        if len(game.get('computer', set())) == 0 or \
                len(game.get('target', set())) == 0:
            return True
        if len(game['computer']) != len(game['target']):
            return True
        for loc in game_instance[1]: # iterating over this frozenset
            if loc not in game.get('target', set()):
                row = loc[0]
                col = loc[1]
                for vertical in [(-1, 0), (1, 0)]:
                    for horizontal in [(0, -1), (0, 1)]:
                        # look for corners
                        tile1 = (row + vertical[0], col + vertical[1])
                        tile2 = (row + horizontal[0], col + horizontal[1])

                        if tile1 in game['wall'] and tile2 in game['wall']:
                                return True
        return False

    def find_shortest_path_to_victory():
        '''
        Uses breadth first search to find the
        shortest path from a starting state to a
        victory condition. Keeps track of sequence
        of moves in parent frame dictionary,
        but returns final state
        '''
        agenda = [start_state]
        visited = {start_state}

        # you only want to skip the search process
        # if the computers are all in corners and
        # not on flags), etc. at the beginning
        if not impossible_to_win(start_state):
            # if starting state has already won
            if set(start_state[1]) == game['target']:
                return start_state
            while agenda: # while this is not empty
                #print(len(agenda))
                curr_state = agenda.pop(0)

                # find neighbors of state (I can just step
                # game, and then convert back and forth
                # from game and hashable state)
                # dictionary of states to moves (left right, etc)
                neighbors = find_neighbors(curr_state)

                for state in neighbors:
                    if state not in visited:
                        agenda.append(state)
                        visited.add(state)
                        # updates path memory with mapping from
                        # state to parent state and associated move
                        path_memory[state] = (curr_state, neighbors[state])

                        if set(state[1]) == game['target']:
                            return state
        return None

    # bfs search through states, connected by moves; store path of moves in state memory

    # 1) convert game to state -> hashable
    start_state = make_state(game)

    # maps states to the state/move combo
    # (tuple) that resulted in that state, wrt the
    # shortest path from the original state
    path_memory = {start_state:None}

    end_state = find_shortest_path_to_victory()
    if end_state is None:
        return None
    else:
        reversed_move_sequence = []
        # loop through path_memory to find sequence of moves
        while end_state != start_state:
            pointer = path_memory[end_state]
            reversed_move_sequence.append(pointer[1])
            end_state = pointer[0]
        return reversed_move_sequence[::-1]

def copy_game_board(game):
    '''
    Given a game board as returned
    by new_game(), returns an unaliased copy
    '''
    new_level = {
        'height': game['height'],
        'width': game['width']
    }
    dimensions = ['height', 'width']
    for key in game:
        if key not in dimensions:
            new_level[key] = game[key].copy()
    return new_level

if __name__ == '__main__':
    pass
    TEST_DIRECTORY = os.path.dirname(__file__)
    filename = 'm1_008'

    with open(os.path.join(TEST_DIRECTORY, 'puzzles', f'{filename}.json')) as f:
        level = json.load(f)
    
    def print_game(game_instance):
        table = dump_game(game_instance)
        for row in table:
            print(row)
        print()

    game = new_game(level)
    print(f'{solve_puzzle(game)}\n')
    # print_game(game)

    # move = ''
    # move = input('Enter your move: ')

    # while move != 'stop':
    #     game = step_game(game, move)
    #     print_game(game)
    #     move = input('Enter your move: ')
    
