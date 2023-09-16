from copy import deepcopy
from itertools import compress
from typing import Self, List

import numpy as np
from numpy import ndarray

from ...constants import Move
from ...snake import Snake

PLAYER_ME = 1
PLAYER_OTHER = 2

MOVE_UP = np.array([0, -1])  # note: vertical axis in opposite direction of constants.py
MOVE_DOWN = np.array([0, 1])
MOVE_LEFT = np.array([-1, 0])
MOVE_RIGHT = np.array([1, 0])
ALL_MOVES = [MOVE_LEFT, MOVE_RIGHT, MOVE_UP, MOVE_DOWN]


# TODO generate list permutations for all possible move sets. The get_valid_moves() function can then select one.

def as_move(move) -> Move:
    assert type(move) == np.ndarray

    if move[0] == 0:
        if move[1] == 1:
            return Move.DOWN
        else:
            return Move.UP
    else:
        if move[0] == 1:
            return Move.RIGHT
        else:
            return Move.LEFT


class Board:
    _target_pos = np.array((0, 0), dtype=int)

    def __init__(self, width: int, height: int):
        """Define an empty board of a given dimension"""
        assert width > 0
        assert height > 0

        self.width: int = width
        self.height: int = height
        self.candies: List[ndarray] = []
        self.grid: ndarray[int] = np.zeros([width, height])
        self.player1_pos: ndarray = np.array((-2, -2), dtype=int)
        self.player2_pos: ndarray = np.array((-2, -2), dtype=int)
        self.player1_head: int = 0
        self.player2_head: int = 0
        self.player1_length: int = 0
        self.player2_length: int = 0

    def spawn(self, pos1: tuple, pos2: tuple):
        """Spawn snakes of length 1 at the given positions"""
        assert isinstance(pos1, tuple)
        assert isinstance(pos2, tuple)
        assert self.player1_head == 0 and self.player2_head == 0, 'players have already spawned'

        self.player1_head = 1
        self.player2_head = -1
        self.player1_length = 1
        self.player2_length = 1
        self.player1_pos = np.array(pos1, dtype=int)
        self.player2_pos = np.array(pos2, dtype=int)
        assert self.is_valid_pos(self.player1_pos), 'invalid spawn pos for P1'
        assert self.is_valid_pos(self.player2_pos), 'invalid spawn pos for P2'

        self.grid[pos1] = self.player1_head
        self.grid[pos2] = self.player2_head
        pass

    def set_state(self, snake1: Snake, snake2: Snake, candies: List[np.array]):
        assert isinstance(snake1, Snake)
        assert isinstance(snake2, Snake)
        assert isinstance(candies, list)

        # clear grid
        self.grid.fill(0)

        self.player1_length = len(snake1.positions)
        self.player2_length = len(snake2.positions)
        self.player1_head = self.player1_length
        self.player2_head = -self.player2_length

        # snake positions are in reverse order (head of the list is tail of the snake)
        # tail = 1, head = p1_head
        for i, pos in enumerate(snake1.positions):
            self.grid[pos[0], pos[1]] = i + 1

        # tail = -1, head = p2_head
        for i, pos in enumerate(snake2.positions):
            self.grid[pos[0], pos[1]] = -i - 1

        # set player positions to the head of the snake (the tail of the list)
        np.copyto(self.player1_pos, snake1.positions[-1], casting='no')
        np.copyto(self.player2_pos, snake2.positions[-1], casting='no')

        self.candies = candies.copy()
        pass

    @property
    def shape(self) -> tuple:
        return self.width, self.height

    def get_free_space(self) -> int:
        return np.sum(self.get_empty_mask() == True, axis=None)  # how to get rid of the warning??

    def is_valid_pos(self, pos: ndarray) -> bool:
        return 0 <= pos[0] < self.width and 0 <= pos[1] < self.height

    def is_candy_pos(self, pos: ndarray) -> bool:
        return pos in self.candies

    # TODO optimize
    def is_empty_pos(self, pos: ndarray) -> bool:
        return self.get_empty_mask()[pos[0], pos[1]] == True

    # TODO optimize
    def get_empty_mask(self) -> ndarray:
        return (self.get_player1_mask() == False) & (self.get_player2_mask() == False)

    def get_player1_mask(self) -> ndarray:
        return self.grid > max(0, self.player1_head - self.player1_length)

    def get_player2_mask(self) -> ndarray:
        return self.grid < min(0, self.player2_head + self.player2_length)

    def get_player_mask(self, player: int) -> ndarray:
        if player == 1:
            return self.get_player1_mask()
        else:
            return self.get_player2_mask()

    def get_candy_mask(self) -> ndarray:
        grid_mask = np.full(self.shape, fill_value=False, dtype=bool)
        grid_mask[self.candies] = True
        return grid_mask

    def has_candy(self) -> bool:
        return len(self.candies) > 0

    def inherit(self, board: Self):
        assert self.shape == board.shape, 'boards must be same size'
        np.copyto(self.player1_pos, board.player1_pos, casting='no')
        np.copyto(self.player2_pos, board.player2_pos, casting='no')
        np.copyto(self.grid, board.grid, casting='no')
        self.player1_length = board.player1_length
        self.player2_length = board.player2_length
        self.player1_head = board.player1_head
        self.player2_head = board.player2_head
        self.candies = deepcopy(board.candies)
        pass

    def get_valid_moves(self, player: int) -> list:
        if player == 1:
            pos = self.player1_pos
        else:
            pos = self.player2_pos

        assert self.is_valid_pos(pos), 'invalid position tuple for player {player}: {pos}'

        can_move_left = pos[0] > 0 and self.is_empty_pos(pos + MOVE_LEFT)
        can_move_right = pos[0] < self.width - 1 and self.is_empty_pos(pos + MOVE_RIGHT)
        can_move_up = pos[1] > 0 and self.is_empty_pos(pos + MOVE_UP)
        can_move_down = pos[1] < self.height - 1 and self.is_empty_pos(pos + MOVE_DOWN)

        return list(compress(ALL_MOVES, [can_move_left, can_move_right, can_move_up, can_move_down]))

    # performing a move increments the turn counter and places a new wall
    def perform_move(self, move: ndarray, player: int):
        assert move[0] in (-1, 0, 1) and \
               move[1] in (-1, 0, 1) and \
               move[0] == 0 or move[1] == 0, 'invalid move vector'

        # TODO remove branching
        if player == 1:
            np.copyto(self._target_pos, self.player1_pos, casting='no')
            np.add(self._target_pos, move, out=self._target_pos)  # compute new pos
            assert self.is_valid_pos(self._target_pos), 'illegal move by P1: invalid position'
            assert self.is_empty_pos(self._target_pos), 'illegal move by P1: suicide'

            # update game state
            np.copyto(self.player1_pos, self._target_pos, casting='no')
            self.player1_head += 1
            if self.is_candy_pos(self.player1_pos):
                self.player1_length += 1
                self.remove_candy(self.player1_pos)
            self.grid[tuple(self.player1_pos)] = self.player1_head
        else:
            np.copyto(self._target_pos, self.player2_pos, casting='no')
            np.add(self._target_pos, move, out=self._target_pos)
            assert self.is_valid_pos(self._target_pos), 'illegal move by P2: invalid position'
            assert self.is_empty_pos(self._target_pos), 'illegal move by P2: suicide'

            # update game state
            np.copyto(self.player2_pos, self._target_pos, casting='no')
            self.player2_head -= 1  # the only difference in logic between the players
            if self.is_candy_pos(self.player2_pos):
                self.player2_length += 1
                self.remove_candy(self.player2_pos)
            self.grid[tuple(self.player2_pos)] = self.player2_head
        pass

    def spawn_candy(self, pos: ndarray):
        assert not (pos in self.candies)
        self.candies.append(pos)

    def remove_candy(self, pos: ndarray):
        self.candies.remove(pos)

    def __len__(self) -> int:
        return self.width * self.height

    def __eq__(self, other) -> bool:
        return self.player1_head == other.player1_head and \
            self.player2_head == other.player2_head and \
            np.array_equal(self.grid, other.grid)

    def __str__(self):
        str_grid = np.full(self.shape, fill_value='_', dtype=str)
        str_grid[self.get_player1_mask()] = 'a'
        str_grid[self.get_player2_mask()] = 'b'
        str_grid[self.get_candy_mask()] = '*'
        if self.player1_length > 0:
            str_grid[tuple(self.player1_pos)] = 'A'
            str_grid[tuple(self.player2_pos)] = 'B'

        str_field = np.pad(str_grid, [(1, 1), (1, 1)], mode='constant')
        str_field[0:, (0, -1)] = '-'
        str_field[(0, -1), 0:] = '|'
        str_field[(0, 0, -1, -1), (0, -1, -1, 0)] = '+'

        # how to join array elems into single string??
        # for now, use array2string and clean up the garbage output by np
        # how to replace multiple chars?
        return '\n' + np.array2string(str_field.T, separator=''). \
            replace('[', ''). \
            replace(']', ''). \
            replace("'", ''). \
            replace(' ', ''). \
            replace('_', ' ')

    def __repr__(self):
        # TODO add turn info
        str_board = str(self)
        return str_board
