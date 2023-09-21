from collections import deque
from copy import deepcopy
from itertools import compress
from typing import List, Deque, TypeVar, Tuple

import numpy as np
from numpy import ndarray

from ...constants import LEFT, RIGHT, UP, DOWN, Move, MOVE_VALUE_TO_DIRECTION
from ...snake import Snake

Self = TypeVar("Self", bound="Board")

ALL_MOVES = (Move.LEFT, Move.RIGHT, Move.UP, Move.DOWN)


# TODO generate list permutations for all possible move sets. The get_valid_moves() function can then select one.

class Board:
    _target_pos = np.array((0, 0), dtype=int)

    def __init__(self, width: int, height: int) -> None:
        """Define an empty board of a given dimension"""
        assert width > 0
        assert height > 0

        self.width: int = width
        self.height: int = height
        self.center: Tuple[int, int] = (int(width / 2), int(height / 2))
        self.grid: ndarray = np.zeros([width, height], dtype=int)
        self.candies: List[Tuple[int, int]] = list()
        self.candy_mask: ndarray = np.full(self.grid.shape, fill_value=False, dtype=bool)
        self.player1_pos: ndarray = np.array((-2, -2), dtype=int)
        self.player2_pos: ndarray = np.array((-2, -2), dtype=int)
        self.player1_head: int = 0
        self.player2_head: int = 0
        self.player1_length: int = 0
        self.player2_length: int = 0
        self.last_player = 1
        self.move_pos_stack: Deque[Tuple] = deque(maxlen=32)
        self.move_head_stack: Deque[int] = deque(maxlen=32)
        self.move_candy_stack: Deque[bool] = deque(maxlen=32)

    def spawn(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> None:
        """Spawn snakes of length 1 at the given positions. Merely for testing purposes."""
        self.set_state(
            snake1=Snake(id=0, positions=np.array([pos1])),
            snake2=Snake(id=1, positions=np.array([pos2])),
            candies=[]
        )

    def set_state(self, snake1: Snake, snake2: Snake, candies: List[np.array]) -> None:
        self.last_player = -1  # set P2 to have moved last

        # clear grid
        self.grid.fill(0)
        self.candy_mask.fill(False)
        self.candies.clear()

        # clear move stacks
        self.move_pos_stack.clear()
        self.move_head_stack.clear()
        self.move_candy_stack.clear()

        self.player1_length = len(snake1.positions)
        self.player2_length = len(snake2.positions)
        self.player1_head = self.player1_length
        self.player2_head = -self.player2_length

        # head = p1_head, tail = p1_head - len + 1
        for i, pos in enumerate(snake1.positions):
            self.grid[pos[0], pos[1]] = self.player1_head - i

        # head = p2_head, tail = p2_head + len - 1
        for i, pos in enumerate(snake2.positions):
            self.grid[pos[0], pos[1]] = self.player2_head + i

        # set player positions to the head of the snake (the tail of the list)
        np.copyto(self.player1_pos, snake1.positions[0], casting='no')
        np.copyto(self.player2_pos, snake2.positions[0], casting='no')

        # spawn candies
        for pos in candies:
            self.spawn_candy(pos)

    def spawn_candy(self, pos: ndarray) -> None:
        pos_tuple = tuple(pos)
        assert pos_tuple not in self.candies
        self.candies.append(pos_tuple)
        self.candy_mask[pos[0], pos[1]] = True

    def remove_candy(self, pos: ndarray) -> None:
        self.candies.remove(tuple(pos))
        self.candy_mask[pos[0], pos[1]] = False

    @property
    def shape(self) -> Tuple[int, int]:
        return self.width, self.height

    def get_free_space(self) -> int:
        return self.get_empty_mask().sum()

    def is_valid_pos(self, pos: ndarray) -> bool:
        return 0 <= pos[0] < self.width and 0 <= pos[1] < self.height

    def is_candy_pos(self, pos: ndarray) -> bool:
        return self.candy_mask[pos[0], pos[1]]

    def is_empty_pos(self, pos: ndarray) -> bool:
        return self.player2_head + self.player2_length <= \
            self.grid[pos[0], pos[1]] <= \
            self.player1_head - self.player1_length

    def get_player_pos(self, player: int) -> ndarray:
        return self.player1_pos if player == 1 else self.player2_pos

    def get_candies(self) -> List[Tuple[int, int]]:
        return self.candies.copy()

    def get_empty_mask(self) -> ndarray:
        return np.logical_and(
            self.player2_head + self.player2_length <= self.grid,
            self.grid <= self.player1_head - self.player1_length
        )

    def get_player1_mask(self) -> ndarray:
        return self.grid > self.player1_head - self.player1_length

    def get_player2_mask(self) -> ndarray:
        return self.grid < self.player2_head + self.player2_length

    def get_player_mask(self, player: int) -> ndarray:
        if player == 1:
            return self.get_player1_mask()
        else:
            return self.get_player2_mask()

    def get_candy_mask(self) -> ndarray:
        return self.candy_mask.view()

    def has_candy(self) -> bool:
        return len(self.candies) > 0

    def can_move(self, player: int) -> bool:
        pos = self.player1_pos if player == 1 else self.player2_pos

        return (pos[0] > 0 and self.is_empty_pos(pos + LEFT)) or \
            (pos[0] < self.width - 1 and self.is_empty_pos(pos + RIGHT)) or \
            (pos[1] < self.height - 1 and self.is_empty_pos(pos + UP)) or \
            (pos[1] > 0 and self.is_empty_pos(pos + DOWN))

    def get_valid_moves(self, player: int) -> List[Move]:
        if player == 1:
            pos = self.player1_pos
        else:
            pos = self.player2_pos

        can_move_left = pos[0] > 0 and self.is_empty_pos(pos + LEFT)
        can_move_right = pos[0] < self.width - 1 and self.is_empty_pos(pos + RIGHT)
        can_move_up = pos[1] < self.height - 1 and self.is_empty_pos(pos + UP)
        can_move_down = pos[1] > 0 and self.is_empty_pos(pos + DOWN)

        return list(compress(ALL_MOVES, [can_move_left, can_move_right, can_move_up, can_move_down]))

    # performing a move increments the turn counter and places a new wall
    def perform_move(self, move: Move, player: int) -> None:
        move_vec = MOVE_VALUE_TO_DIRECTION[move]

        # TODO remove branching
        if player == 1:
            np.copyto(self._target_pos, self.player1_pos, casting='no')
            np.add(self._target_pos, move_vec, out=self._target_pos)  # compute new pos

            # update game state
            ate_candy = self.is_candy_pos(self._target_pos)
            self.move_candy_stack.append(ate_candy)
            self.move_pos_stack.append(tuple(self.player1_pos))
            self.move_head_stack.append(self.grid[self._target_pos[0], self._target_pos[1]])
            np.copyto(self.player1_pos, self._target_pos, casting='no')
            self.player1_head += 1
            self.grid[self.player1_pos[0], self.player1_pos[1]] = self.player1_head

            if ate_candy:
                self.player1_length += 1
                self.remove_candy(self.player1_pos)
        else:
            np.copyto(self._target_pos, self.player2_pos, casting='no')
            np.add(self._target_pos, move_vec, out=self._target_pos)

            # update game state
            ate_candy = self.is_candy_pos(self._target_pos)
            self.move_candy_stack.append(ate_candy)
            self.move_pos_stack.append(tuple(self.player2_pos))
            self.move_head_stack.append(self.grid[self._target_pos[0], self._target_pos[1]])
            np.copyto(self.player2_pos, self._target_pos, casting='no')
            self.player2_head -= 1  # the only difference in logic between the players
            self.grid[self.player2_pos[0], self.player2_pos[1]] = self.player2_head
            if ate_candy:
                self.player2_length += 1
                self.remove_candy(self.player2_pos)

        self.last_player = player

    def undo_move(self, player: int) -> None:
        assert player == self.last_player, 'Last move was performed by the other player'

        ate_candy = self.move_candy_stack.pop()

        # TODO remove branching
        if player == 1:
            if ate_candy:
                self.player1_length -= 1
                self.spawn_candy(self.player1_pos)
            self.grid[self.player1_pos[0], self.player1_pos[1]] = self.move_head_stack.pop()
            np.copyto(self.player1_pos, self.move_pos_stack.pop())
            self.player1_head -= 1
        else:
            if ate_candy:
                self.player2_length -= 1
                self.spawn_candy(self.player2_pos)
            self.grid[self.player2_pos[0], self.player2_pos[1]] = self.move_head_stack.pop()
            np.copyto(self.player2_pos, self.move_pos_stack.pop())
            self.player2_head += 1

        self.last_player = -self.last_player

    def inherit(self, board: Self) -> None:
        assert self.shape == board.shape, 'boards must be same size'
        np.copyto(self.player1_pos, board.player1_pos, casting='no')
        np.copyto(self.player2_pos, board.player2_pos, casting='no')
        np.copyto(self.grid, board.grid, casting='no')
        np.copyto(self.candy_mask, board.candy_mask, casting='no')

        self.player1_length = board.player1_length
        self.player2_length = board.player2_length
        self.player1_head = board.player1_head
        self.player2_head = board.player2_head
        self.move_pos_stack = board.move_pos_stack.copy()
        self.move_head_stack = board.move_head_stack.copy()
        self.move_candy_stack = board.move_candy_stack.copy()

    def copy(self) -> Self:
        return deepcopy(self)

    def __len__(self) -> int:
        return self.width * self.height

    def __eq__(self, other) -> bool:
        return self.width == other.width and \
            self.height == other.height and \
            self.player1_head == other.player1_head and \
            self.player2_head == other.player2_head and \
            self.player1_length == other.player1_length and \
            self.player2_length == other.player2_length and \
            self.last_player == other.last_player and \
            set(self.candies) == set(other.candies) and \
            np.array_equal(self.player1_pos, other.player1_pos) and \
            np.array_equal(self.player2_pos, other.player2_pos) and \
            np.array_equal(self.grid, other.grid) and \
            np.array_equal(self.candy_mask, other.candy_mask) and \
            self.move_pos_stack == other.move_pos_stack and \
            self.move_head_stack == other.move_head_stack and \
            self.move_candy_stack == other.move_candy_stack

    def __str__(self) -> str:
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
        return '\n' + np.array2string(np.flipud(str_field.T), separator=''). \
            replace('[', ''). \
            replace(']', ''). \
            replace("'", ''). \
            replace(' ', ''). \
            replace('_', '·')

    def __repr__(self) -> str:
        # TODO add turn info
        str_board = str(self)
        return str_board


def as_move(move: ndarray) -> Move:
    """Convert ndarray move to Move enum item"""
    if move[0] == 0:
        if move[1] == 1:
            return Move.UP
        else:
            return Move.DOWN
    elif move[0] == 1:
        return Move.RIGHT
    else:
        return Move.LEFT


def distance(pos1, pos2) -> int:
    """L1 distance between the given positions"""
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])


def distance_grid(width: int, height: int, pos) -> ndarray:
    """L1 distance to each cell on the board from the given position"""
    rows = np.abs(np.arange(width) - pos[0])
    cols = np.abs(np.arange(height) - pos[1])
    return rows[:, np.newaxis] + cols[np.newaxis, :]


def player_num(player) -> int:
    return int(1.5 - player / 2)
