from collections import deque
from copy import deepcopy
from itertools import compress
from typing import List, Deque, TypeVar, Tuple, Iterable

import numpy as np
from numpy import ndarray

from ...constants import Move
from ...snake import Snake

Self = TypeVar("Self", bound="Board")
Pos = Tuple[int, int]

ALL_MOVES = (Move.LEFT, Move.RIGHT, Move.UP, Move.DOWN)
POS_OFFSET = np.array((1, 1))

MOVE_TO_DIRECTION = {
    Move.UP: (0, 1),
    Move.DOWN: (0, -1),
    Move.LEFT: (-1, 0),
    Move.RIGHT: (1, 0),
}


class Board:
    _hash: int = 0
    _approx_hash: int = 0
    _wall_hash: int = 0

    def __init__(self, width: int, height: int) -> None:
        """Define an empty board of a given dimension"""
        assert width > 0
        assert height > 0

        self.width: int = width
        self.height: int = height
        self.center: Pos = (int(width / 2) + 1, int(height / 2) + 1)
        self.grid: ndarray = np.pad(
            np.zeros([width, height], dtype=int),
            pad_width=1,
            constant_values=10000
        )
        self.candy_mask: ndarray = np.full(self.grid.shape, fill_value=False, dtype=bool)
        self.candies: List[Pos] = list()
        self.player1_pos: Pos = (-2, -2)
        self.player2_pos: Pos = (-2, -2)
        self.player1_head: int = 0
        self.player2_head: int = 0
        self.player1_length: int = 0
        self.player2_length: int = 0
        self.last_player = 1
        self.move_pos_stack: Deque[Pos] = deque(maxlen=32)
        self.move_head_stack: Deque[int] = deque(maxlen=32)
        self.move_candy_stack: Deque[bool] = deque(maxlen=32)

    def spawn(self, pos1: Pos, pos2: Pos) -> None:
        """Spawn snakes of length 1 at the given positions
        Indices start from 0

        Merely for testing purposes"""
        self.set_state(
            snake1=Snake(id=0, positions=np.array([pos1])),
            snake2=Snake(id=1, positions=np.array([pos2])),
            candies=[]
        )
        self.invalidate()

    def set_state(self, snake1: Snake, snake2: Snake, candies: List[np.array]) -> None:
        self.last_player = -1  # set P2 to have moved last

        # clear grid
        self.grid[1:-1, 1:-1] = 0
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
            self.grid[pos[0] + 1, pos[1] + 1] = self.player1_head - i

        # head = p2_head, tail = p2_head + len - 1
        for i, pos in enumerate(snake2.positions):
            self.grid[pos[0] + 1, pos[1] + 1] = self.player2_head + i

        self.player1_pos = tuple(snake1.positions[0] + POS_OFFSET)
        self.player2_pos = tuple(snake2.positions[0] + POS_OFFSET)

        # spawn candies
        for pos in candies:
            self._spawn_candy(tuple(pos + POS_OFFSET))
        self.invalidate()

    def _spawn_candy(self, pos: Pos) -> None:
        assert min(pos) > 0
        self.candies.append(pos)
        self.candy_mask[pos] = True

    def _remove_candy(self, pos: Pos) -> None:
        self.candies.remove(pos)
        self.candy_mask[pos] = False

    @property
    def shape(self) -> Tuple[int, int]:
        return self.width, self.height

    def count_free_space(self) -> int:
        return self.get_empty_mask().sum()

    def is_valid_pos(self, pos: Pos) -> bool:
        return 0 <= pos[0] < self.width and 0 <= pos[1] < self.height

    def is_candy_pos(self, pos: Pos) -> bool:
        return self.candy_mask[pos]

    def is_empty_pos(self, pos: Pos) -> bool:
        return self.player2_head + self.player2_length <= \
            self.grid[pos] <= \
            self.player1_head - self.player1_length

    def get_player_pos(self, player: int) -> Pos:
        return self.player1_pos if player == 1 else self.player2_pos

    def get_candies(self) -> Tuple[Pos]:
        return tuple(self.candies)

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

        return self.is_empty_pos((pos[0] - 1, pos[1])) or \
            self.is_empty_pos((pos[0] + 1, pos[1])) or \
            self.height - 1 and self.is_empty_pos((pos[0], pos[1] + 1)) or \
            self.is_empty_pos((pos[0], pos[1] - 1))

    def can_do_move(self, move: Move, pos: Pos) -> bool:
        if move == Move.LEFT:
            return self.is_empty_pos((pos[0] - 1, pos[1]))
        elif move == Move.RIGHT:
            return self.is_empty_pos((pos[0] + 1, pos[1]))
        elif move == Move.UP:
            return self.is_empty_pos((pos[0], pos[1] + 1))
        else:
            return self.is_empty_pos((pos[0], pos[1] - 1))

    def can_player1_do_move(self, move: Move) -> bool:
        return self.can_do_move(move, self.player1_pos)

    def can_player2_do_move(self, move: Move) -> bool:
        return self.can_do_move(move, self.player2_pos)

    def iterate_valid_moves(self, player: int, order: Tuple[Move] = ALL_MOVES) -> Iterable[Move]:
        if player == 1:
            return filter(self.can_player1_do_move, order)
        else:
            return filter(self.can_player2_do_move, order)

    def get_valid_moves(self, player: int) -> Tuple[Move]:
        if player == 1:
            pos = self.player1_pos
        else:
            pos = self.player2_pos

        can_moves = [
            self.is_empty_pos((pos[0] - 1, pos[1])),  # left
            self.is_empty_pos((pos[0] + 1, pos[1])),  # right
            self.is_empty_pos((pos[0], pos[1] + 1)),  # up
            self.is_empty_pos((pos[0], pos[1] - 1))  # down
        ]

        return tuple(compress(ALL_MOVES, can_moves))

    def get_valid_moves_ordered(self, player: int, order: Tuple[Move] = ALL_MOVES) -> List[Move]:
        moves = self.get_valid_moves(player=player)
        return [x for _, x in sorted(zip(order, moves))]

    # performing a move increments the turn counter and places a new wall
    def perform_move(self, move: Move, player: int) -> None:
        direction = MOVE_TO_DIRECTION[move]

        # TODO remove branching
        if player == 1:
            target_pos = (self.player1_pos[0] + direction[0], self.player1_pos[1] + direction[1])

            # update game state
            ate_candy = self.is_candy_pos(target_pos)
            self.move_candy_stack.append(ate_candy)
            self.move_pos_stack.append(self.player1_pos)
            self.move_head_stack.append(self.grid[target_pos])
            self.player1_pos = target_pos
            self.player1_head += 1
            self.grid[self.player1_pos] = self.player1_head

            if ate_candy:
                self.player1_length += 1
                self._remove_candy(self.player1_pos)
        else:
            target_pos = (self.player2_pos[0] + direction[0], self.player2_pos[1] + direction[1])

            # update game state
            ate_candy = self.is_candy_pos(target_pos)
            self.move_candy_stack.append(ate_candy)
            self.move_pos_stack.append(self.player2_pos)
            self.move_head_stack.append(self.grid[target_pos])
            self.player2_pos = target_pos
            self.player2_head -= 1  # the only difference in logic between the players
            self.grid[self.player2_pos] = self.player2_head
            if ate_candy:
                self.player2_length += 1
                self._remove_candy(self.player2_pos)

        self.last_player = player
        # invalidate cache
        self._hash = 0
        self._approx_hash = 0
        self._wall_hash = 0

    def undo_move(self, player: int) -> None:
        assert player == self.last_player, 'Last move was performed by the other player'

        ate_candy = self.move_candy_stack.pop()

        # TODO remove branching
        if player == 1:
            if ate_candy:
                self.player1_length -= 1
                self._spawn_candy(self.player1_pos)
            self.grid[self.player1_pos] = self.move_head_stack.pop()
            self.player1_pos = self.move_pos_stack.pop()
            self.player1_head -= 1
        else:
            if ate_candy:
                self.player2_length -= 1
                self._spawn_candy(self.player2_pos)
            self.grid[self.player2_pos] = self.move_head_stack.pop()
            self.player2_pos = self.move_pos_stack.pop()
            self.player2_head += 1

        self.last_player = -self.last_player

        # invalidate cache
        self._hash = 0
        self._approx_hash = 0
        self._wall_hash = 0

    def inherit(self, board: Self) -> None:
        assert self.shape == board.shape, 'boards must be same size'
        self.player1_pos = board.player1_pos
        self.player2_pos = board.player2_pos
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
            self.player1_pos == other.player1_pos and \
            self.player2_pos == other.player2_pos and \
            np.array_equal(self.grid, other.grid) and \
            np.array_equal(self.candy_mask, other.candy_mask) and \
            self.move_pos_stack == other.move_pos_stack and \
            self.move_head_stack == other.move_head_stack and \
            self.move_candy_stack == other.move_candy_stack

    def __hash__(self) -> int:
        """Hash of the exact game state"""
        if self._hash == 0:
            self._hash = hash((_hash_np(self.grid), _hash_np(self.candy_mask), self.last_player))

        return self._hash

    def approx_hash(self) -> int:
        """Hash of the game state only considering blocked cells, player positions, candies, and last player"""
        if self._approx_hash == 0:
            self._approx_hash = hash((
                _hash_np(self.get_empty_mask()),
                _hash_np(self.candy_mask),
                self.player1_pos,
                self.player2_pos,
                self.last_player
            ))
        return self._approx_hash

    def wall_hash(self) -> int:
        """Hash of the game state only considering blocked cells on the grid"""
        if self._wall_hash == 0:
            self._wall_hash = _hash_np(self.get_empty_mask())
        return self._wall_hash

    def invalidate(self) -> None:
        """Clears the cached state of the board. Forces hash recomputation on request"""
        self._hash, self._approx_hash, self._wall_hash = 0, 0, 0

    def __str__(self) -> str:
        str_grid = np.full(self.grid.shape, fill_value='_', dtype=str)
        str_grid[self.get_player1_mask()] = 'a'
        str_grid[self.get_player2_mask()] = 'b'
        str_grid[0:, (0, -1), ] = '-'
        str_grid[(0, -1), 0:] = '|'
        str_grid[(0, 0, -1, -1), (0, -1, -1, 0)] = '+'
        str_grid[self.get_candy_mask()] = '*'
        if self.player1_length > 0:
            str_grid[self.player1_pos] = 'A'
            str_grid[self.player2_pos] = 'B'

        return '\n' + np.array2string(np.flipud(str_grid.T), separator=''). \
            replace('[', ''). \
            replace(']', ''). \
            replace("'", ''). \
            replace(' ', ''). \
            replace('_', '·')

    def __repr__(self) -> str:
        return str(self)


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


def _hash_np(x) -> int:
    return hash(x.data.tobytes())
