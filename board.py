from collections import deque
from copy import deepcopy
from enum import IntEnum, auto, Enum
from itertools import compress, chain
from typing import List, TypeVar, Tuple, Iterator

import numpy as np
from numpy import ndarray

from ...constants import Move
from ...snake import Snake

Self = TypeVar("Self", bound="Board")
Pos = Tuple[int, int]
PosIdx = int
Grid = List[int]
GridMask = List[bool]


class BoardMove(IntEnum):
    UP = auto()
    DOWN = auto()
    LEFT = auto()
    RIGHT = auto()
    __str__ = Enum.__str__


BOARD_MOVE_UP = BoardMove.UP
BOARD_MOVE_DOWN = BoardMove.DOWN
BOARD_MOVE_LEFT = BoardMove.LEFT
BOARD_MOVE_RIGHT = BoardMove.RIGHT


class Direction(IntEnum):
    TOP = auto()
    TOP_RIGHT = auto()
    RIGHT = auto()
    BOTTOM_RIGHT = auto()
    BOTTOM = auto()
    BOTTOM_LEFT = auto()
    LEFT = auto()
    TOP_LEFT = auto()
    __str__ = Enum.__str__


MOVES = (BoardMove.LEFT, BoardMove.RIGHT, BoardMove.UP, BoardMove.DOWN)

MOVE_MAP = {
    BoardMove.UP: Move.UP,
    BoardMove.RIGHT: Move.RIGHT,
    BoardMove.DOWN: Move.DOWN,
    BoardMove.LEFT: Move.LEFT,
}

OPPOSITE_MOVE = {
    BoardMove.UP: BoardMove.DOWN,
    BoardMove.RIGHT: BoardMove.LEFT,
    BoardMove.DOWN: BoardMove.UP,
    BoardMove.LEFT: BoardMove.RIGHT,
}

TURN_LEFT_MOVE = {
    BoardMove.UP: BoardMove.LEFT,
    BoardMove.RIGHT: BoardMove.UP,
    BoardMove.DOWN: BoardMove.RIGHT,
    BoardMove.LEFT: BoardMove.DOWN,
}

TURN_RIGHT_MOVE = {
    BoardMove.UP: BoardMove.RIGHT,
    BoardMove.RIGHT: BoardMove.DOWN,
    BoardMove.DOWN: BoardMove.LEFT,
    BoardMove.LEFT: BoardMove.UP,
}

FIRST_MOVE_ORDER = {m: (m, TURN_LEFT_MOVE[m], TURN_RIGHT_MOVE[m], OPPOSITE_MOVE[m]) for m in MOVES}


class Board:
    __slots__ = (
        'width', 'height',
        'full_width', 'full_height',
        'center',
        'grid_mask',
        'candies', 'spawn_candy', 'remove_candy',
        'player1_pos', 'player2_pos',
        'player1_positions', 'player2_positions',
        'push_player1_position', 'push_player2_position',
        'pop_player1_position', 'pop_player2_position',
        'player1_prev_pos', 'player2_prev_pos',
        'player1_length', 'player2_length',
        'hash',
        'last_player',
        'move_stack', 'push_move_stack', 'pop_move_stack',
        'pos_map',
        'DISTANCE',
        'FOUR_WAY_POSITIONS_COND', 'FOUR_WAY_POSITIONS',
        'FOUR_WAY_POSITIONS_FROM_POS_COND', 'FOUR_WAY_POSITIONS_FROM_POS',
        'EIGHT_WAY_POSITIONS_COND', 'EIGHT_WAY_POSITIONS',
        'EIGHT_WAY_POSITIONS_FROM_POS_COND', 'EIGHT_WAY_POSITIONS_FROM_POS',
        'MOVE_POS_OFFSET',
        'FOUR_WAY_POS_OFFSETS', 'EIGHT_WAY_POS_OFFSETS',
        'DIR_UP_LEFT', 'DIR_UP', 'DIR_UP_RIGHT', 'DIR_RIGHT', 'DIR_DOWN_RIGHT', 'DIR_DOWN', 'DIR_DOWN_LEFT', 'DIR_LEFT',
    )

    def __init__(self, width: int, height: int) -> None:
        """Define an empty board of a given dimension"""
        assert width > 0
        assert height > 0

        self.width, self.height = width, height
        self.full_width, self.full_height = width + 2, height + 2
        self.grid_mask = create_grid(width, height)
        self.candies: List[PosIdx] = []
        self.player1_pos: PosIdx = -10
        self.player2_pos: PosIdx = -10
        self.player1_positions: List[PosIdx] = []
        self.player2_positions: List[PosIdx] = []
        self.push_player1_position, self.push_player2_position = self.player1_positions.append, self.player2_positions.append
        self.pop_player1_position, self.pop_player2_position = self.player1_positions.pop, self.player2_positions.pop
        self.player1_prev_pos: PosIdx = -10
        self.player2_prev_pos: PosIdx = -10
        self.player1_length, self.player2_length = 0, 0
        self.hash: int = 0
        self.last_player = 1
        self.move_stack: List[PosIdx, bool, int] = []
        self.push_move_stack, self.pop_move_stack = self.move_stack.append, self.move_stack.pop
        self.spawn_candy, self.remove_candy = self.candies.append, self.candies.remove
        self.pos_map = tuple([divmod(i, self.full_height) for i in range(0, self.full_width * self.full_height)])
        self.center: PosIdx = self.from_xy(int(width / 2) + 1, int(height / 2) + 1)

        self.DIR_UP_LEFT = -self.full_height + 1
        self.DIR_UP = 1
        self.DIR_UP_RIGHT = self.full_height + 1
        self.DIR_RIGHT = self.full_height
        self.DIR_DOWN_RIGHT = self.full_height - 1
        self.DIR_DOWN = -1
        self.DIR_DOWN_LEFT = -self.full_height - 1
        self.DIR_LEFT = -self.full_height

        self.MOVE_POS_OFFSET = {
            BoardMove.UP: self.DIR_UP,
            BoardMove.RIGHT: self.DIR_RIGHT,
            BoardMove.DOWN: self.DIR_DOWN,
            BoardMove.LEFT: self.DIR_LEFT
        }
        self.FOUR_WAY_POS_OFFSETS = tuple(self.MOVE_POS_OFFSET[m] for m in MOVES)
        self.EIGHT_WAY_POS_OFFSETS = (
            self.DIR_UP_LEFT,
            self.DIR_UP,
            self.DIR_UP_RIGHT,
            self.DIR_RIGHT,
            self.DIR_DOWN_RIGHT,
            self.DIR_DOWN,
            self.DIR_DOWN_LEFT,
            self.DIR_LEFT
        )

        self.DISTANCE = [
            [distance(self.from_index(p1), self.from_index(p2)) for p2 in range(len(self.grid_mask))]
            for p1 in range(len(self.grid_mask))
        ]

        def is_within_bounds(pos):
            if pos < 0 or pos >= len(self.grid_mask):
                return False
            else:
                x, y = self.from_index(pos)
                return 0 < x < self.full_width - 1 and 0 < y < self.full_height - 1

        self.FOUR_WAY_POSITIONS: List[Tuple[PosIdx, ...]] = [
            tuple(p + d for d in self.FOUR_WAY_POS_OFFSETS)
            for p in range(len(self.grid_mask))
        ]

        self.FOUR_WAY_POSITIONS_COND: List[Tuple[PosIdx, ...]] = [
            tuple(filter(is_within_bounds, [p + d for d in self.FOUR_WAY_POS_OFFSETS]))
            for p in range(len(self.grid_mask))
        ]

        def _get_transitional_positions(pos_old: PosIdx, pos_new: PosIdx, pos_options: List) -> Tuple[PosIdx, ...]:
            candidate_positions = pos_options[pos_new]
            if pos_old in candidate_positions:
                return tuple(p for p in candidate_positions if p != pos_old)
            else:
                return candidate_positions

        self.FOUR_WAY_POSITIONS_FROM_POS_COND: List[List] = [
            [
                _get_transitional_positions(pos_old, pos_new, self.FOUR_WAY_POSITIONS_COND)
                for pos_new in range(len(self.grid_mask))
            ]
            for pos_old in range(len(self.grid_mask))
        ]

        self.FOUR_WAY_POSITIONS_FROM_POS: List[List] = [
            [
                _get_transitional_positions(pos_old, pos_new, self.FOUR_WAY_POSITIONS)
                for pos_new in range(len(self.grid_mask))
            ]
            for pos_old in range(len(self.grid_mask))
        ]

        self.EIGHT_WAY_POSITIONS: List[Tuple[PosIdx, ...]] = [
            tuple(p + d for d in self.EIGHT_WAY_POS_OFFSETS)
            for p in range(len(self.grid_mask))
        ]

        self.EIGHT_WAY_POSITIONS_COND: List[Tuple[PosIdx, ...]] = [
            tuple(filter(is_within_bounds, [p + d for d in self.EIGHT_WAY_POS_OFFSETS]))
            for p in range(len(self.grid_mask))
        ]

        self.EIGHT_WAY_POSITIONS_FROM_POS_COND: List[List] = [
            [
                _get_transitional_positions(pos_old, pos_new, self.EIGHT_WAY_POSITIONS_COND)
                for pos_new in range(len(self.grid_mask))
            ]
            for pos_old in range(len(self.grid_mask))
        ]

        self.EIGHT_WAY_POSITIONS_FROM_POS: List[List] = [
            [
                _get_transitional_positions(pos_old, pos_new, self.EIGHT_WAY_POSITIONS)
                for pos_new in range(len(self.grid_mask))
            ]
            for pos_old in range(len(self.grid_mask))
        ]

    def from_pos(self, pos: Pos) -> PosIdx:
        return pos[0] * self.full_height + pos[1]

    def from_xy(self, x: int, y: int) -> PosIdx:
        return x * self.full_height + y

    def from_index(self, index: PosIdx) -> Pos:
        return self.pos_map[index]

    def set_state(self, snake1: Snake, snake2: Snake, candies: List[np.array]) -> None:
        assert len(snake1.positions) > 1
        assert len(snake2.positions) > 1

        # clear grid
        self.grid_mask = create_grid(self.width, self.height)
        self.move_stack = []
        self.push_move_stack, self.pop_move_stack = self.move_stack.append, self.move_stack.pop
        self.candies = [self.from_pos(pos) + self.DIR_UP_RIGHT for pos in candies]
        self.spawn_candy, self.remove_candy = self.candies.append, self.candies.remove  # keep separate from candies assignment

        # player positions
        self.player1_positions = [self.from_pos(pos) + self.DIR_UP_RIGHT for pos in reversed(snake1.positions)]
        self.player2_positions = [self.from_pos(pos) + self.DIR_UP_RIGHT for pos in reversed(snake2.positions)]
        self.push_player1_position, self.push_player2_position, self.pop_player1_position, self.pop_player2_position = \
            self.player1_positions.append, self.player2_positions.append, self.player1_positions.pop, self.player2_positions.pop

        for pos in chain(self.player1_positions, self.player2_positions):
            self.grid_mask[pos] = False

        self.player1_length, self.player2_length = len(snake1.positions), len(snake2.positions)
        self.player1_pos, self.player1_prev_pos = self.player1_positions[-1], self.player1_positions[-2]
        self.player2_pos, self.player2_prev_pos = self.player2_positions[-1], self.player2_positions[-2]

        self.last_player, self.hash = -1, 0

    @property
    def shape(self) -> Tuple[int, int]:
        return self.width, self.height

    def is_empty_pos(self, pos: PosIdx) -> bool:
        return self.grid_mask[pos]

    def count_player_move_partitions(self, player: int) -> int:
        pos = self.player1_pos if player == 1 else self.player2_pos

        # Start from top-left, clockwise
        return count_move_partitions(
            [self.grid_mask[p] for p in self.EIGHT_WAY_POSITIONS[pos]]
        )

    def get_tail_pos(self, player: int) -> PosIdx:
        return self.player1_positions[-self.player1_length] \
            if player == 1 else self.player2_positions[-self.player2_length]

    def get_empty_mask(self) -> GridMask:
        return [*self.grid_mask]

    def get_player1_mask(self) -> GridMask:
        mask = [False] * len(self.grid_mask)
        for pos in self.player1_positions[-self.player1_length:]:
            mask[pos] = True
        return mask

    def get_player2_mask(self) -> GridMask:
        mask = [False] * len(self.grid_mask)
        for pos in self.player2_positions[-self.player2_length:]:
            mask[pos] = True
        return mask

    def can_move(self, player: int) -> bool:
        if player == 1:
            pos, prev_pos = self.player1_pos, self.player1_prev_pos
        else:
            pos, prev_pos = self.player2_pos, self.player2_prev_pos

        pos_options = self.FOUR_WAY_POSITIONS[pos]
        return self.grid_mask[pos_options[0]] or \
            self.grid_mask[pos_options[1]] or \
            self.grid_mask[pos_options[2]] or \
            self.grid_mask[pos_options[3]]
        # def is_empty(p):
        #     return self.lb <= self.grid[p] <= self.ub
        #
        # return any(filter(is_empty, self.FOUR_WAY_POSITIONS_FROM_POS[prev_pos][pos]))

    def count_moves(self, player: int) -> int:
        if player == 1:
            pos, prev_pos = self.player1_pos, self.player1_prev_pos
        else:
            pos, prev_pos = self.player2_pos, self.player2_prev_pos
        return sum([self.grid_mask[p] for p in self.FOUR_WAY_POSITIONS_COND[pos]])

    def iterate_valid_moves(self, player: int, order: Tuple[BoardMove] = MOVES) -> Iterator[BoardMove]:
        if player == 1:
            pos = self.player1_pos
        else:
            pos = self.player2_pos

        def can_do(m):
            return self.grid_mask[pos + self.MOVE_POS_OFFSET[m]]

        return filter(can_do, order)

        # pos = self.player1_pos if player == 1 else self.player2_pos
        # can_moves = (self.lb <= self.grid[pos + self.MOVE_POS_OFFSET[m]] <= self.ub for m in order)
        # return compress(order, can_moves)

    def get_valid_moves_ordered(self, player: int, order: Tuple[BoardMove] = MOVES) -> List[BoardMove]:
        pos = self.player1_pos if player == 1 else self.player2_pos
        moves = list(compress(
            MOVES,
            (self.grid_mask[p] for p in self.FOUR_WAY_POSITIONS[pos])
        ))  # faster than tuple() AND list comprehension
        return [x for _, x in sorted(zip(order, moves))]

    # performing a move increments the turn counter and places a new wall
    def perform_move(self, move: BoardMove, player: int) -> None:
        assert self.last_player != player
        direction = self.MOVE_POS_OFFSET[move]
        if player == 1:
            target_pos = self.player1_pos + direction
            tail_pos = self.player1_positions[-self.player1_length]

            # update game state
            ate_candy = target_pos in self.candies
            self.push_player1_position(target_pos)
            self.push_move_stack((
                self.player1_pos,
                self.player1_prev_pos,
                tail_pos,
                ate_candy,
                self.hash
            ))
            self.player1_prev_pos, self.player1_pos, self.grid_mask[self.player1_pos] = self.player1_pos, target_pos, False
            if ate_candy:
                self.player1_length += 1
                self.remove_candy(self.player1_pos)
            else:
                self.grid_mask[tail_pos] = True
        else:
            target_pos = self.player2_pos + direction
            tail_pos = self.player2_positions[-self.player2_length]

            # update game state
            ate_candy = target_pos in self.candies
            self.push_player2_position(target_pos)
            self.push_move_stack((
                self.player2_pos,
                self.player2_prev_pos,
                tail_pos,
                ate_candy,
                self.hash
            ))
            self.player2_prev_pos, self.player2_pos, self.grid_mask[self.player2_pos] = self.player2_pos, target_pos, False
            if ate_candy:
                self.player2_length += 1
                self.remove_candy(self.player2_pos)
            else:
                self.grid_mask[tail_pos] = True

        self.last_player, self.hash = player, 0

    def undo_move(self, player: int) -> None:
        assert self.last_player == player
        old_pos = self.player1_pos if player == 1 else self.player2_pos

        if player == 1:
            self.grid_mask[old_pos] = True
            self.player1_pos, self.player1_prev_pos, tail_pos, ate_candy, self.hash = \
                self.pop_move_stack()
            if ate_candy:
                self.player1_length -= 1
                self.spawn_candy(old_pos)
            else:
                self.grid_mask[tail_pos] = False
            self.pop_player1_position()
        else:
            self.grid_mask[old_pos] = True
            self.player2_pos, self.player2_prev_pos, tail_pos, ate_candy, self.hash = \
                self.pop_move_stack()
            if ate_candy:
                self.player2_length -= 1
                self.spawn_candy(old_pos)
            else:
                self.grid_mask[tail_pos] = False
            self.pop_player2_position()

        self.last_player = -self.last_player

    def count_free_space_bfs(self, mask: GridMask, pos: PosIdx, max_dist: int, lb: int, prev_pos: PosIdx = 0) -> int:
        mask[pos], free_space, cur_dist, pos_options, queue = \
            False, 1, 0, self.FOUR_WAY_POSITIONS_FROM_POS_COND, deque(maxlen=128)

        while free_space < lb and cur_dist < max_dist:
            for new_pos in pos_options[prev_pos][pos]:
                if mask[new_pos]:
                    mask[new_pos], free_space = False, free_space + 1
                    queue.append((new_pos, pos, cur_dist + 1))
            if not queue:
                break
            pos, prev_pos, cur_dist = queue.popleft()

        return free_space

    def count_free_space_dfs(self, mask: GridMask, pos: PosIdx, lb: int, max_dist: int, distance_map: Tuple[int]):
        stack, free_space, pos_options = [pos], 0, self.FOUR_WAY_POSITIONS_COND

        while stack and free_space < lb:
            pos = stack.pop()
            if not mask[pos] or distance_map[pos] > max_dist:
                continue
            mask[pos], free_space = False, free_space + 1
            stack.extend(pos_options[pos])

        return free_space

    def grid_as_np(self, grid: Grid) -> ndarray:
        a = np.array(grid)
        a.shape = (self.full_width, len(grid) // self.full_width)
        return a

    def copy(self) -> Self:
        return deepcopy(self)

    def __eq__(self, other) -> bool:
        return self.last_player == other.last_player and \
            self.player1_pos == other.player1_pos and \
            self.player2_pos == other.player2_pos and \
            self.player1_length == other.player1_length and \
            self.player2_length == other.player2_length and \
            self.player1_positions[-self.player1_length:] == other.player1_positions[-self.player1_length:] and \
            self.grid_mask == other.grid_mask

    def __hash__(self) -> int:
        if self.hash == 0:
            self.hash = hash((
                tuple(self.grid_mask),
                tuple(self.player1_positions[-self.player1_length:]),
                tuple(self.player2_positions[-self.player2_length:]),
                self.player1_pos,
                self.player2_pos,
                self.last_player
            ))
        return self.hash

    def approx_hash(self) -> int:
        return hash((
            tuple(self.grid_mask),
            self.player1_pos,
            self.player2_pos,
            self.last_player
        ))

    def __str__(self) -> str:
        str_grid = np.full((self.full_width, self.full_height), fill_value='_', dtype=str)
        str_grid[self.grid_as_np(self.get_player1_mask())] = 'a'
        str_grid[self.grid_as_np(self.get_player2_mask())] = 'b'
        str_grid[0:, (0, -1), ] = '-'
        str_grid[(0, -1), 0:] = '|'
        str_grid[(0, 0, -1, -1), (0, -1, -1, 0)] = '+'
        for pos in self.candies:
            str_grid[self.from_index(pos)] = '*'
        if self.player1_length > 0:
            str_grid[self.from_index(self.player1_pos)] = 'A'
            str_grid[self.from_index(self.player2_pos)] = 'B'

        return '\n' + np.array2string(np.flipud(str_grid.T), separator=''). \
            replace('[', ''). \
            replace(']', ''). \
            replace("'", ''). \
            replace(' ', ''). \
            replace('_', '·')

    def __repr__(self) -> str:
        return str(self)


def as_move(move: BoardMove) -> Move:
    return MOVE_MAP[move]


def distance(pos1: Pos, pos2: Pos) -> int:
    """L1 distance between the given positions"""
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])


def count_move_partitions(cells: List[bool]) -> int:
    assert len(cells) == 8

    n_moves = sum((cells[1], cells[3], cells[5], cells[7]))
    if n_moves == 1:
        return 1
    elif n_moves == 2:
        if cells[1] == cells[5]:
            # Tunnel case
            return 2
        else:
            # Corner case
            if cells[3]:  # R
                if cells[5]:  # B
                    return 1 if cells[4] else 2
                else:
                    return 1 if cells[2] else 2
            elif cells[5]:  # B
                return 1 if cells[6] else 2
            else:  # T
                return 1 if cells[0] else 2
    elif n_moves == 3:
        # find the non-empty move cell
        if not cells[1]:  # T
            return 3 - sum((cells[4], cells[6]))
        elif not cells[3]:  # R
            return 3 - sum((cells[0], cells[6]))
        elif not cells[5]:  # B
            return 3 - sum((cells[0], cells[2]))
        else:  # L
            return 3 - sum((cells[2], cells[4]))
    elif n_moves == 0:
        return 0
    else:
        return 1


def create_grid(width, height) -> GridMask:
    first_row = [False] * (height + 2)
    mat = [first_row] + [[False] + [True] * height + [False] for _ in range(width)] + [first_row]
    return list(chain.from_iterable(mat))
