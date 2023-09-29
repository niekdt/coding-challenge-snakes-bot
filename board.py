from collections import deque
from copy import deepcopy
from enum import IntEnum, auto, Enum
from itertools import compress, chain
from typing import List, Deque, TypeVar, Tuple, Iterator

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


MOVES = (BoardMove.LEFT, BoardMove.RIGHT, BoardMove.UP, BoardMove.DOWN)
POS_OFFSET = np.array((1, 1))
# for np access
ROW_OFFSET = (- 1, 1, 0, 0)
COL_OFFSET = (0, 0, 1, -1)

MOVE_MAP = {
    BoardMove.LEFT: Move.LEFT,
    BoardMove.RIGHT: Move.RIGHT,
    BoardMove.UP: Move.UP,
    BoardMove.DOWN: Move.DOWN
}

MOVE_TO_DIRECTION = {
    BoardMove.UP: (0, 1),
    BoardMove.DOWN: (0, -1),
    BoardMove.LEFT: (-1, 0),
    BoardMove.RIGHT: (1, 0),
}

OPPOSITE_MOVE = {
    BoardMove.LEFT: BoardMove.RIGHT,
    BoardMove.RIGHT: BoardMove.LEFT,
    BoardMove.UP: BoardMove.DOWN,
    BoardMove.DOWN: BoardMove.UP
}

TURN_LEFT_MOVE = {
    BoardMove.LEFT: BoardMove.DOWN,
    BoardMove.RIGHT: BoardMove.UP,
    BoardMove.UP: BoardMove.LEFT,
    BoardMove.DOWN: BoardMove.RIGHT
}

TURN_RIGHT_MOVE = {
    BoardMove.LEFT: BoardMove.UP,
    BoardMove.RIGHT: BoardMove.DOWN,
    BoardMove.UP: BoardMove.RIGHT,
    BoardMove.DOWN: BoardMove.LEFT
}

FIRST_MOVE_ORDER = {m: (m, TURN_LEFT_MOVE[m], TURN_RIGHT_MOVE[m], OPPOSITE_MOVE[m]) for m in MOVES}


class Board:
    def __init__(self, width: int, height: int) -> None:
        """Define an empty board of a given dimension"""
        assert width > 0
        assert height > 0

        self.width: int = width
        self.full_width: int = width + 2
        self.height: int = height
        self.full_height: int = height + 2
        self.grid: Grid = create_grid(width, height)
        self.candies: List[PosIdx] = []
        self.player1_pos: PosIdx = -10
        self.player2_pos: PosIdx = -10
        self.player1_head: int = 0
        self.player2_head: int = 0
        self.player1_length: int = 0
        self.player2_length: int = 0
        self.last_player = 1
        self.move_pos_stack: Deque[PosIdx] = deque(maxlen=128)
        self.move_head_stack: Deque[int] = deque(maxlen=128)
        self.move_candy_stack: Deque[bool] = deque(maxlen=128)
        self.pos_map = tuple([divmod(i, self.full_height) for i in range(0, self.full_width * self.full_height)])
        self.center: PosIdx = self.from_xy(int(width / 2) + 1, int(height / 2) + 1)
        self.MOVE_POS_OFFSET = {
            BoardMove.LEFT: -self.full_height,
            BoardMove.RIGHT: self.full_height,
            BoardMove.UP: 1,
            BoardMove.DOWN: -1
        }

    def from_pos(self, pos: Pos) -> PosIdx:
        assert 0 <= pos[0] < self.full_width
        assert 0 <= pos[1] < self.full_height
        return pos[0] * self.full_height + pos[1]

    def from_xy(self, x: int, y: int) -> PosIdx:
        assert 0 <= x < self.full_width
        assert 0 <= y < self.full_height
        return x * self.full_height + y

    def from_index(self, index: PosIdx) -> Pos:
        return self.pos_map[index]

    def index_left(self, index: PosIdx) -> PosIdx:
        return index - self.full_height

    def index_right(self, index: PosIdx) -> PosIdx:
        return index + self.full_height

    def index_up(self, index: PosIdx) -> PosIdx:
        return index + 1

    def index_down(self, index: PosIdx) -> PosIdx:
        return index - 1

    def index_top_left(self, index: PosIdx) -> PosIdx:
        return index - self.full_height + 1

    def index_top_right(self, index: PosIdx) -> PosIdx:
        return index + self.full_height + 1

    def index_bottom_left(self, index: PosIdx) -> PosIdx:
        return index - self.full_height - 1

    def index_bottom_right(self, index: PosIdx) -> PosIdx:
        return index + self.full_height - 1

    def spawn(self, pos1: Pos, pos2: Pos) -> None:
        """Spawn snakes of length 1 at the given positions
        Indices start from 0

        Merely for testing purposes"""
        self.set_state(
            snake1=Snake(id=0, positions=np.array([pos1])),
            snake2=Snake(id=1, positions=np.array([pos2])),
            candies=[]
        )

    def set_state(self, snake1: Snake, snake2: Snake, candies: List[np.array]) -> None:
        self.last_player = -1  # set P2 to have moved last

        # clear grid
        self.grid = create_grid(self.width, self.height)
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
            self.grid[self.from_xy(pos[0] + 1, pos[1] + 1)] = self.player1_head - i

        # head = p2_head, tail = p2_head + len - 1
        for i, pos in enumerate(snake2.positions):
            self.grid[self.from_xy(pos[0] + 1, pos[1] + 1)] = self.player2_head + i

        self.player1_pos = self.from_xy(1 + snake1.positions[0][0], 1 + snake1.positions[0][1])
        self.player2_pos = self.from_xy(1 + snake2.positions[0][0], 1 + snake2.positions[0][1])

        # spawn candies
        for pos in candies:
            self._spawn_candy(self.from_xy(pos[0] + 1, pos[1] + 1))

    def _spawn_candy(self, pos: PosIdx) -> None:
        self.candies.append(pos)

    def _remove_candy(self, pos: PosIdx) -> None:
        self.candies.remove(pos)

    @property
    def shape(self) -> Tuple[int, int]:
        return self.width, self.height

    def is_candy_pos(self, pos: PosIdx) -> bool:
        return pos in self.candies

    def is_empty_pos(self, pos: PosIdx) -> bool:
        return self.player2_head + self.player2_length <= self.grid[pos] <= self.player1_head - self.player1_length

    def is_player_forced(self, player: int) -> bool:
        pos = self.player1_pos if player == 1 else self.player2_pos
        lb = self.player2_head + self.player2_length
        ub = self.player1_head - self.player1_length

        return sum(
            (
                lb <= self.grid[self.index_left(pos)] <= ub,
                lb <= self.grid[self.index_right(pos)] <= ub,
                lb <= self.grid[self.index_up(pos)] <= ub,
                lb <= self.grid[self.index_down(pos)] <= ub
            )
        ) == 1

    def count_player_move_partitions(self, player: int) -> int:
        pos = self.player1_pos if player == 1 else self.player2_pos
        lb = self.player2_head + self.player2_length
        ub = self.player1_head - self.player1_length

        # Start from top-left, clockwise
        cells = (
            lb <= self.grid[self.index_top_left(pos)] <= ub,  # TL = 0
            lb <= self.grid[self.index_up(pos)] <= ub,  # T = 1
            lb <= self.grid[self.index_top_right(pos)] <= ub,  # TR = 2
            lb <= self.grid[self.index_right(pos)] <= ub,  # R = 3
            lb <= self.grid[self.index_bottom_right(pos)] <= ub,  # BR = 4
            lb <= self.grid[self.index_down(pos)] <= ub,  # B = 5
            lb <= self.grid[self.index_bottom_left(pos)] <= ub,  # BL = 6
            lb <= self.grid[self.index_left(pos)] <= ub  # L = 7
        )

        return count_move_partitions(cells)

    def get_candies(self) -> List[PosIdx]:
        return self.candies

    def get_empty_mask(self) -> GridMask:
        lb = self.player2_head + self.player2_length
        ub = self.player1_head - self.player1_length
        return [lb <= v <= ub for v in self.grid]

    def get_player1_mask(self) -> GridMask:
        ub = self.player1_head - self.player1_length
        return [v > ub for v in self.grid]

    def get_player2_mask(self) -> GridMask:
        lb = self.player2_head + self.player2_length
        return [v < lb for v in self.grid]

    def get_player_mask(self, player: int) -> GridMask:
        if player == 1:
            return self.get_player1_mask()
        else:
            return self.get_player2_mask()

    def has_candy(self) -> bool:
        return len(self.candies) > 0

    def can_move(self, player: int) -> bool:
        pos = self.player1_pos if player == 1 else self.player2_pos

        lb = self.player2_head + self.player2_length
        ub = self.player1_head - self.player1_length

        return lb <= self.grid[self.index_left(pos)] <= ub or \
            lb <= self.grid[self.index_right(pos)] <= ub or \
            lb <= self.grid[self.index_up(pos)] <= ub or \
            lb <= self.grid[self.index_down(pos)] <= ub

    def count_moves(self, player: int) -> int:
        pos = self.player1_pos if player == 1 else self.player2_pos

        lb = self.player2_head + self.player2_length
        ub = self.player1_head - self.player1_length

        return sum(
            (
                lb <= self.grid[self.index_left(pos)] <= ub,
                lb <= self.grid[self.index_right(pos)] <= ub,
                lb <= self.grid[self.index_up(pos)] <= ub,
                lb <= self.grid[self.index_down(pos)] <= ub
            )
        )

    def can_do_move(self, move: BoardMove, pos: PosIdx) -> bool:
        move_dir = self.MOVE_POS_OFFSET[move]
        lb = self.player2_head + self.player2_length
        ub = self.player1_head - self.player1_length
        return lb <= self.grid[pos + move_dir] <= ub

    def can_player1_do_move(self, move: BoardMove) -> bool:
        return self.can_do_move(move, self.player1_pos)

    def can_player2_do_move(self, move: BoardMove) -> bool:
        return self.can_do_move(move, self.player2_pos)

    def iterate_valid_moves(self, player: int, order: Tuple[BoardMove] = MOVES) -> Iterator[BoardMove]:
        if player == 1:
            return filter(self.can_player1_do_move, order)
        else:
            return filter(self.can_player2_do_move, order)

    def get_valid_moves(self, player: int) -> List[BoardMove]:
        if player == 1:
            pos = self.player1_pos
        else:
            pos = self.player2_pos

        lb = self.player2_head + self.player2_length
        ub = self.player1_head - self.player1_length

        can_moves = (
            lb <= self.grid[self.index_left(pos)] <= ub,
            lb <= self.grid[self.index_right(pos)] <= ub,
            lb <= self.grid[self.index_up(pos)] <= ub,
            lb <= self.grid[self.index_down(pos)] <= ub
        )

        return list(compress(MOVES, can_moves))  # faster than tuple() AND list comprehension

    def get_valid_moves_ordered(self, player: int, order: Tuple[BoardMove] = MOVES) -> List[BoardMove]:
        moves = self.get_valid_moves(player=player)
        return [x for _, x in sorted(zip(order, moves))]

    # performing a move increments the turn counter and places a new wall
    def perform_move(self, move: BoardMove, player: int) -> None:
        assert self.last_player != player
        direction = self.MOVE_POS_OFFSET[move]

        if player == 1:
            target_pos = self.player1_pos + direction

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
            target_pos = self.player2_pos + direction

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

    def undo_move(self, player: int) -> None:
        assert self.last_player == player
        ate_candy = self.move_candy_stack.pop()

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

    def count_free_space_bfs(self, mask: GridMask, pos: PosIdx, max_dist: int, lb: int) -> int:
        assert isinstance(mask, list)
        mask[pos] = False
        free_space = 0
        queue: Deque[PosIdx] = deque(maxlen=max_dist * 4)
        dqueue: Deque[int] = deque(maxlen=max_dist * 4)
        queue.append(pos)
        dqueue.append(0)

        while queue and free_space < lb:
            cur_pos = queue.popleft()
            cur_dist = dqueue.popleft()
            free_space += 1
            mask[cur_pos] = False

            if cur_dist < max_dist:
                candidate_positions = (
                    self.index_left(cur_pos),
                    self.index_right(cur_pos),
                    self.index_up(cur_pos),
                    self.index_down(cur_pos)
                )
                free_mask = [mask[p] for p in candidate_positions]
                new_positions = list(compress(candidate_positions, free_mask))
                for p in new_positions:
                    mask[p] = False
                queue.extend(new_positions)
                dqueue.extend([cur_dist + 1] * len(new_positions))

        return free_space

    def distance(self, pos1: PosIdx, pos2: PosIdx) -> int:
        return distance(self.from_index(pos1), self.from_index(pos2))

    def count_free_space_dfs(self, mask: GridMask, pos: PosIdx, lb: int, max_dist: int, ref_pos: PosIdx):
        """
        Compute a lower bound on the amount of free space, including the current position
        :param mask: Padded logical matrix
        :param pos: Position to scan from (inclusive)
        :param lb: Count at least `lb` number of free spaces, if possible
        :param max_dist: Maximum L1 distance from the origin to consider
        :param ref_pos: Origin
        :return: Counted free space
        """
        if lb == 0 or self.distance(pos, ref_pos) >= max_dist:
            return 1

        mask[pos] = False

        candidate_positions = (
            self.index_left(pos),
            self.index_right(pos),
            self.index_up(pos),
            self.index_down(pos)
        )

        return 1 + sum(
            [
                self.count_free_space_dfs(mask, pos=p, lb=lb - 1, max_dist=max_dist, ref_pos=ref_pos)
                for p in candidate_positions if mask[p]
            ]
        )

    def grid_as_np(self, grid: Grid) -> ndarray:
        a = np.array(grid)
        a.shape = (self.full_width, len(grid) // self.full_width)
        return a

    def copy(self) -> Self:
        return deepcopy(self)

    def __len__(self) -> int:
        return self.width * self.height

    def __eq__(self, other) -> bool:
        return self.player1_head == other.player1_head and \
            self.player2_head == other.player2_head and \
            self.player1_length == other.player1_length and \
            self.player2_length == other.player2_length and \
            self.last_player == other.last_player and \
            self.player1_pos == other.player1_pos and \
            self.player2_pos == other.player2_pos and \
            self.grid == other.grid

    def __hash__(self) -> int:
        """Hash of the exact game state"""
        return hash((
            tuple(self.grid),
            self.last_player
        ))

    def approx_hash(self) -> int:
        """Hash of the game state only considering blocked cells, player positions, candies, and last player"""
        return hash((
            tuple(self.grid),
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


def count_move_partitions(cells: Tuple[bool, ...]) -> int:
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


def create_grid(width, height) -> Grid:
    wall = 10000
    first_row = [wall] * (height + 2)
    mat = [first_row] + [[wall] + [0] * height + [wall] for i in range(width)] + [first_row]

    return list(chain.from_iterable(mat))
