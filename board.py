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
        self.lb: int = 0
        self.ub: int = 0
        self.last_player = 1
        self.move_pos_stack: Deque[PosIdx] = deque(maxlen=128)
        self.move_head_stack: Deque[int] = deque(maxlen=128)
        self.move_candy_stack: Deque[bool] = deque(maxlen=128)
        self.push_move_pos_stack = self.move_pos_stack.append
        self.push_move_head_stack = self.move_head_stack.append
        self.push_move_candy_stack = self.move_candy_stack.append
        self.pop_move_pos_stack = self.move_pos_stack.pop
        self.pop_move_head_stack = self.move_head_stack.pop
        self.pop_move_candy_stack = self.move_candy_stack.pop
        self.spawn_candy = self.candies.append
        self.remove_candy = self.candies.remove
        self.pos_map = tuple([divmod(i, self.full_height) for i in range(0, self.full_width * self.full_height)])
        self.center: PosIdx = self.from_xy(int(width / 2) + 1, int(height / 2) + 1)
        self.MOVE_POS_OFFSET = {
            BoardMove.LEFT: -self.full_height,
            BoardMove.RIGHT: self.full_height,
            BoardMove.UP: 1,
            BoardMove.DOWN: -1
        }
        self.FOUR_WAY_POS_OFFSETS = (
            -self.full_height,
            self.full_height,
            1,
            -1
        )  # L, R, U, D
        self.FOUR_WAY_POS_OFFSETS = (
            -self.full_height,
            self.full_height,
            1,
            -1
        )
        self.DIR_UP_LEFT = -self.full_height + 1
        self.DIR_UP = 1
        self.DIR_UP_RIGHT = self.full_height + 1
        self.DIR_RIGHT = self.full_height
        self.DIR_DOWN_RIGHT = self.full_height - 1
        self.DIR_DOWN = -1
        self.DIR_DOWN_LEFT = -self.full_height - 1
        self.DIR_LEFT = -self.full_height

        self.EIGHT_WAY_POS_OFFSETS = (
            -self.full_height + 1,  # TL
            1,  # T
            self.full_height + 1,  # TR
            self.full_height,  # R
            self.full_height - 1,  # BR
            -1,  # B
            -self.full_height - 1,  # BL
            -self.full_height  # L
        )

        self.FOUR_WAY_CANDIDATE_POSITIONS = [
            (p + self.DIR_LEFT, p + self.DIR_RIGHT, p + 1, p - 1) for p in range(len(self.grid))
        ]
        self.EIGHT_WAY_CANDIDATE_POSITIONS = [
            (p + self.DIR_UP_LEFT, p + self.DIR_UP, p + self.DIR_UP_RIGHT, p + self.DIR_RIGHT,
             p + self.DIR_DOWN_RIGHT, p + self.DIR_DOWN, p + self.DIR_DOWN_LEFT, p + self.DIR_LEFT)
            for p in range(len(self.grid))
        ]
        self.DISTANCE = [
            [distance(self.from_index(p1), self.from_index(p2)) for p2 in range(len(self.grid))]
            for p1 in range(len(self.grid))
        ]

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
        self.lb = self.player2_head + self.player2_length
        self.ub = self.player1_head - self.player1_length

        # head = p1_head, tail = p1_head - len + 1
        for i, pos in enumerate(snake1.positions):
            self.grid[self.from_pos(pos) + self.DIR_UP_RIGHT] = self.player1_head - i

        # head = p2_head, tail = p2_head + len - 1
        for i, pos in enumerate(snake2.positions):
            self.grid[self.from_pos(pos) + self.DIR_UP_RIGHT] = self.player2_head + i

        self.player1_pos = self.from_pos(snake1.positions[0]) + self.DIR_UP_RIGHT
        self.player2_pos = self.from_pos(snake2.positions[0]) + self.DIR_UP_RIGHT

        # spawn candies
        for pos in candies:
            self.spawn_candy(self.from_pos(pos) + self.DIR_UP_RIGHT)

    @property
    def shape(self) -> Tuple[int, int]:
        return self.width, self.height

    def is_empty_pos(self, pos: PosIdx) -> bool:
        return self.player2_head + self.player2_length <= self.grid[pos] <= self.player1_head - self.player1_length

    def is_player_forced(self, player: int) -> bool:
        pos = self.player1_pos if player == 1 else self.player2_pos

        return sum(
            (
                self.lb <= self.grid[pos + self.DIR_LEFT] <= self.ub,
                self.lb <= self.grid[pos + self.DIR_RIGHT] <= self.ub,
                self.lb <= self.grid[pos + 1] <= self.ub,
                self.lb <= self.grid[pos - 1] <= self.ub
            )
        ) == 1

    def count_player_move_partitions(self, player: int) -> int:
        pos = self.player1_pos if player == 1 else self.player2_pos

        # Start from top-left, clockwise
        return count_move_partitions(
            [self.lb <= self.grid[p] <= self.ub for p in self.EIGHT_WAY_CANDIDATE_POSITIONS[pos]]
        )

    def get_empty_mask(self) -> GridMask:
        return [self.lb <= v <= self.ub for v in self.grid]

    def get_player1_mask(self) -> GridMask:
        return [v > self.ub for v in self.grid]

    def get_player2_mask(self) -> GridMask:
        return [v < self.lb for v in self.grid]

    def can_move(self, player: int) -> bool:
        pos = self.player1_pos if player == 1 else self.player2_pos
        candidate_positions = self.FOUR_WAY_CANDIDATE_POSITIONS[pos]
        return self.lb <= self.grid[candidate_positions[0]] <= self.ub or \
            self.lb <= self.grid[candidate_positions[1]] <= self.ub or \
            self.lb <= self.grid[candidate_positions[2]] <= self.ub or \
            self.lb <= self.grid[candidate_positions[3]] <= self.ub

    def count_moves(self, player: int) -> int:
        pos = self.player1_pos if player == 1 else self.player2_pos
        return sum([self.lb <= self.grid[p] <= self.ub for p in self.FOUR_WAY_CANDIDATE_POSITIONS[pos]])

    def iterate_valid_moves(self, player: int, order: Tuple[BoardMove] = MOVES) -> Iterator[BoardMove]:
        pos = self.player1_pos if player == 1 else self.player2_pos
        def can_do(m): return self.lb <= self.grid[pos + self.MOVE_POS_OFFSET[m]] <= self.ub
        return filter(can_do, order)

        # pos = self.player1_pos if player == 1 else self.player2_pos
        # can_moves = (self.lb <= self.grid[pos + self.MOVE_POS_OFFSET[m]] <= self.ub for m in order)
        # return compress(order, can_moves)

    def get_valid_moves_ordered(self, player: int, order: Tuple[BoardMove] = MOVES) -> List[BoardMove]:
        pos = self.player1_pos if player == 1 else self.player2_pos
        moves = list(compress(
            MOVES,
            (self.lb <= self.grid[p] <= self.ub for p in self.FOUR_WAY_CANDIDATE_POSITIONS[pos])
        ))  # faster than tuple() AND list comprehension
        return [x for _, x in sorted(zip(order, moves))]

    # performing a move increments the turn counter and places a new wall
    def perform_move(self, move: BoardMove, player: int) -> None:
        assert self.last_player != player
        direction = self.MOVE_POS_OFFSET[move]

        if player == 1:
            target_pos = self.player1_pos + direction

            # update game state
            ate_candy = target_pos in self.candies
            self.push_move_candy_stack(ate_candy)
            self.push_move_pos_stack(self.player1_pos)
            self.push_move_head_stack(self.grid[target_pos])
            self.player1_pos = target_pos
            self.player1_head += 1
            self.grid[self.player1_pos] = self.player1_head
            self.ub = self.player1_head - self.player1_length
            if target_pos in self.candies:
                self.player1_length += 1
                self.candies.remove(self.player1_pos)
        else:
            target_pos = self.player2_pos + direction

            # update game state
            ate_candy = target_pos in self.candies
            self.push_move_candy_stack(ate_candy)
            self.push_move_pos_stack(self.player2_pos)
            self.push_move_head_stack(self.grid[target_pos])
            self.player2_pos = target_pos
            self.player2_head -= 1  # the only difference in logic between the players
            self.grid[self.player2_pos] = self.player2_head
            self.lb = self.player2_head + self.player2_length
            if ate_candy:
                self.player2_length += 1
                self.candies.remove(self.player2_pos)

        self.last_player = player

    def undo_move(self, player: int) -> None:
        assert self.last_player == player
        ate_candy = self.pop_move_candy_stack()

        if player == 1:
            if ate_candy:
                self.player1_length -= 1
                self.spawn_candy(self.player1_pos)
            self.grid[self.player1_pos] = self.pop_move_head_stack()
            self.player1_pos = self.pop_move_pos_stack()
            self.player1_head -= 1
            self.ub = self.player1_head - self.player1_length
        else:
            if ate_candy:
                self.player2_length -= 1
                self.spawn_candy(self.player2_pos)
            self.grid[self.player2_pos] = self.pop_move_head_stack()
            self.player2_pos = self.pop_move_pos_stack()
            self.player2_head += 1
            self.lb = self.player2_head + self.player2_length

        self.last_player = -self.last_player

    def count_free_space_bfs(self, mask: GridMask, pos: PosIdx, max_dist: int, lb: int) -> int:
        candidate_pos_cache = self.FOUR_WAY_CANDIDATE_POSITIONS
        mask[pos] = False
        free_space = 0
        queue: Deque[Tuple[PosIdx, int]] = deque(maxlen=max_dist * 4)
        queue.append((pos, 0))

        def is_pos_empty(p): return mask[p]

        while queue and free_space < lb:
            cur_pos, cur_dist = queue.popleft()
            free_space += 1
            mask[cur_pos] = False

            if cur_dist < max_dist:
                # new_positions = list(compress(candidate_positions, [mask[p] for p in candidate_positions])) # Slow
                # new_positions = list(filter(is_pos_empty, candidate_pos_cache[cur_pos])) # Pretty fast
                # for new_pos in [p for p in candidate_pos_cache[cur_pos] if mask[p]]: # Slow
                # for new_pos in list(compress(candidate_positions, [mask[p] for p in candidate_positions])): # Slow^2

                for new_pos in filter(is_pos_empty, candidate_pos_cache[cur_pos]):
                    mask[new_pos] = False
                    queue.append((new_pos, cur_dist + 1))

        return free_space

    def count_free_space_dfs(self, mask: GridMask, pos: PosIdx, lb: int, max_dist: int, distance_map: Tuple[int]):
        """
        Compute a lower bound on the amount of free space, including the current position
        Uses depth-first search using a stack
        :param mask: Padded logical matrix
        :param pos: Position to scan from (inclusive)
        :param lb: Count at least `lb` number of free spaces, if possible
        :param max_dist: Maximum L1 distance from the origin to consider
        :param distance_map: A mapping indicating the distance for the given position
        :return: Counted free space
        """
        candidate_pos_cache = self.FOUR_WAY_CANDIDATE_POSITIONS
        mask[pos] = False
        free_space = 0
        stack: Deque[PosIdx] = deque(maxlen=128)
        stack.append(pos)

        def is_pos_empty(p): return mask[p]

        while stack and free_space < lb:
            cur_pos = stack.pop()

            if distance_map[pos] > max_dist:  # faster than doing the check in is_pos_empty() for some reason
                continue
            free_space += 1
            mask[cur_pos] = False

            for new_pos in filter(is_pos_empty, candidate_pos_cache[cur_pos]):
                mask[new_pos] = False
                stack.append(new_pos)

        return free_space

    def grid_as_np(self, grid: Grid) -> ndarray:
        a = np.array(grid)
        a.shape = (self.full_width, len(grid) // self.full_width)
        return a

    def copy(self) -> Self:
        return deepcopy(self)

    def __len__(self) -> int:
        return self.width * self.height

    def __eq__(self, other) -> bool:
        return self.ub == other.ub and \
            self.lb == other.lb and \
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


def create_grid(width, height) -> Grid:
    wall = 10000
    first_row = [wall] * (height + 2)
    mat = [first_row] + [[wall] + [0] * height + [wall] for i in range(width)] + [first_row]

    return list(chain.from_iterable(mat))
