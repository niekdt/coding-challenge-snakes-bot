from enum import IntEnum, auto, Enum
from functools import lru_cache
from itertools import compress, chain
from typing import List, TypeVar, Tuple, Iterator, Type, Dict

import numpy as np
from numpy import ndarray

from ...constants import Move
from ...game import Game
from ...snake import Snake

Self = TypeVar('Self', bound='Board')
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
    UP_LEFT = auto()
    UP = auto()
    UP_RIGHT = auto()
    RIGHT = auto()
    DOWN_RIGHT = auto()
    DOWN = auto()
    DOWN_LEFT = auto()
    LEFT = auto()
    __str__ = Enum.__str__


MOVES = (BoardMove.LEFT, BoardMove.RIGHT, BoardMove.UP, BoardMove.DOWN)
DIRECTIONS = tuple(d for d in Direction)

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

FIRST_MOVE_ORDER = {m: (m, TURN_LEFT_MOVE[m], TURN_RIGHT_MOVE[m]) for m in MOVES}


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
        'DISTANCE', 'EIGHT_WAY_DISTANCE',
        'DISTANCE_TO_EDGE', 'DISTANCE_TO_CENTER',
        'FOUR_WAY_POSITIONS_COND', 'FOUR_WAY_POSITIONS',
        'FOUR_WAY_POSITIONS_FROM_POS_COND', 'FOUR_WAY_POSITIONS_FROM_POS',
        'EIGHT_WAY_POSITIONS_COND', 'EIGHT_WAY_POSITIONS',
        'EIGHT_WAY_POSITIONS_FROM_POS_COND', 'EIGHT_WAY_POSITIONS_FROM_POS',
        'MOVE_POS_OFFSET', 'MOVE_FROM_TRANS',
        'MOVES_FROM_POS', 'MOVES_FROM_POS_COND', 'MOVES_FROM_POS_TRANS',
        'GAME_POS_OFFSET',
        'TERRITORY1', 'TERRITORY2', 'DELTA_TERRITORY'
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
        self.pos_map = tuple(
            [pos_to_xy(p, self.full_width, self.full_height) for p in range(self.full_width * self.full_height)]
        )
        self.center: PosIdx = self.from_xy(int(width / 2) + 1, int(height / 2) + 1)

        self.MOVE_POS_OFFSET = generate_move_offsets_map(self.full_width, self.full_height)
        self.DISTANCE = generate_l1_distance_lookup(self.full_width, self.full_height)
        self.EIGHT_WAY_DISTANCE = generate_chebyshev_distance_lookup(self.full_width, self.full_height)
        self.DISTANCE_TO_EDGE = generate_edge_distance_lookup(self.full_width, self.full_height)
        self.DISTANCE_TO_CENTER = generate_center_distance_lookup(self.full_width, self.full_height)

        self.GAME_POS_OFFSET = generate_direction_offsets_map(self.full_width, self.full_height)[Direction.UP_RIGHT]

        def _move_from_trans(pos_from, pos_to):
            if abs(pos_to - pos_from) == 1:
                return BoardMove.UP if pos_to > pos_from else BoardMove.DOWN
            else:
                return BoardMove.RIGHT if pos_to > pos_from else BoardMove.LEFT

        self.MOVE_FROM_TRANS: List[List[BoardMove]] = [
            [
                _move_from_trans(p_from, p_to)
                for p_to in range(len(self.grid_mask))
            ]
            for p_from in range(len(self.grid_mask))
        ]

        def is_within_bounds(pos):
            if pos < 0 or pos >= len(self.grid_mask):
                return False
            else:
                x, y = self.from_index(pos)
                return 0 < x < self.full_width - 1 and 0 < y < self.full_height - 1

        self.FOUR_WAY_POSITIONS = generate_4way_positions(self.full_width, self.full_height)
        self.FOUR_WAY_POSITIONS_COND = generate_4way_bounded_positions(self.full_width, self.full_height)

        self.MOVES_FROM_POS_COND: List[List[PosIdx, ...]] = [
            [m for m in MOVES if is_within_bounds(p + self.MOVE_POS_OFFSET[m])]
            for p in range(len(self.grid_mask))
        ]

        def _get_transitional_positions(pos_old: PosIdx, pos_new: PosIdx, pos_options: List) -> Tuple[PosIdx, ...]:
            candidate_positions = pos_options[pos_new]
            if pos_old in candidate_positions:
                return tuple(p for p in candidate_positions if p != pos_old)
            else:
                return candidate_positions

        self.FOUR_WAY_POSITIONS_FROM_POS_COND: List[List[Tuple[PosIdx, ...]]] = [
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

        self.EIGHT_WAY_POSITIONS = generate_8way_positions(self.full_width, self.full_height)
        self.EIGHT_WAY_POSITIONS_COND = generate_8way_bounded_positions(self.full_width, self.full_height)

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

        # Get moves for a given (from, to) position pair, with the first returned move being the last performed move
        self.MOVES_FROM_POS_TRANS: List[List[Tuple[BoardMove, ...], ...]] = [
            [
                tuple(
                    m for m in FIRST_MOVE_ORDER[self.MOVE_FROM_TRANS[pos_old][pos_new]]
                    if is_within_bounds(pos_new + self.MOVE_POS_OFFSET[m])
                )
                for pos_new in range(len(self.grid_mask))
            ]
            for pos_old in range(len(self.grid_mask))
        ]

        self.TERRITORY1, self.TERRITORY2, self.DELTA_TERRITORY = \
            generate_territory_lookup(self.full_width, self.full_height)

    def from_pos(self, pos: Pos) -> PosIdx:
        return pos[0] * self.full_height + pos[1]

    def from_xy(self, x: int, y: int) -> PosIdx:
        return x * self.full_height + y

    def from_index(self, index: PosIdx) -> Pos:
        return self.pos_map[index]

    def set_state(
            self,
            player1_positions: List[PosIdx],
            player2_positions: List[PosIdx],
            candy_positions: List[PosIdx]
    ) -> None:
        # player head is last element
        self.player1_positions = player1_positions
        self.player2_positions = player2_positions
        self.candies = candy_positions

        # clear grid and move stack
        self.grid_mask = create_grid(self.width, self.height)
        self.move_stack = []
        self.push_move_stack = self.move_stack.append
        self.pop_move_stack = self.move_stack.pop
        self.spawn_candy = self.candies.append
        self.remove_candy = self.candies.remove

        self.push_player1_position = self.player1_positions.append
        self.push_player2_position = self.player2_positions.append
        self.pop_player1_position = self.player1_positions.pop
        self.pop_player2_position = self.player2_positions.pop

        for pos in chain(self.player1_positions, self.player2_positions):
            self.grid_mask[pos] = False

        self.player1_length = len(player1_positions)
        self.player2_length = len(player2_positions)
        self.player1_pos = self.player1_positions[-1]
        self.player1_prev_pos = self.player1_positions[-2]
        self.player2_pos = self.player2_positions[-1]
        self.player2_prev_pos = self.player2_positions[-2]

        self.last_player = -1
        self.hash = 0

    def set_state_from_game(self, snake1: Snake, snake2: Snake, candies: List[np.array]) -> None:
        player1_positions = [self.from_pos(pos) + self.GAME_POS_OFFSET for pos in reversed(snake1.positions)]
        player2_positions = [self.from_pos(pos) + self.GAME_POS_OFFSET for pos in reversed(snake2.positions)]
        candy_positions = [self.from_pos(pos) + self.GAME_POS_OFFSET for pos in candies]

        self.set_state(player1_positions, player2_positions, candy_positions)

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

    def get_player_positions(self, player: int) -> List[PosIdx]:
        if player == 1:
            return self.player1_positions[-self.player1_length:]
        else:
            return self.player2_positions[-self.player2_length:]

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

    def are_players_adjacent(self) -> bool:
        return self.DISTANCE[self.player1_pos][self.player2_pos] == 1

    def are_players_near(self) -> bool:
        return self.EIGHT_WAY_DISTANCE[self.player1_pos][self.player2_pos] == 1

    def get_edge_trapped_player(self, player) -> int:
        # Returns 1 if P1 is trapped, -1 if P2 is trapped, 0 if neither are trapped
        if self.are_players_adjacent():
            # players are alongside
            if self.DISTANCE_TO_EDGE[self.player1_pos] == 0:
                # P1 is at the edge, so P2 must not be
                return 1 if self.DISTANCE_TO_EDGE[self.player2_pos] > 0 else 0
            elif self.DISTANCE_TO_EDGE[self.player2_pos] == 0:
                # P1 is not at the edge, and P2 is
                return -1
            else:
                return 0
        return 0

    def can_move(self, player: int) -> bool:
        if player == 1:
            pos_options = self.FOUR_WAY_POSITIONS[self.player1_pos]
        else:
            pos_options = self.FOUR_WAY_POSITIONS[self.player2_pos]

        return self.grid_mask[pos_options[0]] or \
            self.grid_mask[pos_options[1]] or \
            self.grid_mask[pos_options[2]] or \
            self.grid_mask[pos_options[3]]

        # return any(self.grid_mask[p] for p in self.FOUR_WAY_POSITIONS_FROM_POS_COND[prev_pos][pos])
        # def is_empty(p):
        #     return self.lb <= self.grid[p] <= self.ub
        #
        # return any(filter(is_empty, self.FOUR_WAY_POSITIONS_FROM_POS[prev_pos][pos]))

    def count_moves(self, player: int) -> int:
        if player == 1:
            return sum([
                self.grid_mask[p]
                for p in self.FOUR_WAY_POSITIONS_FROM_POS_COND[self.player1_prev_pos][self.player1_pos]
            ])
        else:
            return sum([
                self.grid_mask[p]
                for p in self.FOUR_WAY_POSITIONS_FROM_POS_COND[self.player2_prev_pos][self.player2_pos]
            ])

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

    def perform_move(self, move: BoardMove, player: int) -> None:
        assert self.last_player != player
        direction = self.MOVE_POS_OFFSET[move]
        if player == 1:
            target_pos = self.player1_pos + direction
            assert self.is_empty_pos(target_pos), 'P1 tried illegal move'
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
            self.player1_prev_pos = self.player1_pos
            self.player1_pos = target_pos
            self.grid_mask[self.player1_pos] = False
            if ate_candy:
                self.player1_length += 1
                self.remove_candy(self.player1_pos)
            else:
                self.grid_mask[tail_pos] = True
        else:
            target_pos = self.player2_pos + direction
            assert self.is_empty_pos(target_pos), 'P2 tried illegal move'
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
            self.player2_prev_pos = self.player2_pos
            self.player2_pos = target_pos
            self.grid_mask[self.player2_pos] = False
            if ate_candy:
                self.player2_length += 1
                self.remove_candy(self.player2_pos)
            else:
                self.grid_mask[tail_pos] = True

        self.last_player = player
        self.hash = 0

    def undo_move(self, player: int) -> None:
        assert self.last_player == player
        old_pos = self.player1_pos if player == 1 else self.player2_pos

        if player == 1:
            self.grid_mask[old_pos] = True
            self.player1_pos, self.player1_prev_pos, tail_pos, ate_candy, self.hash = self.pop_move_stack()
            if ate_candy:
                self.player1_length -= 1
                self.spawn_candy(old_pos)
            else:
                self.grid_mask[tail_pos] = False
            self.pop_player1_position()
        else:
            self.grid_mask[old_pos] = True
            self.player2_pos, self.player2_prev_pos, tail_pos, ate_candy, self.hash = self.pop_move_stack()
            if ate_candy:
                self.player2_length -= 1
                self.spawn_candy(old_pos)
            else:
                self.grid_mask[tail_pos] = False
            self.pop_player2_position()

        self.last_player = -self.last_player

    def grid_as_np(self, grid: Grid) -> ndarray:
        a = np.array(grid)
        a.shape = (self.full_width, len(grid) // self.full_width)
        return a

    def copy(self) -> Self:
        other = Board(self.width, self.height)
        other.grid_mask = [*self.grid_mask]
        other.candies = [*self.candies]
        other.move_stack = [*self.move_stack]
        other.last_player = self.last_player
        other.player1_positions = [*self.player1_positions]
        other.player2_positions = [*self.player2_positions]
        other.player1_pos = self.player1_pos
        other.player2_pos = self.player2_pos
        other.player1_length = self.player1_length
        other.player2_length = self.player2_length
        other.player1_prev_pos = self.player1_prev_pos
        other.player2_prev_pos = self.player2_prev_pos
        other.hash = self.hash

        other.push_move_stack = other.move_stack.append
        other.pop_move_stack = other.move_stack.pop
        other.spawn_candy = other.candies.append
        other.remove_candy = other.candies.remove
        other.push_player1_position = other.player1_positions.append
        other.push_player2_position = other.player2_positions.append
        other.pop_player1_position = other.player1_positions.pop
        other.pop_player2_position = other.player2_positions.pop
        return other

    def as_game(self, bot1: Type, bot2: Type) -> Game:
        p1_positions = [
            list(self.from_index(p - self.GAME_POS_OFFSET))
            for p in reversed(self.get_player_positions(player=1))
        ]
        p2_positions = [
            list(self.from_index(p - self.GAME_POS_OFFSET))
            for p in reversed(self.get_player_positions(player=-1))
        ]
        snake1 = Snake(id=0, positions=np.array(p1_positions))
        snake2 = Snake(id=1, positions=np.array(p2_positions))
        candies = [np.array(self.from_index(p - self.GAME_POS_OFFSET)) for p in self.candies]
        return Game(grid_size=self.shape, agents={0: bot1, 1: bot2}, snakes=[snake1, snake2], candies=candies)

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
                tuple(self.player1_positions[-self.player1_length:]),
                tuple(self.player2_positions[-self.player2_length:])
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
            replace('_', '·') + \
            '\nRepr: ' + \
            repr(self)

    def __repr__(self) -> str:
        return f'{self.width:d}x{self.height:d}c[' + \
            ','.join(f'{c:d}' for c in self.candies) + \
            ']a[' + \
            ','.join(f'{p:d}' for p in reversed(self.get_player_positions(player=1))) + \
            ']b[' + \
            ','.join(f'{p:d}' for p in reversed(self.get_player_positions(player=-1))) + \
            ']'


def from_repr(x: str) -> Board:
    size_str, pos_str = x.split('c[')
    w_str, h_str = size_str.split('x')
    w = int(w_str)
    h = int(h_str)
    board = Board(w, h)
    candy_str, players_str = pos_str.split(']a[')
    if candy_str:
        candy_idc = [int(p) for p in candy_str.split(',')]
    else:
        candy_idc = []
    p1_str, p2_str = players_str.split(']b[')
    if p1_str and p2_str:
        p1_idc = [int(p) for p in p1_str.split(',')]
        p2_idc = [int(p) for p in p2_str[:-1].split(',')]
    else:
        return board

    board.set_state(list(reversed(p1_idc)), list(reversed(p2_idc)), candy_idc)
    return board


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


def pos_to_xy(pos: PosIdx, w: int, h: int) -> Tuple[int, int]:
    assert w >= 3
    assert h >= 3
    return divmod(pos, h)


def is_pos_valid(pos: PosIdx, w: int, h: int) -> bool:
    x, y = pos_to_xy(pos, w, h)
    return 0 <= x < w and 0 <= y < h


def is_pos_within_bounds(pos: PosIdx, w: int, h: int) -> bool:
    x, y = pos_to_xy(pos, w, h)
    return 0 < x < w - 1 and 0 < y < h - 1


@lru_cache(maxsize=None)
def generate_l1_distance_lookup(w: int, h: int):
    def manhattan_distance(pos1: Tuple[int, int], pos2: Tuple[int, int]) -> int:
        return int(abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1]))

    return [
        [
            manhattan_distance(pos_to_xy(p1, w, h), pos_to_xy(p2, w, h))
            for p2 in range(w * h)
        ]
        for p1 in range(w * h)
    ]


@lru_cache(maxsize=None)
def generate_chebyshev_distance_lookup(w: int, h: int) -> List[List[int]]:
    def chebyshev_distance(pos1: Tuple[int, int], pos2: Tuple[int, int]) -> int:
        return max(abs(pos1[0] - pos2[0]), abs(pos1[1] - pos2[1]))

    return [
        [
            chebyshev_distance(pos_to_xy(p1, w, h), pos_to_xy(p2, w, h))
            for p2 in range(w * h)
        ]
        for p1 in range(w * h)
    ]


@lru_cache(maxsize=None)
def generate_edge_distance_lookup(w: int, h: int) -> List[PosIdx]:
    def distance_to_edge(p: PosIdx) -> int:
        x, y = pos_to_xy(p, w, h)
        return max(0, min((x - 1, w - 2 - x, y - 1, h - 2 - y)))

    return [
        distance_to_edge(p)
        for p in range(w * h)
    ]


@lru_cache(maxsize=None)
def generate_center_distance_lookup(w: int, h: int) -> List[PosIdx]:
    def distance_to_center(p: PosIdx) -> int:
        x, y = pos_to_xy(p, w, h)
        cx = (w + 1) / 2
        cy = (h + 1) / 2
        return int(abs(x - cx)) + int(abs(y - cy))

    return [
        distance_to_center(p)
        for p in range(w * h)
    ]


def generate_direction_offsets_map(w: int, h: int) -> Dict[Direction, int]:
    assert w >= 3 and h >= 3
    return {
        Direction.UP_LEFT: -h + 1,
        Direction.UP: 1,
        Direction.UP_RIGHT: h + 1,
        Direction.RIGHT: h,
        Direction.DOWN_RIGHT: h - 1,
        Direction.DOWN: -1,
        Direction.DOWN_LEFT: -h - 1,
        Direction.LEFT: -h
    }


@lru_cache(maxsize=None)
def generate_move_offsets_map(w: int, h: int) -> Dict[BoardMove, int]:
    dir_map = generate_direction_offsets_map(w, h)
    return {
        BoardMove.UP: dir_map[Direction.UP],
        BoardMove.RIGHT: dir_map[Direction.RIGHT],
        BoardMove.DOWN: dir_map[Direction.DOWN],
        BoardMove.LEFT: dir_map[Direction.LEFT]
    }


@lru_cache(maxsize=None)
def generate_4way_positions(w: int, h: int) -> List[Tuple[PosIdx, ...]]:
    move_offsets_map = generate_move_offsets_map(w, h)
    pos_offsets = tuple(move_offsets_map[m] for m in MOVES)
    return [
        tuple(p + d for d in pos_offsets)
        for p in range(w * h)
    ]


@lru_cache(maxsize=None)
def generate_8way_positions(w: int, h: int) -> List[Tuple[PosIdx, ...]]:
    dir_offsets_map = generate_direction_offsets_map(w, h)
    pos_offsets = tuple(dir_offsets_map[d] for d in DIRECTIONS)
    return [
        tuple(p + d for d in pos_offsets)
        for p in range(w * h)
    ]


@lru_cache(maxsize=None)
def generate_4way_bounded_positions(w: int, h: int) -> List[Tuple[PosIdx, ...]]:
    move_offsets_map = generate_move_offsets_map(w, h)
    pos_offsets = tuple(move_offsets_map[m] for m in MOVES)

    def is_within_bounds(pos):
        return is_pos_within_bounds(pos, w, h)

    return [
        tuple(filter(is_within_bounds, [p + d for d in pos_offsets]))
        for p in range(w * h)
    ]


@lru_cache(maxsize=None)
def generate_8way_bounded_positions(w: int, h: int) -> List[Tuple[PosIdx, ...]]:
    dir_offsets_map = generate_direction_offsets_map(w, h)
    pos_offsets = tuple(dir_offsets_map[d] for d in DIRECTIONS)

    def is_within_bounds(pos):
        return is_pos_within_bounds(pos, w, h)

    return [
        tuple(filter(is_within_bounds, [p + d for d in pos_offsets]))
        for p in range(w * h)
    ]


@lru_cache(maxsize=None)
def generate_territory_lookup(w: int, h: int) -> Tuple[List[List[int]], List[List[int]], List[List[int]]]:
    from .search.space import count_free_space_bfs_delta
    w0 = w - 2
    h0 = h - 2
    pos_options = generate_4way_bounded_positions(w, h)

    def create_empty_mask():
        return create_grid(w - 2, h - 2)

    def compute_p1_territory(p1: PosIdx, p2: PosIdx) -> int:
        if p1 == p2 or not is_pos_within_bounds(p1, w, h) or not is_pos_within_bounds(p2, w, h):
            return 0

        mask = create_empty_mask()
        _, fs, _ = count_free_space_bfs_delta(mask, pos1=p1, pos2=p2, pos_options=pos_options)
        return fs

    territory1 = [
        [compute_p1_territory(p1, p2) for p2 in range(w * h)]
        for p1 in range(w * h)
    ]
    territory2 = [
        [w0 * h0 - territory1[p1][p2] for p2 in range(w * h)]
        for p1 in range(w * h)
    ]
    delta_territory = [
        [territory1[p1][p2] - territory2[p1][p2] for p2 in range(w * h)]
        for p1 in range(w * h)
    ]

    return territory1, territory2, delta_territory
