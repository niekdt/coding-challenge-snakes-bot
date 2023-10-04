import itertools
from typing import Tuple

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from snakes.constants import Move
from ..board import Board, as_move, count_move_partitions, BoardMove
from ....snake import Snake


@pytest.mark.parametrize('width,height', [(3, 2), (2, 3), (16, 16)])
def test_init(width, height):
    b = Board(width, height)
    assert b.shape == (width, height)
    assert len(b.grid_mask) == (width + 2) * (height + 2)

    for x in range(0, b.full_width):
        assert b.grid_mask[b.from_xy(x, 0)] is False
    for y in range(0, b.full_height):
        assert b.grid_mask[b.from_xy(0, y)] is False
    for x, y in itertools.product(range(b.width), range(b.height)):
        assert b.grid_mask[b.from_xy(1 + x, 1 + y)] is True

    assert np.all(b.grid_as_np(b.get_empty_mask())[1:-1, 1:-1])
    assert not np.any(b.grid_as_np(b.get_player1_mask())[1:-1, 1:-1])
    assert not np.any(b.grid_as_np(b.get_player2_mask())[1:-1, 1:-1])
    assert not b.candies
    assert hash(b) != 0
    assert b.approx_hash() != 0


@pytest.mark.parametrize('width,height', [(3, 2), (2, 3), (16, 16)])
def test_empty_hash(width, height):
    assert hash(Board(width, height)) == hash(Board(width, height))
    assert Board(width, height).approx_hash() == Board(width, height).approx_hash()


@pytest.mark.parametrize('width', [4, 16])
@pytest.mark.parametrize('height', [4, 16])
def test_distance(width, height):
    b = Board(width, height)
    for x1, y1 in itertools.product(range(b.full_width), range(b.full_height)):
        for x2, y2 in itertools.product(range(b.full_width), range(b.full_height)):
            p1 = b.from_xy(x1, y1)
            p2 = b.from_xy(x2, y2)
            assert b.DISTANCE[p1][p2] == abs(x1 - x2) + abs(y1 - y2)
            assert b.DISTANCE[p2][p1] == b.DISTANCE[p1][p2]


@pytest.mark.parametrize('width', [4, 16])
@pytest.mark.parametrize('height', [4, 16])
def test_move_from_trans(width, height):
    b = Board(width, height)
    assert isinstance(b.MOVE_FROM_TRANS, list)

    for x1, y1 in itertools.product(range(1, b.full_width), range(1, b.full_height)):
        for x2, y2 in itertools.product(range(1, b.full_width), range(1, b.full_height)):
            p_from, p_to = b.from_xy(x1, y1), b.from_xy(x2, y2)
            if p_from == p_to or abs(x1 - x2) + abs(y1 - y2) > 1:
                continue

            if x1 == x2:
                ref_move = BoardMove.UP if y2 > y1 else BoardMove.DOWN
            else:
                ref_move = BoardMove.RIGHT if x2 > x1 else BoardMove.LEFT

            assert isinstance(b.MOVE_FROM_TRANS[p_from][p_to], BoardMove)
            assert b.MOVE_FROM_TRANS[p_from][p_to] == ref_move


@pytest.mark.parametrize('width', [4, 16])
@pytest.mark.parametrize('height', [4, 16])
def test_four_way_positions(width, height):
    b = Board(width, height)

    assert isinstance(b.FOUR_WAY_POS_OFFSETS, tuple)
    assert isinstance(b.FOUR_WAY_POSITIONS_COND, list)

    def is_within_bounds(pos):
        if pos < 0 or pos >= len(b.grid_mask):
            return False
        else:
            xx, yy = b.from_index(pos)
            return 0 < xx <= width and 0 < yy <= height

    for x, y in itertools.product(range(1, b.full_width), range(1, b.full_height)):
        ref_positions = list(filter(is_within_bounds, (
            b.from_xy(x - 1, y),
            b.from_xy(x + 1, y),
            b.from_xy(x, y - 1),
            b.from_xy(x, y + 1)
        )))
        p = b.from_xy(x, y)
        positions = b.FOUR_WAY_POSITIONS_COND[p]
        assert isinstance(positions, tuple)
        assert set(positions) == set(ref_positions)


@pytest.mark.parametrize('width', [4, 16])
@pytest.mark.parametrize('height', [4, 16])
def test_four_way_trans_positions(width, height):
    b = Board(width, height)
    for x, y in itertools.product(range(1, b.full_width - 1), range(1, b.full_height - 1)):
        p = b.from_xy(x, y)
        all_positions = b.FOUR_WAY_POSITIONS_COND[p]
        for p_old in all_positions:
            assert isinstance(b.FOUR_WAY_POSITIONS_FROM_POS_COND[p_old], list)
            positions = b.FOUR_WAY_POSITIONS_FROM_POS_COND[p_old][p]
            assert positions is not None
            assert isinstance(positions, tuple)
            ref_positions = set(all_positions)
            ref_positions.remove(p_old)
            assert set(positions) == set(ref_positions)


@pytest.mark.parametrize('width', [4, 16])
@pytest.mark.parametrize('height', [4, 16])
def test_eight_way_positions(width, height):
    b = Board(width, height)

    assert isinstance(b.EIGHT_WAY_POS_OFFSETS, tuple)
    assert isinstance(b.EIGHT_WAY_POSITIONS_COND, list)

    def is_within_bounds(pos):
        if pos < 0 or pos >= len(b.grid_mask):
            return False
        else:
            xx, yy = b.from_index(pos)
            return 0 < xx <= width and 0 < yy <= height

    for x, y in itertools.product(range(1, b.full_width), range(1, b.full_height)):
        ref_positions = list(filter(is_within_bounds, (
            b.from_xy(x + xo, y + yo)
            for xo, yo in itertools.product([-1, 0, 1], repeat=2)
            if xo != 0 or yo != 0
        )))
        p = b.from_xy(x, y)
        positions = b.EIGHT_WAY_POSITIONS_COND[p]
        assert isinstance(positions, tuple)
        assert set(positions) == set(ref_positions)


@pytest.mark.parametrize('width', [4, 16])
@pytest.mark.parametrize('height', [4, 16])
def test_eight_way_trans_positions(width, height):
    b = Board(width, height)
    for x, y in itertools.product(range(1, b.full_width - 1), range(1, b.full_height - 1)):
        p = b.from_xy(x, y)
        all_positions = b.EIGHT_WAY_POSITIONS_COND[p]
        for p_old in all_positions:
            assert isinstance(b.EIGHT_WAY_POSITIONS_FROM_POS_COND[p_old], list)
            positions = b.EIGHT_WAY_POSITIONS_FROM_POS_COND[p_old][p]
            assert positions is not None
            assert isinstance(positions, tuple)
            ref_positions = set(all_positions)
            ref_positions.remove(p_old)
            assert set(positions) == set(ref_positions)


@pytest.mark.parametrize('width', [4, 16])
@pytest.mark.parametrize('height', [4, 16])
def test_moves_from_pos_trans(width, height):
    b = Board(width, height)
    assert isinstance(b.MOVES_FROM_POS_TRANS, list)
    for x1, y1 in itertools.product(range(1, b.full_width), range(1, b.full_height)):
        for x2, y2 in itertools.product(range(1, b.full_width), range(1, b.full_height)):
            p_from, p_to = b.from_xy(x1, y1), b.from_xy(x2, y2)
            if p_from == p_to or abs(x1 - x2) + abs(y1 - y2) > 1:
                continue

            pos_options = b.FOUR_WAY_POSITIONS_FROM_POS_COND[p_from][p_to]
            ref_moves = [b.MOVE_FROM_TRANS[p_to][p] for p in pos_options]

            assert isinstance(b.MOVES_FROM_POS_TRANS[p_from][p_to], tuple)
            assert set(b.MOVES_FROM_POS_TRANS[p_from][p_to]) == set(ref_moves)


@pytest.mark.parametrize('width,height', [(3, 2), (2, 3), (16, 16)])
def test_fully_is_empty_pos(width, height):
    b = Board(width, height)
    for x in range(0, b.full_width):
        assert b.grid_mask[b.from_xy(x, 0)] is False
        assert not b.is_empty_pos(b.from_xy(x, 0))
        assert not b.is_empty_pos(b.from_xy(x, b.height + 1))
    for y in range(0, b.full_height):
        assert b.grid_mask[b.from_xy(0, y)] is False
        assert not b.is_empty_pos(b.from_xy(0, y))
        assert not b.is_empty_pos(b.from_xy(b.width + 1, y))

    for x, y in itertools.product(range(b.width), range(b.height)):
        assert b.grid_mask[b.from_xy(1 + x, 1 + y)] is True
        assert b.is_empty_pos(b.from_xy(1 + x, 1 + y))


@pytest.mark.parametrize('width,height', [(3, 2), (2, 3), (16, 16)])
def test_fully_get_empty_mask(width, height):
    b = Board(width, height)
    ref_mask = np.full(b.shape, fill_value=True)
    assert_array_equal(b.grid_as_np(b.get_empty_mask())[1:-1, 1:-1], ref_mask)


@pytest.mark.parametrize('width,height', [(3, 2), (16, 16)])
def test_spawn_candy(width, height):
    b = Board(width, height)
    assert not b.candies
    b.candies.append(b.from_xy(1, 2))
    assert b.candies
    assert b.from_xy(1, 2) in b.candies


@pytest.mark.parametrize('width,height', [(3, 2), (16, 16)])
def test_remove_candy(width, height):
    b = Board(width, height)
    b.candies.append(b.from_xy(1, 2))
    b.candies.remove(b.from_xy(1, 2))
    assert not b.candies
    assert not b.from_xy(1, 2) in b.candies


@pytest.mark.parametrize('width,height', [(3, 2), (2, 3), (16, 16)])
def test_is_candy_pos(width, height):
    b = Board(width, height)
    for x, y in itertools.product(range(b.width), range(b.height)):
        assert not b.from_xy(x, y) in b.candies

    candy_pos = (1, 1)
    b.candies.append(b.from_pos(candy_pos))
    assert b.candies
    assert b.from_pos(candy_pos) in b.candies

    for x, y in itertools.product(range(b.width), range(b.height)):
        if x == candy_pos[0] and y == candy_pos[1]:
            continue
        else:
            assert not b.from_xy(x, y) in b.candies


def test_set_state():
    b = Board(3, 2)
    b.set_state(
        snake1=Snake(id=0, positions=np.array([[1, 0], [0, 0]])),
        snake2=Snake(id=1, positions=np.array([[1, 1], [2, 1]])),
        candies=[]
    )
    assert b.player1_length == 2
    assert b.player2_length == 2
    assert b.player1_pos == b.from_xy(2, 1)
    assert b.player2_pos == b.from_xy(2, 2)
    assert b.player1_prev_pos == b.from_xy(1, 1)
    assert b.player2_prev_pos == b.from_xy(3, 2)
    assert b.get_tail_pos(player=1) == b.player1_prev_pos
    assert b.get_tail_pos(player=-1) == b.player2_prev_pos
    assert b.last_player == -1
    assert_array_equal(
        b.grid_as_np(b.grid_mask)[1:-1, 1:-1],
        np.array([[False, True], [False, False], [True, False]])
    )
    assert not b.candies

    # reuse board
    b0 = b.copy()
    b.set_state(
        snake1=Snake(id=0, positions=np.array([[0, 0], [1, 0]])),
        snake2=Snake(id=1, positions=np.array([[2, 1], [1, 1]])),
        candies=[]
    )
    assert b.player1_length == 2
    assert b.player2_length == 2
    assert b.player1_pos == b.from_xy(1, 1)
    assert b.player2_pos == b.from_xy(3, 2)
    assert b.player1_prev_pos == b.from_xy(2, 1)
    assert b.player2_prev_pos == b.from_xy(2, 2)
    assert b.get_tail_pos(player=1) == b.player1_prev_pos
    assert b.get_tail_pos(player=-1) == b.player2_prev_pos
    assert b.last_player == -1
    assert_array_equal(
        b.grid_as_np(b.grid_mask)[1:-1, 1:-1],
        np.array([[False, True], [False, False], [True, False]])
    )
    assert not b.candies
    assert hash(b) != hash(b0)


@pytest.mark.parametrize('width,height', [(3, 2), (16, 16)])
def test_get_is_empty_pos(width, height):
    b = Board(width, height)
    b.set_state(
        snake1=Snake(id=0, positions=np.array([[0, 1], [0, 0]])),
        snake2=Snake(id=1, positions=np.array([[2, 0], [2, 1]])),
        candies=[]
    )
    assert not b.is_empty_pos(b.from_xy(1, 1))
    assert not b.is_empty_pos(b.from_xy(1, 2))
    assert not b.is_empty_pos(b.from_xy(3, 1))
    assert not b.is_empty_pos(b.from_xy(3, 2))

    assert b.is_empty_pos(b.from_xy(2, 1))


@pytest.mark.parametrize('width,height', [(3, 2), (16, 16)])
def test_get_empty_mask(width, height):
    b = Board(width, height)
    b.set_state(
        snake1=Snake(id=0, positions=np.array([[0, 1], [0, 0]])),
        snake2=Snake(id=1, positions=np.array([[2, 0], [2, 1]])),
        candies=[]
    )
    ref_mask = np.full(b.shape, fill_value=True)
    ref_mask[(0, 0)] = False
    ref_mask[(0, 1)] = False
    ref_mask[(2, 0)] = False
    ref_mask[(2, 1)] = False
    assert_array_equal(b.grid_as_np(b.get_empty_mask())[1:-1, 1:-1], ref_mask)


def test_perform_move():
    b = Board(3, 3)
    b.set_state(
        snake1=Snake(id=0, positions=np.array([[1, 0], [0, 0]])),
        snake2=Snake(id=1, positions=np.array([[1, 2], [2, 2]])),
        candies=[]
    )
    b0 = b.copy()
    # force hash computation
    assert hash(b) == hash(b0)
    assert b.approx_hash() == b0.approx_hash()

    # move P1 right
    b.perform_move(move=BoardMove.RIGHT, player=1)
    assert b.player1_pos == b.from_xy(3, 1)
    assert b.player2_pos == b.from_xy(2, 3)
    assert b.player1_prev_pos == b.from_xy(2, 1)
    assert b.player2_prev_pos == b.from_xy(3, 3)
    assert not b.is_empty_pos(b.from_xy(3, 1))
    assert not b.is_empty_pos(b.from_xy(2, 1))
    assert b.get_tail_pos(player=1) == b.player1_prev_pos
    assert b.get_tail_pos(player=-1) == b.player2_prev_pos
    assert b.is_empty_pos(b.from_xy(1, 1))

    assert hash(b) != hash(b0)
    assert b.approx_hash() != b0.approx_hash()

    # move P2 left
    b.perform_move(move=BoardMove.LEFT, player=-1)
    assert b.player2_pos == b.from_xy(1, 3)
    assert b.player1_pos == b.from_xy(3, 1)
    assert b.player1_prev_pos == b.from_xy(2, 1)
    assert b.player2_prev_pos == b.from_xy(2, 3)
    assert not b.is_empty_pos(b.from_xy(1, 3))
    assert not b.is_empty_pos(b.from_xy(2, 3))
    assert b.get_tail_pos(player=1) == b.player1_prev_pos
    assert b.get_tail_pos(player=-1) == b.player2_prev_pos
    assert b.is_empty_pos(b.from_xy(3, 3))

    # move P1 up
    b.perform_move(move=BoardMove.UP, player=1)
    assert b.player1_pos == b.from_xy(3, 2)
    assert b.player1_prev_pos == b.from_xy(3, 1)
    assert not b.is_empty_pos(b.from_xy(3, 2))
    assert not b.is_empty_pos(b.from_xy(3, 1))
    assert b.is_empty_pos(b.from_xy(2, 1))


def test_perform_move_candy():
    b = Board(3, 3)
    b.set_state(
        snake1=Snake(id=0, positions=np.array([[1, 0], [0, 0]])),
        snake2=Snake(id=1, positions=np.array([[1, 2], [2, 2]])),
        candies=[]
    )
    candy_pos = (2, 2)
    b.candies.append(b.from_pos(candy_pos))
    b.perform_move(move=BoardMove.UP, player=1)
    assert not b.candies  # candy should have been eaten
    assert b.player1_length == 3
    assert b.player2_length == 2
    # P1
    assert not b.is_empty_pos(b.from_xy(2, 2))
    assert not b.is_empty_pos(b.from_xy(2, 1))
    assert not b.is_empty_pos(b.from_xy(1, 1))
    # P2
    assert not b.is_empty_pos(b.from_xy(2, 3))
    assert not b.is_empty_pos(b.from_xy(3, 3))


def test_undo_move():
    b = Board(3, 3)
    b.set_state(
        snake1=Snake(id=0, positions=np.array([[1, 0], [0, 0]])),
        snake2=Snake(id=1, positions=np.array([[1, 2], [2, 2]])),
        candies=[]
    )
    assert not b.is_empty_pos(b.from_xy(2, 1))  # P1 pos
    assert not b.is_empty_pos(b.from_xy(1, 1))  # P1 prev pos
    assert not b.is_empty_pos(b.from_xy(2, 3))  # P2 pos
    assert not b.is_empty_pos(b.from_xy(3, 3))  # P2 prev pos

    b_start = b.copy()
    b.perform_move(move=BoardMove.RIGHT, player=1)
    b.undo_move(player=1)
    assert b == b_start
    assert b.player1_length == 2
    assert b.player2_length == 2
    assert b.player1_pos == b.from_xy(2, 1)
    assert b.player2_pos == b.from_xy(2, 3)
    assert b.player1_prev_pos == b.from_xy(1, 1)
    assert b.player2_prev_pos == b.from_xy(3, 3)
    assert b.is_empty_pos(b.from_xy(3, 1))  # P1 moved back out of here
    assert not b.is_empty_pos(b.from_xy(2, 1))  # P1 pos
    assert not b.is_empty_pos(b.from_xy(1, 1))  # P1 prev pos
    assert not b.is_empty_pos(b.from_xy(2, 3))  # P2 pos
    assert not b.is_empty_pos(b.from_xy(3, 3))  # P2 prev pos
    assert hash(b) == hash(b_start)
    assert b.approx_hash() == b_start.approx_hash()

    b.perform_move(move=BoardMove.RIGHT, player=1)
    assert b.player1_pos == b.from_xy(3, 1)
    assert not b.is_empty_pos(b.from_xy(3, 1))  # cur pos
    assert not b.is_empty_pos(b.from_xy(2, 1))  # prev pos
    assert b.is_empty_pos(b.from_xy(1, 1))

    b_ref2 = b.copy()
    b.perform_move(move=BoardMove.LEFT, player=-1)
    assert b.player2_pos == b.from_xy(1, 3)
    assert not b.is_empty_pos(b.from_xy(1, 3))  # cur pos
    assert not b.is_empty_pos(b.from_xy(2, 3))  # prev pos
    assert b.is_empty_pos(b.from_xy(3, 3))

    b.undo_move(player=-1)
    assert b == b_ref2
    assert b.player1_length == 2
    assert b.player2_length == 2
    assert b.player1_pos == b.from_xy(3, 1)
    assert b.player2_pos == b.from_xy(2, 3)
    assert b.player1_prev_pos == b.from_xy(2, 1)
    assert b.player2_prev_pos == b.from_xy(3, 3)
    assert b.is_empty_pos(b.from_xy(1, 3))  # P2 moved back out of here
    assert not b.is_empty_pos(b.from_xy(2, 3))
    assert not b.is_empty_pos(b.from_xy(3, 3))
    assert hash(b) == hash(b_ref2)
    assert b.approx_hash() == b_ref2.approx_hash()

    b.undo_move(player=1)
    assert b == b_start
    assert hash(b) == hash(b_start)
    assert b.approx_hash() == b_start.approx_hash()


def test_undo_move_candy():
    b = Board(3, 3)
    b.set_state(
        snake1=Snake(id=0, positions=np.array([[1, 0], [0, 0]])),
        snake2=Snake(id=1, positions=np.array([[1, 2], [2, 2]])),
        candies=[]
    )
    candy_pos = (2, 2)
    b.candies.append(b.from_pos(candy_pos))
    b_start = b.copy()

    b.perform_move(move=BoardMove.UP, player=1)
    assert not b.is_empty_pos(b.from_xy(1, 1))
    assert not b.candies
    b.undo_move(player=1)
    assert b.candies
    assert b.from_pos(candy_pos) in b.candies
    assert not b.is_empty_pos(b.from_xy(1, 1))
    assert b == b_start
    assert hash(b) == hash(b_start)
    assert b.approx_hash() == b_start.approx_hash()


def test_print():
    b = Board(3, 2)
    assert str(b) == '\n+---+\n|···|\n|···|\n+---+'

    b.set_state(
        snake1=Snake(id=0, positions=np.array([[1, 0], [0, 0]])),
        snake2=Snake(id=1, positions=np.array([[1, 1], [2, 1]])),
        candies=[]
    )
    assert str(b) == '\n+---+\n|·Bb|\n|aA·|\n+---+'

    b.perform_move(BoardMove.RIGHT, player=1)
    assert str(b) == '\n+---+\n|·Bb|\n|·aA|\n+---+'

    b.perform_move(BoardMove.LEFT, player=-1)
    assert str(b) == '\n+---+\n|Bb·|\n|·aA|\n+---+'


def test_move_generation():
    b = Board(4, 3)
    b.set_state(
        snake1=Snake(id=0, positions=np.array([[1, 0], [0, 0]])),
        snake2=Snake(id=1, positions=np.array([[3, 2], [3, 1]])),
        candies=[]
    )
    moves1 = b.get_valid_moves_ordered(player=1)
    assert set(moves1) == {BoardMove.RIGHT, BoardMove.UP}

    moves2 = b.get_valid_moves_ordered(player=-1)
    assert set(moves2) == {BoardMove.LEFT}

    # perform a move and recheck the options
    b.perform_move(BoardMove.UP, player=1)
    moves12 = b.get_valid_moves_ordered(player=1)
    assert set(moves12) == {BoardMove.LEFT, BoardMove.RIGHT, BoardMove.UP}

    b.perform_move(BoardMove.LEFT, player=2)
    moves22 = b.get_valid_moves_ordered(2)
    assert set(moves22) == {BoardMove.LEFT, BoardMove.DOWN}


def test_set_state_candy():
    # one candy
    b = Board(3, 2)
    snake1 = Snake(id=0, positions=np.array([[1, 0], [0, 0]]))
    snake2 = Snake(id=1, positions=np.array([[1, 1], [2, 1]]))
    b.set_state(snake1=snake1, snake2=snake2, candies=[(1, 1)])
    b1 = b.copy()
    assert b.candies
    assert b.from_xy(2, 2) in b.candies

    # no candies (reuse board)
    b.set_state(snake1=snake1, snake2=snake2, candies=[])
    assert not b.candies
    assert not b.from_xy(2, 2) in b.candies

    # two candies
    b.set_state(snake1=snake1, snake2=snake2, candies=[(1, 1), (0, 1)])
    assert b.candies
    assert b.from_xy(2, 2) in b.candies
    assert b.from_xy(1, 2) in b.candies


@pytest.mark.parametrize('board_move,move', [
    (BoardMove.LEFT, Move.LEFT),
    (BoardMove.RIGHT, Move.RIGHT),
    (BoardMove.UP, Move.UP),
    (BoardMove.DOWN, Move.DOWN)
])
def test_as_move(board_move, move):
    assert as_move(board_move) == move


@pytest.mark.parametrize('size', [2, 3, 5])
@pytest.mark.parametrize('lb', [1, 2, 3, 5, 10])
def test_count_free_space_dfs(size, lb):
    b = Board(size, size)

    # test without lb
    space = b.count_free_space_dfs(mask=b.get_empty_mask(), pos=b.from_xy(1, 1), lb=1000, max_dist=100, distance_map=b.DISTANCE[b.from_xy(1, 1)])
    assert space == size ** 2

    # test with lb
    space = b.count_free_space_dfs(mask=b.get_empty_mask(), pos=b.from_xy(1, 1), lb=lb, max_dist=100, distance_map=b.DISTANCE[b.from_xy(1, 1)])
    assert space >= min(lb, size ** 2)

    # insert wall
    b.grid_mask[b.from_xy(2, 1)] = False
    b.grid_mask[b.from_xy(2, 2)] = False
    space = b.count_free_space_dfs(mask=b.get_empty_mask(), pos=b.from_xy(1, 1), lb=1000, max_dist=100, distance_map=b.DISTANCE[b.from_xy(1, 1)])
    assert space == size ** 2 - 2

    space = b.count_free_space_dfs(mask=b.get_empty_mask(), pos=b.from_xy(1, 1), lb=lb, max_dist=100, distance_map=b.DISTANCE[b.from_xy(1, 1)])
    assert space >= min(lb, size ** 2 - 2)

    if size >= 3:
        # insert void
        for y in range(b.full_height):
            b.grid_mask[b.from_xy(2, y)] = False
        space = b.count_free_space_dfs(mask=b.get_empty_mask(), pos=b.from_xy(1, 1), lb=1000, max_dist=100, distance_map=b.DISTANCE[b.from_xy(1, 1)])
        assert space == size
        space = b.count_free_space_dfs(mask=b.get_empty_mask(), pos=b.from_xy(1, 1), lb=lb, max_dist=100, distance_map=b.DISTANCE[b.from_xy(1, 1)])
        assert space >= min(lb, size)


@pytest.mark.parametrize('size', [2, 3, 4, 5])
@pytest.mark.parametrize('lb', [10, 5, 3, 2, 1])
def test_count_free_space_bfs(size, lb):
    b = Board(size, size)

    # test without restrictions
    space = b.count_free_space_bfs(mask=b.get_empty_mask(), pos=b.from_xy(1, 1), max_dist=1000, lb=1000)
    assert space == size ** 2

    # test with lb
    space = b.count_free_space_bfs(mask=b.get_empty_mask(), pos=b.from_xy(1, 1), max_dist=1000, lb=lb)
    assert min(lb, size ** 2) <= space <= size ** 2

    if size >= 3:
        # insert void
        for y in range(b.full_height):
            b.grid_mask[b.from_xy(2, y)] = False
        space = b.count_free_space_bfs(mask=b.get_empty_mask(), pos=b.from_xy(1, 1), max_dist=1000, lb=1000)
        assert space == size
        space = b.count_free_space_bfs(mask=b.get_empty_mask(), pos=b.from_xy(1, 1), max_dist=1000, lb=lb)
        assert space >= min(lb, size)


@pytest.mark.parametrize('size', [15])
@pytest.mark.parametrize('max_dist', [1, 2, 3, 4, 5])
@pytest.mark.parametrize('lb', [10, 5, 3, 2, 1])
def test_count_free_space_bfs_dist(size, max_dist, lb):
    b = Board(size, size)

    def max_area(d):
        return 2 * (d + 1) ** 2 - 2 * (d + 1) + 1

    center = (int(size / 2) + 1, int(size / 2) + 1)
    space = b.count_free_space_bfs(mask=b.get_empty_mask(), pos=b.from_pos(center), max_dist=max_dist, lb=1000)
    assert space == min(size ** 2, max_area(max_dist))

    # test with lb
    space2 = b.count_free_space_bfs(mask=b.get_empty_mask(), pos=b.from_pos(center), max_dist=max_dist, lb=lb)
    assert min(lb, min(size ** 2, max_area(max_dist))) <= space2 <= min(size ** 2, max_area(max_dist))


def shift(x: Tuple, n: int) -> Tuple:
    return x[n:] + x[:n]


@pytest.mark.parametrize('offset', [0, 2, 4, 6])
@pytest.mark.parametrize('tl', [False, True])
@pytest.mark.parametrize('tr', [False, True])
@pytest.mark.parametrize('br', [False, True])
@pytest.mark.parametrize('bl', [False, True])
def test_count_move_partitions_m1_forced(offset, tl, tr, br, bl):
    assert count_move_partitions(
        shift((tl, False, tr, True, br, False, bl, False), offset)
    ) == 1


@pytest.mark.parametrize('offset', [0, 2, 4, 6])
@pytest.mark.parametrize('tl', [False, True])
@pytest.mark.parametrize('br', [False, True])
@pytest.mark.parametrize('bl', [False, True])
def test_count_move_partitions_m2_uni(offset, tl, br, bl):
    assert count_move_partitions(
        shift((tl, True, True, True, br, False, bl, False), offset)
    ) == 1


@pytest.mark.parametrize('v', [False, True])
@pytest.mark.parametrize('tl', [False, True])
@pytest.mark.parametrize('tr', [False, True])
@pytest.mark.parametrize('br', [False, True])
@pytest.mark.parametrize('bl', [False, True])
def test_count_move_partitions_m2_bump(v, tl, tr, br, bl):
    assert count_move_partitions((tl, v, tr, not v, br, v, bl, not v)) == 2


@pytest.mark.parametrize('offset', [0, 2, 4, 6])
@pytest.mark.parametrize('tl', [False, True])
@pytest.mark.parametrize('tr', [False, True])
@pytest.mark.parametrize('bl', [False, True])
def test_count_move_partitions_m2_corner(offset, tl, tr, bl):
    assert count_move_partitions(
        shift((tl, False, tr, True, False, True, bl, False), offset)
    ) == 2


@pytest.mark.parametrize('offset', [0, 2, 4, 6])
@pytest.mark.parametrize('br', [False, True])
@pytest.mark.parametrize('bl', [False, True])
def test_count_move_partitions_m3_uni(offset, br, bl):
    assert count_move_partitions(
        shift((True, True, True, True, br, False, bl, True), offset)
    ) == 1


@pytest.mark.parametrize('offset', [0, 2, 4, 6])
@pytest.mark.parametrize('left', [False, True])
@pytest.mark.parametrize('br', [False, True])
@pytest.mark.parametrize('bl', [False, True])
def test_count_move_partitions_m3_tunnel(offset, left, br, bl):
    assert count_move_partitions(
        shift((left, True, not left, True, br, False, bl, True), offset)
    ) == 2


@pytest.mark.parametrize('offset', [0, 2, 4, 6])
@pytest.mark.parametrize('br', [False, True])
@pytest.mark.parametrize('bl', [False, True])
def test_count_move_partitions_m3_tri(offset, br, bl):
    assert count_move_partitions(
        shift((False, True, False, True, br, False, bl, True), offset)
    ) == 3
