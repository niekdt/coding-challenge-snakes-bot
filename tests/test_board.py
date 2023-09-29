import itertools
from typing import Tuple

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from snakes.constants import Move
from ..board import Board, as_move, count_move_partitions, BoardMove
from ....snake import Snake


def test_init():
    b = Board(2, 3)
    assert b.shape == (2, 3)
    assert len(b) == 2 * 3
    assert len(b.grid) == 4 * 5

    for x in range(0, b.full_width):
        assert b.grid[b.from_xy(x, 0)] != 0
    for y in range(0, b.full_height):
        assert b.grid[b.from_xy(0, y)] != 0
    for x, y in itertools.product(range(b.width), range(b.height)):
        assert b.grid[b.from_xy(1 + x, 1 + y)] == 0

    assert np.all(b.grid_as_np(b.get_empty_mask())[1:-1, 1:-1])
    assert not np.any(b.grid_as_np(b.get_player1_mask())[1:-1, 1:-1])
    assert not np.any(b.grid_as_np(b.get_player2_mask())[1:-1, 1:-1])
    assert not b.candies
    assert hash(b) != 0
    assert b.approx_hash() != 0


def test_empty_hash():
    assert hash(Board(4, 4)) == hash(Board(4, 4))
    assert Board(4, 4).approx_hash() == Board(4, 4).approx_hash()

    assert hash(Board(4, 4)) != hash(Board(4, 5))


def test_spawn():
    b = Board(4, 4)
    b.spawn(pos1=(1, 2), pos2=(2, 3))

    assert b.player1_pos == b.from_xy(2, 3)
    assert b.player2_pos == b.from_xy(3, 4)
    assert np.sum(b.grid_as_np(b.get_player1_mask())[1:-1, 1:-1]) == 1
    assert np.sum(b.grid_as_np(b.get_player2_mask())[1:-1, 1:-1]) == 1
    assert not b.is_empty_pos(b.player1_pos)
    assert not b.is_empty_pos(b.player2_pos)
    assert b.get_player1_mask()[b.from_xy(2, 3)]
    assert b.get_player2_mask()[b.from_xy(3, 4)]


def test_is_empty_pos():
    b = Board(3, 2)
    for x in range(0, b.full_width):
        assert b.grid[b.from_xy(x, 0)] != 0
        assert not b.is_empty_pos(b.from_xy(x, 0))
        assert not b.is_empty_pos(b.from_xy(x, b.height + 1))
    for y in range(0, b.full_height):
        assert b.grid[b.from_xy(0, y)] != 0
        assert not b.is_empty_pos(b.from_xy(0, y))
        assert not b.is_empty_pos(b.from_xy(b.width + 1, y))

    for x, y in itertools.product(range(b.width), range(b.height)):
        assert b.grid[b.from_xy(1 + x, 1 + y)] == 0
        assert b.is_empty_pos(b.from_xy(1 + x, 1 + y))

    b.spawn(pos1=(0, 0), pos2=(2, 1))
    assert not b.is_empty_pos(b.from_xy(1, 1))
    assert not b.is_empty_pos(b.from_xy(3, 2))

    assert b.is_empty_pos(b.from_xy(1, 2))


def test_get_empty_mask():
    b = Board(3, 2)
    ref_mask = np.full(b.shape, fill_value=True)
    assert_array_equal(b.grid_as_np(b.get_empty_mask())[1:-1, 1:-1], ref_mask)

    b.spawn(pos1=(0, 0), pos2=(2, 1))
    ref_mask[(0, 0)] = False
    ref_mask[(2, 1)] = False
    assert_array_equal(b.grid_as_np(b.get_empty_mask())[1:-1, 1:-1], ref_mask)


def test_spawn_candy():
    b = Board(3, 2)
    b0 = b.copy()
    assert not b.candies
    b.candies.append(b.from_xy(1, 2))
    assert b.candies
    assert b.from_xy(1, 2) in b.candies


def test_remove_candy():
    b = Board(3, 2)
    b.candies.append(b.from_xy(1, 2))
    b0 = b.copy()
    b.candies.remove(b.from_xy(1, 2))
    assert not b.candies
    assert not b.from_xy(1, 2) in b.candies


def test_is_candy_pos():
    b = Board(3, 2)
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


def test_perform_move():
    b = Board(3, 3)
    b.spawn(pos1=(0, 0), pos2=(2, 2))
    b0 = b.copy()
    # force hash computation
    assert hash(b) == hash(b0)
    assert b.approx_hash() == b0.approx_hash()

    # move P1
    b.perform_move(move=BoardMove.RIGHT, player=1)
    assert b.player1_pos == b.from_xy(2, 1)
    assert b.is_empty_pos(b.from_xy(1, 1))
    assert not b.is_empty_pos(b.from_xy(2, 1))
    assert not b.is_empty_pos(b.from_xy(3, 3))

    assert hash(b) != hash(b0)
    assert b.approx_hash() != b0.approx_hash()

    # move P2
    b.perform_move(move=BoardMove.LEFT, player=2)
    assert b.player2_pos == b.from_xy(2, 3)
    assert b.is_empty_pos(b.from_xy(1, 1))
    assert b.is_empty_pos(b.from_xy(3, 3))
    assert not b.is_empty_pos(b.from_xy(2, 1))
    assert not b.is_empty_pos(b.from_xy(2, 3))

    # move P1 to center
    b.perform_move(move=BoardMove.UP, player=1)
    assert b.player1_pos == b.from_xy(2, 2)


def test_perform_move_candy():
    b = Board(3, 3)
    b.spawn(pos1=(0, 0), pos2=(2, 2))
    candy_pos = (2, 1)
    b.candies.append(b.from_pos(candy_pos))
    b.perform_move(move=BoardMove.RIGHT, player=1)
    assert not b.candies  # candy should have been eaten
    assert b.player1_length == 2
    assert b.player2_length == 1


def test_undo_move():
    b = Board(3, 3)
    b.spawn(pos1=(0, 0), pos2=(2, 2))
    b_start = b.copy()
    with pytest.raises(Exception):
        b.undo_move(player=-1)
    b.perform_move(move=BoardMove.RIGHT, player=1)
    b.undo_move(player=1)
    assert b == b_start
    assert hash(b) == hash(b_start)
    assert b.approx_hash() == b_start.approx_hash()

    with pytest.raises(Exception):
        b.undo_move(player=1)
    with pytest.raises(Exception):
        b.undo_move(player=-1)

    b.perform_move(move=BoardMove.RIGHT, player=1)
    with pytest.raises(Exception):
        b.undo_move(player=-1)  # cannot undo because P1 moved last
    b_ref2 = b.copy()
    b.perform_move(move=BoardMove.LEFT, player=-1)

    b.undo_move(player=-1)
    assert b == b_ref2
    assert hash(b) == hash(b_ref2)
    assert b.approx_hash() == b_ref2.approx_hash()

    b.undo_move(player=1)
    assert b == b_start
    assert hash(b) == hash(b_start)
    assert b.approx_hash() == b_start.approx_hash()

    with pytest.raises(Exception):
        b.undo_move(player=1)
    with pytest.raises(Exception):
        b.undo_move(player=-1)


def test_undo_move_candy():
    b = Board(3, 3)
    b.spawn(pos1=(0, 0), pos2=(2, 2))
    candy_pos = (2, 1)
    b.candies.append(b.from_pos(candy_pos))
    b_start = b.copy()

    b.perform_move(move=BoardMove.RIGHT, player=1)
    assert not b.candies
    b.undo_move(player=1)
    assert b.candies
    assert b.from_pos(candy_pos) in b.candies
    assert b == b_start
    assert hash(b) == hash(b_start)
    assert b.approx_hash() == b_start.approx_hash()


def test_print():
    b = Board(3, 2)
    assert str(b) == '\n+---+\n|···|\n|···|\n+---+'
    b.spawn(pos1=(1, 0), pos2=(2, 1))
    assert str(b) == '\n+---+\n|··B|\n|·A·|\n+---+'

    b.player1_length = 2
    b.player2_length = 2
    b.perform_move(BoardMove.RIGHT, player=1)
    b.perform_move(BoardMove.LEFT, player=-1)
    assert str(b) == '\n+---+\n|·Bb|\n|·aA|\n+---+'


def test_move_generation():
    b = Board(3, 2)
    b.spawn(pos1=(1, 0), pos2=(2, 1))
    moves1 = b.get_valid_moves_ordered(player=1)
    assert len(moves1) == 3
    assert BoardMove.LEFT in moves1
    assert BoardMove.RIGHT in moves1
    assert BoardMove.UP in moves1

    moves2 = b.get_valid_moves_ordered(player=-1)
    assert len(moves2) == 2
    assert BoardMove.LEFT in moves2
    assert BoardMove.DOWN in moves2

    # perform a move and recheck the options
    b.perform_move(BoardMove.LEFT, player=1)
    moves12 = b.get_valid_moves_ordered(player=1)
    assert len(moves12) == 2
    assert BoardMove.RIGHT in moves12
    assert BoardMove.UP in moves12

    b.perform_move(BoardMove.LEFT, player=2)
    moves22 = b.get_valid_moves_ordered(2)
    assert len(moves22) == 3
    assert BoardMove.LEFT in moves22
    assert BoardMove.RIGHT in moves22
    assert BoardMove.DOWN in moves22


def test_set_state():
    b = Board(2, 2)
    b.set_state(
        snake1=Snake(id=0, positions=np.array([[0, 0]])),
        snake2=Snake(id=1, positions=np.array([[1, 1]])),
        candies=[]
    )
    assert b.player1_head == 1
    assert b.player2_head == -1
    assert b.player1_length == 1
    assert b.player2_length == 1
    assert b.last_player == -1
    assert_array_equal(
        b.grid_as_np(b.grid)[1:-1, 1:-1],
        np.array([[1, 0], [0, -1]])
    )
    assert not b.candies

    # reuse board
    b0 = b.copy()
    b.set_state(
        snake1=Snake(id=0, positions=np.array([[0, 1], [0, 0]])),
        snake2=Snake(id=1, positions=np.array([[1, 1]])),
        candies=[]
    )
    assert b.player1_head == 2
    assert b.player2_head == -1
    assert b.player1_length == 2
    assert b.player2_length == 1
    assert b.last_player == -1
    assert_array_equal(
        b.grid_as_np(b.grid)[1:-1, 1:-1],
        np.array([[1, 2], [0, -1]])
    )
    assert not b.candies
    assert hash(b) != hash(b0)
    assert b.approx_hash() != b0.approx_hash()

    b3 = Board(2, 2)
    b3.set_state(
        snake1=Snake(id=0, positions=np.array([[0, 1], [0, 0]])),
        snake2=Snake(id=1, positions=np.array([[1, 0], [1, 1]])),
        candies=[]
    )
    assert b3.player1_head == 2
    assert b3.player2_head == -2
    assert_array_equal(
        b.grid_as_np(b3.grid)[1:-1, 1:-1],
        np.array([[1, 2], [-2, -1]])
    )
    assert not b.candies


def test_set_state_candy():
    # one candy
    b = Board(2, 2)
    b.set_state(
        snake1=Snake(id=0, positions=np.array([[0, 0]])),
        snake2=Snake(id=1, positions=np.array([[1, 1]])),
        candies=[(1, 0)]
    )
    b1 = b.copy()
    assert b.candies
    assert b.from_xy(2, 1) in b.candies

    # no candies (reuse board)
    b.set_state(
        snake1=Snake(id=0, positions=np.array([[0, 0]])),
        snake2=Snake(id=1, positions=np.array([[1, 1]])),
        candies=[]
    )
    assert not b.candies
    assert not b.from_xy(2, 1) in b.candies

    # two candies
    b.set_state(
        snake1=Snake(id=0, positions=np.array([[0, 0]])),
        snake2=Snake(id=1, positions=np.array([[1, 1]])),
        candies=[(1, 0), (0, 1)]
    )
    assert b.candies
    assert b.from_xy(2, 1) in b.candies
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
    space = b.count_free_space_dfs(mask=b.get_empty_mask(), pos=b.from_xy(1, 1), lb=1000, max_dist=100, ref_pos=b.from_xy(1, 1))
    assert space == size ** 2

    # test with lb
    space = b.count_free_space_dfs(mask=b.get_empty_mask(), pos=b.from_xy(1, 1), lb=lb, max_dist=100, ref_pos=b.from_xy(1, 1))
    assert space >= min(lb, size ** 2)

    # insert wall
    b.grid[b.from_xy(2, 1)] = 100
    b.grid[b.from_xy(2, 2)] = 100
    space = b.count_free_space_dfs(mask=b.get_empty_mask(), pos=b.from_xy(1, 1), lb=1000, max_dist=100, ref_pos=b.from_xy(1, 1))
    assert space == size ** 2 - 2

    space = b.count_free_space_dfs(mask=b.get_empty_mask(), pos=b.from_xy(1, 1), lb=lb, max_dist=100, ref_pos=b.from_xy(1, 1))
    assert space >= min(lb, size ** 2 - 2)

    if size >= 3:
        # insert void
        for y in range(b.full_height):
            b.grid[b.from_xy(2, y)] = 100
        space = b.count_free_space_dfs(mask=b.get_empty_mask(), pos=b.from_xy(1, 1), lb=1000, max_dist=100, ref_pos=b.from_xy(1, 1))
        assert space == size
        space = b.count_free_space_dfs(mask=b.get_empty_mask(), pos=b.from_xy(1, 1), lb=lb, max_dist=100, ref_pos=b.from_xy(1, 1))
        assert space >= min(lb, size)


@pytest.mark.parametrize('size', [2, 3, 4, 5])
@pytest.mark.parametrize('lb', [10, 5, 3, 2, 1])
def test_count_free_space_bfs(size, lb):
    b = Board(size, size)

    # test without restrictions
    space = b.count_free_space_bfs(mask=b.get_empty_mask(), pos=b.from_xy(1, 1), max_dist=size * 2, lb=1000)
    assert space == size ** 2

    # test with lb
    space = b.count_free_space_bfs(mask=b.get_empty_mask(), pos=b.from_xy(1, 1), max_dist=size * 2, lb=lb)
    assert min(lb, size ** 2) <= space <= size ** 2

    if size >= 3:
        # insert void
        for y in range(b.full_height):
            b.grid[b.from_xy(2, y)] = 100
        space = b.count_free_space_bfs(mask=b.get_empty_mask(), pos=b.from_xy(1, 1), max_dist=size * 2, lb=1000)
        assert space == size
        space = b.count_free_space_bfs(mask=b.get_empty_mask(), pos=b.from_xy(1, 1), max_dist=size * 2, lb=lb)
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
