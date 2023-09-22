import itertools

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from ..board import Board, as_move
from ....constants import Move, MOVE_VALUE_TO_DIRECTION, MOVES
from ....snake import Snake


def test_init():
    b = Board(8, 6)
    assert b.shape == (8, 6)
    assert len(b) == 8 * 6

    assert np.all(b.get_empty_mask()[1:-1, 1:-1])
    assert not np.any(b.get_player1_mask()[1:-1, 1:-1])
    assert not np.any(b.get_player2_mask()[1:-1, 1:-1])
    assert not b.has_candy()
    assert hash(b) != 0
    assert b.approx_hash() != 0
    assert b.wall_hash() != 0


def test_empty_hash():
    assert hash(Board(4, 4)) == hash(Board(4, 4))
    assert Board(4, 4).approx_hash() == Board(4, 4).approx_hash()
    assert Board(4, 4).wall_hash() == Board(4, 4).wall_hash()

    assert hash(Board(4, 4)) != hash(Board(4, 5))


def test_spawn():
    b = Board(4, 4)
    b.spawn(pos1=(1, 2), pos2=(2, 3))

    assert b.player1_pos == (2, 3)
    assert b.player2_pos == (3, 4)
    assert np.sum(b.get_player1_mask()[1:-1, 1:-1]) == 1
    assert np.sum(b.get_player2_mask()[1:-1, 1:-1]) == 1
    assert not b.is_empty_pos(b.player1_pos)
    assert not b.is_empty_pos(b.player2_pos)


def test_is_valid_pos():
    b = Board(3, 2)
    for x, y in itertools.product(range(b.width), range(b.height)):
        assert b.is_valid_pos((x, y))

    assert not b.is_valid_pos((-1, 0))
    assert not b.is_valid_pos((0, -1))
    assert not b.is_valid_pos((b.width, 0))
    assert not b.is_valid_pos((0, b.height))


def test_get_empty_mask():
    b = Board(3, 2)
    ref_mask = np.full(b.shape, fill_value=True)
    assert_array_equal(b.get_empty_mask()[1:-1, 1:-1], ref_mask)

    b.spawn(pos1=(0, 0), pos2=(2, 1))
    ref_mask[(0, 0)] = False
    ref_mask[(2, 1)] = False
    assert_array_equal(b.get_empty_mask()[1:-1, 1:-1], ref_mask)


def test_is_empty_pos():
    b = Board(3, 2)
    for x in range(0, b.width + 1):
        assert not b.is_empty_pos((x, 0))
        assert not b.is_empty_pos((x, b.height + 1))
    for y in range(0, b.height + 1):
        assert not b.is_empty_pos((0, y))
        assert not b.is_empty_pos((b.width + 1, y))

    for x, y in itertools.product(range(b.width), range(b.height)):
        assert b.is_empty_pos((x + 1, y + 1))

    b.spawn(pos1=(0, 0), pos2=(2, 1))
    assert not b.is_empty_pos((1, 1))
    assert not b.is_empty_pos((3, 2))

    assert b.is_empty_pos((1, 2))


def test_spawn_candy():
    b = Board(3, 2)
    b0 = b.copy()
    assert not b.has_candy()
    assert not b.candy_mask[(1, 2)]
    b._spawn_candy((1, 2))
    assert b.has_candy()
    assert b.candy_mask[(1, 2)]

    assert hash(b) != hash(b0)
    assert b.approx_hash() != b0.approx_hash()
    assert b.wall_hash() == b0.wall_hash()


def test_remove_candy():
    b = Board(3, 2)
    b._spawn_candy((1, 2))
    b0 = b.copy()
    b._remove_candy((1, 2))
    assert not b.has_candy()
    assert not b.candy_mask[(1, 2)]

    assert hash(b) != hash(b0)
    assert b.approx_hash() != b0.approx_hash()
    assert b.wall_hash() == b0.wall_hash()


def test_get_candy_mask():
    b = Board(3, 2)
    assert b.grid.shape == b.candy_mask.shape
    ref_mask = np.full(b.grid.shape, fill_value=False)
    assert np.array_equal(b.get_candy_mask(), ref_mask)
    candy_pos = (1, 1)
    b._spawn_candy(candy_pos)

    ref_mat = np.full(b.grid.shape, fill_value=False)
    ref_mat[tuple(candy_pos)] = True
    assert np.array_equal(b.get_candy_mask(), ref_mat)


def test_is_candy_pos():
    b = Board(3, 2)
    for x, y in itertools.product(range(b.width), range(b.height)):
        assert not b.is_candy_pos((x, y))

    candy_pos = (1, 1)
    b._spawn_candy(candy_pos)
    assert b.has_candy()
    assert b.is_candy_pos(candy_pos)

    for x, y in itertools.product(range(b.width), range(b.height)):
        if x == candy_pos[0] and y == candy_pos[1]:
            continue
        else:
            assert not b.is_candy_pos((x, y))


def test_free_space():
    b = Board(8, 6)
    assert b.count_free_space() == 8 * 6

    b.spawn(pos1=(1, 2), pos2=(2, 3))
    assert b.count_free_space() == 8 * 6 - 2

    b._spawn_candy((1, 1))
    assert b.count_free_space() == 8 * 6 - 2  # should not affect free space


def test_perform_move():
    b = Board(3, 3)
    b.spawn(pos1=(0, 0), pos2=(2, 2))
    b0 = b.copy()
    # force hash computation
    assert hash(b) == hash(b0)
    assert b.approx_hash() == b0.approx_hash()
    assert b.wall_hash() == b.wall_hash()

    # move P1
    b.perform_move(move=Move.RIGHT, player=1)
    assert b.player1_pos == (2, 1)
    assert b.is_empty_pos((1, 1))
    assert not b.is_empty_pos((2, 1))
    assert not b.is_empty_pos((3, 3))

    assert hash(b) != hash(b0)
    assert b.approx_hash() != b0.approx_hash()
    assert b.wall_hash() != b0.wall_hash()

    # move P2
    b.perform_move(move=Move.LEFT, player=2)
    assert b.player2_pos == (2, 3)
    assert b.is_empty_pos((1, 1))
    assert b.is_empty_pos((3, 3))
    assert not b.is_empty_pos((2, 1))
    assert not b.is_empty_pos((2, 3))

    # move P1 to center
    b.perform_move(move=Move.UP, player=1)
    assert b.player1_pos == (2, 2)


def test_perform_move_candy():
    b = Board(3, 3)
    b.spawn(pos1=(0, 0), pos2=(2, 2))
    candy_pos = (2, 1)
    b._spawn_candy(candy_pos)
    b.perform_move(move=Move.RIGHT, player=1)
    assert not b.has_candy()  # candy should have been eaten
    assert b.player1_length == 2
    assert b.player2_length == 1


def test_undo_move():
    b = Board(3, 3)
    b.spawn(pos1=(0, 0), pos2=(2, 2))
    b_start = b.copy()
    with pytest.raises(Exception):
        b.undo_move(player=-1)
    b.perform_move(move=Move.RIGHT, player=1)
    b.undo_move(player=1)
    assert b == b_start
    assert hash(b) == hash(b_start)
    assert b.approx_hash() == b_start.approx_hash()
    assert b.wall_hash() == b_start.wall_hash()

    with pytest.raises(Exception):
        b.undo_move(player=1)
    with pytest.raises(Exception):
        b.undo_move(player=-1)

    b.perform_move(move=Move.RIGHT, player=1)
    with pytest.raises(Exception):
        b.undo_move(player=-1)  # cannot undo because P1 moved last
    b_ref2 = b.copy()
    b.perform_move(move=Move.LEFT, player=-1)

    b.undo_move(player=-1)
    assert b == b_ref2
    assert hash(b) == hash(b_ref2)
    assert b.approx_hash() == b_ref2.approx_hash()
    assert b.wall_hash() == b_ref2.wall_hash()

    b.undo_move(player=1)
    assert b == b_start
    assert hash(b) == hash(b_start)
    assert b.approx_hash() == b_start.approx_hash()
    assert b.wall_hash() == b_start.wall_hash()

    with pytest.raises(Exception):
        b.undo_move(player=1)
    with pytest.raises(Exception):
        b.undo_move(player=-1)


def test_undo_move_candy():
    b = Board(3, 3)
    b.spawn(pos1=(0, 0), pos2=(2, 2))
    candy_pos = (2, 1)
    b._spawn_candy(candy_pos)
    b_start = b.copy()

    b.perform_move(move=Move.RIGHT, player=1)
    assert not b.has_candy()
    b.undo_move(player=1)
    assert b.has_candy()
    assert b.is_candy_pos(candy_pos)
    assert b == b_start
    assert hash(b) == hash(b_start)
    assert b.approx_hash() == b_start.approx_hash()
    assert b.wall_hash() == b_start.wall_hash()


def test_print():
    b = Board(3, 2)
    assert str(b) == '\n+---+\n|···|\n|···|\n+---+'
    b.spawn(pos1=(1, 0), pos2=(2, 1))
    assert str(b) == '\n+---+\n|··B|\n|·A·|\n+---+'

    b.player1_length = 2
    b.player2_length = 2
    b.perform_move(Move.RIGHT, player=1)
    b.perform_move(Move.LEFT, player=2)
    assert str(b) == '\n+---+\n|·Bb|\n|·aA|\n+---+'


def test_move_generation():
    b = Board(3, 2)
    b.spawn(pos1=(1, 0), pos2=(2, 1))
    moves1 = b.get_valid_moves(player=1)
    assert len(moves1) == 3
    assert Move.LEFT in moves1
    assert Move.RIGHT in moves1
    assert Move.UP in moves1

    moves2 = b.get_valid_moves(player=-1)
    assert len(moves2) == 2
    assert Move.LEFT in moves2
    assert Move.DOWN in moves2

    # perform a move and recheck the options
    b.perform_move(Move.LEFT, player=1)
    moves12 = b.get_valid_moves(player=1)
    assert len(moves12) == 2
    assert Move.RIGHT in moves12
    assert Move.UP in moves12

    b.perform_move(Move.LEFT, player=2)
    moves22 = b.get_valid_moves(2)
    assert len(moves22) == 3
    assert Move.LEFT in moves22
    assert Move.RIGHT in moves22
    assert Move.DOWN in moves22


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
        b.grid[1:-1, 1:-1],
        np.array([[1, 0], [0, -1]])
    )
    assert not b.has_candy()

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
        b.grid[1:-1, 1:-1],
        np.array([[1, 2], [0, -1]])
    )
    assert not b.has_candy()
    assert hash(b) != hash(b0)
    assert b.approx_hash() != b0.approx_hash()
    assert b.wall_hash() != b0.wall_hash()

    b3 = Board(2, 2)
    b3.set_state(
        snake1=Snake(id=0, positions=np.array([[0, 1], [0, 0]])),
        snake2=Snake(id=1, positions=np.array([[1, 0], [1, 1]])),
        candies=[]
    )
    assert b3.player1_head == 2
    assert b3.player2_head == -2
    assert_array_equal(
        b3.grid[1:-1, 1:-1],
        np.array([[1, 2], [-2, -1]])
    )
    assert not b.has_candy()


def test_set_state_candy():
    # one candy
    b = Board(2, 2)
    b.set_state(
        snake1=Snake(id=0, positions=np.array([[0, 0]])),
        snake2=Snake(id=1, positions=np.array([[1, 1]])),
        candies=[(1, 0)]
    )
    b1 = b.copy()
    assert b.has_candy()
    assert b.is_candy_pos((2, 1))

    # no candies (reuse board)
    b.set_state(
        snake1=Snake(id=0, positions=np.array([[0, 0]])),
        snake2=Snake(id=1, positions=np.array([[1, 1]])),
        candies=[]
    )
    assert not b.has_candy()
    assert not b.is_candy_pos((2, 1))
    assert hash(b) != hash(b1)
    assert b.approx_hash() != b1.approx_hash()
    assert b.wall_hash() == b1.wall_hash()  # wall hash should be equal since only diff is candies

    # two candies
    b.set_state(
        snake1=Snake(id=0, positions=np.array([[0, 0]])),
        snake2=Snake(id=1, positions=np.array([[1, 1]])),
        candies=[(1, 0), (0, 1)]
    )
    assert b.has_candy()
    assert b.is_candy_pos((2, 1))
    assert b.is_candy_pos((1, 2))


@pytest.mark.parametrize('move', MOVES)
def test_as_move(move):
    assert as_move(MOVE_VALUE_TO_DIRECTION[move]) == move
