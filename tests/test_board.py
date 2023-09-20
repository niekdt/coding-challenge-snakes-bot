import itertools
from copy import deepcopy

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from ..board import Board, as_move
from ....constants import Move, LEFT, DOWN, RIGHT, UP
from ....snake import Snake


def test_init():
    b = Board(8, 6)
    assert b.shape == (8, 6)
    assert len(b) == 8 * 6

    assert np.all(b.get_empty_mask() == True)
    assert np.all(b.get_player1_mask() == False)
    assert np.all(b.get_player2_mask() == False)
    assert not b.has_candy()


def test_spawn():
    b = Board(4, 4)
    b.spawn(pos1=(1, 2), pos2=(2, 3))

    assert_array_equal(b.player1_pos, np.array((1, 2)))
    assert_array_equal(b.player2_pos, np.array((2, 3)))
    assert np.sum(b.get_player1_mask()) == 1
    assert np.sum(b.get_player2_mask()) == 1
    assert not b.is_empty_pos(b.player1_pos)
    assert not b.is_empty_pos(b.player2_pos)


def test_is_valid_pos():
    b = Board(3, 2)
    for x, y in itertools.product(range(b.width), range(b.height)):
        assert b.is_valid_pos(np.array((x, y)))

    assert not b.is_valid_pos(np.array((-1, 0)))
    assert not b.is_valid_pos(np.array((0, -1)))
    assert not b.is_valid_pos(np.array((b.width, 0)))
    assert not b.is_valid_pos(np.array((0, b.height)))


def test_get_empty_mask():
    b = Board(3, 2)
    ref_mask = np.full(b.shape, fill_value=True)
    assert_array_equal(b.get_empty_mask(), ref_mask)

    b.spawn(pos1=(0, 0), pos2=(2, 1))
    ref_mask[(0, 0)] = False
    ref_mask[(2, 1)] = False
    assert_array_equal(b.get_empty_mask(), ref_mask)


def test_is_empty_pos():
    b = Board(3, 2)
    for x, y in itertools.product(range(b.width), range(b.height)):
        assert b.is_empty_pos(np.array((x, y)))

    b.spawn(pos1=(0, 0), pos2=(2, 1))
    assert not b.is_empty_pos(np.array((0, 0)))
    assert not b.is_empty_pos(np.array((2, 1)))

    assert b.is_empty_pos(np.array((0, 1)))


def test_spawn_candy():
    b = Board(3, 2)
    assert not b.has_candy()
    assert not b.candy_mask[(0, 1)]
    b.spawn_candy(np.array((0, 1)))
    assert b.has_candy()
    assert b.candy_mask[(0, 1)]


def test_remove_candy():
    b = Board(3, 2)
    b.spawn_candy(np.array((0, 1)))
    b.remove_candy(np.array((0, 1)))
    assert not b.has_candy()
    assert not b.candy_mask[(0, 1)]


def test_get_candy_mask():
    b = Board(3, 2)
    ref_mask = np.full(b.shape, fill_value=False)
    assert np.array_equal(b.get_candy_mask(), ref_mask)
    candy_pos = np.array([1, 1])
    b.spawn_candy(candy_pos)

    ref_mat = np.full(b.shape, fill_value=False)
    ref_mat[tuple(candy_pos)] = True
    assert np.array_equal(b.get_candy_mask(), ref_mat)


def test_is_candy_pos():
    b = Board(3, 2)
    for x, y in itertools.product(range(b.width), range(b.height)):
        assert not b.is_candy_pos(np.array((x, y)))

    candy_pos = np.array([1, 1])
    b.spawn_candy(candy_pos)
    assert b.has_candy()
    assert b.is_candy_pos(candy_pos)

    for x, y in itertools.product(range(b.width), range(b.height)):
        if x == candy_pos[0] and y == candy_pos[1]:
            continue
        else:
            assert not b.is_candy_pos(np.array((x, y)))


def test_free_space():
    b = Board(8, 6)
    assert b.get_free_space() == 8 * 6

    b.spawn(pos1=(1, 2), pos2=(2, 3))
    assert b.get_free_space() == 8 * 6 - 2

    b.spawn_candy(np.array((0, 0)))
    assert b.get_free_space() == 8 * 6 - 2 # should not affect free space


def test_perform_move():
    b = Board(3, 3)
    b.spawn(pos1=(0, 0), pos2=(2, 2))

    # move P1
    with pytest.raises(AssertionError):
        b.perform_move(move=LEFT, player=1)  # cannot move left
    with pytest.raises(AssertionError):
        b.perform_move(move=DOWN, player=1)  # cannot move down

    b.perform_move(move=RIGHT, player=1)
    assert_array_equal(b.player1_pos, np.array((1, 0)))

    # move P2
    with pytest.raises(AssertionError):
        b.perform_move(move=RIGHT, player=2)  # cannot move right
    with pytest.raises(AssertionError):
        b.perform_move(move=UP, player=2)  # cannot move up
    b.perform_move(move=LEFT, player=2)
    assert_array_equal(b.player2_pos, np.array((1, 2)))

    # move P1 to center
    b.perform_move(move=UP, player=1)
    assert_array_equal(b.player1_pos, np.array((1, 1)))

    # attempt to move P2 to center (suicide)
    with pytest.raises(AssertionError):
        b.perform_move(move=DOWN, player=2)


def test_perform_move_candy():
    b = Board(3, 3)
    b.spawn(pos1=(0, 0), pos2=(2, 2))
    candy_pos = np.array([1, 0])
    b.spawn_candy(candy_pos)
    b.perform_move(move=RIGHT, player=1)
    assert not b.has_candy()  # candy should have been eaten
    assert b.player1_length == 2
    assert b.player2_length == 1


def test_undo_move():
    b = Board(3, 3)
    b.spawn(pos1=(0, 0), pos2=(2, 2))
    b_start = b.copy()
    with pytest.raises(AssertionError):
        b.undo_move(player=-1)
    b.perform_move(move=RIGHT, player=1)
    b.undo_move(player=1)
    assert b == b_start
    with pytest.raises(AssertionError):
        b.undo_move(player=1)
    with pytest.raises(AssertionError):
        b.undo_move(player=-1)

    b.perform_move(move=RIGHT, player=1)
    with pytest.raises(AssertionError):
        b.undo_move(player=-1)  # cannot undo because P1 moved last
    b_ref2 = b.copy()
    b.perform_move(move=LEFT, player=-1)

    b.undo_move(player=-1)
    assert b == b_ref2

    b.undo_move(player=1)
    assert b == b_start

    with pytest.raises(AssertionError):
        b.undo_move(player=1)
    with pytest.raises(AssertionError):
        b.undo_move(player=-1)


def test_undo_move_candy():
    b = Board(3, 3)
    b.spawn(pos1=(0, 0), pos2=(2, 2))
    candy_pos = np.array([1, 0])
    b.spawn_candy(candy_pos)
    b_start = b.copy()

    b.perform_move(move=RIGHT, player=1)
    assert not b.has_candy()
    b.undo_move(player=1)
    assert b.has_candy()
    assert b.is_candy_pos(candy_pos)
    assert b == b_start


def test_print():
    b = Board(3, 2)
    assert str(b) == '\n+---+\n|···|\n|···|\n+---+'
    b.spawn(pos1=(1, 0), pos2=(2, 1))
    assert str(b) == '\n+---+\n|··B|\n|·A·|\n+---+'

    b.player1_length = 2
    b.player2_length = 2
    b.perform_move(RIGHT, player=1)
    b.perform_move(LEFT, player=2)
    assert str(b) == '\n+---+\n|·Bb|\n|·aA|\n+---+'


def test_move_generation():
    b = Board(3, 2)
    b.spawn(pos1=(1, 0), pos2=(2, 1))
    moves1 = list(map(tuple, b.get_valid_moves(1)))
    assert len(moves1) == 3
    assert tuple(LEFT) in moves1
    assert tuple(RIGHT) in moves1
    assert tuple(UP) in moves1

    moves2 = list(map(tuple, b.get_valid_moves(2)))
    assert len(moves2) == 2
    assert tuple(LEFT) in moves2
    assert tuple(DOWN) in moves2

    # perform a move and recheck the options
    b.perform_move(LEFT, 1)
    moves12 = list(map(tuple, b.get_valid_moves(1)))
    assert len(moves12) == 2
    assert tuple(RIGHT) in moves12
    assert tuple(UP) in moves12

    b.perform_move(LEFT, 2)
    moves22 = list(map(tuple, b.get_valid_moves(2)))
    assert len(moves22) == 3
    assert tuple(LEFT) in moves22
    assert tuple(RIGHT) in moves22
    assert tuple(DOWN) in moves22


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
        b.grid,
        np.array([[1, 0], [0, -1]])
    )
    assert not b.has_candy()

    # reuse board
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
        b.grid,
        np.array([[1, 2], [0, -1]])
    )
    assert not b.has_candy()

    b3 = Board(2, 2)
    b3.set_state(
        snake1=Snake(id=0, positions=np.array([[0, 1], [0, 0]])),
        snake2=Snake(id=1, positions=np.array([[1, 0], [1, 1]])),
        candies=[]
    )
    assert b3.player1_head == 2
    assert b3.player2_head == -2
    assert_array_equal(
        b3.grid,
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
    assert b.has_candy()
    assert b.is_candy_pos(np.array((1, 0)))

    # no candies (reuse board)
    b.set_state(
        snake1=Snake(id=0, positions=np.array([[0, 0]])),
        snake2=Snake(id=1, positions=np.array([[1, 1]])),
        candies=[]
    )
    assert not b.has_candy()
    assert not b.is_candy_pos(np.array((1, 0)))

    # two candies
    b.set_state(
        snake1=Snake(id=0, positions=np.array([[0, 0]])),
        snake2=Snake(id=1, positions=np.array([[1, 1]])),
        candies=[(1, 0), (0, 1)]
    )
    assert b.has_candy()
    assert b.is_candy_pos(np.array((1, 0)))
    assert b.is_candy_pos(np.array((0, 1)))


def test_as_move():
    assert as_move(LEFT) == Move.LEFT
    assert as_move(RIGHT) == Move.RIGHT
    assert as_move(UP) == Move.UP
    assert as_move(DOWN) == Move.DOWN
