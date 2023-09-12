import numpy as np
import pytest

from ..board import Board, MOVE_LEFT, MOVE_UP, MOVE_RIGHT, MOVE_DOWN, as_move
from snakes.constants import Move
from snakes.snake import Snake


def test_init():
    b = Board(8, 6)
    assert b.shape == (8, 6)
    assert b.size == 8 * 6

    assert np.all(b.get_empty_mask() == True)
    assert b.get_free_space() == 8 * 6
    assert np.all(b.get_player1_mask() == False)
    assert np.all(b.get_player2_mask() == False)
    assert b.is_empty_pos((0, 0))
    assert not b.has_candy()


def test_spawn():
    b = Board(8, 8)
    b.spawn(pos1=(1, 2), pos2=(2, 3))

    assert np.array_equal(b.player1_pos, np.array((1, 2)))
    assert np.array_equal(b.player2_pos, np.array((2, 3)))
    assert np.sum(b.get_player1_mask()) == 1
    assert np.sum(b.get_player2_mask()) == 1
    assert not b.is_empty_pos(b.player1_pos)
    assert not b.is_empty_pos(b.player2_pos)
    assert b.get_free_space() == 8 * 8 - 2

    # candies
    assert np.all(b.get_candy_mask() == False)
    assert not b.has_candy()


def test_move():
    b = Board(3, 3)
    b.spawn(pos1=(0, 0), pos2=(2, 2))

    # move P1
    with pytest.raises(AssertionError):
        b.perform_move(move=MOVE_LEFT, player=1)  # cannot move left
    with pytest.raises(AssertionError):
        b.perform_move(move=MOVE_UP, player=1)  # cannot move up

    b.perform_move(move=MOVE_RIGHT, player=1)
    assert np.array_equal(b.player1_pos, np.array((1, 0)))

    with pytest.raises(AssertionError):
        b.perform_move(move=MOVE_RIGHT, player=1)  # cannot move twice in a row

    # move P2
    with pytest.raises(AssertionError):
        b.perform_move(move=MOVE_RIGHT, player=2)  # cannot move right
    with pytest.raises(AssertionError):
        b.perform_move(move=MOVE_DOWN, player=2)  # cannot move down
    b.perform_move(move=MOVE_LEFT, player=2)
    assert np.array_equal(b.player2_pos, np.array((1, 2)))

    with pytest.raises(AssertionError):
        b.perform_move(move=MOVE_LEFT, player=2)  # cannot move twice in a row

    # move P1 to center
    b.perform_move(move=MOVE_DOWN, player=1)
    assert np.array_equal(b.player1_pos, np.array((1, 1)))

    # attempt to move P2 to center (suicide)
    with pytest.raises(AssertionError):
        b.perform_move(move=MOVE_UP, player=2)


def test_print():
    b = Board(3, 2)
    assert b.__str__() == '+---+\n|   |\n|   |\n+---+'
    b.spawn(pos1=(1, 0), pos2=(2, 1))
    assert b.__str__() == '+---+\n| A |\n|  B|\n+---+'

    b.player1_length = 2
    b.player2_length = 2
    b.perform_move(MOVE_RIGHT, player=1)
    b.perform_move(MOVE_LEFT, player=2)
    assert b.__str__() == '+---+\n| aA|\n| Bb|\n+---+'


def test_move_generation():
    b = Board(3, 2)
    b.spawn(pos1=(1, 0), pos2=(2, 1))
    moves1 = list(map(tuple, b.get_valid_moves(1)))
    assert len(moves1) == 3
    assert tuple(MOVE_LEFT) in moves1
    assert tuple(MOVE_RIGHT) in moves1
    assert tuple(MOVE_DOWN) in moves1

    moves2 = list(map(tuple, b.get_valid_moves(2)))
    assert len(moves2) == 2
    assert tuple(MOVE_LEFT) in moves2
    assert tuple(MOVE_UP) in moves2

    # perform a move and recheck the options
    b.perform_move(MOVE_LEFT, 1)
    moves12 = list(map(tuple, b.get_valid_moves(1)))
    assert len(moves12) == 2
    assert tuple(MOVE_RIGHT) in moves12
    assert tuple(MOVE_DOWN) in moves12

    b.perform_move(MOVE_LEFT, 2)
    moves22 = list(map(tuple, b.get_valid_moves(2)))
    assert len(moves22) == 3
    assert tuple(MOVE_LEFT) in moves22
    assert tuple(MOVE_RIGHT) in moves22
    assert tuple(MOVE_UP) in moves22


def test_set_state():
    b = Board(2, 2)
    snakes = [Snake(id=0, positions=np.array([[0, 0]])), Snake(id=1, positions=np.array([[1, 1]]))]
    b.set_state(snakes=snakes, candies=[], turn=1)
    assert b.turn == 1
    assert b.player1_turn == 1
    assert b.player2_turn == 1
    assert np.array_equal(b.grid, np.array([[1, 0], [0, -1]]))
    assert len(b.candies) == 0

    b2 = Board(2, 2)
    snakes2 = [Snake(id=0, positions=np.array([[0, 0], [0, 1]])), Snake(id=1, positions=np.array([[1, 1]]))]
    b2.set_state(snakes=snakes2, candies=[], turn=2)
    assert b2.turn == 2
    assert b2.player1_turn == 2
    assert b2.player2_turn == 1
    assert np.array_equal(b2.grid, np.array([[1, 2], [0, -1]]))
    assert len(b2.candies) == 0

    b3 = Board(2, 2)
    snakes3 = [Snake(id=0, positions=np.array([[0, 0], [0, 1]])), Snake(id=1, positions=np.array([[1, 1], [1, 0]]))]
    b3.set_state(snakes=snakes3, candies=[], turn=3)
    assert b3.turn == 3
    assert b3.player1_turn == 2
    assert b3.player2_turn == 2
    assert np.array_equal(b3.grid, np.array([[1, 2], [-2, -1]]))
    assert len(b3.candies) == 0


def test_move_conversion():
    assert Move.LEFT == as_move(MOVE_LEFT)
    assert Move.RIGHT == as_move(MOVE_RIGHT)
    assert Move.UP == as_move(MOVE_UP)
    assert Move.DOWN == as_move(MOVE_DOWN)
