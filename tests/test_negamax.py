import contextlib
import os
import time
from math import inf

import numpy as np
import pytest

from snakes.bots.niekdt.board import Board
from snakes.bots.niekdt.eval import death, candy_dist, best
from snakes.bots.niekdt.search.choose import best_move, has_single_best_move, get_best_moves_count
from snakes.bots.niekdt.search.negamax import negamax_moves, negamax_ab_moves
from snakes.constants import Move
from snakes.snake import Snake


@pytest.mark.parametrize('depth', [1, 2, 3])
@pytest.mark.parametrize('search', [negamax_moves, negamax_ab_moves])
def test_forced_move(depth, search):
    board = Board(2, 2)
    board.set_state(
        snake1=Snake(id=0, positions=np.array([[0, 1]])),
        snake2=Snake(id=1, positions=np.array([[1, 1]])),
        candies=[]
    )

    moves = search(board, depth=depth, eval_fun=death.evaluate)
    assert len(moves) == 1
    assert Move.DOWN in moves
    assert moves[Move.DOWN] == 0


@pytest.mark.parametrize('depth', [1, 2, 3, 4])
@pytest.mark.parametrize('search', [negamax_moves, negamax_ab_moves])
def test_winning_move(depth, search):
    board = Board(3, 3)
    board.set_state(
        snake1=Snake(id=0, positions=np.array([[1, 1], [0, 1], [0, 0]])),
        snake2=Snake(id=1, positions=np.array([[0, 2]])),
        candies=[]
    )
    ref_board = board.copy()

    moves = search(board, depth=1, eval_fun=death.evaluate)
    assert moves == {Move.RIGHT: 0, Move.UP: inf, Move.DOWN: 0}

    assert board == ref_board


@pytest.mark.parametrize('depth', [1, 2, 3, 4, 5, 6, 7])
@pytest.mark.parametrize('size', [4, 8])
@pytest.mark.parametrize('search', [negamax_moves])
def test_goto_candy(depth, search, size):
    board = Board(size, size)
    board.set_state(
        snake1=Snake(id=0, positions=np.array([[1, 1]])),
        snake2=Snake(id=1, positions=np.array([[1, 2]])),
        candies=[np.array((size - 1, 1))]
    )

    moves = search(board, depth=depth, eval_fun=candy_dist.evaluate)

    assert has_single_best_move(moves)
    assert best_move(moves) == Move.RIGHT


@pytest.mark.parametrize('depth', [1, 6, 7, 10])
@pytest.mark.parametrize('search', [negamax_moves, negamax_ab_moves])
def test_goto_candy_far(depth, search):
    board = Board(16, 16)
    board.set_state(
        snake1=Snake(id=0, positions=np.array([[13, 1], [13, 0], [12, 0]])),
        snake2=Snake(id=1, positions=np.array([[6, 7], [6, 6]])),
        candies=[np.array((0, 12)), np.array((1, 9)), np.array((4, 2))]
    )

    moves = search(board, depth=depth, eval_fun=best.evaluate)
    print(moves)

    assert best_move(moves) in [Move.LEFT, Move.UP]
    assert moves[Move.LEFT] == moves[Move.UP]


@pytest.mark.parametrize('depth', [1, 6, 7, 10])
@pytest.mark.parametrize('search', [negamax_moves, negamax_ab_moves])
def test_goto_candy_near(depth, search):
    board = Board(16, 16)
    board.set_state(
        snake1=Snake(id=0, positions=np.array([[5, 14], [4, 14]])),
        snake2=Snake(id=1, positions=np.array([[13, 7], [13, 6]])),
        candies=[np.array((6, 14)), np.array((7, 14)), np.array((6, 9))]
    )

    moves = search(board, depth=depth, eval_fun=candy_dist.evaluate)
    print(moves)

    assert best_move(moves) == Move.RIGHT
    assert has_single_best_move(moves)


@pytest.mark.parametrize('depth', [1, 6, 7, 10])
@pytest.mark.parametrize('search', [negamax_moves, negamax_ab_moves])
def test_goto_candy_near2(depth, search):
    if depth > 10 and search == negamax_moves:
        pytest.skip()

    if not (depth >= 10 and search == negamax_ab_moves):
        pytest.skip()

    board = Board(16, 9)
    board.set_state(
        snake1=Snake(id=1, positions=np.array([[3, 1], [3, 0], [4, 0]])),
        snake2=Snake(id=0, positions=np.array([[14, 6], [14, 5]])),
        candies=[np.array((2, 2)), np.array((14, 8))]
    )
    print(board)

    moves_ref = negamax_moves(board, depth=depth, eval_fun=best.evaluate)
    print(f'Ref moves: {moves_ref}')
    moves = search(board, depth=depth, eval_fun=best.evaluate)

    assert moves == moves_ref
    print('>' * 80)
    print(moves)

    assert get_best_moves_count(moves) < 3


@pytest.mark.parametrize('search', [negamax_moves, negamax_ab_moves])
def test_computation_time(search):
    depth = 8
    eval_fun = candy_dist.evaluate
    board = Board(16, 16)
    board.set_state(
        snake1=Snake(id=0, positions=np.array([[0, 0]])),
        snake2=Snake(id=1, positions=np.array([[4, 4]])),
        candies=[np.array((2, 0))]
    )

    start = time.time()
    with open(os.devnull, 'w') as f, contextlib.redirect_stdout(f):
        moves = search(board, depth=depth, eval_fun=eval_fun)
        move = best_move(moves)
    assert (time.time() - start) < .23

