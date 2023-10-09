import gc

import numpy as np
import pytest

from snakes.bots.niekdt.eval.annotation import AnnotatedBoard
from snakes.bots.niekdt.search.pvs import pvs_moves
from ..board import Board, BoardMove
from ..eval import death, candy_dist
from ..search.choose import best_move, has_single_best_move
from ..search.negamax import negamax_moves, negamax_ab_moves
from ....snake import Snake


@pytest.fixture(autouse=True)
def cleanup():
    gc.collect()
    gc.disable()


@pytest.mark.parametrize('depth', [1, 2, 4, 7])
@pytest.mark.parametrize('search', [negamax_moves, negamax_ab_moves, pvs_moves])
def test_forced_move(depth, search):
    board = Board(3, 2)
    board.set_state_from_game(
        snake1=Snake(id=0, positions=np.array([[1, 0], [0, 0]])),
        snake2=Snake(id=1, positions=np.array([[1, 1], [0, 1]])),
        candies=[]
    )

    moves = search(board, depth=depth, eval_fun=death.evaluate, move_history=dict())
    assert set(moves) == {BoardMove.RIGHT}


@pytest.mark.parametrize('depth', [1, 2, 4, 7])
@pytest.mark.parametrize('search', [pvs_moves])
def test_winning_suicide(depth, search):
    board = Board(4, 4)
    board.set_state_from_game(
        snake1=Snake(id=0, positions=np.array([[3, 1], [3, 0], [2, 0], [1, 0], [0, 0]])),
        snake2=Snake(id=1, positions=np.array([[2, 3], [3, 3]])),
        candies=[]
    )
    ref_board = board.copy()

    moves = search(board, depth=depth, eval_fun=death.evaluate, move_history=dict())

    assert board == ref_board
    assert set(moves) == {BoardMove.DOWN}


@pytest.mark.parametrize('depth', [1, 2, 4, 7])
@pytest.mark.parametrize('size', [4, 16])
@pytest.mark.parametrize('search', [negamax_moves])
def test_goto_candy(depth, search, size):
    board = Board(size, size)
    board.set_state_from_game(
        snake1=Snake(id=0, positions=np.array([[2, 1], [1, 1]])),
        snake2=Snake(id=1, positions=np.array([[2, 2], [1, 2]])),
        candies=[np.array((size - 1, 1))]
    )

    aboard = AnnotatedBoard(board)
    aboard.moves = [BoardMove.RIGHT]
    moves = search(board, depth=depth, eval_fun=candy_dist.evaluate, move_history=dict())

    assert has_single_best_move(moves)
    assert best_move(moves) == BoardMove.RIGHT


@pytest.mark.parametrize('search,depth', [(negamax_moves, 10), (negamax_ab_moves, 16), (pvs_moves, 13)])
@pytest.mark.timeout(5)
def test_computation_time(search, depth):
    board = Board(16, 16)
    board.set_state_from_game(
        snake1=Snake(id=0, positions=np.array([[1, 0], [0, 0]])),
        snake2=Snake(id=1, positions=np.array([[3, 4], [4, 4]])),
        candies=[np.array((2, 0))]
    )
    print(board)

    search(board, depth=depth, eval_fun=candy_dist.evaluate, move_history=dict())
