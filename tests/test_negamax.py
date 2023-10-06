import gc

import numpy as np
import pytest

from snakes.bots.niekdt.search.pvs import pvs_moves
from ..board import Board, BoardMove
from ..eval import death, candy_dist, best
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

    moves = search(board, depth=depth, eval_fun=candy_dist.evaluate, move_history=dict())

    assert has_single_best_move(moves)
    assert best_move(moves) == BoardMove.RIGHT


@pytest.mark.parametrize('depth', [1, 2, 4, 7])
@pytest.mark.parametrize('search', [negamax_moves, negamax_ab_moves, pvs_moves])
def test_goto_candy_far(depth, search):
    board = Board(16, 16)
    board.set_state_from_game(
        snake1=Snake(id=0, positions=np.array([[13, 1], [13, 0], [12, 0]])),
        snake2=Snake(id=1, positions=np.array([[6, 7], [6, 6]])),
        candies=[np.array((0, 12)), np.array((1, 9)), np.array((4, 2))]
    )

    moves = search(board, depth=depth, eval_fun=best.evaluate, move_history=dict())
    print(moves)

    assert best_move(moves) in [BoardMove.LEFT, BoardMove.UP]
    if search == negamax_moves:
        assert moves[BoardMove.LEFT] == moves[BoardMove.UP]


@pytest.mark.parametrize('depth', [1, 2, 4, 7])
@pytest.mark.parametrize('search', [negamax_moves, negamax_ab_moves, pvs_moves])
def test_goto_candy_near(depth, search):
    board = Board(16, 16)
    board.set_state_from_game(
        snake1=Snake(id=0, positions=np.array([[5, 14], [4, 14]])),
        snake2=Snake(id=1, positions=np.array([[13, 7], [13, 6]])),
        candies=[np.array((6, 14)), np.array((7, 14)), np.array((6, 9))]
    )

    moves = search(board, depth=depth, eval_fun=candy_dist.evaluate, move_history=dict())
    print(moves)

    assert best_move(moves) == BoardMove.RIGHT
    assert has_single_best_move(moves)


@pytest.mark.parametrize('depth', [1, 2, 4, 7, 14])
@pytest.mark.parametrize('search', [negamax_moves, negamax_ab_moves, pvs_moves])
def test_goto_candy_near2(depth, search):
    if depth > 10 and search == negamax_moves:
        pytest.skip()

    board = Board(16, 9)
    board.set_state_from_game(
        snake1=Snake(id=1, positions=np.array([[3, 1], [3, 0], [4, 0]])),
        snake2=Snake(id=0, positions=np.array([[14, 6], [14, 5]])),
        candies=[np.array((2, 2)), np.array((14, 8))]
    )

    moves = search(board, depth=depth, eval_fun=best.evaluate, move_history=dict())

    assert best_move(moves) in (BoardMove.LEFT, BoardMove.UP)


@pytest.mark.parametrize('search,depth', [(negamax_moves, 10), (negamax_ab_moves, 12), (pvs_moves, 12)])
def test_search_extension_issue(search, depth):
    board = Board(16, 16)
    board.set_state_from_game(
        snake1=Snake(id=1, positions=np.array(
            [[13, 11], [13, 10], [13, 9], [13, 8], [13, 7], [13, 6], [13, 5], [12, 5], [11, 5], [10, 5], [9, 5], [8, 5],
             [7, 5], [7, 4], [7, 3], [6, 3], [5, 3], [5, 4], [5, 5], [5, 6], [5, 7], [5, 8], [4, 8], [3, 8], [2, 8],
             [1, 8], [1, 7], [1, 6], [1, 5], [1, 4], [1, 3], [2, 3], [2, 4], [2, 5], [2, 6], [2, 7], [3, 7], [4, 7],
             [4, 6], [4, 5], [4, 4], [4, 3], [4, 2], [4, 1], [5, 1], [6, 1]])),
        snake2=Snake(id=0, positions=np.array(
            [[10, 14], [9, 14], [8, 14], [8, 13], [8, 12], [8, 11], [9, 11], [9, 10], [9, 9], [9, 8], [9, 7], [8, 7],
             [8, 6], [7, 6], [6, 6], [6, 7], [7, 7], [7, 8], [8, 8], [8, 9], [8, 10], [7, 10], [7, 11], [6, 11],
             [5, 11], [4, 11], [4, 12], [3, 12], [2, 12], [1, 12], [1, 13], [2, 13], [3, 13]])),
        candies=[np.array([12, 12]), np.array([10, 5]), np.array([10, 15])]
    )

    moves = search(board, depth=depth, eval_fun=best.evaluate, move_history=dict())

    # unclear which move is best, but RIGHT is not picked due to pruning
    assert best_move(moves) in (BoardMove.LEFT, BoardMove.UP)


@pytest.mark.parametrize('search,depth', [(pvs_moves, 6)])
def test_trap8(search, depth):
    board = Board(16, 16)
    board.set_state_from_game(
        snake1=Snake(id=1, positions=np.array(
            [[14, 11], [14, 12], [13, 12], [12, 12], [11, 12], [10, 12], [10, 13], [9, 13], [8, 13], [7, 13], [7, 12],
             [7, 11], [7, 10], [8, 10], [9, 10], [10, 10], [11, 10], [12, 10], [13, 10], [14, 10], [14, 9], [13, 9],
             [12, 9], [11, 9], [10, 9], [10, 8]])),
        snake2=Snake(id=0, positions=np.array(
            [[1, 10], [1, 11], [1, 12], [1, 13], [1, 14], [2, 14], [3, 14], [4, 14], [5, 14], [6, 14], [6, 13], [6, 12],
             [6, 11], [6, 10], [6, 9], [6, 8], [6, 7], [6, 6], [6, 5]])),
        candies=[np.array([2, 9]), np.array([4, 10]), np.array([4, 9])]
    )

    moves = search(board, depth=depth, eval_fun=best.evaluate, move_history=dict())
    print(moves)

    assert best_move(moves) == BoardMove.RIGHT


@pytest.mark.parametrize('search,depth', [(pvs_moves, 12)])
@pytest.mark.timeout(2)
def test_wall_search(search, depth):
    board = Board(16, 16)
    board.set_state_from_game(
        snake1=Snake(id=1, positions=np.array(
            [[15, 11], [14, 11], [14, 12], [13, 12], [12, 12], [11, 12], [10, 12], [10, 13], [9, 13], [8, 13], [7, 13],
             [7, 12], [7, 11], [7, 10], [8, 10], [9, 10], [10, 10], [11, 10], [12, 10], [13, 10], [14, 10], [14, 9],
             [13, 9], [12, 9], [11, 9], [10, 9], [10, 8]])),
        snake2=Snake(id=0, positions=np.array(
            [[1, 10], [1, 11], [1, 12], [1, 13], [1, 14], [2, 14], [3, 14], [4, 14], [5, 14], [6, 14], [6, 13], [6, 12],
             [6, 11], [6, 10], [6, 9], [6, 8], [6, 7], [6, 6], [6, 5]])),
        candies=[np.array([2, 9]), np.array([4, 10]), np.array([4, 9])]
    )
    moves = search(board, depth=depth, eval_fun=best.evaluate, move_history=dict())

    assert best_move(moves) == BoardMove.DOWN


@pytest.mark.parametrize('search,depth', [(negamax_moves, 10), (negamax_ab_moves, 12), (pvs_moves, 12)])
@pytest.mark.timeout(2)
def test_follow_tail(search, depth):
    board = Board(16, 16)
    board.set_state_from_game(
        snake1=Snake(id=1, positions=np.array(
            [[7, 5], [8, 5], [9, 5], [10, 5], [10, 6], [9, 6], [8, 6], [8, 7], [8, 8], [8, 9], [7, 9], [7, 8], [7, 7],
             [6, 7], [5, 7], [4, 7], [3, 7], [2, 7], [2, 8], [2, 9], [2, 10], [2, 11], [2, 12], [2, 13], [1, 13],
             [0, 13], [0, 14], [1, 14], [2, 14], [3, 14]])),
        snake2=Snake(id=0, positions=np.array(
            [[9, 14], [10, 14], [10, 13], [9, 13], [8, 13], [8, 12], [8, 11], [9, 11], [9, 10], [9, 9], [9, 8], [10, 8],
             [10, 7], [11, 7], [11, 6], [12, 6], [12, 5], [11, 5], [11, 4], [10, 4], [10, 3], [10, 2], [10, 1], [9, 1],
             [9, 2], [8, 2], [7, 2], [6, 2], [6, 3], [7, 3], [8, 3], [8, 4], [7, 4], [6, 4], [5, 4], [4, 4], [4, 5],
             [4, 6], [5, 6], [5, 5], [6, 5], [6, 6]])),
        candies=[]
    )

    moves = search(board, depth=depth, eval_fun=best.evaluate, move_history=dict())
    assert moves[BoardMove.UP] > -10 ** 5


@pytest.mark.parametrize('search,depth', [(negamax_moves, 10), (negamax_ab_moves, 12), (pvs_moves, 12)])
def test_losing_position_gap(search, depth):
    board = Board(16, 16)
    board.set_state_from_game(
        snake1=Snake(id=1, positions=np.array([[4, 6], [4, 7], [3, 7], [3, 8], [2, 8], [2, 9], [2, 10], [3, 10], [4, 10], [5, 10], [6, 10], [6, 11], [6, 12], [5, 12], [5, 13], [5, 14], [6, 14], [7, 14], [7, 13], [7, 12]])),
        snake2=Snake(id=0, positions=np.array([[6, 5], [7, 5], [8, 5], [9, 5], [9, 6], [9, 7], [8, 7], [8, 8], [7, 8], [7, 9], [8, 9], [8, 10], [9, 10], [10, 10], [11, 10], [12, 10], [13, 10], [13, 9], [12, 9]])),
        candies=[np.array([9, 12]), np.array([5, 4]), np.array([7, 2])]
    )

    moves = search(board, depth=depth, eval_fun=best.evaluate, move_history=dict())
    assert set(moves) == {BoardMove.DOWN, BoardMove.RIGHT, BoardMove.LEFT}
    assert moves[BoardMove.RIGHT] < moves[BoardMove.DOWN]
    assert moves[BoardMove.RIGHT] < moves[BoardMove.LEFT]


@pytest.mark.parametrize('search,depth', [(negamax_moves, 10), (negamax_ab_moves, 12), (pvs_moves, 12)])
def test_losing_position_gap_tunnel_escape(search, depth):
    board = Board(16, 16)
    board.set_state_from_game(
        snake1=Snake(id=1, positions=np.array([[3, 2], [3, 3], [3, 4], [3, 5], [3, 6], [3, 7], [2, 7], [1, 7], [0, 7], [0, 8], [1, 8], [2, 8], [3, 8], [4, 8], [5, 8], [5, 9], [5, 10], [6, 10], [7, 10], [8, 10], [9, 10], [9, 9], [8, 9]])),
        snake2=Snake(id=0, positions=np.array([[5, 2], [5, 3], [5, 4], [5, 5], [5, 6], [6, 6], [7, 6], [7, 7], [7, 8], [8, 8], [8, 7], [8, 6], [8, 5], [9, 5], [10, 5], [11, 5]])),
        candies=[np.array([2, 13]), np.array([11, 4]), np.array([2, 6])]
    )

    moves = search(board, depth=depth, eval_fun=best.evaluate, move_history=dict())
    assert set(moves) == {BoardMove.LEFT, BoardMove.RIGHT, BoardMove.DOWN}
    assert moves[BoardMove.RIGHT] > moves[BoardMove.DOWN]
    assert moves[BoardMove.RIGHT] > moves[BoardMove.LEFT]


@pytest.mark.parametrize('search,depth', [(negamax_moves, 10), (negamax_ab_moves, 12), (pvs_moves, 12)])
def test_winning_gap(search, depth):
    board = Board(16, 16)
    board.set_state_from_game(
        snake1=Snake(id=1, positions=np.array([[8, 6], [9, 6], [10, 6], [10, 7], [10, 8], [10, 9], [11, 9], [12, 9], [13, 9], [13, 10], [12, 10], [11, 10], [11, 11], [11, 12], [10, 12], [9, 12], [8, 12], [8, 11], [7, 11], [6, 11], [5, 11], [5, 10], [5, 9], [5, 8], [5, 7], [4, 7], [4, 6], [5, 6]])),
        snake2=Snake(id=0, positions=np.array([[8, 0], [7, 0], [7, 1], [8, 1], [8, 2], [8, 3], [7, 3], [7, 4], [7, 5], [6, 5], [6, 4], [6, 3], [5, 3], [5, 2], [5, 1], [4, 1], [3, 1], [3, 0], [2, 0], [1, 0], [0, 0], [0, 1], [0, 2], [1, 2], [1, 1], [2, 1], [2, 2], [3, 2]])),
        candies=[]
    )
    print(board)

    moves = search(board, depth=depth, eval_fun=best.evaluate, move_history=dict())
    assert set(moves) == {BoardMove.DOWN, BoardMove.LEFT, BoardMove.UP}
    assert moves[BoardMove.DOWN] > moves[BoardMove.LEFT]
    assert moves[BoardMove.DOWN] > moves[BoardMove.UP]


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
