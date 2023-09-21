from math import inf
from typing import Dict

from snakes.constants import Move

from ..board import Board


def negamax_moves(board: Board, depth: int, eval_fun: callable) -> Dict[Move, float]:
    moves = board.get_valid_moves(player=1)
    assert len(moves) > 0, 'no possible moves!'

    move_values = dict()
    for move in moves:
        board.perform_move(move, player=1)
        move_values[move] = -negamax(board, depth=depth - 1, player=-1, eval_fun=eval_fun)
        board.undo_move(player=1)

    return move_values


def negamax(board: Board, depth: int, player: int, eval_fun: callable) -> float:
    if depth == 0:
        return eval_fun(board, player=player)

    moves = board.get_valid_moves(player=player)
    if len(moves) == 0:  # current player is stuck
        return -inf  # TODO compute game score, as we may still have won if the other player died first

    best_value = -inf
    for move in moves:
        board.perform_move(move, player=player)
        best_value = max(
            best_value,
            -negamax(board, depth=depth - 1, player=-player, eval_fun=eval_fun)
        )
        board.undo_move(player=player)
    return best_value


def negamax_ab_moves(board: Board, depth: int, eval_fun: callable) -> Dict[Move, float]:
    print(f'D{depth} search')
    moves = board.get_valid_moves(player=1)
    assert len(moves) > 0, 'no possible moves!'

    alpha = -inf
    beta = inf
    best_move = Move.UP
    best_value = -inf
    for move in moves:
        print(f'== Evaluate {move} for alpha = {alpha} ==')
        board.perform_move(move, player=1)
        value = -negamax_ab(
            board,
            depth=depth - 1,
            player=-1,
            alpha=-beta,
            beta=-alpha,
            eval_fun=eval_fun
        )
        print(f'Got value {value}')
        board.undo_move(player=1)
        if value > best_value:
            best_move = move
            best_value = value
        alpha = max(alpha, value)

    return dict([(best_move, best_value)])


def negamax_ab(board: Board, depth: int, player: int, alpha: float, beta: float, eval_fun: callable) -> float:
    """
    Negamax with alpha-beta pruning
    :param board: The game state
    :param depth: Remaining depth to search
    :param player: Current player
    :param alpha: Guaranteed lower bound
    :param beta: Guaranteed upper bound
    :param eval_fun: Leaf evaluation function
    :return: Game position score
    """
    indent = ' ' * (16 - depth)
    # print(f'{indent}D{depth:02d} P{player:2d}: entering')
    if depth == 0:
        s = eval_fun(board, player=player)
        # print(board)
        # print(f'{indent}D{0:02d} P{player:2d}: leaf node score = {s}')
        return s

    moves = board.get_valid_moves(player=player)
    if len(moves) == 0:  # current player is stuck
        return -inf  # TODO compute game score, as we may still have won if the other player died first

    best_value = -inf
    for move in moves:
        board.perform_move(move, player=player)
        value = -negamax_ab(board, depth=depth - 1, player=-player, alpha=-beta, beta=-alpha, eval_fun=eval_fun)
        board.undo_move(player=player)
        # print(f'{indent}D{depth:02d} P{player:2d}: got {value} for {move}')
        best_value = max(best_value, value)
        alpha = max(alpha, best_value)
        if alpha >= beta:
            # print(f'{indent}D{depth:02d} P{player:2d}: prune with value {best_value} because alpha >= beta ({alpha} >= {beta})')
            return best_value
    return best_value
