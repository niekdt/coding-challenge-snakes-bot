from math import inf

from snakes.constants import Move
from ..board import Board, as_move


def negamax_moves(board: Board, depth: int, eval_fun: callable) -> dict[Move, float]:
    move_vecs = board.get_valid_moves(player=1)

    move_values = dict()
    for i, m in enumerate(move_vecs):
        board.perform_move(m, player=1)
        move = as_move(m)
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
    for m in moves:
        board.perform_move(m, player=player)
        best_value = max(
            best_value,
            -negamax(board, depth=depth - 1, player=-player, eval_fun=eval_fun)
        )
        board.undo_move(player=player)
    return best_value


def negamax_ab_moves(board: Board, depth: int, eval_fun: callable) -> dict[Move, float]:
    move_vecs = board.get_valid_moves(player=1)

    alpha = -inf
    beta = inf
    move_values = dict()
    for i, m in enumerate(move_vecs):
        board.perform_move(m, player=1)
        move = as_move(m)
        value = -negamax_ab(
            board,
            depth=depth - 1,
            player=-1,
            alpha=-beta,
            beta=-alpha,
            eval_fun=eval_fun
        )
        board.undo_move(player=1)
        move_values[move] = value
        alpha = max(alpha, value)

    return move_values


def negamax_ab(board: Board, depth: int, player: int, alpha: float, beta: float, eval_fun: callable) -> float:
    if depth == 0:
        s = eval_fun(board, player=player)
        # print(board)
        # print(f'Score: {s}')
        return s

    moves = board.get_valid_moves(player=player)
    if len(moves) == 0:  # current player is stuck
        return -inf  # TODO compute game score, as we may still have won if the other player died first

    best_value = -inf
    for m in moves:
        board.perform_move(m, player=player)
        best_value = max(
            best_value,
            -negamax_ab(board, depth=depth - 1, player=-player, alpha=-beta, beta=-alpha, eval_fun=eval_fun)
        )
        board.undo_move(player=player)
        alpha = max(alpha, best_value)
        if alpha >= beta:
            break
    return best_value
