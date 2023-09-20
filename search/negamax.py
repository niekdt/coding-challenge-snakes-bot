from math import inf

from ..board import Board


def negamax(board: Board, depth: int, maximize: bool, eval_fun: callable) -> float:
    player = 2 - maximize
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
            -negamax(board, depth=depth - 1, maximize=not maximize, eval_fun=eval_fun)
        )
        board.undo_move(player=player)
    return best_value


def negamax_ab(board: Board, depth: int, maximize: bool, alpha: float, beta: float, eval_fun: callable) -> float:
    player = 2 - maximize
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
            -negamax_ab(board, depth=depth - 1, maximize=not maximize, alpha=-beta, beta=-alpha, eval_fun=eval_fun)
        )
        board.undo_move(player=player)
        alpha = max(alpha, best_value)
        if alpha >= beta:
            break
    return best_value
