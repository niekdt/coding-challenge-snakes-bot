from copy import deepcopy
from math import inf

from board import Board


def minimax(board: Board, depth: int, maximize: bool, eval_fun) -> float:
    if depth == 0:
        return eval_fun(board)

    moves = board.get_valid_moves(player=maximize)

    if len(moves) == 0:
        return eval_fun(board)

    if maximize is True:
        value = -inf
        for m in moves:
            new_board = deepcopy(board)
            new_board.perform_move(m, player=maximize)
            value = max(value, minimax(new_board, depth=depth - 1, maximize=False, eval_fun=eval_fun))
        return value
    else:
        value = inf
        for m in moves:
            new_board = deepcopy(board)
            new_board.perform_move(m, player=maximize)
            value = min(value, minimax(new_board, depth=depth - 1, maximize=True, eval_fun=eval_fun))
        return value
