from copy import deepcopy
from math import inf

from ..board import Board


def negamax(board: Board, depth: int, maximize: bool, eval_fun: callable) -> float:
    player = 2 - maximize
    print(f'\t= D{depth} for P{player} =')
    if depth == 0:
        print(f'\tReached max depth: computing heuristic for P{player}')
        return eval_fun(board, player=player)

    moves = board.get_valid_moves(player=player)

    if len(moves) == 0:  # game state reached a dead-end
        print('\tReached dead end: no moves')
        return -inf  # TODO compute game score, as we may still have won if the other player died first

    best_value = -inf
    for m in moves:
        new_board = deepcopy(board)
        new_board.perform_move(m, player=player)
        value = -negamax(new_board, depth=depth - 1, maximize=not maximize, eval_fun=eval_fun)
        best_value = max(best_value, value)
    return best_value
