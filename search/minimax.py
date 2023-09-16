from copy import deepcopy
from math import inf

from ..board import Board


def minimax(board: Board, depth: int, maximize: bool, eval_fun: callable) -> float:
    player = 2 - maximize
    print(f'\t= D{depth} for player {player} =')
    if depth == 0:
        print('\tReached max depth')
        return eval_fun(board, player=player)

    moves = board.get_valid_moves(player=player)

    if len(moves) == 0:  # game state reached a dead-end
        print('\tReached dead end: no moves')
        return -inf  # TODO compute game score, as we may still have won if the other player died first

    if maximize is True:
        value = -inf
        for m in moves:
            new_board = deepcopy(board)
            new_board.perform_move(m, player=player)
            value = max(value, minimax(new_board, depth=depth - 1, maximize=False, eval_fun=eval_fun))
        return value
    else:
        value = inf
        for m in moves:
            new_board = deepcopy(board)
            new_board.perform_move(m, player=player)
            value = min(value, minimax(new_board, depth=depth - 1, maximize=True, eval_fun=eval_fun))
        return value
