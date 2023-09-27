from math import isinf, inf
from random import choice
from typing import Dict

from snakes.bots.niekdt.board import BoardMove


def best_move(move_values: Dict[BoardMove, float], tolerance: float = .001) -> BoardMove:
    """Select the highest scored move. Breaks ties at random."""
    best_value = max(move_values.values())
    if isinf(best_value):
        if best_value > 0:
            if __debug__:
                print('Choosing first winning move')
            return next(m for m, v in move_values.items() if v == inf)
        else:
            if __debug__:
                print('Choosing forced losing move')
            return list(move_values.keys())[0]
    else:
        best_moves = [m for m, v in move_values.items() if abs(v - best_value) < tolerance]
        if len(best_moves) > 1:
            if __debug__:
                print(f'Choosing randomly between {len(best_moves)} moves with same score.')
            return choice(best_moves)
        else:
            return best_moves[0]


def single_best_move(move_values: Dict[BoardMove, float], tolerance: float = .001) -> BoardMove:
    """Returns the highest scored move. Throws an error when there is a score tie."""
    best_value = max(move_values.values())
    if isinf(best_value):
        moves = [m for m, v in move_values.items() if v == inf]
    else:
        moves = [m for m, v in move_values.items() if abs(v - best_value) < tolerance]

    assert len(moves) == 1, 'multiple moves with same score'
    return moves[0]


def has_single_best_move(move_values: Dict[BoardMove, float], tolerance: float = .001) -> bool:
    return get_best_moves_count(move_values, tolerance) == 1


def get_best_moves_count(move_values: Dict[BoardMove, float], tolerance: float = .001) -> int:
    best_value = max(move_values.values())
    if isinf(best_value):
        moves = [m for m, v in move_values.items() if v == inf]
    else:
        moves = [m for m, v in move_values.items() if abs(v - best_value) < tolerance]

    return len(moves)
