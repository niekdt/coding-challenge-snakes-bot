from math import isinf, inf
from random import choice

from snakes.constants import Move


def best_move(move_values: dict[Move, float], tolerance: float = .001) -> Move:
    """Select the highest scored move. Breaks ties at random."""
    best_value = max(move_values.values())
    if isinf(best_value):
        print('Choosing first winning move')
        return next(m for m, v in move_values.items() if v == inf)
    else:
        best_moves = [m for m, v in move_values.items() if abs(v - best_value) < tolerance]
        if len(best_moves) > 1:
            print(f'Choosing randomly between {len(best_moves)} moves with same score.')
            return choice(best_moves)
        else:
            return best_moves[0]


def single_best_move(move_values: dict[Move, float], tolerance: float = .001) -> Move:
    """Returns the highest scored move. Throws an error when there is a score tie."""
    best_value = max(move_values.values())
    if isinf(best_value):
        moves = [m for m, v in move_values.items() if v == inf]
    else:
        moves = [m for m, v in move_values.items() if abs(v - best_value) < tolerance]

    assert len(moves) == 1, 'multiple moves with same score'
    return moves[0]


def assert_single_best_move(move_values: dict[Move, float], tolerance: float = .001) -> None:
    single_best_move(move_values, tolerance)
