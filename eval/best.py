from functools import lru_cache
from math import log

from ..board import Board


@lru_cache(maxsize=None)
def evaluate(board: Board, player: int) -> float:
    p1_pos = board.player1_pos
    p2_pos = board.player2_pos
    p1_dist = board.DISTANCE[p1_pos]
    p2_dist = board.DISTANCE[p2_pos]

    # length
    score = 20000 * (board.player1_length - board.player2_length)
    # candy dist
    score += 10 * sum([p2_dist[candy_pos] - p1_dist[candy_pos] for candy_pos in board.candies])

    # candy mode
    if player == 1 and board.player1_length < 12 or player == -1 and board.player2_length < 12:
        return player * score

    if player == 1:
        delta_space, fs0, fs1 = board.count_free_space_bfs_delta(board.get_empty_mask(), pos1=p1_pos, pos2=p2_pos)
    else:
        delta_space, fs1, fs0 = board.count_free_space_bfs_delta(board.get_empty_mask(), pos1=p2_pos, pos2=p1_pos)
        delta_space *= -1

    score += int(10000 * log(fs0 / fs1))

    # positional bonus
    if player == 1:
        score += 100 * board.DELTA_TERRITORY[p1_pos][p2_pos]
    else:
        score -= 100 * board.DELTA_TERRITORY[p2_pos][p1_pos]

    return player * score
