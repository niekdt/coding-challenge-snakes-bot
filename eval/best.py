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
    score = 1000 * (board.player1_length - board.player2_length) ** 3
    # candy dist
    score += 10 * sum([p2_dist[candy_pos] - p1_dist[candy_pos] for candy_pos in board.candies])

    # candy mode
    if board.player1_length < 10 or board.player2_length < 10:
        return player * score

    # free space lower bound
    lb = 32

    if player == 1:
        delta_space, fs0, fs1 = board.count_free_space_bfs_delta(board.get_empty_mask(), pos1=p1_pos, pos2=p2_pos)
    else:
        delta_space, fs1, fs0 = board.count_free_space_bfs_delta(board.get_empty_mask(), pos1=p2_pos, pos2=p1_pos)
        delta_space *= -1

    score += int(10000 * log(fs0 / fs1))

    # distance to center
    score += 5 * (p2_dist[board.center] - p1_dist[board.center])

    return player * score
