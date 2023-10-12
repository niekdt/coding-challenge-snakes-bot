from functools import lru_cache

from ..board import Board


@lru_cache(maxsize=None)
def evaluate(board: Board, player: int) -> float:
    # length
    score = 1000 * (board.player1_length - board.player2_length) ** 3
    p1_pos = board.player1_pos
    p2_pos = board.player2_pos
    p1_dist = board.DISTANCE[p1_pos]
    p2_dist = board.DISTANCE[p2_pos]

    # candy dist
    score += 10 * sum([p2_dist[candy_pos] - p1_dist[candy_pos] for candy_pos in board.candies])

    # free space lower bound
    lb = 32

    #f1 = min(lb, board.count_free_space_dfs(board.get_empty_mask(), pos=p1_pos, lb=lb, max_dist=6, distance_map=p1_dist))
    #f2 = min(lb, board.count_free_space_dfs(board.get_empty_mask(), pos=p2_pos, lb=lb, max_dist=6, distance_map=p2_dist))
    #f1 = min(lb, board.count_free_space_bfs(board.get_empty_mask(), pos=p1_pos, prev_pos=board.player1_prev_pos, lb=lb, max_dist=10))
    #f2 = min(lb, board.count_free_space_bfs(board.get_empty_mask(), pos=p2_pos, prev_pos=board.player2_prev_pos, lb=lb, max_dist=10))
    if player == 1:
        delta_space, _, _ = board.count_free_space_bfs_delta(board.get_empty_mask(), pos1=p1_pos, pos2=p2_pos, max_dist=32, delta_lb=32)
    else:
        delta_space, _, _ = board.count_free_space_bfs_delta(board.get_empty_mask(), pos1=p2_pos, pos2=p1_pos, max_dist=32, delta_lb=32)
        delta_space *= -1
    if abs(delta_space) < 2:
        delta_space = 0
    score += 10 * delta_space

    # distance to center
    score += 5 * (p2_dist[board.center] - p1_dist[board.center])

    return player * score
