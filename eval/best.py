from ..board import Board


def evaluate(board: Board, player: int) -> float:
    # length
    score = 1000 * (board.player1_length - board.player2_length)
    p1_pos = board.player1_pos
    p2_pos = board.player2_pos
    p1_dist = board.DISTANCE[p1_pos]
    p2_dist = board.DISTANCE[p2_pos]

    # candy dist
    if board.candies:
        d1 = sum([p1_dist[candy_pos] for candy_pos in board.candies])
        d2 = sum([p2_dist[candy_pos] for candy_pos in board.candies])

        score += 10 * (d2 - d1)

    # free space lower bound
    lb = 16

    f1 = min(lb, board.count_free_space_dfs(board.get_empty_mask(), pos=p1_pos, lb=lb, max_dist=6, distance_map=p1_dist))
    f2 = min(lb, board.count_free_space_dfs(board.get_empty_mask(), pos=p2_pos, lb=lb, max_dist=6, distance_map=p2_dist))
    # f1 = min(lb, board.count_free_space_bfs(board.get_empty_mask(), pos=p1_pos, lb=lb, max_dist=6))
    # f2 = min(lb, board.count_free_space_bfs(board.get_empty_mask(), pos=p2_pos, lb=lb, max_dist=6))
    score += 10000 * (f1 - f2)

    # distance to center
    score += 5 * (p2_dist[board.center] - p1_dist[board.center])

    return player * score
