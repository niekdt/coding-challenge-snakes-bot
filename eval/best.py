from ..board import distance, Board, count_free_space_dfs


def evaluate(board: Board, player: int) -> float:
    # length
    score = 1000 * (board.player1_length - board.player2_length)

    # candy dist
    if board.has_candy():
        d1 = sum([distance(board.player1_pos, candy_pos) for candy_pos in board.get_candies()])
        d2 = sum([distance(board.player2_pos, candy_pos) for candy_pos in board.get_candies()])

        score += 10 * (d2 - d1)

    # free space lower bound
    lb = 16
    # mask = board.get_empty_mask()
    f1 = min(lb, count_free_space_dfs(board.get_empty_mask(), pos=board.player1_pos, lb=lb, max_dist=6, ref_pos=board.player1_pos))
    f2 = min(lb, count_free_space_dfs(board.get_empty_mask(), pos=board.player2_pos, lb=lb, max_dist=6, ref_pos=board.player2_pos))
    score += 10000 * (f1 - f2)

    # distance to center
    score += 5 * (distance(board.player2_pos, board.center) - distance(board.player1_pos, board.center))

    return player * score
