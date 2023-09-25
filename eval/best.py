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
    mask = board.get_empty_mask()
    lb = 16
    f1 = min(count_free_space_dfs(mask.copy(), pos=board.player1_pos, lb=lb), lb)
    f2 = min(count_free_space_dfs(mask, pos=board.player2_pos, lb=lb), lb)
    score += 10000 * (f1 - f2)

    # distance to center
    score += 5 * (distance(board.player2_pos, board.center) - distance(board.player1_pos, board.center))

    return player * score
