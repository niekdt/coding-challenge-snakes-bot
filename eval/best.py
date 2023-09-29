from ..board import Board


def evaluate(board: Board, player: int) -> float:
    # length
    score = 1000 * (board.player1_length - board.player2_length)

    # candy dist
    if board.candies:
        d1 = sum([board.DISTANCE[board.player1_pos][candy_pos] for candy_pos in board.candies])
        d2 = sum([board.DISTANCE[board.player2_pos][candy_pos] for candy_pos in board.candies])

        score += 10 * (d2 - d1)

    # free space lower bound
    lb = 16

    f1 = min(lb, board.count_free_space_dfs(board.get_empty_mask(), pos=board.player1_pos, lb=lb, max_dist=6, ref_pos=board.player1_pos))
    f2 = min(lb, board.count_free_space_dfs(board.get_empty_mask(), pos=board.player2_pos, lb=lb, max_dist=6, ref_pos=board.player2_pos))
    score += 10000 * (f1 - f2)

    # distance to center
    score += 5 * (board.DISTANCE[board.player2_pos][board.center] - board.DISTANCE[board.player1_pos][board.center])

    return player * score
