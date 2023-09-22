from ..board import distance, Board


def evaluate(board: Board, player: int) -> float:
    # length
    score = 1000 * (board.player1_length - board.player2_length)

    # candy dist
    if board.has_candy():
        d1 = sum([distance(board.player1_pos, candy_pos) for candy_pos in board.get_candies()])
        d2 = sum([distance(board.player2_pos, candy_pos) for candy_pos in board.get_candies()])

        score += 10 * (d2 - d1)

    score += 5 * (distance(board.player2_pos, board.center) - distance(board.player1_pos, board.center))

    return player * score
