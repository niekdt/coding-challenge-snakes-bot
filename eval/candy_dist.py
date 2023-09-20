from snakes.bots.niekdt.board import distance


def evaluate(board, player) -> float:
    score = 2 * (board.player1_length - board.player2_length)
    if board.has_candy():
        d1 = min([distance(board.player1_pos, candy_pos) for candy_pos in board.get_candies()])
        d2 = min([distance(board.player2_pos, candy_pos) for candy_pos in board.get_candies()])

        score += (d2 - d1) / 16

    return player * score
