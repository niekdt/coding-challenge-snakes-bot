def evaluate(board, player) -> float:
    score = 10 * (board.player1_length - board.player2_length)
    if board.has_candy():
        d1 = sum([board.distance(board.player1_pos, candy_pos) for candy_pos in board.get_candies()])
        d2 = sum([board.distance(board.player2_pos, candy_pos) for candy_pos in board.get_candies()])

        score += (d2 - d1) / board.width

    return player * score
