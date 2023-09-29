def evaluate(board, player) -> float:
    score = 10 * (board.player1_length - board.player2_length)
    if board.candies:
        d1 = sum([board.DISTANCE[board.player1_pos][candy_pos] for candy_pos in board.candies])
        d2 = sum([board.DISTANCE[board.player2_pos][candy_pos] for candy_pos in board.candies])

        score += (d2 - d1) / board.width

    return player * score
