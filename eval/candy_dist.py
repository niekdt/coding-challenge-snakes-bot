from snakes.bots.niekdt.board import distance


def evaluate(board, player) -> float:
    if board.has_candy():
        d1 = min([distance(board.player1_pos, candy_pos) for candy_pos in board.get_candies()])
        d2 = min([distance(board.player2_pos, candy_pos) for candy_pos in board.get_candies()])

        return player * (d1 - d2) / 16
    else:
        return 0
