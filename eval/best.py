from math import inf, sqrt

from snakes.bots.niekdt.board import distance


def evaluate(board, player) -> float:
    # death
    if not board.can_move(player):
        return -inf

    # length
    score = 10 * (board.player1_length - board.player2_length)

    # candy dist
    if board.has_candy():
        d1 = sum([sqrt(distance(board.player1_pos, candy_pos)) for candy_pos in board.get_candies()])
        d2 = sum([sqrt(distance(board.player2_pos, candy_pos)) for candy_pos in board.get_candies()])

        score += (d1 - d2) / 16

    return player * score
