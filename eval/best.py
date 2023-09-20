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
        d1 = min([distance(board.player1_pos, candy_pos) for candy_pos in board.get_candies()])
        d2 = min([distance(board.player2_pos, candy_pos) for candy_pos in board.get_candies()])

        score += (d2 - d1) / board.width

    return player * score
