from math import inf, sqrt

from snakes.bots.niekdt.board import distance


def evaluate(board, player) -> float:
    # death
    if not board.can_move(player):
        return -inf

    # length
    score = board.player1_length - board.player2_length if player == 1 else board.player2_length - board.player1_length
    score *= 10

    # candy dist
    if board.has_candy():
        pos = board.get_player_pos(player)
        other_pos = board.get_player_pos(3 - player)
        d = sum([sqrt(distance(pos, candy_pos)) for candy_pos in board.get_candies()])
        other_d = sum([sqrt(distance(other_pos, candy_pos)) for candy_pos in board.get_candies()])

        score += (other_d - d) / 16

    return score
