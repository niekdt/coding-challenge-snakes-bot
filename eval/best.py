from math import inf

from ..board import distance, Board


def evaluate(board: Board, player: int) -> float:
    assert player in (-1, 1)
    # death
    if not board.can_move(player):
        return -inf

    # length
    score = 1000 * (board.player1_length - board.player2_length)

    # candy dist
    if board.has_candy():
        d1 = min([distance(board.player1_pos, candy_pos) for candy_pos in board.get_candies()])
        d2 = min([distance(board.player2_pos, candy_pos) for candy_pos in board.get_candies()])

        score += (d2 - d1)

    # score += 10 * (distance(board.player2_pos, board.center) - distance(board.player1_pos, board.center))
    score += distance(board.player1_pos, board.player2_pos)

    return player * score
