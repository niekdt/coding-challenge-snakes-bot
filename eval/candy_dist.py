from snakes.bots.niekdt.board import distance


def evaluate(board, player) -> float:
    if board.has_candy():
        pos = board.get_player_pos(player)
        other_pos = board.get_player_pos(3 - player)
        return (min([distance(pos, candy_pos) for candy_pos in board.get_candies()]) -
                min([distance(other_pos, candy_pos) for candy_pos in board.get_candies()])) / board.width
    else:
        return 0
