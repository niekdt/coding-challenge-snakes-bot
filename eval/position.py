from snakes.bots.niekdt.board import Board


def evaluate(board: Board, player: int) -> float:
    if player == 1:
        return board.DELTA_TERRITORY[board.player1_pos][board.player2_pos]
    else:
        return board.DELTA_TERRITORY[board.player2_pos][board.player1_pos]
