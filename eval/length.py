def evaluate(board, player) -> float:
    """Score based on the snake length discrepancy"""
    return board.player1_length - board.player2_length if player == 1 else board.player2_length - board.player1_length
