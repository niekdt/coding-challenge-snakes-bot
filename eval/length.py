def evaluate(board, player) -> float:
    """Score based on the snake length discrepancy"""
    return player * (board.player1_length - board.player2_length)
