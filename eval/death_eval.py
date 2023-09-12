from math import inf


# Simple evaluation for unavoidable death in the current turn.
# -inf: if the player has no valid move options, i.e., will die this turn
# 0: otherwise
def evaluate(board, maximize) -> float:
    return -inf if len(board.get_valid_moves(player=maximize)) == 0 else 0
