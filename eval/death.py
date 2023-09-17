from math import inf


# Simple evaluation for unavoidable death in the current turn.
# -inf: if the player has no valid move options, i.e., will die this turn
# 0: otherwise
def evaluate(board, player) -> float:
    return -inf if board.can_move(player) == 0 else 0
