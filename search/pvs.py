from math import inf
from typing import Dict

from snakes.bots.niekdt.board import Board, ALL_MOVES, FIRST_MOVE_ORDER
from snakes.bots.niekdt.search.negamax import MOVE_HISTORY
from snakes.constants import Move


def pvs_moves(
    board: Board,
    depth: int,
    eval_fun: callable,
    move_history: Dict = MOVE_HISTORY
) -> Dict[Move, float]:
    # suicide
    if board.player1_length > 2 * board.player2_length:
        raise Exception('ayy lmao')

    board_hash = board.approx_hash()
    move_order = move_history.get(board_hash, ALL_MOVES)
    moves = board.get_valid_moves_ordered(player=1, order=move_order)

    alpha = -inf
    beta = inf
    best_move = Move.UP

    best_value = -inf
    for move in moves:
        if __debug__:
            print(f'== Evaluate {move} for alpha = {alpha} ==')
        board.perform_move(move, player=1)
        value = -pvs(
            board,
            my_move=Move.LEFT,
            opponent_move=move,
            depth_left=depth - 1,
            depth=1,
            player=-1,
            alpha=-beta,
            beta=-alpha,
            eval_fun=eval_fun,
            move_history=move_history
        )
        if __debug__:
            print(f'\tGot value {value}')
        board.undo_move(player=1)
        if value > best_value:
            best_move = move
            best_value = value
        alpha = max(alpha, value)

    return {best_move: best_value}


def pvs(
        board: Board,
        my_move: Move,
        opponent_move: Move,
        depth_left: int,
        depth: int,
        player: int,
        alpha: float,
        beta: float,
        eval_fun: callable,
        move_history: Dict
) -> float:
    if depth_left <= 0 and board.count_moves(player=player) == 1:
        depth_left += 2

    if depth_left == 0 or depth == 32:
        return eval_fun(board, player=player)

    # what if we suicide?
    if player == 1:
        if board.player1_length > 2 * board.player2_length:
            return inf
    else:
        if board.player2_length > 2 * board.player1_length:
            return 999999

    if not board.can_move(player):
        return -999999

    board_hash = board.approx_hash()
    move_order = move_history.get(board_hash, FIRST_MOVE_ORDER[my_move])
    moves = board.iterate_valid_moves(player=player, order=move_order)

    move_scores = {
        Move.LEFT: -inf,
        Move.RIGHT: -inf,
        Move.UP: -inf,
        Move.DOWN: -inf
    }

    # do first move
    move = next(moves, Move.UP)
    board.perform_move(move, player=player)
    value = -pvs(
        board,
        my_move=opponent_move,
        opponent_move=move,
        depth_left=depth_left - 1,
        depth=depth + 1,
        player=-player,
        alpha=-beta,
        beta=-alpha,
        eval_fun=eval_fun,
        move_history=move_history
    )
    board.undo_move(player=player)
    move_scores[move] = value
    alpha = max(alpha, value)
    if alpha >= beta:
        move_history[board_hash] = sorted(ALL_MOVES, key=lambda m: move_scores[m], reverse=True)
        return alpha

    # do remaining moves
    for move in moves:
        board.perform_move(move, player=player)
        value = -pvs(
            board,
            my_move=opponent_move,
            opponent_move=move,
            depth_left=depth_left - 1,
            depth=depth + 1,
            player=-player,
            alpha=-alpha - 1,
            beta=-alpha,
            eval_fun=eval_fun,
            move_history=move_history
        )
        if alpha < value < beta:
            # search again with full [alpha, beta] window
            value = -pvs(
                board,
                my_move=opponent_move,
                opponent_move=move,
                depth_left=depth_left - 1,
                depth=depth + 1,
                player=-player,
                alpha=-beta,
                beta=-alpha,
                eval_fun=eval_fun,
                move_history=move_history
            )
        board.undo_move(player=player)
        move_scores[move] = value
        alpha = max(alpha, value)
        if alpha >= beta:
            break

    move_history[board_hash] = sorted(ALL_MOVES, key=lambda m: move_scores[m], reverse=True)
    return alpha
