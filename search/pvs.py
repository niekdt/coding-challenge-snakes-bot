from math import inf
from typing import Dict

from snakes.bots.niekdt.board import Board, ALL_MOVES
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
            depth=depth - 1,
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

    return dict([(best_move, best_value)])


def pvs(
        board: Board,
        depth: int,
        player: int,
        alpha: float,
        beta: float,
        eval_fun: callable,
        move_history: Dict
) -> float:
    if depth == 0:
        return eval_fun(board, player=player)

    # suicide
    if player == 1:
        alpha = inf if board.player1_length > 2 * board.player2_length else alpha
    else:
        alpha = inf if board.player2_length > 2 * board.player1_length else alpha

    if alpha == inf:
        return alpha
    elif not board.can_move(player):
        return -999999

    board_hash = board.approx_hash()
    move_order = move_history.get(board_hash, ALL_MOVES)
    moves = board.iterate_valid_moves(player=player, order=move_order)

    move_scores = dict({
        Move.LEFT: -inf,
        Move.RIGHT: -inf,
        Move.UP: -inf,
        Move.DOWN: -inf
    })

    # do first move
    move = next(moves, Move.UP)
    board.perform_move(move, player=player)
    value = -pvs(
        board,
        depth=depth - 1,
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
            depth=depth - 1,
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
                depth=depth - 1,
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
