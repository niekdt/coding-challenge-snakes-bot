from math import inf
from typing import Dict

from ..board import Board, MOVES, BoardMove


def negamax_moves(
        board: Board,
        depth: int,
        eval_fun: callable,
        move_history: Dict
) -> Dict[BoardMove, float]:
    # suicide
    if board.player1_length > 2 * board.player2_length:
        raise Exception('ayy lmao')

    moves = board.get_valid_moves_ordered(player=1)
    move_values = dict()
    for move in moves:
        board.perform_move(move, player=1)
        move_values[move] = -negamax(board, depth=depth - 1, player=-1, eval_fun=eval_fun)
        board.undo_move(player=1)

    return move_values


def negamax(board: Board, depth: int, player: int, eval_fun: callable) -> float:
    if depth == 0:
        return eval_fun(board, player=player)

    # suicide
    if player == 1:
        best_value = inf if board.player1_length > 2 * board.player2_length else -inf
    else:
        best_value = inf if board.player2_length > 2 * board.player1_length else -inf

    if best_value == inf:
        return best_value

    moves = board.get_valid_moves_ordered(player=player)
    for move in moves:
        board.perform_move(move, player=player)
        best_value = max(
            best_value,
            -negamax(board, depth=depth - 1, player=-player, eval_fun=eval_fun)
        )
        board.undo_move(player=player)
    return best_value


def negamax_ab_moves(
        board: Board,
        depth: int,
        eval_fun: callable,
        move_history: Dict
) -> Dict[BoardMove, float]:
    # suicide
    if board.player1_length > 2 * board.player2_length:
        raise Exception('ayy lmao')

    board_hash = board.approx_hash()
    move_order = move_history.get(board_hash, MOVES)
    moves = board.get_valid_moves_ordered(player=1, order=move_order)

    alpha = -inf
    beta = inf
    best_move = BoardMove.UP

    best_value = -inf
    for move in moves:
        if __debug__:
            print(f'== Evaluate {move} for alpha = {alpha} ==')
        board.perform_move(move, player=1)
        value = -negamax_ab(
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


def negamax_ab(
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

    board_hash = board.approx_hash()
    move_order = move_history.get(board_hash, MOVES)
    moves = board.iterate_valid_moves(player=player, order=move_order)

    move_scores = dict({
        BoardMove.LEFT: -inf,
        BoardMove.RIGHT: -inf,
        BoardMove.UP: -inf,
        BoardMove.DOWN: -inf
    })

    for move in moves:
        board.perform_move(move, player=player)
        value = -negamax_ab(
            board,
            depth=depth - 1,
            player=-player,
            alpha=-beta,
            beta=-alpha,
            eval_fun=eval_fun,
            move_history=move_history
        )
        move_scores[move] = value
        board.undo_move(player=player)
        alpha = max(alpha, value)
        if alpha >= beta:
            break

    move_history[board_hash] = sorted(MOVES, key=lambda m: move_scores[m], reverse=True)
    return alpha
