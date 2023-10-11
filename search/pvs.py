from math import inf
from typing import Dict

from snakes.bots.niekdt.board import Board, MOVES, FIRST_MOVE_ORDER, BoardMove, BOARD_MOVE_UP, BOARD_MOVE_LEFT, \
    BOARD_MOVE_RIGHT, BOARD_MOVE_DOWN

MAX_DEPTH = 64


def pvs_moves(
        board: Board,
        depth: int,
        eval_fun: callable,
        move_history: Dict
) -> Dict[BoardMove, float]:
    player1_last_move = board.MOVE_FROM_TRANS[board.player1_prev_pos][board.player1_pos]
    player2_last_move = board.MOVE_FROM_TRANS[board.player2_prev_pos][board.player2_pos]

    # suicide
    if board.player1_length > 2 * board.player2_length:
        return {board.MOVE_FROM_TRANS[board.player1_pos][board.player1_prev_pos]: inf}

    board_hash = hash(board)
    move_order = move_history.get(board_hash, FIRST_MOVE_ORDER[player1_last_move])
    moves = list(board.iterate_valid_moves(player=1, order=move_order))

    alpha = -inf
    beta = inf
    best_move = BOARD_MOVE_UP
    best_value = -inf
    score_history = dict()

    for move in moves:
        if best_value == -inf and move == moves[-1]:
            if __debug__:
                print('Skipping last root move evaluation because all other moves sucked')
            best_move = move
            # best_value = 0
            break

        if __debug__:
            print(f'== Evaluate {move} with alpha={alpha} ==')
        board.perform_move(move, player=1)
        value = -pvs(
            board,
            my_move=player2_last_move,
            opponent_move=move,
            depth_left=depth - 1,
            depth=1,
            player=-1,
            alpha=-beta,
            beta=-alpha,
            eval_fun=eval_fun,
            move_history=move_history,
            score_history=score_history
        )
        if __debug__:
            print(f'\tGot score {value}')
        board.undo_move(player=1)
        if value > best_value:
            best_move = move
            best_value = value
        alpha = max(alpha, value)

    return {best_move: best_value}


def pvs(
        board: Board,
        my_move: BoardMove,
        opponent_move: BoardMove,
        depth_left: int,
        depth: int,
        player: int,
        alpha: float,
        beta: float,
        eval_fun: callable,
        move_history: Dict,
        score_history: Dict
) -> float:
    if depth_left == 0 or (depth_left <= 2 and not is_quiet_node(board)):
        return qsearch(
            board,
            my_move=my_move,
            opponent_move=opponent_move,
            depth_left=16,
            depth=depth,
            player=player,
            alpha=alpha,
            beta=beta,
            eval_fun=eval_fun,
            move_history=move_history,
            score_history=score_history
        )

    # what if we suicide?
    if player == 1:
        if board.player1_length > 2 * board.player2_length:
            return 2 << 29
    else:
        if board.player2_length > 2 * board.player1_length:
            return 2 << 29

    if not board.can_move(player):
        return -2 << 30

    board_hash = hash(board)
    move_order = move_history.get(board_hash, FIRST_MOVE_ORDER[my_move])
    moves = board.iterate_valid_moves(player=player, order=move_order)

    move_scores = {
        BOARD_MOVE_LEFT: -inf,
        BOARD_MOVE_RIGHT: -inf,
        BOARD_MOVE_UP: -inf,
        BOARD_MOVE_DOWN: -inf
    }

    # do first move
    move = next(moves, BOARD_MOVE_UP)
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
        move_history=move_history,
        score_history=score_history
    )
    board.undo_move(player=player)
    move_scores[move] = value
    alpha = max(alpha, value)
    if alpha >= beta:
        move_history[board_hash] = sorted(MOVES, key=lambda m: move_scores[m], reverse=True)
        return alpha

    # do remaining moves
    for move in moves:
        # search with zero window
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
            move_history=move_history,
            score_history=score_history
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
                move_history=move_history,
                score_history=score_history
            )
        board.undo_move(player=player)
        move_scores[move] = value
        alpha = max(alpha, value)
        if alpha >= beta:
            break

    move_history[board_hash] = sorted(MOVES, key=lambda m: move_scores[m], reverse=True)
    return alpha


def qsearch(
        board: Board,
        my_move: BoardMove,
        opponent_move: BoardMove,
        depth_left: int,
        depth: int,
        player: int,
        alpha: float,
        beta: float,
        eval_fun: callable,
        move_history: Dict,
        score_history: Dict
) -> float:
    if not board.can_move(player):
        return -2 << 30  # do not use inf as it can lead to pruning issues (not sure why)
    # print(f'Q-search for D{depth_left} into depth {depth}')
    assert depth < MAX_DEPTH
    if depth_left == 0 or is_quiet_node(board) or depth >= MAX_DEPTH:
        return eval_fun(board, player=player)

    moves = board.iterate_valid_moves(player=player, order=FIRST_MOVE_ORDER[my_move])

    for move in moves:
        board.perform_move(move, player=player)
        value = -qsearch(
            board,
            my_move=opponent_move,
            opponent_move=move,
            depth_left=depth_left - 1,
            depth=depth + 1,
            player=-player,
            alpha=-beta,
            beta=-alpha,
            eval_fun=eval_fun,
            move_history=move_history,
            score_history=score_history
        )
        board.undo_move(player=player)
        alpha = max(alpha, value)
        if alpha >= beta:
            break

    return alpha


def is_quiet_node(board: Board) -> bool:
    return board.DISTANCE[board.player1_pos][board.player2_pos] > 3 and \
        board.count_moves(player=1) > 1 and \
        board.count_moves(player=-1) > 1 and \
        board.count_player_move_partitions(player=1) <= 1 and \
        board.count_player_move_partitions(player=-1) <= 1

