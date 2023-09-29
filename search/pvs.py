from math import inf
from typing import Dict, Tuple

from snakes.bots.niekdt.board import Board, MOVES, FIRST_MOVE_ORDER, BoardMove

MAX_DEPTH = 32


def pvs_moves(
        board: Board,
        depth: int,
        eval_fun: callable,
        move_history: Dict,
        root_moves: Tuple[BoardMove] = MOVES
) -> Dict[BoardMove, float]:
    # suicide
    if board.player1_length > 2 * board.player2_length:
        raise Exception('ayy lmao')

    board_hash = hash(board)
    move_order = move_history.get(board_hash, root_moves)
    moves = board.get_valid_moves_ordered(player=1, order=move_order)

    alpha = -inf
    beta = inf
    best_move = BoardMove.UP
    score_history = dict()

    best_value = -inf
    for move in moves:
        if best_value == -inf and move == moves[-1]:
            if __debug__:
                print('Skipping last root move evaluation because all other moves sucked')
            # best_move = move
            # best_value = 0
            # break

        if __debug__:
            print(f'== Evaluate {move} with alpha={alpha} ==')
        board.perform_move(move, player=1)
        value = -pvs(
            board,
            my_move=BoardMove.LEFT,
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
        BoardMove.LEFT: -inf,
        BoardMove.RIGHT: -inf,
        BoardMove.UP: -inf,
        BoardMove.DOWN: -inf
    }

    # do first move
    move = next(moves, BoardMove.UP)
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
    return board.distance(board.player1_pos, board.player2_pos) > 2 and \
        board.count_moves(player=1) > 1 and \
        board.count_moves(player=-1) > 1 and \
        board.count_player_move_partitions(player=1) <= 1 and \
        board.count_player_move_partitions(player=-1) <= 1

