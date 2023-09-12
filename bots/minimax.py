from abc import ABC
from copy import deepcopy

from ..board import Board, as_move
from ..eval.death import evaluate
from ..search.minimax import minimax
from snakes.bot import Bot
from snakes.constants import Move


class Minimax(Bot, ABC):
    def __init__(self, id, grid_size, depth=3, eval_fun=evaluate):
        super().__init__(id, grid_size)
        self.depth = depth
        self.eval_fun = eval_fun

    @property
    def name(self):
        return 'Minimax'

    def determine_next_move(self, snakes, candies, turn) -> Move:
        board = Board(width=self.grid_size[0], height=self.grid_size[1])
        board.set_state(snakes=snakes, candies=candies, turn=turn)

        snake = next(s for s in snakes if s.id == self.id)

        player = 1 if snake == snakes[0] else 2

        moves = board.get_valid_moves(player)

        move_values = [0.0, ] * len(moves)
        for i, m in enumerate(moves):
            new_board = deepcopy(board)
            new_board.perform_move(m, player=player)
            move_values[i] = minimax(new_board, depth=self.depth, maximize=True, eval_fun=self.eval_fun)

        # select best move
        best_value = max(move_values)
        best_move = moves[move_values.index(best_value)]

        # convert to MOVE
        return as_move(best_move)
