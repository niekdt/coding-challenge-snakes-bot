import time
from copy import deepcopy
from typing import List

import numpy as np

from ..eval import death
from ....snake import Snake
from ....bot import Bot
from ....constants import Move
from ..board import Board, as_move
from ..search.minimax import minimax


class MinimaxBot(Bot):
    def __init__(
            self, id: int,
            grid_size: tuple[int, int],
            depth: int = 3,
            eval_fun: callable = death.evaluate
    ) -> None:
        super().__init__(id, grid_size)
        assert depth >= 1
        self.depth: int = depth
        self.eval_fun: callable = eval_fun

    @property
    def name(self) -> str:
        return 'Sneek-minimax'

    @property
    def contributor(self) -> str:
        return 'niekdt'

    def determine_next_move(self, snake: Snake, other_snakes: List[Snake], candies: List[np.array]) -> Move:
        start = time.time()
        board = Board(width=self.grid_size[0], height=self.grid_size[1])
        board.set_state(snake1=snake, snake2=other_snakes[0], candies=candies)
        print('\n== Determine next move using minimax search ==')
        print('Initial game state:', end='')
        print(board)

        moves = board.get_valid_moves(player=1)

        move_values = [0.0, ] * len(moves)
        for i, m in enumerate(moves):
            new_board = deepcopy(board)
            new_board.perform_move(m, player=1)
            move_values[i] = minimax(new_board, depth=self.depth, maximize=True, eval_fun=self.eval_fun)
            print(f'\t Root {as_move(m)} yielded score {move_values[i]}')

        # select best move
        best_value = max(move_values)
        best_move = moves[move_values.index(best_value)]

        end = time.time()
        m = as_move(best_move)
        print(f'== Decided on {m} in {(end - start) * 1000:.2f} ms ==')
        return m
