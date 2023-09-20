import math
import time
from random import choice
from typing import List
from math import inf

import numpy as np

from ..eval import death
from ..search.choose import best_move
from ....snake import Snake
from ....bot import Bot
from ....constants import Move
from ..board import Board, as_move
from ..search.negamax import negamax_ab, negamax_ab_moves


class NegamaxAbBot(Bot):
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
        self.board = Board(width=self.grid_size[0], height=self.grid_size[1])

    @property
    def name(self) -> str:
        return 'Snek'

    @property
    def contributor(self) -> str:
        return 'niekdt'

    def determine_next_move(self, snake: Snake, other_snakes: List[Snake], candies: List[np.array]) -> Move:
        print('=' * 80)
        start = time.time()
        self.board.set_state(snake1=snake, snake2=other_snakes[0], candies=candies)
        print(snake)
        print(other_snakes[0])
        print(candies)
        print('Initial game state:', end='')
        print(self.board)

        move_values = negamax_ab_moves(self.board, depth=self.depth, eval_fun=self.eval_fun)

        print('Root move evaluations:')
        print(move_values)
        move = best_move(move_values)

        print(f'== Decided on {move} in {(time.time() - start) * 1000:.2f} ms ==')
        return move
