import time
from typing import List, Tuple

import numpy as np

from ..board import Board
from ..eval import death
from ..search.choose import best_move, get_best_moves_count
from ..search.pvs import pvs_moves
from ....bot import Bot
from ....constants import Move
from ....snake import Snake


class NegamaxAbBot(Bot):
    def __init__(
        self, id: int,
        grid_size: Tuple[int, int],
        depth: int = 3,
        eval_fun: callable = death.evaluate
    ) -> None:
        super().__init__(id, grid_size)
        assert depth >= 1
        self.depth: int = depth
        self.eval_fun: callable = eval_fun
        self.board = Board(width=self.grid_size[0], height=self.grid_size[1])
        self.move_history = dict()

    @property
    def name(self) -> str:
        return 'Snek'

    @property
    def contributor(self) -> str:
        return 'niekdt'

    def determine_next_move(self, snake: Snake, other_snakes: List[Snake], candies: List[np.array]) -> Move:
        start = time.time()
        self.board.set_state(snake1=snake, snake2=other_snakes[0], candies=candies)
        if __debug__:
            print('=' * 80)
            print(snake)
            print(other_snakes[0])
            print(candies)
            print('Initial game state:', end='')
            print(self.board)

        move_values = pvs_moves(
            self.board,
            depth=self.depth,
            eval_fun=self.eval_fun,
            move_history=self.move_history
        )

        if __debug__:
            print('Root move evaluations:')
            print(move_values)
            if get_best_moves_count(move_values) == 3:
                print('Indecisive!!')
        move = best_move(move_values)

        if __debug__:
            print(f'== Decided on {move} in {(time.time() - start) * 1000:.2f} ms ==')
        return move
