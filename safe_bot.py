from typing import List, Tuple

import numpy as np

from .bots.negamax import NegamaxBot
from .eval import best
from ...bot import Bot
from ...constants import Move
from ...snake import Snake


class SafeSnek(Bot):
    def __init__(self, id: int, grid_size: Tuple[int, int]) -> None:
        super().__init__(id, grid_size)
        best.evaluate.cache_clear()
        self.bot = NegamaxBot(id=id, grid_size=grid_size, depth=8, eval_fun=best.evaluate)

    @property
    def name(self):
        return 'SafeSnek'

    @property
    def contributor(self):
        return 'niekdt'

    def determine_next_move(self, snake: Snake, other_snakes: List[Snake], candies: List[np.array]) -> Move:
        return self.bot.determine_next_move(snake, other_snakes, candies)
