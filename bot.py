from typing import List

import numpy as np

from .bots.negamax import NegamaxBot
from .bots.negamax_ab import NegamaxAbBot
from .eval import best
from ...bot import Bot
from ...constants import Move
from ...snake import Snake


class Snek(Bot):
    def __init__(self, id: int, grid_size: tuple[int, int]) -> None:
        super().__init__(id, grid_size)
        self.bot = NegamaxAbBot(id=id, grid_size=grid_size, depth=7, eval_fun=best.evaluate)

    @property
    def name(self):
        return 'Snek'

    @property
    def contributor(self):
        return 'niekdt'

    def determine_next_move(self, snake: Snake, other_snakes: List[Snake], candies: List[np.array]) -> Move:
        return self.bot.determine_next_move(snake, other_snakes, candies)
