import numpy as np

from ..bots.minimax import MinimaxBot
from snakes.constants import Move
from snakes.snake import Snake


def test_forced_move():
    bot = MinimaxBot(id=0, grid_size=(2, 2), depth=1)

    move = bot.determine_next_move(
        snake=Snake(id=0, positions=np.array([[0, 1]])),
        other_snakes=[Snake(id=1, positions=np.array([[1, 1]]))],
        candies=[]
    )
    assert move == Move.DOWN  # only option
