import numpy as np

from ..bots.minimax import Minimax
from snakes.constants import Move
from snakes.snake import Snake


def test_forced_move():
    bot = Minimax(id=0, grid_size=(2, 2))
    snakes = [
        Snake(id=0, positions=np.array([[0, 1]])),
        Snake(id=1, positions=np.array([[1, 1]]))
    ]
    candies = []

    move = bot.determine_next_move(snakes, candies, 1)
    assert move == Move.UP  # only option
