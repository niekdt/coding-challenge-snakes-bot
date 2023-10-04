import gc
import random

import numpy as np
import pytest

from snakes.bots import Slifer, Snek
from snakes.bots.niekdt.eval import best
from snakes.game import Game, RoundType
from snakes.snake import Snake


@pytest.fixture(autouse=True)
def cleanup():
    # warm-up
    test_play_deep_game(grid=16, seed=1, max_turns=400)
    best.evaluate.cache_clear()
    gc.collect()
    gc.disable()


@pytest.mark.parametrize('grid', [16])
@pytest.mark.parametrize('seed', [1] * 6)
@pytest.mark.parametrize('max_turns', [500])  # time to beat: 4.2s for BFS, 7.7s for DFS
def test_play_deep_game(grid, seed, max_turns):
    random.seed(seed)
    grid_size = (grid, grid)
    snake1 = Snake(id=0, positions=np.array([[0, 0], [0, 1]]))
    snake2 = Snake(id=1, positions=np.array([[2, 2], [2, 1]]))
    snakes = [snake1, snake2]

    agents = {0: Snek, 1: Slifer}
    game = Game(grid_size=grid_size, agents=agents, round_type=RoundType.TURNS, snakes=snakes)

    while not game.finished() and game.turns < max_turns:
        game.update()
