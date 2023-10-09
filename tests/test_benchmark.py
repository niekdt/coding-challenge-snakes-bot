import gc
import random

import numpy as np
import pytest

from snakes.bots import Slifer, Snek
from snakes.bots.niekdt.eval import best
from snakes.game import Game
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
@pytest.mark.parametrize('max_turns', [500])  # time to beat: 12.9s
def test_play_deep_game(grid, seed, max_turns):
    random.seed(seed)
    grid_size = (grid, grid)
    snake1 = Snake(id=0, positions=np.array([[0, 0], [0, 1]]))
    snake2 = Snake(id=1, positions=np.array([[2, 2], [2, 1]]))
    game = Game(grid_size=grid_size, agents={0: Snek, 1: Slifer}, snakes=[snake1, snake2])

    while not game.finished() and game.turns < max_turns:
        game.update()

    assert game.finished()
    assert game.scores[0] > game.scores[1]
    assert game.scores[0] == 64
    assert game.scores[1] == 20
