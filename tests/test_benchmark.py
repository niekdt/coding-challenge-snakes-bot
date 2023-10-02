import gc
import random

import numpy as np
import pytest

from snakes.bots import bots
from snakes.bots.niekdt.eval import best
from snakes.game import Game, RoundType
from snakes.snake import Snake


@pytest.fixture(autouse=True)
def cleanup():
    # warm-up
    test_play_deep_game(grid=16, seed=1, bot='Snek', max_turns=100)
    best.evaluate.cache_clear()
    gc.collect()
    gc.disable()


@pytest.mark.parametrize('grid', [16])
@pytest.mark.parametrize('seed', [1] * 6)
@pytest.mark.parametrize('bot', ['Snek'])
@pytest.mark.parametrize('max_turns', [100])  # time to beat: 6.45s for BFS, 5.7 for DFS
def test_play_deep_game(grid, seed, bot, max_turns):
    random.seed(seed)
    grid_size = (grid, grid)
    snake1 = Snake(id=0, positions=np.array([[0, 0], [0, 1]]))
    snake2 = Snake(id=1, positions=np.array([[2, 2], [2, 1]]))
    snakes = [snake1, snake2]

    bot_names = [Bot(id=i, grid_size=(1, 1)).name for i, Bot in enumerate(bots)]
    bot_i = int(np.where(np.array(bot_names) == bot)[0])
    agents = {0: bots[bot_i], 1: bots[bot_i]}
    game = Game(grid_size=grid_size, agents=agents, round_type=RoundType.TURNS, snakes=snakes)

    while not game.finished() and game.turns < max_turns:
        game.update()
