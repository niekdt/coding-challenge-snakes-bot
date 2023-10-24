import gc
import glob
import itertools
import random
from typing import List

import pytest

from snakes.bots.niekdt.board import Board
from snakes.bots.niekdt.eval import best, annotation
from snakes.bots.niekdt.search.pvs import pvs_moves


def find_positions(path) -> List[str]:
    return glob.glob(f'{path}/*.png', recursive=True)


@pytest.fixture(autouse=True)
def cleanup():
    # warm-up
    best.evaluate.cache_clear()
    gc.collect()
    gc.disable()


@pytest.fixture(scope='session')
def boards(request) -> List[Board]:
    paths = find_positions(request.param)

    def load_board(file) -> List[Board]:
        aboard = annotation.from_png(file)
        return [b.board for b in aboard.orientations()]

    return list(itertools.chain.from_iterable([load_board(file) for file in paths]))


@pytest.mark.parametrize('boards', ['forced-win', 'forced-loss'], indirect=True)
def test_search(boards):
    random.seed(1)
    for board in boards:
        best.evaluate.cache_clear()
        pvs_moves(board.copy(), depth=1, eval_fun=best.evaluate, move_history=dict())
