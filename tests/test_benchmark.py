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
    gc.collect()
    gc.disable()


@pytest.fixture(scope='session')
def boards() -> List[Board]:
    paths = find_positions('forced-win') + \
            find_positions('forced-loss') + \
            find_positions('best-choice') + \
            find_positions('neutral')

    def load_board(file) -> List[Board]:
        aboard = annotation.from_png(file)
        return [b.board for b in aboard.orientations()]

    return list(itertools.chain.from_iterable([load_board(file) for file in paths]))


# time to beat: 9.45s
@pytest.mark.parametrize('seed', [1] * 6)
@pytest.mark.parametrize('repeat', [10])
def test_search(seed, repeat, boards):
    random.seed(seed)
    for board in boards:
        for i in range(repeat):
            pvs_moves(board.copy(), depth=1, eval_fun=best.evaluate, move_history=dict())
