import glob
from math import inf
from typing import List

import pytest

from snakes.bots import Snek
from snakes.bots.niekdt.eval import annotation


def find_all_positions() -> List[str]:
    return glob.glob('**/*.png', recursive=True)


def find_positions(path) -> List[str]:
    return glob.glob(f'{path}/*.png', recursive=True)


# @pytest.mark.parametrize('file', ['forced-win\\corner-h.png'])
@pytest.mark.parametrize('file', find_all_positions())
@pytest.mark.parametrize('bot', [Snek])
def test_move(file, bot):
    aboard = annotation.from_png(file)
    if len(aboard.moves) == 0 or len(aboard.moves) == 3:
        pytest.skip('position has no defined moves')

    for orientation in aboard.orientations():
        orientation.assert_determine_move(bot=bot)


@pytest.mark.parametrize('file', find_positions('forced-win'))
def test_forced_win_outcome(file):
    aboard = annotation.from_png(file)
    aboard.eval = inf
    assert len(aboard.moves) == 1, 'problem with position: can only have one best move'

    for orientation in aboard.orientations():
        game = orientation.play_game()
        orientation.assert_game_outcome(game)
