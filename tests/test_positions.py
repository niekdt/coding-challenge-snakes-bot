import glob
from math import inf
from typing import List

import pytest

from snakes.bots import Snek
from snakes.bots.niekdt.eval import annotation, best


def find_all_positions() -> List[str]:
    return glob.glob('**/*.png', recursive=True)


def find_positions(path) -> List[str]:
    return glob.glob(f'{path}/*.png', recursive=True)


@pytest.mark.parametrize('file', ['best-choice\\gap-17.png', 'best-choice\\gap-21.png', 'best-choice\\fill-gap.png'])
# @pytest.mark.parametrize('file', find_all_positions())
@pytest.mark.parametrize('eval_fun', [best.evaluate])
def test_eval(file, eval_fun):
    aboard = annotation.from_png(file)
    board = aboard.board
    moves = list(board.iterate_valid_moves(player=1))

    move_values = dict(zip(moves, [-inf] * 4))
    for m in moves:
        print(f'\n\nPerform move {m}')
        board.perform_move(m, player=1)
        print(board)
        value = eval_fun(aboard.board, player=1)
        print(f'VALUE: {value:,d}')
        move_values[m] = value
        board.undo_move(player=1)
    best_value = max(move_values.values())

    print('-' * 50)
    print('Move scores:')
    print(move_values)
    best_moves = [k for k in move_values.keys() if move_values[k] == best_value]

    assert set(aboard.moves).issubset(best_moves)


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
