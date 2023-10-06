import pathlib
from os import path

import numpy as np
import pytest

from snakes.bots.niekdt.board import Board, BoardMove
from snakes.bots.niekdt.eval import annotation
from snakes.bots.niekdt.eval.annotation import AnnotatedBoard
from snakes.snake import Snake


def test_io():
    board = Board(16, 16)
    snake1 = Snake(id=1, positions=np.array([[8, 6], [9, 6], [10, 6], [10, 7], [10, 8], [10, 9], [11, 9], [12, 9], [13, 9], [13, 10], [12, 10], [11, 10], [11, 11], [11, 12], [10, 12], [9, 12], [8, 12], [8, 11], [7, 11], [6, 11], [5, 11], [5, 10], [5, 9], [5, 8], [5, 7], [4, 7], [4, 6], [5, 6]]))
    snake2 = Snake(id=0, positions=np.array([[8, 0], [7, 0], [7, 1], [8, 1], [8, 2], [8, 3], [7, 3], [7, 4], [7, 5], [6, 5], [6, 4], [6, 3], [5, 3], [5, 2], [5, 1], [4, 1], [3, 1], [3, 0], [2, 0], [1, 0], [0, 0], [0, 1], [0, 2], [1, 2], [1, 1], [2, 1], [2, 2], [3, 2]]))
    board.set_state_from_game(snake1=snake1, snake2=snake2, candies=[])

    aboard = AnnotatedBoard(board)
    aboard.save('test')
    assert path.exists('test.png')
    aboard2 = annotation.from_png('test.png')
    assert aboard == aboard2

    # with candies
    board.set_state_from_game(snake1=snake1, snake2=snake2, candies=[np.array([2, 13]), np.array([11, 4]), np.array([2, 6])])
    aboard.save('test-candies')
    aboard2c = annotation.from_png('test-candies.png')
    assert aboard == aboard2c

    # with single move
    aboard.moves = [BoardMove.DOWN]
    aboard.save('test-candies-move')
    aboard2cm = annotation.from_png('test-candies-move.png')
    assert aboard == aboard2cm

    # with two moves
    aboard.moves = [BoardMove.DOWN, BoardMove.UP]
    aboard.save('test-candies-moves2')
    aboard2cm2 = annotation.from_png('test-candies-moves2.png')
    assert aboard == aboard2cm2

    # with three moves
    aboard.moves = [BoardMove.DOWN, BoardMove.UP, BoardMove.LEFT]
    aboard.save('test-candies-moves3')
    aboard2cm3 = annotation.from_png('test-candies-moves3.png')
    assert aboard == aboard2cm3


@pytest.mark.parametrize('folder', ['forced-win', 'forced-loss', 'best-choice', 'two-choice'])
def test_annotations(folder):
    files = list(pathlib.Path(folder).glob('*.png'))
    for file in files:
        aboard = annotation.from_png(str(file))
        assert aboard.board.shape == (16, 16), f'problem with board image {file}'
