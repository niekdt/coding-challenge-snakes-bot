from os import path
from typing import List

import numpy as np
from numpy import ndarray

from ..board import Board, BoardMove, PosIdx


class AnnotatedBoard:
    def __init__(self, board: Board) -> None:
        self.board = board
        self.moves: List[BoardMove] = []
        self.eval = 0

    def show(self) -> None:
        img = self.as_image()
        img.show()

    def save(self, name: str) -> None:
        img = self.as_image()
        file_name = f'{name}.png'
        img.save(file_name)
        file_path = path.abspath(file_name)
        print(f'Saved board to image file "{file_path}"')

    def as_image(self):
        from PIL import Image
        color_grid = np.full((self.board.width + 2, self.board.height + 2, 3), fill_value=0, dtype=np.uint8)
        r, g, b = 0, 1, 2

        for i, pos in enumerate(reversed(self.board.get_player_positions(player=1))):
            x, y = self.board.from_index(pos)
            if i == 0:
                color_grid[(x, y, r)] = 255
            else:
                color_grid[(x, y, r)] = 180 - i * (150 / self.board.player1_length)

        for i, pos in enumerate(reversed(self.board.get_player_positions(player=-1))):
            x, y = self.board.from_index(pos)
            if i == 0:
                color_grid[(x, y, b)] = 255
            else:
                color_grid[(x, y, b)] = 180 - i * (150 / self.board.player2_length)

        for pos in self.board.candies:
            x, y = self.board.from_index(pos)
            color_grid[(x, y, g)] = 191

        for m in self.moves:
            x, y = self.board.from_index(self.board.player1_pos + self.board.MOVE_POS_OFFSET[m])
            color_grid[(x, y, g)] += 64

        return Image.fromarray(np.rot90(color_grid[1:-1, 1:-1, :]))

    def __eq__(self, other) -> bool:
        return self.board == other.board and set(self.moves) == set(other.moves) and self.eval == other.eval

    def __str__(self) -> str:
        return str(self.board)

    def __repr__(self) -> str:
        return repr(self.board) + 'm[' + ','.join(map(repr, self.moves)) + ']'


def from_image(img) -> AnnotatedBoard:
    color_grid = np.pad(np.rot90(np.asarray(img), axes=(1, 0)), pad_width=((1,), (1,), (0,)), constant_values=255)
    assert color_grid.ndim == 3
    assert color_grid.shape[2] == 3
    board = Board(color_grid.shape[0] - 2, color_grid.shape[1] - 2)

    # find player positions
    pos1_idc = _gather_player_positions(board, dim=0, color_grid=color_grid)
    pos2_idc = _gather_player_positions(board, dim=2, color_grid=color_grid)

    # find candies
    x_list, y_list = np.where(color_grid[1:-1, 1:-1, 1] >= 191)
    candy_idc = [board.from_xy(x + 1, y + 1) for x, y in zip(x_list, y_list)]
    assert len(candy_idc) <= 3

    # find candidate moves
    x_list, y_list = np.where(np.logical_or(
        np.logical_and(color_grid[1:-1, 1:-1, 1] > 0, color_grid[1:-1, 1:-1, 1] < 180),
        color_grid[1:-1, 1:-1, 1] == 255
    ))
    move_idc = [board.from_xy(x + 1, y + 1) for x, y in zip(x_list, y_list)]
    assert len(move_idc) <= 3
    moves = [board.MOVE_FROM_TRANS[pos1_idc[-1]][pos] for pos in move_idc]

    board.set_state(pos1_idc, pos2_idc, candy_idc)

    aboard = AnnotatedBoard(board)
    aboard.moves = moves
    return aboard


def _gather_player_positions(board: Board, dim, color_grid: ndarray) -> List[PosIdx]:
    x, y = np.where(color_grid[1:-1, 1:-1, dim] == 255)
    cell_pos = (int(x) + 1, int(y) + 1, dim)
    cell_value = 255
    player_idc = []
    while cell_value > 0:
        x = cell_pos[0]
        y = cell_pos[1]
        player_idc.insert(0, board.from_xy(x, y))
        # scan for next player cell
        cell_positions = (
            (x - 1, y, dim),
            (x + 1, y, dim),
            (x, y - 1, dim),
            (x, y + 1, dim)
        )

        cell_value = max(int(color_grid[pos]) for pos in cell_positions if color_grid[pos] < cell_value)
        cell_opts = list(filter(lambda pos: color_grid[pos] == cell_value, cell_positions))
        assert cell_value == 0 or len(cell_opts) == 1, 'image grid contain player cells with the same value'
        cell_pos = cell_opts[0]
    return player_idc


def from_png(file: str) -> AnnotatedBoard:
    from PIL import Image
    with Image.open(file) as img:
        return from_image(img)