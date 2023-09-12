from random import choice
from typing import List, Tuple

import numpy as np

from ...bot import Bot
from ...constants import Move, MOVE_VALUE_TO_DIRECTION
from ...snake import Snake


def is_on_grid(pos: np.array, grid_size: Tuple[int, int]) -> bool:
    """
    Check if a position is still on the grid
    """
    return 0 <= pos[0] < grid_size[0] and 0 <= pos[1] < grid_size[1]


def collides(pos: np.array, snakes: List[Snake]) -> bool:
    """
    Check if a position is occupied by any of the snakes
    """
    for snake in snakes:
        if snake.collides(pos):
            return True
    return False


class SneekBot(Bot):
    """
    Moves randomly, but makes sure it doesn't collide with other snakes
    """

    @property
    def name(self):
        return 'Sneek'

    @property
    def contributor(self):
        return 'niekdt'

    def determine_next_move(self, snake: Snake, other_snakes: List[Snake], candies: List[np.array]) -> Move:
        moves = self._determine_possible_moves(snake, other_snakes[0])
        return self.choose_move(moves)

    def _determine_possible_moves(self, snake, other_snake) -> List[Move]:
        """
        Return a list with all moves that we want to do. Later we'll choose one from this list randomly. This method
        will be used during unit-testing
        """
        # highest priority, a move that is on the grid
        on_grid = [move for move in MOVE_VALUE_TO_DIRECTION
                   if is_on_grid(snake[0] + MOVE_VALUE_TO_DIRECTION[move], self.grid_size)]
        if not on_grid:
            return list(Move)

        # then avoid collisions with other snakes
        collision_free = [move for move in on_grid
                          if is_on_grid(snake[0] + MOVE_VALUE_TO_DIRECTION[move], self.grid_size)
                          and not collides(snake[0] + MOVE_VALUE_TO_DIRECTION[move], [snake, other_snake])]
        if collision_free:
            return collision_free
        else:
            return on_grid

    def choose_move(self, moves: List[Move]) -> Move:
        """
        Randomly pick a move
        """
        return choice(moves)
