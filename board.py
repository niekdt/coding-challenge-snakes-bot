import unittest as ut
from itertools import compress
from typing import Self

import numpy as np
from numpy import ndarray

from game import Snake

MOVE_UP = np.array([0, -1])  # note: the reverse of constants.py
MOVE_DOWN = np.array([0, 1])
MOVE_LEFT = np.array([-1, 0])
MOVE_RIGHT = np.array([1, 0])
ALL_MOVES = [MOVE_LEFT, MOVE_RIGHT, MOVE_UP, MOVE_DOWN]


# TODO generate list permutations for all possible move sets. The get_valid_moves() function can then select one.

class Board:
    def __init__(self, width: int, height: int):
        assert width > 0
        assert height > 0

        self.width = width
        self.height = height
        self.candies = []
        self.grid = np.zeros([width, height])
        self.player1_turn = 1
        self.player2_turn = 1
        self.player1_pos = ()
        self.player2_pos = ()
        self.player1_length = 1
        self.player2_length = 1

    def spawn(self, pos1: tuple, pos2: tuple):
        assert type(pos1) is tuple
        assert type(pos2) is tuple
        assert len(self.player1_pos) == 0, 'players have already spawned'
        assert self.is_valid_pos(pos1), 'invalid spawn pos for P1'
        assert self.is_valid_pos(pos2), 'invalid spawn pos for P2'

        # TODO decide on whether pos should be tuple or ndarray. Currently it is converted to ndarray by perform_move()
        self.player1_pos = pos1
        self.player2_pos = pos2
        self.grid[pos1] = self.player1_turn
        self.grid[pos2] = -self.player2_turn
        pass

    # turn 1: P1 is about to move (P1=1, P2=1)
    # turn 2: P2 is about to move, P1 has moved (P1=2, P2=1)
    # turn 3: P1 is about to move, P2 has moved (P1=2, P2=2)
    # turn 4: P2 is about to move, P1 has moved (P1=3, P2=2)
    # etc.
    def set_state(self, snakes, candies, turn):
        assert len(snakes) == 2
        assert turn >= 1

        # clear grid
        self.grid.fill(0)

        self.player1_turn = turn // 2 + 1
        self.player2_turn = (turn + 1) // 2

        snake1 = snakes[0]
        snake2 = snakes[1]

        assert self.player1_turn >= len(snake1.positions)
        assert self.player2_turn >= len(snake2.positions)

        # snake positions are in reverse order (head is last element)
        for i, pos in enumerate(snake1.positions):
            self.grid[pos[0], pos[1]] = self.player1_turn - len(snake1.positions) + i + 1

        for i, pos in enumerate(snake2.positions):
            self.grid[pos[0], pos[1]] = -self.player2_turn + len(snake2.positions) - i - 1

        self.candies = candies.copy()
        pass

    @property
    def shape(self) -> tuple:
        return self.width, self.height

    @property
    def turn(self) -> int:
        return self.player1_turn + self.player2_turn - 1

    @property
    def size(self) -> int:
        return self.width * self.height

    def get_free_space(self) -> int:
        return np.sum(self.get_empty_mask() == True, axis=None)  # how to get rid of the warning??

    def is_valid_pos(self, pos: tuple) -> bool:
        return 0 <= pos[0] < self.width and 0 <= pos[1] < self.height

    def is_candy_pos(self, pos: tuple) -> bool:
        return pos in self.candies

    # TODO optimize
    def is_empty_pos(self, pos: tuple) -> bool:
        return self.get_empty_mask()[pos[0], pos[1]] == True

    # TODO optimize
    def get_empty_mask(self) -> ndarray:
        return (self.get_player1_mask() == False) & (self.get_player2_mask() == False)

    def get_player1_mask(self) -> ndarray:
        return self.grid > max(0, self.player1_turn - self.player1_length)

    def get_player2_mask(self) -> ndarray:
        return self.grid < min(0, -self.player2_turn + self.player2_length)

    def get_player_mask(self, player: int) -> ndarray:
        if player == 1:
            return self.get_player1_mask()
        else:
            return self.get_player2_mask()

    def get_candy_mask(self) -> ndarray:
        grid_mask = np.full(self.shape, fill_value=False, dtype=bool)
        grid_mask[self.candies] = True
        return grid_mask

    def has_candy(self) -> bool:
        return len(self.candies) > 0

    def inherit(self, board: Self):
        self.player1_pos = board.player1_pos
        self.player2_pos = board.player2_pos
        self.player1_length = board.player1_length
        self.player2_length = board.player2_length
        self.player1_turn = board.player1_turn
        self.player2_turn = board.player2_turn
        self.candies = board.candies
        self.grid = board.grid
        pass

    def get_valid_moves(self, player: int) -> list:
        if player == 1:
            pos = self.player1_pos
        else:
            pos = self.player2_pos

        can_move_left = pos[0] > 0 and self.is_empty_pos(pos + MOVE_LEFT)
        can_move_right = pos[0] < self.width - 1 and self.is_empty_pos(pos + MOVE_RIGHT)
        can_move_up = pos[1] > 0 and self.is_empty_pos(pos + MOVE_UP)
        can_move_down = pos[1] < self.height - 1 and self.is_empty_pos(pos + MOVE_DOWN)

        return list(compress(ALL_MOVES, [can_move_left, can_move_right, can_move_up, can_move_down]))

    # performing a move increments the turn counter and places a new wall
    def perform_move(self, move: ndarray, player: int):
        assert move[0] in (-1, 0, 1) and \
               move[1] in (-1, 0, 1) and \
               move[0] == 0 or move[1] == 0, 'invalid move vector'

        # TODO remove branching
        if player == 1:
            target_pos = self.player1_pos + move
            assert self.player1_turn == self.player2_turn, 'P1 already moved'
            assert self.is_valid_pos(target_pos), 'illegal move by P1: invalid position'
            assert self.is_empty_pos(target_pos), 'illegal move by P1: suicide'
            self.player1_turn += 1
            self.player1_pos += move  # update pos
            if self.is_candy_pos(self.player1_pos):
                self.player1_length += 1
                self.remove_candy(self.player1_pos)
            self.grid[tuple(self.player1_pos)] = self.player1_turn
        else:
            target_pos = self.player2_pos + move
            assert self.player2_turn == self.player1_turn - 1, 'P2 already moved'
            assert self.is_valid_pos(target_pos), 'illegal move by P2: invalid position'
            assert self.is_empty_pos(target_pos), 'illegal move by P2: suicide'
            self.player2_turn += 1
            self.player2_pos = target_pos
            if self.is_candy_pos(self.player2_pos):
                self.player2_length += 1
                self.remove_candy(self.player2_pos)
            self.grid[tuple(self.player2_pos)] = -self.player2_turn
        pass

    def spawn_candy(self, pos: tuple):
        assert not (pos in self.candies)
        self.candies.append(pos)

    def remove_candy(self, pos: tuple):
        self.candies.remove(pos)

    def __eq__(self, other) -> bool:
        return self.player1_turn == other.player1_turn and \
            self.player2_turn == other.player2_turn and \
            np.array_equal(self.grid, other.grid)

    def __str__(self):
        str_grid = np.full(self.shape, fill_value='_', dtype=str)
        str_grid[self.get_player1_mask()] = 'a'
        str_grid[self.get_player2_mask()] = 'b'
        str_grid[self.get_candy_mask()] = '*'
        if len(self.player1_pos) > 0:
            str_grid[tuple(self.player1_pos)] = 'A'
            str_grid[tuple(self.player2_pos)] = 'B'

        str_field = np.pad(str_grid, [(1, 1), (1, 1)], mode='constant')
        str_field[0:, (0, -1)] = '-'
        str_field[(0, -1), 0:] = '|'
        str_field[(0, 0, -1, -1), (0, -1, -1, 0)] = '+'

        # how to join array elems into single string??
        # for now, use array2string and clean up the garbage output by np
        # how to replace multiple chars?
        return np.array2string(str_field.T, separator=''). \
            replace('[', ''). \
            replace(']', ''). \
            replace("'", ''). \
            replace(' ', ''). \
            replace('_', ' ')

    def __repr__(self):
        # TODO add turn info
        str_board = self.__str__()
        return str_board


# Unit tests
class TestBoard(ut.TestCase):

    def test_init(self):
        b = Board(8, 6)
        self.assertEqual(b.shape, (8, 6))
        self.assertEqual(b.size, 8 * 6)

        self.assertTrue(np.all(b.get_empty_mask() == True))
        self.assertEqual(b.get_free_space(), 8 * 6)
        self.assertTrue(np.all(b.get_player1_mask() == False))
        self.assertTrue(np.all(b.get_player2_mask() == False))
        self.assertTrue(b.is_empty_pos((0, 0)))
        self.assertFalse(b.has_candy())

    def test_spawn(self):
        b = Board(8, 8)
        b.spawn(pos1=(1, 2), pos2=(2, 3))

        self.assertTrue(np.array_equal(b.player1_pos, np.array((1, 2))))
        self.assertTrue(np.array_equal(b.player2_pos, np.array((2, 3))))
        self.assertEqual(np.sum(b.get_player1_mask()), 1)
        self.assertEqual(np.sum(b.get_player2_mask()), 1)
        self.assertFalse(b.is_empty_pos(b.player1_pos))
        self.assertFalse(b.is_empty_pos(b.player2_pos))
        self.assertEqual(b.get_free_space(), 8 * 8 - 2)

        # candies
        self.assertTrue(np.all(b.get_candy_mask() == False))
        self.assertFalse(b.has_candy())

    def test_move(self):
        b = Board(3, 3)
        b.spawn(pos1=(0, 0), pos2=(2, 2))

        # move P1
        with self.assertRaises(AssertionError):
            b.perform_move(move=MOVE_LEFT, player=1)  # cannot move left
        with self.assertRaises(AssertionError):
            b.perform_move(move=MOVE_UP, player=1)  # cannot move up

        b.perform_move(move=MOVE_RIGHT, player=1)
        self.assertTrue(np.array_equal(b.player1_pos, np.array((1, 0))))

        with self.assertRaises(AssertionError):
            b.perform_move(move=MOVE_RIGHT, player=1)  # cannot move twice in a row

        # move P2
        with self.assertRaises(AssertionError):
            b.perform_move(move=MOVE_RIGHT, player=2)  # cannot move right
        with self.assertRaises(AssertionError):
            b.perform_move(move=MOVE_DOWN, player=2)  # cannot move down
        b.perform_move(move=MOVE_LEFT, player=2)
        self.assertTrue(np.array_equal(b.player2_pos, np.array((1, 2))))

        with self.assertRaises(AssertionError):
            b.perform_move(move=MOVE_LEFT, player=2)  # cannot move twice in a row

        # move P1 to center
        b.perform_move(move=MOVE_DOWN, player=1)
        self.assertTrue(np.array_equal(b.player1_pos, np.array((1, 1))))

        # attempt to move P2 to center (suicide)
        with self.assertRaises(AssertionError):
            b.perform_move(move=MOVE_UP, player=2)

    def test_print(self):
        b = Board(3, 2)
        self.assertEqual(
            b.__str__(),
            '+---+\n|   |\n|   |\n+---+'
        )
        b.spawn(pos1=(1, 0), pos2=(2, 1))
        self.assertEqual(
            b.__str__(),
            '+---+\n| A |\n|  B|\n+---+'
        )

        b.player1_length = 2
        b.player2_length = 2
        b.perform_move(MOVE_RIGHT, player=1)
        b.perform_move(MOVE_LEFT, player=2)
        self.assertEqual(
            b.__str__(),
            '+---+\n| aA|\n| Bb|\n+---+'
        )

    def test_move_generation(self):
        b = Board(3, 2)
        b.spawn(pos1=(1, 0), pos2=(2, 1))
        moves1 = list(map(tuple, b.get_valid_moves(1)))
        self.assertEqual(len(moves1), 3)
        self.assertTrue(tuple(MOVE_LEFT) in moves1)
        self.assertTrue(tuple(MOVE_RIGHT) in moves1)
        self.assertTrue(tuple(MOVE_DOWN) in moves1)

        moves2 = list(map(tuple, b.get_valid_moves(2)))
        self.assertEqual(len(moves2), 2)
        self.assertTrue(tuple(MOVE_LEFT) in moves2)
        self.assertTrue(tuple(MOVE_UP) in moves2)

        # perform a move and recheck the options
        b.perform_move(MOVE_LEFT, 1)
        moves12 = list(map(tuple, b.get_valid_moves(1)))
        self.assertEqual(len(moves12), 2)
        self.assertTrue(tuple(MOVE_RIGHT) in moves12)
        self.assertTrue(tuple(MOVE_DOWN) in moves12)

        b.perform_move(MOVE_LEFT, 2)
        moves22 = list(map(tuple, b.get_valid_moves(2)))
        self.assertEqual(len(moves22), 3)
        self.assertTrue(tuple(MOVE_LEFT) in moves22)
        self.assertTrue(tuple(MOVE_RIGHT) in moves22)
        self.assertTrue(tuple(MOVE_UP) in moves22)

    def test_set_state(self):
        b = Board(2, 2)
        snakes = [Snake(id=0, positions=np.array([[0, 0]])), Snake(id=1, positions=np.array([[1, 1]]))]
        b.set_state(snakes=snakes, candies=[], turn=1)
        self.assertEqual(b.turn, 1)
        self.assertEqual(b.player1_turn, 1)
        self.assertEqual(b.player2_turn, 1)
        self.assertTrue(np.array_equal(b.grid, np.array([[1, 0], [0, -1]])))
        self.assertEqual(len(b.candies), 0)

        b2 = Board(2, 2)
        snakes2 = [Snake(id=0, positions=np.array([[0, 0], [0, 1]])), Snake(id=1, positions=np.array([[1, 1]]))]
        b2.set_state(snakes=snakes2, candies=[], turn=2)
        self.assertEqual(b2.turn, 2)
        self.assertEqual(b2.player1_turn, 2)
        self.assertEqual(b2.player2_turn, 1)
        self.assertTrue(np.array_equal(b2.grid, np.array([[1, 2], [0, -1]])))
        self.assertEqual(len(b2.candies), 0)

        b3 = Board(2, 2)
        snakes3 = [Snake(id=0, positions=np.array([[0, 0], [0, 1]])), Snake(id=1, positions=np.array([[1, 1], [1, 0]]))]
        b3.set_state(snakes=snakes3, candies=[], turn=3)
        self.assertEqual(b3.turn, 3)
        self.assertEqual(b3.player1_turn, 2)
        self.assertEqual(b3.player2_turn, 2)
        self.assertTrue(np.array_equal(b3.grid, np.array([[1, 2], [-2, -1]])))
        self.assertEqual(len(b3.candies), 0)


if __name__ == '__main__':
    ut.main()
