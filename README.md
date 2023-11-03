# Snek
Snek is my bot submission for the multiplayer snakes tournament organized by [nobleans-playerground](https://github.com/nobleans-playground/coding-challenge-snakes).

Snek uses [principal variation search](https://www.chessprogramming.org/Principal_Variation_Search) to search for move sequences that lead to favorable or undesirable game states, and prunes part of the game tree to save computation time. Snek does not have a heart, but it does have a custom [game board representation](board.py) optimized for tree-search algorithms.

# Setup
This repository is implemented to be a submodule of https://github.com/nobleans-playground/coding-challenge-snakes.
See the respective [README](https://github.com/nobleans-playground/coding-challenge-snakes/blob/main/README.md) for instructions.

# Development
Some of the unit tests are time-intensive so it's useful to run some in parallel:
```sh
pip install pytest-xdist
```
