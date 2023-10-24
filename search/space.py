from collections import deque
from typing import List, Tuple

from snakes.bots.niekdt.board import PosIdx


def count_free_space_bfs(
        mask: List[bool],
        pos: PosIdx,
        max_dist: int,
        lb: int,
        pos_pos_options: List[List[Tuple[PosIdx, ...]]],
        prev_pos: PosIdx = 0
) -> int:
    mask[pos] = False
    free_space = 1
    cur_dist = 0
    queue = deque(maxlen=128)

    while free_space < lb and cur_dist < max_dist:
        for new_pos in pos_pos_options[prev_pos][pos]:
            if mask[new_pos]:
                mask[new_pos] = False
                free_space += 1
                queue.append((new_pos, pos, cur_dist + 1))
        if not queue:
            break
        pos, prev_pos, cur_dist = queue.popleft()

    return free_space


def count_free_space_bfs_delta(
        mask: List[bool],
        pos1: PosIdx,
        pos2: PosIdx,
        pos_options: List[Tuple[PosIdx, ...]],
        min_dist: int = 1000,
        max_dist: int = 1000,
        delta_lb: int = 1000
) -> Tuple[int, int, int]:
    mask[pos1] = False
    mask[pos2] = False
    pos = pos1
    player = 0
    free_space = [1, 1]
    delta_space = 0
    cur_dist = 0
    queue = deque(maxlen=256)
    queue.append((1, pos2, 0))

    while cur_dist < min_dist or (abs(delta_space) < delta_lb and cur_dist < max_dist):
        for new_pos in pos_options[pos]:
            if mask[new_pos]:
                mask[new_pos] = False
                free_space[player] += 1
                queue.append((player, new_pos, cur_dist + 1))
        assert len(queue) < 200
        if not queue:
            break
        player, pos, cur_dist = queue.popleft()
        delta_space = free_space[0] - free_space[1]

    return delta_space, free_space[0], free_space[1]


def count_free_space_dfs(
        mask: List[bool],
        pos: PosIdx,
        pos_options: List[Tuple[PosIdx, ...]],
        lb: int,
        max_dist: int,
        distance_map: Tuple[int]
) -> int:
    stack = [pos]
    free_space = 0

    while stack and free_space < lb:
        pos = stack.pop()
        if not mask[pos] or distance_map[pos] > max_dist:
            continue
        mask[pos] = False
        free_space += 1
        stack.extend(pos_options[pos])

    return free_space
