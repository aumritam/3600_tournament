#----------------------------------------------------------------------------------------







#----------------------------------------------------------------------------------------
"""
Practical territorial heuristic.

This is intentionally closer to your friend's "score-now / steal-now" style
than the more abstract line-planning heuristic.


from typing import Tuple

from game.board import Board
from game.enums import Direction, Cell, BOARD_SIZE, CARPET_POINTS_TABLE

ALL_DIRS = [Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT]

EARLY_CUTOFF = 28
MID_CUTOFF = 10

W_POINTS_LEAD = 3.0
W_OWN_TRAIL = 2.7
W_STEAL_CHAIN = 2.8
W_DENY_OPP = 2.2
W_TRAIL_BUILD = 1.4
W_CUTOFF = 1.0
W_RAT_EV = 0.5
W_TURNS_DELTA = 0.05


def heuristic(b: Board, rat_belief) -> float:
    my_pts = b.player_worker.get_points()
    opp_pts = b.opponent_worker.get_points()
    pt_lead = my_pts - opp_pts
    turns_left = b.player_worker.turns_left

    early = turns_left > EARLY_CUTOFF
    mid = MID_CUTOFF < turns_left <= EARLY_CUTOFF
    trailing = pt_lead < 0

    if early:
        steal_scale = 0.8
        rat_scale = 0.35
        cutoff_scale = 1.8
        trail_scale = 1.5
    elif mid:
        steal_scale = 1.9
        rat_scale = 0.6
        cutoff_scale = 0.3
        trail_scale = 1.0
    else:
        steal_scale = 2.6 if trailing else 1.9
        rat_scale = 0.25
        cutoff_scale = 0.0
        trail_scale = 0.25

    my_loc = b.player_worker.get_location()
    opp_loc = b.opponent_worker.get_location()

    score = 0.0
    score += W_POINTS_LEAD * pt_lead
    score += W_OWN_TRAIL * trail_scale * _primed_chain_value(b, my_loc, ours=True)
    score += W_STEAL_CHAIN * steal_scale * _primed_chain_value(b, my_loc, ours=False)
    score -= W_DENY_OPP * steal_scale * _primed_chain_value(b, opp_loc, ours=False)
    score += W_TRAIL_BUILD * trail_scale * _best_trail_behind(b)

    if early:
        score += W_CUTOFF * cutoff_scale * _cutoff_score(b, my_loc, opp_loc)

    if rat_belief is not None:
        best_pos, _ = rat_belief.best_search_target()
        best_prob = rat_belief.belief_at(best_pos)
        score += W_RAT_EV * rat_scale * max(0.0, 6.0 * best_prob - 2.0)

    score += W_TURNS_DELTA * (b.player_worker.turns_left - b.opponent_worker.turns_left)
    return score


def _primed_chain_value(b: Board, loc: Tuple[int, int], ours: bool) -> float:
    total = 0.0
    for direction in ALL_DIRS:
        length = _chain_length_from(b, loc, direction, target_cell=Cell.PRIMED)
        if length >= 2:
            carpet_points = CARPET_POINTS_TABLE.get(min(length, 7), 0)
            total += carpet_points
            total += 0.6 * length

            end = _walk(loc, direction, length)
            if end is not None:
                bonus = 0.4 if ours else 1.0
                total += bonus * _adjacent_foreign_prime_bonus(b, end, direction)
    return total


def _best_trail_behind(b: Board) -> float:
    best = 0
    loc = b.player_worker.get_location()
    for direction in ALL_DIRS:
        best = max(best, _chain_length_from(b, loc, _opposite(direction), target_cell=Cell.PRIMED))
    return float(best)


def _cutoff_score(b: Board, my_loc: Tuple[int, int], opp_loc: Tuple[int, int]) -> float:
    my_space = _reachable_open_space(b, my_loc, limit=18)
    opp_space = _reachable_open_space(b, opp_loc, limit=18)
    dx = abs(my_loc[0] - opp_loc[0])
    dy = abs(my_loc[1] - opp_loc[1])
    centrality = 3.5 - (abs(my_loc[0] - 3.5) + abs(my_loc[1] - 3.5)) / 2.0
    return 0.15 * (my_space - opp_space) + 0.1 * (dx + dy) + 0.2 * centrality


def _adjacent_foreign_prime_bonus(b: Board, end: Tuple[int, int], direction: Direction) -> float:
    bonus = 0.0
    for side in _perpendiculars(direction):
        nxt = _step(end, side)
        if _in_bounds(nxt) and b.get_cell(nxt) == Cell.PRIMED:
            bonus += 1.0
    return bonus


def _reachable_open_space(b: Board, start: Tuple[int, int], limit: int = 18) -> int:
    seen = {start}
    q = [start]
    count = 0
    while q and count < limit:
        cur = q.pop(0)
        for d in ALL_DIRS:
            nxt = _step(cur, d)
            if nxt in seen or not _in_bounds(nxt):
                continue
            if b.is_cell_blocked(nxt):
                continue
            if b.get_cell(nxt) in (Cell.PRIMED, Cell.CARPET):
                continue
            seen.add(nxt)
            q.append(nxt)
            count += 1
            if count >= limit:
                break
    return count


def _chain_length_from(b: Board, loc: Tuple[int, int], direction: Direction, target_cell: Cell) -> int:
    length = 0
    cur = loc
    while True:
        cur = _step(cur, direction)
        if not _in_bounds(cur):
            break
        if b.get_cell(cur) != target_cell:
            break
        if cur == b.player_worker.get_location() or cur == b.opponent_worker.get_location():
            break
        length += 1
        if length >= 7:
            break
    return length


def _walk(loc: Tuple[int, int], direction: Direction, steps: int):
    cur = loc
    for _ in range(steps):
        cur = _step(cur, direction)
        if not _in_bounds(cur):
            return None
    return cur


def _step(loc: Tuple[int, int], direction: Direction) -> Tuple[int, int]:
    x, y = loc
    if direction == Direction.UP:
        return (x, y - 1)
    if direction == Direction.DOWN:
        return (x, y + 1)
    if direction == Direction.LEFT:
        return (x - 1, y)
    return (x + 1, y)


def _opposite(direction: Direction) -> Direction:
    if direction == Direction.UP:
        return Direction.DOWN
    if direction == Direction.DOWN:
        return Direction.UP
    if direction == Direction.LEFT:
        return Direction.RIGHT
    return Direction.LEFT


def _perpendiculars(direction: Direction):
    if direction in (Direction.UP, Direction.DOWN):
        return [Direction.LEFT, Direction.RIGHT]
    return [Direction.UP, Direction.DOWN]


def _in_bounds(loc: Tuple[int, int]) -> bool:
    return 0 <= loc[0] < BOARD_SIZE and 0 <= loc[1] < BOARD_SIZE
"""