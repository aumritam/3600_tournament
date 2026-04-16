"""
heuristic.py — board position evaluator for tree search.

Scores a position based on opportunity available right now:
- Points already scored
- Chain we can roll immediately (discounted by opponent threat)
- Runway extending our best chain
- Open space for future building
- Minus opponent's immediate scoring opportunity
"""

from game.board import Board
from game.enums import Direction, Cell, BOARD_SIZE, CARPET_POINTS_TABLE

ALL_DIRS = [Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT]

OPPOSITE = {
    Direction.UP:    Direction.DOWN,
    Direction.DOWN:  Direction.UP,
    Direction.LEFT:  Direction.RIGHT,
    Direction.RIGHT: Direction.LEFT,
}

def runway_prime_points(b: Board, direction: Direction, loc=None) -> int:
    opp_loc = b.opponent_worker.get_location()
    cur = loc if loc is not None else b.player_worker.get_location()
    points = 0
    steps = 0
    while steps < 7:
        cur = _step(cur, direction)
        if not _in_bounds(cur):
            break
        if b._blocked_mask & (1 << (cur[1] * BOARD_SIZE + cur[0])):
            break
        if b.get_cell(cur) == Cell.CARPET:
            break
        if cur == opp_loc:
            break
        if b.get_cell(cur) == Cell.SPACE:
            points += 1
        steps += 1
    return points

def _step(loc, direction: Direction):
    x, y = loc
    if direction == Direction.UP:
        return (x, y - 1)
    if direction == Direction.DOWN:
        return (x, y + 1)
    if direction == Direction.LEFT:
        return (x - 1, y)
    return (x + 1, y)

def _in_bounds(loc) -> bool:
    return 0 <= loc[0] < BOARD_SIZE and 0 <= loc[1] < BOARD_SIZE

def _best_rollable_chain(board, loc):
    """
    From loc, look in all 4 directions and count consecutive primed cells.
    Returns (best_length, best_direction, chain_end_pos)
    chain_end_pos is the far end of the best chain.
    """
    best_length = 0
    best_direction = None
    best_end = None

    for d in ALL_DIRS:
        cur = loc
        length = 0
        last = None
        while length < 7:
            nxt = _step(cur, d)
            if not _in_bounds(nxt):
                break
            if board.get_cell(nxt) != Cell.PRIMED:
                break
            cur = nxt
            last = nxt
            length += 1

        if length > best_length:
            best_length = length
            best_direction = d
            best_end = last

    return best_length, best_direction, best_end

def _raw_carpet_value(length):
    """
    Points earned if we roll a carpet of this length right now.
    """
    return float(CARPET_POINTS_TABLE.get(length, 0))

def _end_approach_threat(board, loc, best_direction, best_end):
    opp_loc = board.opponent_worker.get_location()

    roll_pos = _step(best_end, best_direction)
    dist = abs(opp_loc[0] - roll_pos[0]) + abs(opp_loc[1] - roll_pos[1])

    if dist == 0:   return 0.1   # opponent already at roll position
    elif dist == 1: return 0.5   # one step away, real immediate threat
    else:           return 1.0   # tree handles it


def _perpendicular_threat(board, loc, best_direction, best_length):
    opp_loc = board.opponent_worker.get_location()
    perp_dirs = ([Direction.LEFT, Direction.RIGHT]
                 if best_direction in (Direction.UP, Direction.DOWN)
                 else [Direction.UP, Direction.DOWN])

    for pd in perp_dirs:
        cur = opp_loc
        length = 0
        while length < 7:
            cur = _step(cur, pd)
            if not _in_bounds(cur): break
            if board.get_cell(cur) != Cell.PRIMED: break

            # Check if this primed cell is in our chain
            chain_cur = loc
            for _ in range(best_length):
                chain_cur = _step(chain_cur, best_direction)
                if chain_cur == cur:
                    dist = length
                    if dist == 0:   return 0.1   # opponent already at roll position, immediate threat
                    elif dist == 1: return 0.5   # one step away, real immediate threat
                    else:           return 1.0   # let the tree handle it
            length += 1

    return 1.0


def _threat_discount(board, loc, best_direction, best_end, best_length):
    threat1 = _end_approach_threat(board, loc, best_direction, best_end)
    threat2 = _perpendicular_threat(board, loc, best_direction, best_length)
    return min(threat1, threat2)


def _corridor_openness(board, loc, direction):
    x, y = loc
    open_count = 0
    total = 0

    if direction == Direction.DOWN:
        for row in range(y + 1, BOARD_SIZE):
            for col in range(BOARD_SIZE):
                if board._blocked_mask & (1 << (row * BOARD_SIZE + col)): continue
                total += 1
                if board.get_cell((col, row)) == Cell.SPACE: open_count += 1
    elif direction == Direction.UP:
        for row in range(0, y):
            for col in range(BOARD_SIZE):
                if board._blocked_mask & (1 << (row * BOARD_SIZE + col)): continue
                total += 1
                if board.get_cell((col, row)) == Cell.SPACE: open_count += 1
    elif direction == Direction.RIGHT:
        for col in range(x + 1, BOARD_SIZE):
            for row in range(BOARD_SIZE):
                if board._blocked_mask & (1 << (row * BOARD_SIZE + col)): continue
                total += 1
                if board.get_cell((col, row)) == Cell.SPACE: open_count += 1
    else:  # LEFT
        for col in range(0, x):
            for row in range(BOARD_SIZE):
                if board._blocked_mask & (1 << (row * BOARD_SIZE + col)): continue
                total += 1
                if board.get_cell((col, row)) == Cell.SPACE: open_count += 1

    if total == 0: return 0.0
    return open_count / total


def heuristic(board, rat_belief=None) -> float:
    my_pts  = board.player_worker.get_points()
    opp_pts = board.opponent_worker.get_points()
    my_loc  = board.player_worker.get_location()
    opp_loc = board.opponent_worker.get_location()

    score = float(my_pts - opp_pts)
    print(f"score dif {score}")

    # Our best rollable chain value, discounted by opponent threat
    best_length, best_dir, best_end = _best_rollable_chain(board, my_loc)

    if best_length > 0:
        raw_value = _raw_carpet_value(best_length)
        if raw_value > 0:
            discount = _threat_discount(board, my_loc, best_dir, best_end, best_length)
            score += raw_value * discount

    # Runway extending our committed chain
    print(f"best direction {best_dir}")
    if best_dir is not None and best_end is not None:
        extension = runway_prime_points(board, OPPOSITE[best_dir], loc=best_end)
        score += extension * 0.5

    # Best open runway in any direction (positional openness)
    best_runway = max(runway_prime_points(board, d) for d in ALL_DIRS)
    #score += best_runway * 0.2
    

    # Corridor openness — how much open space in our building direction
    if best_dir is not None:
        openness = _corridor_openness(board, my_loc, best_dir)
        #score += openness * .8
    else:
        best_openness = 0.0
        for d in ALL_DIRS:
            o = _corridor_openness(board, my_loc, d)
            best_openness = max(best_openness, o)
        #score += best_openness * 1.5

    # Subtract opponent's best rollable chain
    opp_length, _, _ = _best_rollable_chain(board, opp_loc)
    opp_raw = _raw_carpet_value(opp_length)
    #score -= opp_raw * 0.5

    """
    for d in ALL_DIRS:
        print(f"corridor {d.name}: {_corridor_openness(board, my_loc, d):.2f}", flush=True)
    """

    # Subtract opponent's best rollable chain — only meaningful chains
    opp_length, opp_dir, opp_end = _best_rollable_chain(board, opp_loc)
    if opp_length >= 3:
        opp_raw = _raw_carpet_value(opp_length)
        score -= opp_raw * 0.5

    return score

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