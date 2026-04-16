"""
Hybrid search helpers and shallow tree search.

Key changes from your previous version:
- carpet-first move ordering with NO dropped length-3 carpets
- branch cap to keep search stable
- rat belief is copied/predicted down the tree
- strong greedy helpers used by the agent before tree search
"""
#---------------------------------------------------------------------------------
from game.board import Board
from game.enums import MoveType, Direction, Cell, BOARD_SIZE, CARPET_POINTS_TABLE
from game.move import Move

from .heuristic import heuristic

TIME_RESERVE = 8.0
INF = 1e9

OPPOSITE = {
    Direction.UP: Direction.DOWN,
    Direction.DOWN: Direction.UP,
    Direction.LEFT: Direction.RIGHT,
    Direction.RIGHT: Direction.LEFT,
}

def runway_carpet_length(b: Board, direction: Direction, loc=None) -> int:
    opp_loc = b.opponent_worker.get_location()
    cur = loc if loc is not None else b.player_worker.get_location()
    length = 0
    while length < 7:
        cur = _step(cur, direction)
        if not _in_bounds(cur):
            break
        if b._blocked_mask & (1 << (cur[1] * BOARD_SIZE + cur[0])):
            break
        if b.get_cell(cur) == Cell.CARPET:
            break
        if cur == opp_loc:
            break
        length += 1
    return length

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

def trail_length_behind(b: Board, direction: Direction, loc=None) -> int:
    cur = loc if loc is not None else b.player_worker.get_location()
    opp_dir = OPPOSITE[direction]
    length = 0
    while length < 7:
        cur = _step(cur, opp_dir)
        if not _in_bounds(cur):
            break
        if b.get_cell(cur) != Cell.PRIMED:
            break
        length += 1
    return length

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

def _order_moves(b, moves):
    best_carpets = {}
    for m in moves:
        if m.move_type == MoveType.CARPET:
            if m.direction not in best_carpets or m.roll_length > best_carpets[m.direction].roll_length:
                best_carpets[m.direction] = m
    carpets = sorted(best_carpets.values(), key=lambda m: m.roll_length, reverse=True)
    primes = [m for m in moves if m.move_type == MoveType.PRIME]
    plains = [m for m in moves if m.move_type == MoveType.PLAIN]
    return (carpets + primes + plains)[:8]

def expectiminimax(
    b: Board,
    rat_belief,
    depth: int,
    alpha: float,
    beta: float,
    time_left_fn
) -> float:
    
    if depth == 0 or b.is_game_over():
        print("calcing huristic")
        value = heuristic(b, rat_belief)
        print(value)
        return heuristic(b, rat_belief)
    if time_left_fn() < TIME_RESERVE:
        return heuristic(b, rat_belief)

    next_belief = rat_belief.copy()
    next_belief.predict()

    moves = b.get_valid_moves(enemy=False, exclude_search=True)
    moves = _order_moves(b, moves)
    print(f"possible moves: {moves}")

    best = -INF
    for m in moves:
        print(m)
        child = b.forecast_move(m, check_ok=False)
        if child is None:
            continue
        child.reverse_perspective()
        val = -expectiminimax(child, next_belief, depth-1, alpha, beta, time_left_fn)
        best = max(best, val)
        alpha = max(alpha, val)
        if beta <= alpha:
            break
    return best
    
def iterative_deepening(
b: Board,
rat_belief,
max_depth: int,
time_left_fn,
ranked_moves: list,
) -> list:
    if not ranked_moves:
        return ranked_moves
    
    # ── Tier 1 — 1 best from each category, always evaluated ──────────────
    best_carpet = next((m for _, m in ranked_moves if m.move_type == MoveType.CARPET), None)
    best_prime  = next((m for _, m in ranked_moves if m.move_type == MoveType.PRIME), None)
    best_plain  = next((m for _, m in ranked_moves if m.move_type == MoveType.PLAIN), None)
    #best_search = next((m for _, m in ranked_moves if m.move_type == MoveType.SEARCH), None)

    tier1 = [m for m in [best_carpet, best_prime, best_plain] if m is not None] #add back best search
    tier1_ids = set(id(m) for m in tier1)

    # ── Tier 2 — top 6 from ranked list, deduplicated against tier 1 ──────
    tier2 = [m for _, m in ranked_moves if id(m) not in tier1_ids and m.move_type != MoveType.SEARCH][:6]
    tier2_ids = set(id(m) for m in tier2)

    # ── Tier 3 — everything else, depth 1 quick look ──────────────────────
    full_ids = tier1_ids | tier2_ids
    tier3 = [m for _, m in ranked_moves if id(m) not in full_ids]
    
    print(f"Tier 3: {[str(m) for m in tier3]}", flush=True)

    """
    turns_left = b.player_worker.turns_left
    time_available = time_left_fn() - TIME_RESERVE
    time_per_turn = time_available / max(turns_left, 1)

    if time_per_turn > 4.0:
        depth = max_depth
    elif time_per_turn > 1.5:
        depth = 2
    else:
        depth = 1
    """

    depth = 4

    #Tier 3
    promoted = []
    for m in tier3:
        if time_left_fn() < TIME_RESERVE + 1.0:
            break
        child = b.forecast_move(m, check_ok=True)
        if child is None:
            print("broken")
            continue
        child.reverse_perspective()
        val = -expectiminimax(child, rat_belief, 0, -INF, INF, time_left_fn)
        promoted.append((val, m))

    # ── Evaluate tier 1 + tier 2 at full depth ────────────────────────────
    full_results = []
    print(f"Tier 2: {[str(m) for m in tier1 + tier2]}", flush=True)
    for m in tier1 + tier2:
        print(m)
        print(time_left_fn())
        if time_left_fn() < TIME_RESERVE:
            break
        child = b.forecast_move(m, check_ok=True)
        if child is None:
            continue
        child.reverse_perspective()
        val = -expectiminimax(child, rat_belief, depth - 1, -INF, INF, time_left_fn)
        full_results.append((val, m))

     # ── Promote tier 3 moves that beat tier 1/2 minimum ───────────────────
    if full_results:
        tier_min = min(v for v, _ in full_results)
        for val, m in promoted:
            if val > tier_min:
                if time_left_fn() < TIME_RESERVE:
                    break
                child = b.forecast_move(m, check_ok=True)
                if child is None:
                    continue
                child.reverse_perspective()
                full_val = expectiminimax(child, rat_belief, depth - 1, -INF, INF, False, time_left_fn)
                full_results.append((full_val, m))
                
    full_results.sort(key=lambda x: x[0], reverse=True)
    return full_results

#--------------------------------------------------------------------------------
"""
from collections.abc import Callable
from game.board import Board
from game.enums import MoveType, Direction, Cell, BOARD_SIZE, CARPET_POINTS_TABLE
from .heuristic import heuristic

TIME_RESERVE = 8.0
INF = 1e9

OPPOSITE = {
    Direction.UP: Direction.DOWN,
    Direction.DOWN: Direction.UP,
    Direction.LEFT: Direction.RIGHT,
    Direction.RIGHT: Direction.LEFT,
}


def expectiminimax(
    b: Board,
    rat_belief,
    depth: int,
    alpha: float,
    beta: float,
    is_maximizing: bool,
    time_left_fn: Callable,
) -> float:
    if depth == 0 or b.is_game_over():
        return heuristic(b, rat_belief)
    if time_left_fn() < TIME_RESERVE:
        return heuristic(b, rat_belief)

    next_belief = rat_belief.copy()
    next_belief.predict()

    moves = b.get_valid_moves(enemy=False, exclude_search=True)
    moves = prioritize_moves(b, moves)[:12]

    if is_maximizing:
        best = -INF
        for m in moves:
            child = b.forecast_move(m, check_ok=False)
            if child is None:
                continue
            child.reverse_perspective()
            val = expectiminimax(child, next_belief, depth - 1, alpha, beta, False, time_left_fn)
            best = max(best, val)
            alpha = max(alpha, val)
            if beta <= alpha:
                break
        return best

    worst = INF
    for m in moves:
        child = b.forecast_move(m, check_ok=False)
        if child is None:
            continue
        child.reverse_perspective()
        val = expectiminimax(child, next_belief, depth - 1, alpha, beta, True, time_left_fn)
        worst = min(worst, val)
        beta = min(beta, val)
        if beta <= alpha:
            break
    return worst


def iterative_deepening(
    b: Board,
    rat_belief,
    max_depth: int,
    time_left_fn: Callable,
    moves: list,
):
    best_move = moves[0] if moves else None

    for depth in range(1, max_depth + 1):
        if time_left_fn() < TIME_RESERVE + 0.8:
            break

        candidate = None
        best_val = -INF

        for m in moves[:16]:
            if time_left_fn() < TIME_RESERVE:
                break
            child = b.forecast_move(m, check_ok=True)
            if child is None:
                continue
            child.reverse_perspective()
            val = expectiminimax(child, rat_belief, depth - 1, -INF, INF, False, time_left_fn)
            if val > best_val:
                best_val = val
                candidate = m

        if candidate is not None:
            best_move = candidate

    return best_move


def prioritize_moves(b: Board, moves: list, committed_direction=None) -> list:
    carpets = [m for m in moves if m.move_type == MoveType.CARPET]
    carpets.sort(key=lambda m: (m.roll_length, CARPET_POINTS_TABLE.get(m.roll_length, -99)), reverse=True)

    primes = [m for m in moves if m.move_type == MoveType.PRIME]
    plains = [m for m in moves if m.move_type == MoveType.PLAIN]

    def prime_score(m):
        runway = runway_length(b, m.direction)
        trail = trail_length_behind(b, m.direction)

        score = 0.75 * runway + 4.0 * trail
        if committed_direction is not None and m.direction == committed_direction:
            score += 5.0
        return score

    def plain_score(m):
        new_pos = _step(b.player_worker.get_location(), m.direction)
        opp = b.opponent_worker.get_location()

        score = -2.0
        score -= 0.25 * (abs(new_pos[0] - opp[0]) + abs(new_pos[1] - opp[1]))

        # tiny bonus for central squares so it does not drift to edges
        score -= 0.1 * (abs(new_pos[0] - 3.5) + abs(new_pos[1] - 3.5))

        # heavy penalty for stepping onto our own carpet — wasted movement
        if b.get_cell(new_pos) == Cell.CARPET:
            score -= 5.0

        return score

    primes.sort(key=prime_score, reverse=True)
    plains.sort(key=plain_score, reverse=True)
    return carpets + primes + plains


def best_greedy_carpet(b: Board, min_length: int = 5):
    moves = b.get_valid_moves(enemy=False, exclude_search=True)
    carpets = [m for m in moves if m.move_type == MoveType.CARPET and m.roll_length >= min_length]
    if not carpets:
        return None
    carpets.sort(
        key=lambda m: (m.roll_length, CARPET_POINTS_TABLE.get(m.roll_length, -99)),
        reverse=True,
    )
    return carpets[0]


def trail_length_behind(b: Board, direction: Direction) -> int:
    loc = b.player_worker.get_location()
    opp_dir = OPPOSITE[direction]
    cur = loc
    length = 0
    while True:
        cur = _step(cur, opp_dir)
        if not _in_bounds(cur):
            break
        if b.get_cell(cur) != Cell.PRIMED:
            break
        length += 1
        if length >= 7:
            break
    return length


def runway_length(b: Board, direction: Direction) -> int:
    opp_loc = b.opponent_worker.get_location()
    cur = b.player_worker.get_location()
    length = 0
    while length < 7:
        cur = _step(cur, direction)
        if not _in_bounds(cur):
            break
        if b.is_cell_blocked(cur):
            break
        if b.get_cell(cur) in (Cell.PRIMED, Cell.CARPET):
            break
        if cur == opp_loc:
            break
        length += 1
    return length


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
"""