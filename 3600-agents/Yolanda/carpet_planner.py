"""
carpet_planner.py  —  line scoring and carpet strategy.

Core job: given the current board state, answer two questions:
  1. What are the best lines to build toward?  (score_all_lines)
  2. What is the single best move to make right now?  (best_move)

Key rules confirmed from board.py / enums.py:
  - CARPET_POINTS_TABLE: {1:-1, 2:2, 3:4, 4:6, 5:10, 6:15, 7:21}
  - Prime step:  worker leaves current cell as PRIMED, moves into next cell
      -> current cell must be SPACE (not already primed/carpeted)
      -> destination cell must not be blocked/primed/occupied
  - Plain step:  move into adjacent cell
      -> destination must not be blocked/primed/occupied
  - Carpet roll of length k: walk k contiguous PRIMED cells in one direction
      -> each cell gets carpeted, worker ends on last cell
      -> cells cannot be occupied by either worker
  - Primed squares block plain/prime movement (you cannot walk onto them)
  - Either player can roll carpet over any primed line
"""

import numpy as np
from game.enums import (
    BOARD_SIZE, CARPET_POINTS_TABLE, Cell, Direction, MoveType,
    loc_after_direction
)
from game.move import Move

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DIRECTIONS = [Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT]

# Net value of carpeting a line of length n:
# carpet points minus the 1pt cost of each prime step
# (primes already placed are sunk cost — only future primes matter)
CARPET_NET = {n: CARPET_POINTS_TABLE[n] for n in range(1, 8)}

# Direction vectors for iteration
DIR_DELTA = {
    Direction.UP:    (0, -1),
    Direction.DOWN:  (0,  1),
    Direction.LEFT:  (-1, 0),
    Direction.RIGHT: (1,  0),
}

# Opposite direction
OPPOSITE = {
    Direction.UP:    Direction.DOWN,
    Direction.DOWN:  Direction.UP,
    Direction.LEFT:  Direction.RIGHT,
    Direction.RIGHT: Direction.LEFT,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def in_bounds(pos: tuple) -> bool:
    return 0 <= pos[0] < BOARD_SIZE and 0 <= pos[1] < BOARD_SIZE


def cells_in_line(start: tuple, direction: Direction, length: int) -> list:
    """Return list of `length` cells starting at `start` going in `direction`."""
    cells = []
    pos = start
    dx, dy = DIR_DELTA[direction]
    for _ in range(length):
        pos = (pos[0] + dx, pos[1] + dy)
        if not in_bounds(pos):
            return []   # line goes off board
        cells.append(pos)
    return cells


def get_cell_type(board, pos: tuple) -> Cell:
    """Read cell type from board bitmasks directly (faster than get_cell)."""
    bit = 1 << (pos[1] * BOARD_SIZE + pos[0])
    if board._primed_mask & bit:  return Cell.PRIMED
    if board._carpet_mask & bit:  return Cell.CARPET
    if board._blocked_mask & bit: return Cell.BLOCKED
    return Cell.SPACE


def manhattan(a: tuple, b: tuple) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


# ---------------------------------------------------------------------------
# Line representation
# ---------------------------------------------------------------------------

class Line:
    """
    A candidate carpet line: a straight run of n cells in one direction.

    Attributes
    ----------
    cells       : list of (x,y) — the cells in the line, in order
    direction   : which way the line runs
    n_primed    : how many cells are already primed
    n_space     : how many cells are still open space
    n_blocked   : how many cells are blocked/carpeted (makes line invalid)
    carpet_pts  : points earned when rolled  (CARPET_POINTS_TABLE[len])
    prime_cost  : points spent priming remaining space cells (1 per cell)
    net_value   : carpet_pts + prime_cost  (priming gains points, doesn't cost them)
    """

    def __init__(self, cells: list, direction: Direction, board):
        self.cells     = cells
        self.direction = direction
        self.length    = len(cells)

        n_primed = n_space = n_bad = 0
        for pos in cells:
            ct = get_cell_type(board, pos)
            if ct == Cell.PRIMED:   n_primed += 1
            elif ct == Cell.SPACE:  n_space  += 1
            else:                   n_bad    += 1   # blocked or already carpeted

        self.n_primed  = n_primed
        self.n_space   = n_space
        self.n_blocked = n_bad
        self.valid     = (n_bad == 0)   # line is only viable if fully clear

        self.carpet_pts = CARPET_POINTS_TABLE[self.length]
        self.prime_cost = n_space       # 1pt per prime step still needed
        self.net_value  = self.carpet_pts + self.prime_cost

    def is_ready(self) -> bool:
        """All cells are primed — can be rolled immediately."""
        return self.valid and self.n_primed == self.length

    def turns_to_prime(self, worker_pos: tuple) -> int:
        """
        Rough lower bound on turns needed to finish priming this line.
        = number of unprimed cells  (you prime one cell per prime-step turn)
        + travel distance to reach the line
        Does NOT account for path-finding around blocked/primed cells.
        """
        if not self.valid:
            return 999
        # Closest unprimed cell to walk to
        unprimed = [c for c in self.cells if get_cell_type_static(c) == Cell.SPACE]
        if not unprimed:
            return 0
        min_dist = min(manhattan(worker_pos, c) for c in unprimed)
        return self.n_space + min_dist

    def __repr__(self):
        return (f"Line(len={self.length}, dir={self.direction.name}, "
                f"primed={self.n_primed}/{self.length}, "
                f"net={self.net_value:+d}, valid={self.valid})")


# Static version for turns_to_prime (no board reference needed after init)
def get_cell_type_static(pos):
    # placeholder — Line.turns_to_prime is called after __init__ where
    # we already have cell type info; in practice use the board reference
    return Cell.SPACE


# ---------------------------------------------------------------------------
# Main planner class
# ---------------------------------------------------------------------------

class CarpetPlanner:
    """
    Scores all viable carpet lines on the board and recommends moves.

    Usage (from agent.py each turn)
    --------------------------------
        planner = CarpetPlanner()          # once at init

        move = planner.best_move(board, board.player_worker.get_location())
    """

    def __init__(self):
        pass   # stateless — all computation is per-turn from board state

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score_all_lines(self, board, worker_pos: tuple) -> list:
        """
        Enumerate and score every valid horizontal/vertical line of length 2-7.
        Returns list of Line objects sorted by priority score descending.
        """
        lines = []
        for y in range(BOARD_SIZE):
            for x in range(BOARD_SIZE):
                start = (x, y)
                for direction in [Direction.RIGHT, Direction.DOWN]:
                    for length in range(2, BOARD_SIZE + 1):
                        cells = cells_in_line(start, direction, length)
                        if len(cells) < length:
                            break   # went off board, longer won't work either
                        line = Line(cells, direction, board)
                        if line.valid:
                            lines.append(line)

        # Score each line and sort
        scored = []
        for line in lines:
            score = self._priority_score(line, worker_pos, board)
            scored.append((score, line))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [line for _, line in scored]

    def best_move(self, board, worker_pos: tuple) -> Move:
        """
        Return the single best Move to make this turn.

        Decision order:
          1. Roll carpet if any ready line is reachable
          2. Prime toward the best target line
          3. Plain step toward the best target line
          4. Fallback: any valid move
        """
        # Check for immediate carpet rolls first
        roll = self._best_carpet_roll(board, worker_pos)
        if roll is not None:
            return roll

        # Find best line to build toward
        lines = self.score_all_lines(board, worker_pos)
        target = lines[0] if lines else None

        if target is not None:
            # Try to prime toward target
            prime = self._prime_toward(board, worker_pos, target)
            if prime is not None:
                return prime

            # Can't prime right now — plain step toward target
            plain = self._step_toward(board, worker_pos, target)
            if plain is not None:
                return plain

        # Fallback: any valid non-search move
        valid = board.get_valid_moves(exclude_search=True)
        if valid:
            return valid[0]

        return None   # should never happen

    def ready_lines(self, board, worker_pos: tuple) -> list:
        """
        Return all lines that are fully primed and can be rolled right now
        from the current worker position, sorted by carpet points descending.
        """
        result = []
        valid_moves = board.get_valid_moves(exclude_search=True)
        carpet_moves = {(m.direction, m.roll_length): m
                        for m in valid_moves if m.move_type == MoveType.CARPET}

        for length in range(2, BOARD_SIZE + 1):
            for direction in DIRECTIONS:
                cells = cells_in_line(worker_pos, direction, length)
                if len(cells) < length:
                    continue
                line = Line(cells, direction, board)
                if line.is_ready():
                    key = (direction, length)
                    if key in carpet_moves:
                        result.append((CARPET_POINTS_TABLE[length], line, carpet_moves[key]))

        result.sort(key=lambda x: x[0], reverse=True)
        return result

    def opponent_threatens(self, board, opp_pos: tuple) -> list:
        """
        Lines the opponent can roll on their next turn.
        Used by heuristic to penalise leaving long primed lines unattended.
        """
        return self.ready_lines(board, opp_pos)

    # ------------------------------------------------------------------
    # Internal: scoring
    # ------------------------------------------------------------------

    def _priority_score(self, line: Line, worker_pos: tuple, board) -> float:
        """
        Score a line for planning priority. Higher = more urgent to pursue.

        Components:
          net_value       — carpet pts minus remaining prime cost
          length bonus    — longer lines get disproportionate reward
          proximity       — closer lines are preferred (discount by distance)
          ownership       — lines we've already partially primed are preferred
          opponent risk   — lines the opponent can reach sooner penalised
        """
        if not line.valid:
            return -999.0

        # Base: net carpet value
        score = float(line.net_value)

        # Length bonus: 7-line is much better than two 3-lines
        score += line.length ** 2.0
        
        # Already invested primes are a sunk-cost bonus
        # (we've already paid for those squares)
        completion_ratio = line.n_primed / line.length
        if completion_ratio >= 0.5:
            score += line.n_primed * 5.0
        elif completion_ratio >= 0.25:
            score += line.n_primed * 3.0
        else:
            score += line.n_primed * 1.5

        # Proximity discount: prefer reachable lines
        closest_cell = min(line.cells, key=lambda c: manhattan(worker_pos, c))
        dist = manhattan(worker_pos, closest_cell)
        score -= dist * 0.8

        # Penalise if opponent is closer to this line than we are
        opp_pos = board.opponent_worker.get_location()
        opp_dist = _dist_to_line(opp_pos, line)
        my_dist  = _dist_to_line(worker_pos, line)
        if opp_dist < my_dist:
            poach_risk = (my_dist - opp_dist) / max(1, line.length)
            poach_risk *= (1.0 - completion_ratio * 0.7)
            score -= poach_risk * 3.0

        return score

    # ------------------------------------------------------------------
    # Internal: move construction
    # ------------------------------------------------------------------

    def _best_carpet_roll(self, board, worker_pos: tuple):
        """Return the highest-value carpet Move available right now, or None."""
        ready = self.ready_lines(board, worker_pos)
        if ready:
            _, _, move = ready[0]
            return move
        return None

    def _prime_toward(self, board, worker_pos: tuple, target: Line):
        """
        Return a PRIME move that advances toward the target line, or None.

        A prime move leaves the current cell as primed and steps forward.
        We can only prime if the current cell is SPACE.
        """
        current_cell = get_cell_type(board, worker_pos)
        if current_cell != Cell.SPACE:
            return None   # can't prime from here

        # Find the unprimed cell in the target that is closest to us
        unprimed = [c for c in target.cells
                    if get_cell_type(board, c) == Cell.SPACE]
        if not unprimed:
            return None

        goal = min(unprimed, key=lambda c: manhattan(worker_pos, c))

        # Pick the prime direction that gets us closest to goal
        best_dir = self._best_direction_toward(board, worker_pos, goal,
                                                move_type=MoveType.PRIME)
        if best_dir is None:
            return None
        return Move.prime(best_dir)

    def _step_toward(self, board, worker_pos: tuple, target: Line):
        """
        Return a PLAIN move stepping toward the target line, or None.
        Used when we can't prime (current cell is already primed/carpet).
        """
        closest = min(target.cells, key=lambda c: manhattan(worker_pos, c))
        best_dir = self._best_direction_toward(board, worker_pos, closest,
                                                move_type=MoveType.PLAIN)
        if best_dir is None:
            return None
        return Move.plain(best_dir)

    def _best_direction_toward(
        self, board, worker_pos: tuple, goal: tuple, move_type: MoveType
    ):
        """
        Among valid moves of move_type, return the direction that minimises
        manhattan distance to goal after one step.
        """
        valid_moves = board.get_valid_moves(exclude_search=True)
        candidates = [m for m in valid_moves if m.move_type == move_type]

        best_dir = None
        best_dist = manhattan(worker_pos, goal)   # must improve on current dist

        for move in candidates:
            next_pos = loc_after_direction(worker_pos, move.direction)
            d = manhattan(next_pos, goal)
            if d < best_dist:
                best_dist = d
                best_dir = move.direction

        return best_dir

def _dist_to_line(pos: tuple, line: Line) -> int:
    return min(manhattan(pos, c) for c in line.cells)