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
# carpet points plus points earned priming remaining cells
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
 
# Axis grouping — UP/DOWN share vertical axis, LEFT/RIGHT share horizontal
AXIS = {
    Direction.UP:    (Direction.UP, Direction.DOWN),
    Direction.DOWN:  (Direction.UP, Direction.DOWN),
    Direction.LEFT:  (Direction.LEFT, Direction.RIGHT),
    Direction.RIGHT: (Direction.LEFT, Direction.RIGHT),
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
    prime_cost  : number of remaining prime steps needed
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
        self.prime_cost = n_space       # remaining prime steps needed
        self.net_value  = self.carpet_pts + self.prime_cost
 
    def is_ready(self) -> bool:
        """All cells are primed — can be rolled immediately."""
        return self.valid and self.n_primed == self.length
 
    def turns_to_prime(self, worker_pos: tuple) -> int:
        if not self.valid:
            return 999
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
    return Cell.SPACE
 
 
# ---------------------------------------------------------------------------
# Main planner class
# ---------------------------------------------------------------------------
 
class CarpetPlanner:
    """
    Scores all viable carpet lines on the board and recommends moves.
 
    Key design: _prime_toward enforces ON-AXIS priming — it only primes
    in the target line's direction or opposite, preventing the zigzag
    priming that scatters cells across the board instead of building lines.
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
          1. Roll carpet if any ready line meets minimum length
          2. Prime toward the best target line (on-axis only)
          3. Plain step to reposition toward target line axis
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
            # Try to prime toward target (on-axis only)
            prime = self._prime_toward(board, worker_pos, target)
            if prime is not None:
                return prime
 
            # Can't prime on-axis — step toward line axis to reposition
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
          net_value       — carpet pts plus remaining prime gains
          length bonus    — longer lines get disproportionate reward
          commitment      — heavily invested lines get large bonus to prevent abandonment
          proximity       — closer lines are preferred
          opponent risk   — lines the opponent can reach sooner are penalised
        """
        if not line.valid:
            return -999.0
 
        # Base: net carpet value (carpet pts + priming gains)
        score = float(line.net_value)
 
        # Length bonus: strongly prefer long lines
        score += line.length ** 2.0
 
        # Commitment bonus — once invested, lock in hard to prevent abandonment
        completion_ratio = line.n_primed / line.length
        if completion_ratio >= 0.5:
            score += line.n_primed * 15.0   # heavily committed — almost unbeatable
        elif completion_ratio >= 0.25:
            score += line.n_primed * 8.0    # partially committed
        else:
            score += line.n_primed * 1.5    # just started
 
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
        """Return the highest-value carpet Move available, or None.
        Only rolls lines of minimum length to avoid short carpet waste."""
        ready = self.ready_lines(board, worker_pos)
        turns_left = board.player_worker.turns_left
        min_length = 3 if turns_left <= 8 else 4
        ready = [(pts, line, move) for pts, line, move in ready
                 if line.length >= min_length]
        if ready:
            _, _, move = ready[0]
            return move
        return None
 
    def _prime_toward(self, board, worker_pos: tuple, target: Line):
        """
        Return a PRIME move that advances toward the target line, or None.
 
        KEY DESIGN: Only primes on the target line's axis (direction or opposite).
        This prevents zigzag priming where cells are scattered instead of forming
        a straight line. If we can't prime on-axis, returns None so the caller
        uses a plain step to reposition onto the axis first.
 
        A prime move leaves the current cell as PRIMED and steps forward.
        We can only prime if the current cell is SPACE.
        """
        current_cell = get_cell_type(board, worker_pos)
        if current_cell != Cell.SPACE:
            return None   # can't prime from here
 
        # Find unprimed cells in the target line
        unprimed = [c for c in target.cells
                    if get_cell_type(board, c) == Cell.SPACE]
        if not unprimed:
            return None
 
        goal = min(unprimed, key=lambda c: manhattan(worker_pos, c))
 
        valid_moves = board.get_valid_moves(exclude_search=True)
        prime_moves = [m for m in valid_moves if m.move_type == MoveType.PRIME]
 
        # Only allow priming along the line's axis
        axis_dirs = AXIS[target.direction]
        axis_primes = [m for m in prime_moves if m.direction in axis_dirs]
 
        # Among on-axis primes, pick the one that gets closest to goal
        best_dir = None
        best_dist = manhattan(worker_pos, goal)
        for m in axis_primes:
            after = loc_after_direction(worker_pos, m.direction)
            d = manhattan(after, goal)
            if d < best_dist:
                best_dist = d
                best_dir = m.direction
 
        if best_dir is not None:
            return Move.prime(best_dir)
 
        # If on the axis but blocked along axis, allow any prime as fallback
        if self._on_line_axis(worker_pos, target):
            fallback_dir = self._best_direction_toward(
                board, worker_pos, goal, move_type=MoveType.PRIME)
            if fallback_dir is not None:
                return Move.prime(fallback_dir)
 
        # Not on axis and can't reach goal on-axis — caller should reposition
        return None
 
    def _on_line_axis(self, pos: tuple, line: Line) -> bool:
        """
        Check if pos shares the same row (horizontal line) or column
        (vertical line) as the target line.
        """
        if line.direction in (Direction.RIGHT, Direction.LEFT):
            # Horizontal line — check if we share the same row (y coordinate)
            line_y = line.cells[0][1]
            return pos[1] == line_y
        else:
            # Vertical line — check if we share the same column (x coordinate)
            line_x = line.cells[0][0]
            return pos[0] == line_x
 
    def _step_toward(self, board, worker_pos: tuple, target: Line):
        """
        Return a PLAIN move stepping toward the target line's axis, or None.
        Used when we can't prime (current cell is already primed/carpet)
        or when we need to reposition onto the line's axis before priming.
 
        Prioritizes moving onto the axis so subsequent primes will be on-axis.
        """
        # Target the nearest point on the line's axis
        axis_goal = self._nearest_axis_entry(worker_pos, target)
 
        if axis_goal != worker_pos:
            best_dir = self._best_direction_toward(board, worker_pos, axis_goal,
                                                    move_type=MoveType.PLAIN)
            if best_dir is not None:
                return Move.plain(best_dir)
 
        # Fall back to stepping toward closest cell in the line
        closest = min(target.cells, key=lambda c: manhattan(worker_pos, c))
        best_dir = self._best_direction_toward(board, worker_pos, closest,
                                                move_type=MoveType.PLAIN)
        if best_dir is None:
            return None
        return Move.plain(best_dir)
 
    def _nearest_axis_entry(self, worker_pos: tuple, line: Line) -> tuple:
        """
        Find the point on the line's axis closest to worker_pos.
        For a horizontal line: (clamped_worker_x, line_y)
        For a vertical line:   (line_x, clamped_worker_y)
        This is where we want to reposition before starting to prime.
        """
        if line.direction in (Direction.RIGHT, Direction.LEFT):
            # Horizontal line — target same row as line
            line_y = line.cells[0][1]
            min_x = min(c[0] for c in line.cells)
            max_x = max(c[0] for c in line.cells)
            target_x = max(min_x, min(max_x, worker_pos[0]))
            return (target_x, line_y)
        else:
            # Vertical line — target same column as line
            line_x = line.cells[0][0]
            min_y = min(c[1] for c in line.cells)
            max_y = max(c[1] for c in line.cells)
            target_y = max(min_y, min(max_y, worker_pos[1]))
            return (line_x, target_y)
 
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