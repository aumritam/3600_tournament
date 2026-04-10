"""
search.py  —  expectiminimax with alpha-beta pruning + iterative deepening.
 
Design decisions based on benchmarks:
  - forecast_move costs ~0.04ms, get_valid_moves ~0.007ms
  - Branching factor ~8-10 mid-game (excluding search moves)
  - Depth 4 ~ 0.8s, depth 5 ~ 7s  (before pruning)
  - Alpha-beta pruning typically cuts to sqrt(b^d) in best case
  - Iterative deepening: always have a best move ready, go deeper while time allows
 
Key design choices:
  - Search moves are NOT expanded in the tree — the rat is stochastic and
    the HMM handles that reasoning. We pass rat_belief EV into the heuristic
    instead. This keeps branching factor low.
  - We DO model the opponent's turn as a minimizing player (not expectation)
    because the opponent is adversarial, not random.
  - board.reverse_perspective() is called after each forecast so player_worker
    always refers to whoever is moving next — matches how the engine works.
  - Move ordering: carpet rolls first (likely best), then primes, then plains.
    Good ordering dramatically improves alpha-beta cutoffs.
"""
 
import time
from game.move import Move
from game.enums import MoveType, Direction
from .heuristic import evaluate
 
# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
 
INF = float('inf')
 
# How much of the remaining time budget to use per turn
# Leave headroom for HMM update and other overhead
TIME_FRACTION = 0.85
 
# Minimum depth we always complete before checking time
MIN_DEPTH = 2
 
# Maximum depth we ever search (safety cap)
MAX_DEPTH = 6
 
 
# ---------------------------------------------------------------------------
# Move ordering
# ---------------------------------------------------------------------------
 
def order_moves(moves: list) -> list:
    good_carpets = sorted(
        [m for m in moves if m.move_type == MoveType.CARPET and m.roll_length >= 4],
        key=lambda m: -m.roll_length
    )
    bad_carpets = [m for m in moves if m.move_type == MoveType.CARPET and m.roll_length < 3]
    primes = [m for m in moves if m.move_type == MoveType.PRIME]
    plains = [m for m in moves if m.move_type == MoveType.PLAIN]
    return good_carpets + primes + plains + bad_carpets
 
 
# ---------------------------------------------------------------------------
# Core search
# ---------------------------------------------------------------------------
 
class Searcher:
    """
    Iterative deepening expectiminimax with alpha-beta pruning.
 
    Usage
    -----
        searcher = Searcher()
        best_move, best_score = searcher.search(board, rat_belief, time_left_func)
    """
 
    def __init__(self):
        self.nodes_searched = 0
        self.best_move_at_depth = {}   # depth -> Move (for debugging)
 
    def search(self, board, rat_belief, time_left_func) -> tuple:
        """
        Run iterative deepening search and return (best_move, best_score).
 
        Parameters
        ----------
        board          : Board (player_worker = us, already perspective-correct)
        rat_belief     : RatBelief — passed to heuristic at leaf nodes
        time_left_func : callable returning seconds remaining this turn
        """
        self.nodes_searched = 0
        self.start_time = time.perf_counter()
 
        # Compute time budget for this search
        total_remaining = time_left_func()
        turns_left = board.player_worker.turns_left
        per_turn_share = total_remaining / max(turns_left, 1)
        self.time_budget = min(per_turn_share * TIME_FRACTION, 8.0)
        self.deadline = self.start_time + self.time_budget
        self.rat_belief = rat_belief
 
        # Get and order moves at the root
        root_moves = order_moves(board.get_valid_moves(exclude_search=True))
 
        if not root_moves:
            return None, 0.0
 
        # Always have a fallback move
        best_move  = root_moves[0]
        best_score = -INF
 
        # Iterative deepening
        for depth in range(1, MAX_DEPTH + 1):
            # Check time before starting a new depth
            if depth > MIN_DEPTH and self._out_of_time():
                break
 
            move, score = self._search_root(board, root_moves, depth)
 
            if move is not None:
                best_move  = move
                best_score = score
                self.best_move_at_depth[depth] = move
 
            # If we found a near-perfect score, stop early
            if best_score > 900:
                break
 
        return best_move, best_score
 
    def _search_root(self, board, moves, depth) -> tuple:
        """Root-level search — returns (best_move, best_score)."""
        best_move  = None
        best_score = -INF
        alpha = -INF
        beta  =  INF
 
        for move in moves:
            if self._out_of_time() and depth > MIN_DEPTH:
                break
 
            child = board.forecast_move(move, check_ok=False)
            if child is None:
                continue
 
            child.reverse_perspective()
 
            score = -self._negamax(child, depth - 1, -beta, -alpha)
 
            if score > best_score:
                best_score = score
                best_move  = move
 
            alpha = max(alpha, score)
 
        return best_move, best_score
 
    def _negamax(self, board, depth: int, alpha: float, beta: float) -> float:
        """
        Negamax with alpha-beta pruning.
 
        After reverse_perspective(), player_worker is always the side to move.
        We maximize their score. The sign flip at each level handles the
        min/max alternation automatically.
 
        Returns the score from the perspective of the CURRENT player_worker.
        """
        self.nodes_searched += 1
 
        # Terminal conditions
        if board.is_game_over() or depth == 0:
            return evaluate(board, self.rat_belief)
 
        if self._out_of_time():
            return evaluate(board, self.rat_belief)
 
        moves = order_moves(board.get_valid_moves(exclude_search=True))
 
        if not moves:
            return evaluate(board, self.rat_belief)
 
        best = -INF
 
        for move in moves:
            child = board.forecast_move(move, check_ok=False)
            if child is None:
                continue
 
            child.reverse_perspective()
 
            score = -self._negamax(child, depth - 1, -beta, -alpha)
 
            best  = max(best, score)
            alpha = max(alpha, score)
 
            if alpha >= beta:
                break   # beta cutoff
 
        return best
 
    def _out_of_time(self) -> bool:
        return time.perf_counter() >= self.deadline
 
    def stats(self) -> str:
        elapsed = time.perf_counter() - self.start_time
        return (f"nodes={self.nodes_searched}, "
                f"time={elapsed:.3f}s, "
                f"depths={list(self.best_move_at_depth.keys())}")