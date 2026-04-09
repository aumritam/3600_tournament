"""
rat_belief.py  —  HMM for tracking the rat's position.

Calibrated exactly to the engine's rat.py and board.py:
  - Bitmasks: bit_index = y * 8 + x  (confirmed in board.py)
  - Noise enum: Noise.SQUEAK=0, Noise.SCRATCH=1, Noise.SQUEAL=2
  - Transition matrix T[i, j] = P(rat moves i -> j),  row-stochastic
  - Distance error offsets: (-1, 0, +1, +2) with probs (0.12, 0.70, 0.12, 0.06)
  - Noise probs per Cell type match NOISE_PROBS in rat.py exactly
  - Sensor is sampled from the PLAYER'S worker position (rat.py: estimate_distance)
  - board._primed_mask / _carpet_mask / _blocked_mask are the real attribute names
"""

import numpy as np
from game.enums import Cell, Noise, BOARD_SIZE

# ---------------------------------------------------------------------------
# Constants (mirrored from rat.py / enums.py)
# ---------------------------------------------------------------------------

NOISE_EMISSION = {
    #             squeak  scratch  squeal
    Cell.BLOCKED: np.array([0.50, 0.30, 0.20]),
    Cell.SPACE:   np.array([0.70, 0.15, 0.15]),
    Cell.PRIMED:  np.array([0.10, 0.80, 0.10]),
    Cell.CARPET:  np.array([0.10, 0.10, 0.80]),
}

DISTANCE_OFFSETS = (-1, 0, 1, 2)
DISTANCE_PROBS   = (0.12, 0.70, 0.12, 0.06)

N = BOARD_SIZE * BOARD_SIZE  # 64


# ---------------------------------------------------------------------------
# Index helpers  (matches board.py: bit_index = y * BOARD_SIZE + x)
# ---------------------------------------------------------------------------

def pos_to_idx(pos: tuple) -> int:
    return pos[1] * BOARD_SIZE + pos[0]


def idx_to_pos(idx: int) -> tuple:
    return (idx % BOARD_SIZE, idx // BOARD_SIZE)


def manhattan(a: tuple, b: tuple) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


# ---------------------------------------------------------------------------
# Prior: rat starts at (0,0) then walks T 1000 steps analytically
# ---------------------------------------------------------------------------

def compute_prior(T: np.ndarray) -> np.ndarray:
    """
    (T.T)^1000 @ e_{(0,0)}  via matrix repeated-squaring.
    T is row-stochastic: T[i,j] = P(i->j), so belief update is Tt @ b.
    """
    e0 = np.zeros(N, dtype=np.float64)
    e0[pos_to_idx((0, 0))] = 1.0

    Tt = T.T.astype(np.float64)
    result = e0.copy()
    mat = Tt.copy()
    n = 1000
    while n > 0:
        if n & 1:
            result = mat @ result
        mat = mat @ mat
        n >>= 1
    result = np.maximum(result, 0.0)
    result /= result.sum()
    return result


# ---------------------------------------------------------------------------
# Floor-type lookup using board's real bitmasks
# ---------------------------------------------------------------------------

def get_floor_type(board, idx: int) -> Cell:
    """
    Determine cell type for HMM emission using board bitmasks.
    Priority order matches board.get_cell(): primed > carpet > blocked > space.
    Bit layout: bit_index = y * BOARD_SIZE + x  (same as idx here).
    """
    bit = 1 << idx
    if board._primed_mask & bit:
        return Cell.PRIMED
    if board._carpet_mask & bit:
        return Cell.CARPET
    if board._blocked_mask & bit:
        return Cell.BLOCKED
    return Cell.SPACE


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class RatBelief:
    """
    Hidden Markov Model tracking the rat across the 8x8 board.

    Typical per-turn call from agent.py
    ------------------------------------
        noise, est_dist = sensor_data          # Noise enum + int
        rb.update(board, noise, est_dist,
                  board.player_worker.get_location(),
                  opp_guess=opp_loc, opp_found=opp_result)

        if rb.should_search():
            best_pos, ev = rb.best_search_target()
            return Move.search(best_pos)
    """

    def __init__(self, T: np.ndarray):
        """
        Parameters
        ----------
        T : np.ndarray  shape (64, 64)
            Row-stochastic transition matrix from the engine.
            May arrive as a JAX array -- we convert to numpy once here.
        """
        # JAX arrays arrive here; convert once and keep as numpy
        T = np.asarray(T, dtype=np.float64)
        assert T.shape == (N, N), f"Expected ({N},{N}), got {T.shape}"

        self.T  = T
        self.Tt = T.T.copy()          # belief update: b' = Tt @ b

        self.belief = compute_prior(T)

        # Cache per-cell floor-type emission vector (64 x 3).
        # Updated lazily whenever board state changes (primed/carpet cells).
        self._emission_cache = None
        self._cache_board_state = None  # (primed_mask, carpet_mask, blocked_mask)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self,
        board,
        noise: Noise,
        est_distance: int,
        worker_pos: tuple,
        opp_guess=None,
        opp_found=None,
    ):
        """
        Full HMM step for one turn:
          1. Predict   (rat moved before our turn)
          2. Observe   (noise + distance sensor)
          3. Incorporate opponent search result if available

        Parameters
        ----------
        board        : Board from engine
        noise        : sensor_data[0]  (Noise enum)
        est_distance : sensor_data[1]  (int)
        worker_pos   : board.player_worker.get_location()
        opp_guess    : board.opponent_search[0]  -- None if opponent didn't search
        opp_found    : board.opponent_search[1]  -- True if opponent caught the rat
        """
        self.predict()
        self.observe(board, noise, est_distance, worker_pos)

        if opp_guess is not None:
            if opp_found:
                # Rat caught -> new rat already walked 1000 steps, reset
                self.reset()
            else:
                # Opponent missed -> zero out that cell
                self.belief[pos_to_idx(opp_guess)] = 0.0
                self._safe_normalize()

    def predict(self):
        """Time update: propagate belief through transition model."""
        self.belief = self.Tt @ self.belief
        self._safe_normalize()

    def observe(self, board, noise: Noise, est_distance: int, worker_pos: tuple):
        """
        Measurement update: reweight by noise + distance likelihoods.
        Always call predict() before observe() each turn.
        """
        noise_lk = self._noise_likelihood(board, noise)
        dist_lk  = self._distance_likelihood(est_distance, worker_pos)
        self.belief *= noise_lk * dist_lk
        self._safe_normalize()

    def note_my_miss(self, pos: tuple):
        """Call this right after YOUR own failed search move."""
        self.belief[pos_to_idx(pos)] = 0.0
        self._safe_normalize()

    def reset(self):
        """Reset to prior after any rat capture (yours or opponent's)."""
        self.belief = compute_prior(self.T)
        self._emission_cache = None   # board state may have changed

    # ------------------------------------------------------------------
    # Query interface
    # ------------------------------------------------------------------

    def search_ev(self, pos: tuple) -> float:
        """
        Expected value of searching pos.
        EV = 4*P + (-2)*(1-P) = 6P - 2.   Positive when P > 1/3.
        """
        return 6.0 * float(self.belief[pos_to_idx(pos)]) - 2.0

    def best_search_target(self):
        """Return (pos, ev) for the cell with highest search EV."""
        best_idx = int(np.argmax(self.belief))
        best_pos = idx_to_pos(best_idx)
        return best_pos, self.search_ev(best_pos)

    def should_search(self, min_ev: float = 0.0) -> bool:
        """True if best-cell EV > min_ev.  Default: any +EV cell (P > 1/3)."""
        _, ev = self.best_search_target()
        return ev > min_ev

    def top_k_targets(self, k: int = 5) -> list:
        """
        Top-k (pos, ev) pairs sorted by EV descending.
        Useful for passing search candidates into minimax.
        """
        k = min(k, N)
        indices = np.argpartition(self.belief, -k)[-k:]
        indices = indices[np.argsort(self.belief[indices])[::-1]]
        return [(idx_to_pos(int(i)), self.search_ev(idx_to_pos(int(i)))) for i in indices]

    def belief_at(self, pos: tuple) -> float:
        return float(self.belief[pos_to_idx(pos)])

    def belief_grid(self) -> np.ndarray:
        """8x8 view of the belief distribution -- useful for debugging."""
        return self.belief.reshape(BOARD_SIZE, BOARD_SIZE)

    # ------------------------------------------------------------------
    # Internal: likelihoods
    # ------------------------------------------------------------------

    def _noise_likelihood(self, board, noise: Noise) -> np.ndarray:
        """
        64-element vector: L[i] = P(noise | floor_type at cell i).
        Uses a cached (64, 3) emission matrix rebuilt when board state changes.
        """
        state_key = (board._primed_mask, board._carpet_mask, board._blocked_mask)
        if self._cache_board_state != state_key:
            self._rebuild_emission_cache(board)
            self._cache_board_state = state_key

        noise_idx = int(noise)   # Noise.SQUEAK=0, SCRATCH=1, SQUEAL=2
        return self._emission_cache[:, noise_idx]

    def _rebuild_emission_cache(self, board):
        """Build (64, 3) emission matrix from current board state."""
        cache = np.empty((N, 3), dtype=np.float64)
        for idx in range(N):
            cell_type = get_floor_type(board, idx)
            cache[idx] = NOISE_EMISSION[cell_type]
        self._emission_cache = cache

    def _distance_likelihood(self, est_distance: int, worker_pos: tuple) -> np.ndarray:
        """
        64-element vector: L[i] = P(est_distance reported | rat at cell i).

        Error model from rat.py:
            reported = max(0, actual + offset)
            offsets (-1, 0, +1, +2) with probs (0.12, 0.70, 0.12, 0.06)

        Clipping means multiple (actual, offset) pairs can produce the same
        reported value -- we sum all their probabilities.
        """
        lk = np.zeros(N, dtype=np.float64)
        for idx in range(N):
            pos = idx_to_pos(idx)
            actual = manhattan(pos, worker_pos)
            p = 0.0
            for offset, prob in zip(DISTANCE_OFFSETS, DISTANCE_PROBS):
                reported = actual + offset
                if reported < 0:
                    reported = 0   # engine clips negatives (rat.py line: d if d > 0 else 0)
                if reported == est_distance:
                    p += prob
            lk[idx] = p
        return lk

    def _safe_normalize(self):
        total = self.belief.sum()
        if total > 1e-15:
            self.belief /= total
        else:
            # Complete information collapse -- fall back to uniform
            self.belief = np.ones(N, dtype=np.float64) / N
