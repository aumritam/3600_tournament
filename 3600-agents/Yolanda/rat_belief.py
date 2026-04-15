"""
Hybrid rat belief model.

Keeps the engine-aligned logic from the stronger version, with one small
addition: a lightweight copy() helper so search can advance belief in
hypothetical future turns.
"""

#-----------------------------------------------------------------------------

import numpy as np
from game.enums import Cell, Noise, BOARD_SIZE

NOISE_EMISSION = {
    Cell.BLOCKED: np.array([0.50, 0.30, 0.20]),
    Cell.SPACE:   np.array([0.70, 0.15, 0.15]),
    Cell.PRIMED:  np.array([0.10, 0.80, 0.10]),
    Cell.CARPET:  np.array([0.10, 0.10, 0.80]),
}

DISTANCE_OFFSETS = (-1, 0, 1, 2)
DISTANCE_PROBS   = (0.12, 0.70, 0.12, 0.06)

N = BOARD_SIZE * BOARD_SIZE

def compute_prior(T: np.ndarray) -> np.ndarray:
    e0 = np.zeros(N, dtype=np.float64)
    e0[0] = 1.0

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
    total = result.sum()
    if total > 0:
        result /= total
    else:
        result[:] = 1.0 / N
    return result

def get_floor_type(board, idx: int) -> Cell:
    bit = 1 << idx
    if board._primed_mask & bit:
        return Cell.PRIMED
    if board._carpet_mask & bit:
        return Cell.CARPET
    if board._blocked_mask & bit:
        return Cell.BLOCKED
    return Cell.SPACE

def pos_to_idx(pos: tuple) -> int:
    return pos[1] * BOARD_SIZE + pos[0]

def idx_to_pos(idx: int) -> tuple:
    return (idx % BOARD_SIZE, idx // BOARD_SIZE)

def manhattan(a: tuple, b: tuple) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

class RatBelief:
    def __init__(self, T: np.ndarray):
        T = np.asarray(T, dtype=np.float64)
        self.T = T
        self.Tt = T.T.copy()
        self.belief = compute_prior(T)
        self._emission_cache = None
        self._cache_board_state = (-1, -1, -1)

    def update(self, board, noise, est_distance, worker_pos, opp_guess=None, opp_found=None):
        self.predict()

        self.observe(board, noise, est_distance, worker_pos)

        if opp_guess is not None:
            if opp_found:
                self.reset()
            else:
                self.belief[pos_to_idx(opp_guess)] = 0.0
                self._safe_normalize()

    def predict(self):
        self.belief = self.Tt @ self.belief
        self._safe_normalize()

    def _safe_normalize(self):
        total = self.belief.sum()
        if total > 1e-15:
            self.belief /= total
        else:
            self.belief = np.ones(N, dtype=np.float64) / N

    def observe(self, board, noise, est_distance, worker_pos):
        noise_lk = self._noise_likelihood(board, noise)
        dist_lk = self._distance_likelihood(est_distance, worker_pos)
        self.belief *= noise_lk * dist_lk
        self._safe_normalize()

    def _noise_likelihood(self, board, noise):
        state_key = (board._primed_mask, board._carpet_mask, board._blocked_mask)
        if self._cache_board_state != state_key:
            self._rebuild_emission_cache(board)
            self._cache_board_state = state_key
        return self._emission_cache[:, int(noise)]
    
    def _rebuild_emission_cache(self, board):
        cache = np.empty((N, 3), dtype=np.float64)
        for idx in range(N):
            cache[idx] = NOISE_EMISSION[get_floor_type(board, idx)]
        self._emission_cache = cache

    def _distance_likelihood(self, est_distance: int, worker_pos: tuple) -> np.ndarray:
        lk = np.zeros(N, dtype=np.float64)
        for idx in range(N):
            actual = manhattan(idx_to_pos(idx), worker_pos)
            p = 0.0
            for offset, prob in zip(DISTANCE_OFFSETS, DISTANCE_PROBS):
                reported = actual + offset
                if reported < 0:
                    reported = 0
                if reported == est_distance:
                    p += prob
            lk[idx] = p
        return lk
    
    def reset(self):
        self.belief = compute_prior(self.T)
        self._emission_cache = None
        self._cache_board_state = (-1, -1, -1)

    def note_my_miss(self, pos: tuple):
        self.belief[pos_to_idx(pos)] = 0.0
        self._safe_normalize()

    def belief_at(self, pos: tuple) -> float:
        return float(self.belief[pos_to_idx(pos)])
    
    def best_search_target(self):
        best_idx = int(np.argmax(self.belief))
        best_pos = idx_to_pos(best_idx)
        return best_pos, self.search_ev(best_pos)
    
    def search_ev(self, pos: tuple) -> float:
        p = float(self.belief[pos_to_idx(pos)])
        return p * 4.0 - (1.0 - p) * 2.0
    

#-----------------------------------------------------------------------------

"""
def pos_to_idx(pos: tuple) -> int:
    return pos[1] * BOARD_SIZE + pos[0]


def idx_to_pos(idx: int) -> tuple:
    return (idx % BOARD_SIZE, idx // BOARD_SIZE)


def manhattan(a: tuple, b: tuple) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def compute_prior(T: np.ndarray) -> np.ndarray:
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
    total = result.sum()
    if total > 0:
        result /= total
    else:
        result[:] = 1.0 / N
    return result


def get_floor_type(board, idx: int) -> Cell:
    bit = 1 << idx
    if board._primed_mask & bit:
        return Cell.PRIMED
    if board._carpet_mask & bit:
        return Cell.CARPET
    if board._blocked_mask & bit:
        return Cell.BLOCKED
    return Cell.SPACE


class RatBelief:
    def __init__(self, T: np.ndarray):
        T = np.asarray(T, dtype=np.float64)
        assert T.shape == (N, N), f"Expected ({N},{N}), got {T.shape}"

        self.T = T
        self.Tt = T.T.copy()
        self.belief = compute_prior(T)
        self._emission_cache = None
        self._cache_board_state = (-1, -1, -1)

    def copy(self):
        new_rb = RatBelief.__new__(RatBelief)
        new_rb.T = self.T
        new_rb.Tt = self.Tt
        new_rb.belief = self.belief.copy()
        new_rb._emission_cache = None if self._emission_cache is None else self._emission_cache.copy()
        new_rb._cache_board_state = self._cache_board_state
        return new_rb

    def update(self, board, noise: Noise, est_distance: int, worker_pos: tuple,
               opp_guess=None, opp_found=None):
        self.predict()
        self.observe(board, noise, est_distance, worker_pos)

        if opp_guess is not None:
            if opp_found:
                self.reset()
            else:
                self.belief[pos_to_idx(opp_guess)] = 0.0
                self._safe_normalize()

    def predict(self):
        self.belief = self.Tt @ self.belief
        self._safe_normalize()

    def observe(self, board, noise: Noise, est_distance: int, worker_pos: tuple):
        noise_lk = self._noise_likelihood(board, noise)
        dist_lk = self._distance_likelihood(est_distance, worker_pos)
        self.belief *= noise_lk * dist_lk
        self._safe_normalize()

    def note_my_miss(self, pos: tuple):
        self.belief[pos_to_idx(pos)] = 0.0
        self._safe_normalize()

    def reset(self):
        self.belief = compute_prior(self.T)
        self._emission_cache = None
        self._cache_board_state = (-1, -1, -1)

    def search_ev(self, pos: tuple) -> float:
        p = float(self.belief[pos_to_idx(pos)])
        return p * 4.0 - (1.0 - p) * 2.0

    def best_search_target(self):
        best_idx = int(np.argmax(self.belief))
        best_pos = idx_to_pos(best_idx)
        return best_pos, self.search_ev(best_pos)

    def best_search_probability(self) -> float:
        best_pos, _ = self.best_search_target()
        return self.belief_at(best_pos)

    def belief_at(self, pos: tuple) -> float:
        return float(self.belief[pos_to_idx(pos)])

    def top_k_targets(self, k: int = 5) -> list:
        k = min(k, N)
        indices = np.argpartition(self.belief, -k)[-k:]
        indices = indices[np.argsort(self.belief[indices])[::-1]]
        return [(idx_to_pos(int(i)), float(self.belief[int(i)])) for i in indices]

    def _noise_likelihood(self, board, noise: Noise) -> np.ndarray:
        state_key = (board._primed_mask, board._carpet_mask, board._blocked_mask)
        if self._cache_board_state != state_key:
            self._rebuild_emission_cache(board)
            self._cache_board_state = state_key
        return self._emission_cache[:, int(noise)]

    def _rebuild_emission_cache(self, board):
        cache = np.empty((N, 3), dtype=np.float64)
        for idx in range(N):
            cache[idx] = NOISE_EMISSION[get_floor_type(board, idx)]
        self._emission_cache = cache

    def _distance_likelihood(self, est_distance: int, worker_pos: tuple) -> np.ndarray:
        lk = np.zeros(N, dtype=np.float64)
        for idx in range(N):
            actual = manhattan(idx_to_pos(idx), worker_pos)
            p = 0.0
            for offset, prob in zip(DISTANCE_OFFSETS, DISTANCE_PROBS):
                reported = actual + offset
                if reported < 0:
                    reported = 0
                if reported == est_distance:
                    p += prob
            lk[idx] = p
        return lk

    def _safe_normalize(self):
        total = self.belief.sum()
        if total > 1e-15:
            self.belief /= total
        else:
            self.belief = np.ones(N, dtype=np.float64) / N

"""