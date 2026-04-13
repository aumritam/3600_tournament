"""
Hybrid player agent.

What it keeps from your version:
- strong rat HMM
- search as fallback, not the main personality

What it takes from your friend's version:
- carpet-first opening
- committed big-line behavior
- stricter rat-search discipline
- less pointless late-game plains
"""

import random
from collections.abc import Callable
from typing import Optional, Tuple

import numpy as np

from game import board
from game.enums import BOARD_SIZE, MoveType, Direction, Cell
from game.move import Move

from .rat_belief import RatBelief
from .search import (
    iterative_deepening,
    prioritize_moves,
    best_greedy_carpet,
    trail_length_behind,
    runway_length,
)

TIME_RESERVE = 8.0
MAX_DEPTH = 4

EARLY_CUTOFF = 28
MID_CUTOFF = 10

RAT_THRESHOLD = 0.75
MAX_SAME_CELL = 1
MIN_RUNWAY = 1
CUTOFF_BONUS = 4
FORCE_CARPET_LEN = 2
FORCE_CARPET_RUNWAY = 1


class PlayerAgent:
    def __init__(self, board_state, transition_matrix=None, time_left: Callable = None):
        if transition_matrix is not None:
            T = np.array(transition_matrix, dtype=np.float64)
        else:
            n = BOARD_SIZE * BOARD_SIZE
            T = np.ones((n, n), dtype=np.float64) / n

        self.rat_belief = RatBelief(T)
        self.turns_played = 0
        self.searches_made = 0
        self.prime_direction: Optional[Direction] = None
        self.last_search_loc = None
        self.search_same_count = 0

    def commentate(self) -> str:
        best_loc, _ = self.rat_belief.best_search_target()
        best_prob = self.rat_belief.belief_at(best_loc)
        return (
            f"Turns played: {self.turns_played} | "
            f"Searches made: {self.searches_made} | "
            f"Prime direction: {getattr(self.prime_direction, 'name', None)} | "
            f"Best rat belief: {best_loc} @ {best_prob:.2%}"
        )

    def play(self, board_state: board.Board, sensor_data: Tuple, time_left: Callable) -> Move:
        self.turns_played += 1
        noise, est_distance = sensor_data

        my_loc = board_state.player_worker.get_location()
        opp_loc = board_state.opponent_worker.get_location()
        turns_left = board_state.player_worker.turns_left
        my_pts = board_state.player_worker.get_points()
        opp_pts = board_state.opponent_worker.get_points()
        leading = my_pts > opp_pts

        opp_guess, opp_found = board_state.opponent_search
        self.rat_belief.update(
            board_state,
            noise,
            est_distance,
            my_loc,
            opp_guess=opp_guess,
            opp_found=opp_found,
        )

        my_last_loc, my_last_hit = board_state.player_search
        if my_last_loc is not None and not my_last_hit:
            self.rat_belief.note_my_miss(my_last_loc)

        best_loc, _ = self.rat_belief.best_search_target()
        best_prob = self.rat_belief.belief_at(best_loc)

        if best_loc == self.last_search_loc:
            self.search_same_count += 1
        else:
            self.search_same_count = 0

        dist_me = abs(my_loc[0] - best_loc[0]) + abs(my_loc[1] - best_loc[1])
        dist_opp = abs(opp_loc[0] - best_loc[0]) + abs(opp_loc[1] - best_loc[1])

        can_search = (
            best_prob >= RAT_THRESHOLD
            and self.search_same_count < MAX_SAME_CELL
            and not (leading and turns_left <= MID_CUTOFF)
            and dist_me <= dist_opp
        )
        if can_search:
            self.last_search_loc = best_loc
            self.searches_made += 1
            return Move.search(best_loc)

        forced_carpet = self._forced_carpet_if_dead_end(board_state)
        if forced_carpet is not None:
            self._update_direction_after_carpet(forced_carpet)
            return forced_carpet

        if turns_left > EARLY_CUTOFF:
            greedy = best_greedy_carpet(board_state, min_length=5)
        elif turns_left > MID_CUTOFF:
            greedy = best_greedy_carpet(board_state, min_length=4)
        else:
            greedy = best_greedy_carpet(board_state, min_length=2)

        if greedy is not None:
            self._update_direction_after_carpet(greedy)
            return greedy

        if turns_left > EARLY_CUTOFF:
            move = self._early_phase(board_state, opp_loc)
        elif turns_left > MID_CUTOFF:
            move = self._mid_phase(board_state, time_left)
        else:
            move = self._late_phase(board_state, time_left)

        if move is not None:
            return move

        valid = board_state.get_valid_moves(enemy=False, exclude_search=True)
        return valid[0] if valid else Move.search(best_loc)

    def _early_phase(self, board_state, opp_loc) -> Optional[Move]:
        valid = board_state.get_valid_moves(enemy=False, exclude_search=True)

        forced_carpet = self._forced_carpet_if_dead_end(board_state)
        if forced_carpet is not None:
            self._update_direction_after_carpet(forced_carpet)
            return forced_carpet

        primes = {m.direction: m for m in valid if m.move_type == MoveType.PRIME}

        if self.prime_direction in primes:
            runway = runway_length(board_state, self.prime_direction)
            trail = trail_length_behind(board_state, self.prime_direction)
            if trail < 6 and runway >= MIN_RUNWAY:
                return primes[self.prime_direction]
            self.prime_direction = None

        best_dir = None
        best_score = -1e9
        my_loc = board_state.player_worker.get_location()

        for direction, move in primes.items():
            runway = runway_length(board_state, direction)
            if runway < MIN_RUNWAY:
                continue

            trail = trail_length_behind(board_state, direction)
            cutoff = 0
            dx = opp_loc[0] - my_loc[0]
            dy = opp_loc[1] - my_loc[1]
            if direction == Direction.RIGHT and dx > 0:
                cutoff = CUTOFF_BONUS
            elif direction == Direction.LEFT and dx < 0:
                cutoff = CUTOFF_BONUS
            elif direction == Direction.DOWN and dy > 0:
                cutoff = CUTOFF_BONUS
            elif direction == Direction.UP and dy < 0:
                cutoff = CUTOFF_BONUS

            score = 2.0 * runway + 4.0 * trail + cutoff
            if score > best_score:
                best_score = score
                best_dir = direction

        if best_dir is not None:
            self.prime_direction = best_dir
            return primes[best_dir]
        
        return None

    def _mid_phase(self, board_state, time_left) -> Optional[Move]:
        valid = board_state.get_valid_moves(enemy=False, exclude_search=True)
        moves = prioritize_moves(board_state, valid, self.prime_direction)

        forced_carpet = self._forced_carpet_if_dead_end(board_state)
        if forced_carpet is not None:
            self._update_direction_after_carpet(forced_carpet)
            return forced_carpet

        prime_moves = [m for m in moves if m.move_type == MoveType.PRIME]
        if self.prime_direction is not None:
            for m in prime_moves:
                if m.direction == self.prime_direction:
                    runway = runway_length(board_state, m.direction)
                    trail = trail_length_behind(board_state, m.direction)
                    if runway >= MIN_RUNWAY and trail < 7:
                        return m
                    
        forced_carpet = self._forced_carpet_if_dead_end(board_state)
        if forced_carpet is not None:
            self._update_direction_after_carpet(forced_carpet)
            return forced_carpet

        viable_primes = [
        m for m in prime_moves
        if runway_length(board_state, m.direction) >= MIN_RUNWAY
        ]

        if viable_primes:
            viable_primes.sort(
                key=lambda m: 4.0 * trail_length_behind(board_state, m.direction)
                    + 1.0 * runway_length(board_state, m.direction)
                    + (3.5 if self.prime_direction is not None and m.direction == self.prime_direction else 0.0),
                reverse=True,
            )
            self.prime_direction = viable_primes[0].direction
            return viable_primes[0]

        return self._tree_fallback(board_state, time_left)

    def _late_phase(self, board_state, time_left) -> Optional[Move]:
        valid = board_state.get_valid_moves(enemy=False, exclude_search=True)

        carpet_moves = [m for m in valid if m.move_type == MoveType.CARPET]
        if carpet_moves:
            carpet_moves.sort(key=lambda m: m.roll_length, reverse=True)
            self._update_direction_after_carpet(carpet_moves[0])
            return carpet_moves[0]

        steal_prime = self._prime_into_enemy_chain(board_state, valid)
        if steal_prime is not None:
            self.prime_direction = steal_prime.direction
            return steal_prime

        prime_moves = [m for m in valid if m.move_type == MoveType.PRIME]
        if prime_moves:
            prime_moves.sort(
                key=lambda m: trail_length_behind(board_state, m.direction) + 0.8 * runway_length(board_state, m.direction),
                reverse=True,
            )
            self.prime_direction = prime_moves[0].direction
            return prime_moves[0]

        return self._tree_fallback(board_state, time_left)
    
    def _forced_carpet_if_dead_end(self, board_state) -> Optional[Move]:
        valid = board_state.get_valid_moves(enemy=False, exclude_search=True)
        carpet_moves = [m for m in valid if m.move_type == MoveType.CARPET]
        if not carpet_moves or self.prime_direction is None:
            return None

        carpet_moves.sort(key=lambda m: m.roll_length, reverse=True)
        best_carpet = carpet_moves[0]

        trail = trail_length_behind(board_state, self.prime_direction)
        runway = runway_length(board_state, self.prime_direction)

        valid_prime_moves = [m for m in valid if m.move_type == MoveType.PRIME]
        valid_prime_dirs = {m.direction for m in valid_prime_moves}

        next_pos = _step(board_state.player_worker.get_location(), self.prime_direction)
        hits_carpet = _in_bounds(next_pos) and board_state.get_cell(next_pos) == Cell.CARPET
        hits_primed = _in_bounds(next_pos) and board_state.get_cell(next_pos) == Cell.PRIMED

        dead_end = (
            runway <= FORCE_CARPET_RUNWAY
            or hits_carpet
            or hits_primed
        )

        must_turn = self.prime_direction not in valid_prime_dirs

        if trail < FORCE_CARPET_LEN:
            return None

        my_loc = board_state.player_worker.get_location()
        bit = 1 << (my_loc[1] * BOARD_SIZE + my_loc[0])
        standing_on_space = not (
            board_state._primed_mask & bit
            or board_state._carpet_mask & bit
            or board_state._blocked_mask & bit
        )

        # CASE 1: Wall / carpet / blocked forward
        if dead_end:
            if standing_on_space and self.prime_direction in valid_prime_dirs:
                return None  # prime one more tile
            return best_carpet

        # CASE 2: Must turn → cash out chains of 3+
        if must_turn:
            if trail >= FORCE_CARPET_LEN:
                if standing_on_space and runway > 0:
                    return None
                return best_carpet
            return None

        return None

    def _tree_fallback(self, board_state, time_left) -> Optional[Move]:
        turns_left = board_state.player_worker.turns_left
        t_left = time_left()
        time_per_turn = max(0.0, (t_left - TIME_RESERVE) / max(turns_left, 1))

        if time_per_turn > 4.0:
            depth = MAX_DEPTH
        elif time_per_turn > 1.5:
            depth = 2
        else:
            depth = 1

        moves = board_state.get_valid_moves(enemy=False, exclude_search=True)
        moves = prioritize_moves(board_state, moves, self.prime_direction)
        if not moves:
            return None

        best_move = iterative_deepening(board_state, self.rat_belief, depth, time_left, moves)
        if best_move is None:
            best_move = random.choice(moves)

        if best_move.move_type == MoveType.PRIME:
            self.prime_direction = best_move.direction
        elif best_move.move_type == MoveType.CARPET:
            self._update_direction_after_carpet(best_move)
        return best_move

    def _prime_into_enemy_chain(self, board_state, valid_moves) -> Optional[Move]:
        prime_moves = [m for m in valid_moves if m.move_type == MoveType.PRIME]
        best = None
        best_score = -1e9
        for m in prime_moves:
            nxt = _step(board_state.player_worker.get_location(), m.direction)
            score = 0.0
            for side in _perpendiculars(m.direction):
                adj = _step(nxt, side)
                if _in_bounds(adj) and board_state.get_cell(adj) == Cell.PRIMED:
                    score += 2.0
            score += 0.5 * runway_length(board_state, m.direction)
            if score > best_score:
                best_score = score
                best = m
        return best if best_score > 0 else None

    def _plain_toward_opponent_score(self, board_state, direction: Direction) -> float:
        my_loc = board_state.player_worker.get_location()
        opp_loc = board_state.opponent_worker.get_location()
        nxt = _step(my_loc, direction)
        return -(abs(nxt[0] - opp_loc[0]) + abs(nxt[1] - opp_loc[1]))

    def _update_direction_after_carpet(self, move: Move):
        if move.move_type == MoveType.CARPET:
            self.prime_direction = move.direction


def _step(loc, direction: Direction):
    x, y = loc
    if direction == Direction.UP:
        return (x, y - 1)
    if direction == Direction.DOWN:
        return (x, y + 1)
    if direction == Direction.LEFT:
        return (x - 1, y)
    return (x + 1, y)


def _perpendiculars(direction: Direction):
    if direction in (Direction.UP, Direction.DOWN):
        return [Direction.LEFT, Direction.RIGHT]
    return [Direction.UP, Direction.DOWN]


def _in_bounds(loc) -> bool:
    return 0 <= loc[0] < BOARD_SIZE and 0 <= loc[1] < BOARD_SIZE
