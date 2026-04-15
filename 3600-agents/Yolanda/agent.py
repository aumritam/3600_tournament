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
#-----------------------------------------------------------------------------

from collections.abc import Callable
from typing import Optional, Tuple

import numpy as np

from game import board
from game.enums import BOARD_SIZE, MoveType, Direction, Cell, CARPET_POINTS_TABLE
from game.move import Move

from .rat_belief import RatBelief

from .search import (
    trail_length_behind,
    runway_carpet_length,
    runway_prime_points,
    _step
)

TIME_RESERVE = 8.0
MAX_DEPTH = 3

class PlayerAgent:
    def __init__(self, board_state, transition_matrix=None, time_left: Callable = None):
        if transition_matrix is not None:
            T = np.array(transition_matrix, dtype=np.float64)
        else:
            n = BOARD_SIZE * BOARD_SIZE
            T = np.ones((n, n), dtype=np.float64) / n

        self.rat_belief = RatBelief(T)
        self.turns_played = 0
        self.target_line = None  
        self.last_pos = None

    def play(self, board_state: board.Board, sensor_data: Tuple, time_left: Callable) -> Move:
        self.turns_played += 1
        noise, est_distance = sensor_data

        my_loc = board_state.player_worker.get_location()
        opp_loc = board_state.opponent_worker.get_location()
        turns_left = board_state.player_worker.turns_left

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

        #Currently Ranking All Possible Moves And Picking The Top 1
        ranked = self._rank_moves(board_state, my_loc, opp_loc, turns_left)
        print(ranked)

        if ranked:
            self.last_pos = my_loc
            return ranked[0][1]  #Picking the top 1
        
        valid = board_state.get_valid_moves(enemy=False, exclude_search=True)
        return valid[0] if valid else Move.search((0, 0))
    
    def _rank_moves(self, board_state, my_loc, opp_loc, turns_left):
        # TODO: Once all moves are scored on standardized EV scale,
        # run tree search on top N candidates to re-rank based on
        # lookahead. Tree search will handle multi-move sequences
        # like plain stepping over carpet to reach new priming areas.

        my_pts = board_state.player_worker.get_points()
        opp_pts = board_state.opponent_worker.get_points()

        valid = board_state.get_valid_moves(enemy=False, exclude_search=True) #excludes search moves (check if valid)
        print(f"Valid moves: {valid}")

        best_carpets = {}
        for move in valid:
            if move.move_type == MoveType.CARPET:
                if move.direction not in best_carpets or move.roll_length > best_carpets[move.direction].roll_length:
                    best_carpets[move.direction] = move
    
        non_carpets = [m for m in valid if m.move_type != MoveType.CARPET]
        valid = non_carpets + list(best_carpets.values())

        scored = []
        for move in valid:
            score = self._score_move(board_state, move, my_loc, opp_loc, turns_left, my_pts, opp_pts)
            scored.append((score, move))

        best_loc, _ = self.rat_belief.best_search_target()
        search_score = self._score_search(board_state, best_loc, my_loc, opp_loc, turns_left, my_pts, opp_pts)
        scored.append((search_score, Move.search(best_loc)))

        scored.sort(key=lambda x: x[0], reverse=True)
        return scored


    def _score_move(self, board_state, move, my_loc, opp_loc, turns_left, my_pts, opp_pts):
        if move.move_type == MoveType.CARPET:
            return self._score_carpet(board_state, move, my_loc, opp_loc, turns_left, my_pts, opp_pts)
        elif move.move_type == MoveType.PRIME:
            return self._score_prime(board_state, move, my_loc, opp_loc, turns_left, my_pts, opp_pts)
        elif move.move_type == MoveType.PLAIN:
            return self._score_plain(board_state, move, my_loc, opp_loc, turns_left, my_pts, opp_pts)
        return -999.0
    
    # Basic straight table number
    def _score_carpet(self, board_state, move, my_loc, opp_loc, turns_left, my_pts, opp_pts):
        return float(CARPET_POINTS_TABLE.get(move.roll_length, 0))
    
    def _score_prime(self, board_state, move, my_loc, opp_loc, turns_left, my_pts, opp_pts):
        runway = runway_carpet_length(board_state, move.direction)
        runway_pts = runway_prime_points(board_state, move.direction)
        trail = trail_length_behind(board_state, move.direction)
        
        potential_length = min(trail + runway, turns_left)
        carpet_value = float(CARPET_POINTS_TABLE.get(potential_length, 0))
        trail_bonus = trail * 3.0
        score = (carpet_value + float(runway_pts)) * 0.8 + trail_bonus
        
        return score
    
    #Def might need to tweek later
    def _score_plain(self, board_state, move, my_loc, opp_loc, turns_left, my_pts, opp_pts):
        new_pos = _step(my_loc, move.direction)
        
        # Check if we're genuinely trapped with no open adjacent cells
        valid = board_state.get_valid_moves(enemy=False, exclude_search=True)
        prime_moves = [m for m in valid if m.move_type == MoveType.PRIME]
        plain_to_space = [m for m in valid if m.move_type == MoveType.PLAIN 
                        and board_state.get_cell(_step(my_loc, m.direction)) == Cell.SPACE]
        
        genuinely_trapped = not prime_moves and not plain_to_space
        
        # If standing on carpet, priority is to escape to open space for priming
        if board_state.get_cell(my_loc) == Cell.CARPET:
            if board_state.get_cell(new_pos) == Cell.SPACE:
                return 5.0  # escape carpet to resume priming
            if board_state.get_cell(new_pos) == Cell.CARPET:
                if genuinely_trapped:
                    nearest_open = self._nearest_open_cell(board_state, new_pos)
                    if nearest_open is not None:
                        dist = abs(new_pos[0] - nearest_open[0]) + abs(new_pos[1] - nearest_open[1])
                        return 3.0 + (1.0 / max(dist, 1))
                    return 3.0
                return -2.0
        
        # Penalize stepping onto carpet
        if board_state.get_cell(new_pos) == Cell.CARPET:
            if genuinely_trapped:
                nearest_open = self._nearest_open_cell(board_state, new_pos)
                if nearest_open is not None:
                    dist = abs(new_pos[0] - nearest_open[0]) + abs(new_pos[1] - nearest_open[1])
                    return 3.0 + (1.0 / max(dist, 1))
                return 3.0
            if self.last_pos is not None and new_pos == self.last_pos:
                return -8.0
            return -1.0
        
        # Base penalty for plain move
        return -0.5
    
    def _score_search(self, board_state, best_loc, my_loc, opp_loc, turns_left, my_pts, opp_pts):
        best_prob = self.rat_belief.belief_at(best_loc)
        ev = 6.0 * best_prob - 2.0
        return ev
    
    def _nearest_open_cell(self, board_state, pos):
        best = None
        best_dist = 999
        for y in range(BOARD_SIZE):
            for x in range(BOARD_SIZE):
                if board_state.get_cell((x, y)) == Cell.SPACE:
                    d = abs(pos[0] - x) + abs(pos[1] - y)
                    if d < best_dist:
                        best_dist = d
                        best = (x, y)
        return best

#--------------------------------------------------------------------------------
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
MAX_DEPTH = 3

EARLY_CUTOFF = 28
MID_CUTOFF = 10

RAT_THRESHOLD = 0.75
MAX_SAME_CELL = 1
MIN_RUNWAY = 1
CUTOFF_BONUS = 4
FORCE_CARPET_LEN = 3
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

        can_search = (
            best_prob >= RAT_THRESHOLD
            and self.search_same_count < MAX_SAME_CELL
            and not (leading and turns_left <= MID_CUTOFF)
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
            greedy = best_greedy_carpet(board_state, min_length=3)

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

"""