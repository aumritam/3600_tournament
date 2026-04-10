"""
agent.py  —  main entry point for the CS3600 carpet game.
 
Pipeline each turn:
  1. HMM update  (rat_belief)
  2. Search decision  — is a rat search +EV enough to skip the tree?
  3. Minimax search   (search.py) using heuristic.py + carpet_planner.py
  4. Return best move
"""
 
from collections.abc import Callable
from typing import Tuple
 
from game.board import Board
from game.move import Move
from game.enums import MoveType
 
from .rat_belief import RatBelief
from .search import Searcher
 
# Search EV threshold to skip minimax and just search for the rat.
# If the best cell has EV above this, it's so good we just take it.
# 2.0 means P > 2/3 — very confident the rat is there.
RAT_SEARCH_THRESHOLD = 2.0
 
 
class PlayerAgent:
 
    def __init__(
        self,
        board: Board,
        transition_matrix=None,
        time_left: Callable = None,
    ):
        # HMM — compute_prior runs T^1000 here (~0.15ms)
        self.rat_belief = RatBelief(transition_matrix)
 
        # Minimax searcher — stateless across turns, reused each call
        self.searcher = Searcher()
 
        # Turn counter for commentate
        self.my_turn = 0
 
    def play(
        self,
        board: Board,
        sensor_data: Tuple,
        time_left: Callable,
    ) -> Move:
        self.my_turn += 1
 
        # ---- 1. Unpack sensor data ------------------------------------
        noise, est_distance = sensor_data
        opp_loc, opp_found  = board.opponent_search
 
        # ---- 2. HMM update --------------------------------------------
        self.rat_belief.update(
            board,
            noise,
            est_distance,
            board.player_worker.get_location(),
            opp_guess=opp_loc,
            opp_found=opp_found,
        )
 
        # Zero out our own last missed search
        my_last_loc, my_last_hit = board.player_search
        if my_last_loc is not None and not my_last_hit:
            self.rat_belief.note_my_miss(my_last_loc)
 
        # ---- 3. High-confidence rat search ----------------------------
        # If we're very sure where the rat is, just take it — don't waste
        # minimax time deliberating on an obvious +EV play
        best_rat_pos, best_rat_ev = self.rat_belief.best_search_target()
        if best_rat_ev >= RAT_SEARCH_THRESHOLD:
            return Move.search(best_rat_pos)
 
        # ---- 4. Minimax search ----------------------------------------
        move, score = self.searcher.search(board, self.rat_belief, time_left)
 
        # ---- 5. Fallback ----------------------------------------------
        # Should never happen, but returning None = instant loss
        if move is None:
            valid = board.get_valid_moves(exclude_search=False)
            return valid[0] if valid else Move.search((0, 0))
 
        # ---- 6. Check if a moderate rat search beats the tree's choice -
        # If the tree returned a non-search move but the rat EV is decent,
        # prefer the search. Threshold lower than above since tree already
        # considered carpeting alternatives.
        if move.move_type != MoveType.SEARCH:
            if best_rat_ev >= 0.0 and self.rat_belief.should_search():
                # Only override if rat EV meaningfully beats tree score
                # (avoids wasteful searches when we're mid-line)
                turns_left = board.player_worker.turns_left
                # Late game: be more willing to search (fewer carpet turns left)
                rat_override_threshold = 1.0 if turns_left > 15 else 0.0
                if best_rat_ev >= rat_override_threshold:
                    return Move.search(best_rat_pos)
 
        return move
 
    def commentate(self) -> str:
        best_pos, best_ev = self.rat_belief.best_search_target()
        depths = list(self.searcher.best_move_at_depth.keys())
        return (
            f"Turn {self.my_turn} done. "
            f"Search depths reached: {depths}. "
            f"Rat peak belief at {best_pos}, EV={best_ev:.2f}. "
            f"Last search stats: {self.searcher.stats()}"
        )