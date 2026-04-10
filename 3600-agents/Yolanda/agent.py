"""
agent.py  —  main entry point for the CS3600 carpet game.
"""
 
from collections.abc import Callable
from typing import Tuple
 
from game.board import Board
from game.move import Move
from game.enums import MoveType
 
from .rat_belief import RatBelief
from .search import Searcher
 
 
class PlayerAgent:
 
    def __init__(
        self,
        board: Board,
        transition_matrix=None,
        time_left: Callable = None,
    ):
        self.rat_belief = RatBelief(transition_matrix)
        self.searcher = Searcher()
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
 
        my_last_loc, my_last_hit = board.player_search
        if my_last_loc is not None and not my_last_hit:
            self.rat_belief.note_my_miss(my_last_loc)
 
        # ---- 3. Dynamic rat search threshold --------------------------
        margin = board.player_worker.get_points() - board.opponent_worker.get_points()
        turns_left = board.player_worker.turns_left
        if margin > 0:
            rat_search_threshold = 3.0
        else:
            rat_search_threshold = max(0.0, 2.0 + margin / max(turns_left, 1))
 
        # ---- 4. High-confidence rat search ----------------------------
        best_rat_pos, best_rat_ev = self.rat_belief.best_search_target()
        if best_rat_ev >= rat_search_threshold:
            return Move.search(best_rat_pos)
 
        # ---- 5. Minimax search ----------------------------------------
        move, score = self.searcher.search(board, self.rat_belief, time_left)
 
        # ---- 6. Fallback ----------------------------------------------
        if move is None:
            valid = board.get_valid_moves(exclude_search=False)
            return valid[0] if valid else Move.search((0, 0))
 
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