"""
heuristic.py  —  board evaluation for the minimax tree.

Returns a single float: higher = better for the current player (player_worker).
Called at leaf nodes in the minimax tree — needs to be fast.

Core components:
  1. Score differential          — actual points scored so far
  2. Carpet potential            — value of lines in progress, discounted by
                                   completion risk and turns remaining
  3. Opponent threat             — penalise lines the opponent can poach
  4. Rat opportunity             — expected value of best search target

All weights are modulated by a single urgency dial:
  urgency = (my_points - opp_points) / turns_remaining

  urgency >> 0  ->  ahead, play conservatively (block, discount risky lines)
  urgency << 0  ->  behind, play aggressively (chase big lines, search more)
  urgency ~ 0   ->  neutral, balanced weights
"""

from game.enums import CARPET_POINTS_TABLE, BOARD_SIZE, Cell
from .carpet_planner import CarpetPlanner, Line, get_cell_type, manhattan

# ---------------------------------------------------------------------------
# Tunable weights  (adjust after watching real games)
# ---------------------------------------------------------------------------

W_SCORE           = 1.0   # score differential — anchor, always 1.0
W_CARPET_BASE     = 0.6   # weight on our best lines net value
W_CARPET_PRIMED   = 0.4   # extra credit per already-primed cell
W_OPP_THREAT      = 0.5   # penalty per point of opponent carpet value
W_RAT_BASE        = 0.5   # rat EV weight when neutral
URGENCY_SCALE     = 2.0   # how aggressively urgency shifts weights
TOP_N_LINES       = 1     # how many top lines to consider
MIN_TURNS_FOR_LINE = 2    # ignore incomplete lines if fewer turns remain

# ---------------------------------------------------------------------------
# Main heuristic
# ---------------------------------------------------------------------------

_planner = CarpetPlanner()


def evaluate(board, rat_belief=None) -> float:
    """
    Evaluate board from perspective of player_worker.
    Higher = better for us.
    """
    my_worker  = board.player_worker
    opp_worker = board.opponent_worker
    my_pos     = my_worker.get_location()
    opp_pos    = opp_worker.get_location()

    turns_left = max(my_worker.turns_left, 1)
    my_pts     = my_worker.get_points()
    opp_pts    = opp_worker.get_points()
    margin     = my_pts - opp_pts

    # urgency: positive = ahead, negative = behind, scaled by time
    urgency = max(-3.0, min(3.0, margin / turns_left))

    score  = W_SCORE * margin
    score += _carpet_potential(board, my_pos, opp_pos, turns_left, urgency)
    score -= _opponent_threat(board, opp_pos, my_pos, turns_left, urgency)

    if rat_belief is not None:
        score += _rat_component(rat_belief, urgency)

    return score


# ---------------------------------------------------------------------------
# Components
# ---------------------------------------------------------------------------

def _carpet_potential(board, my_pos, opp_pos, turns_left, urgency) -> float:
    if turns_left < MIN_TURNS_FOR_LINE:
        return 0.0

    lines = _planner.score_all_lines(board, my_pos)
    if not lines:
        return 0.0

    # ahead -> discount risky lines more steeply
    # behind -> value potential more optimistically
    risk_aversion = max(0.2, 1.0 + urgency * URGENCY_SCALE)

    total = 0.0
    for line in lines[:TOP_N_LINES]:
        if not line.valid:
            continue

        turns_needed = line.n_space + max(0, _dist_to_line(my_pos, line) - 1)
        if turns_needed >= turns_left:
            continue

        val = line.net_value
        val += line.n_primed * W_CARPET_PRIMED

        # discount by how complete the line is
        completion_ratio = line.n_primed / line.length
        val *= (0.3 + 0.7 * completion_ratio)

        # discount by time pressure
        val *= max(0.0, 1.0 - (turns_needed / turns_left))

        # apply risk aversion
        val /= risk_aversion

        # poaching penalty
        opp_dist = _dist_to_line(opp_pos, line)
        my_dist  = _dist_to_line(my_pos,  line)
        if opp_dist < my_dist:
            poach_risk = (my_dist - opp_dist) / max(1, line.length)
            poach_risk *= (1.0 - completion_ratio * 0.7)
            val *= max(0.0, 1.0 - poach_risk)

        total += val * W_CARPET_BASE

    return total


def _opponent_threat(board, opp_pos, my_pos, turns_left, urgency) -> float:
    lines = _planner.score_all_lines(board, opp_pos)
    if not lines:
        return 0.0

    # ahead -> weight threat higher (block more)
    # behind -> weight lower (focus on own scoring)
    threat_weight = max(0.1, W_OPP_THREAT * (1.0 + urgency * 0.5))

    total = 0.0
    for line in lines[:TOP_N_LINES]:
        if not line.valid:
            continue

        turns_needed = line.n_space + max(0, _dist_to_line(opp_pos, line) - 1)
        if turns_needed >= turns_left:
            continue

        val = line.net_value
        val += line.n_primed * W_CARPET_PRIMED

        completion_ratio = line.n_primed / line.length
        val *= (0.3 + 0.7 * completion_ratio)
        val *= max(0.0, 1.0 - (turns_needed / turns_left))

        total += val * threat_weight

    return total


def _rat_component(rat_belief, urgency) -> float:
    _, best_ev = rat_belief.best_search_target()
    if best_ev <= 0:
        return 0.0

    # behind -> more willing to gamble on rat
    # ahead  -> more conservative
    rat_weight = max(0.1, min(1.5, W_RAT_BASE * (1.0 - urgency * 0.3)))
    return rat_weight * best_ev


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _dist_to_line(pos: tuple, line: Line) -> int:
    return min(manhattan(pos, c) for c in line.cells)
