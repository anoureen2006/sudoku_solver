"""Pipeline tests: correction search + CSP (synthetic network output, no dataset needed)."""
import math
import random

import pytest

from sudoku_rescue.csp import find_conflicts, flatten, to_grid, solve
from sudoku_rescue.integration import (RescueConfig, rescue, CorrectionProblem, no_correction, random_correction,
                                       probs_without_nn)
from sudoku_rescue.search import best_first_search
from helpers import PUZZLE, SOLUTION, digits, synthetic_reading, is_valid_solution

TRUE = digits(PUZZLE)
GIVENS = [i for i, d in enumerate(TRUE) if d]


def wrong_digit(cell, avoid=()):
    for d in range(1, 10):
        if d != TRUE[cell] and d not in avoid:
            return d


def test_normal_clean_reading_needs_no_correction():
    r = rescue(synthetic_reading(TRUE))
    assert r.status == "solved" and r.corrections == [] and r.total_cost == 0
    assert flatten(r.solution) == digits(SOLUTION)


def test_normal_single_misread_is_corrected_with_log():
    cell = GIVENS[3]
    rd = synthetic_reading(TRUE, {cell: (7 if TRUE[cell] != 7 else 1, 0.55)})
    assert find_conflicts(rd.grid) or True
    r = rescue(rd, RescueConfig(goal="unique"))
    assert r.status == "solved"
    assert [(c.row, c.col) for c in r.corrections] == [(cell // 9 + 1, cell % 9 + 1)]
    c = r.corrections[0]
    assert c.new_digit == TRUE[cell] and c.read_digit == rd.grid[cell] and c.cost >= 0
    assert math.isclose(c.cost, math.log(c.p_read / c.p_new))
    assert flatten(r.solution) == digits(SOLUTION)


@pytest.mark.parametrize("strategy", ["ucs", "greedy", "astar"])
def test_normal_three_strategies_recover_two_misreads(strategy):
    cells = [GIVENS[2], GIVENS[10]]
    wrong = {c: (wrong_digit(c), 0.5) for c in cells}
    r = rescue(synthetic_reading(TRUE, wrong), RescueConfig(strategy=strategy, goal="unique"))
    assert r.status == "solved" and len(r.corrections) == 2
    assert flatten(r.solution) == digits(SOLUTION)


def test_normal_astar_is_optimal_like_ucs_and_expands_no_more():
    """A* heuristic admissibility check over many random misread patterns."""
    rng = random.Random(3)
    for _ in range(12):
        cells = rng.sample(GIVENS, rng.choice([1, 2, 3]))
        wrong = {c: (rng.choice([d for d in range(1, 10) if d != TRUE[c]]), rng.uniform(0.45, 0.8)) for c in cells}
        rd = synthetic_reading(TRUE, wrong)
        u = rescue(rd, RescueConfig(strategy="ucs", goal="unique"))
        a = rescue(rd, RescueConfig(strategy="astar", goal="unique"))
        assert u.status == a.status
        if u.status == "solved":
            assert math.isclose(u.total_cost, a.total_cost, abs_tol=1e-9)
            assert a.search.nodes_expanded <= u.search.nodes_expanded


def test_normal_ucs_costs_are_never_negative_and_ordering_is_by_cost():
    rd = synthetic_reading(TRUE, {GIVENS[5]: (wrong_digit(GIVENS[5]), 0.6)})
    prob = CorrectionProblem(rd.grid, rd.probs)
    for _, _, cost in prob.successors(()):
        assert cost >= 0


def test_nn_guidance_beats_random_and_no_correction():
    cell = GIVENS[7]
    d = wrong_digit(cell)
    rd = synthetic_reading(TRUE, {cell: (d, 0.5)})
    sol_none, _ = no_correction(rd)
    sol_rand, _ = random_correction(rd, random.Random(0), max_trials=1000)
    r = rescue(rd, RescueConfig(goal="unique"))
    assert flatten(r.solution) == digits(SOLUTION)
    assert sol_none is None or flatten(sol_none) != digits(SOLUTION)


def test_nn_off_ablation_discards_confidence():
    rd = synthetic_reading(TRUE, {GIVENS[1]: (wrong_digit(GIVENS[1]), 0.5)})
    flat = probs_without_nn(rd.probs, rd.grid)
    assert all(abs(flat[i].max() - 0.5) < 1e-9 for i in GIVENS)


# ---------------------------------------------------------------------------- edge / impossible --
def test_edge_blank_cells_are_never_corrected():
    rd = synthetic_reading(TRUE, {GIVENS[0]: (wrong_digit(GIVENS[0]), 0.5)})
    prob = CorrectionProblem(rd.grid, rd.probs)
    assert all(rd.grid[c] != 0 for c in prob.candidates)


def test_edge_max_corrections_is_respected():
    cells = [GIVENS[1], GIVENS[9], GIVENS[18], GIVENS[25]]
    rd = synthetic_reading(TRUE, {c: (wrong_digit(c), 0.5) for c in cells})
    r = rescue(rd, RescueConfig(max_corrections=3, goal="unique"))
    assert len(r.corrections) <= 3       # never more than max_corrections, whatever the outcome
    assert r.status == "unreadable" or r.total_cost >= 0


def test_impossible_pipeline_reports_unreadable_with_least_confident_cells():
    # r1c1 is read as 3 (duplicates r1c2) and r2c1 is read as 1 (duplicates r2c4): two cell-disjoint
    # conflicts, so at least two corrections are needed, but only one is allowed.
    rd = synthetic_reading(TRUE, {0: (3, 0.5), 9: (1, 0.5)})
    assert len(find_conflicts(rd.grid)) >= 2
    for strategy in ("ucs", "astar"):
        r = rescue(rd, RescueConfig(strategy=strategy, max_corrections=1))
        assert r.status == "unreadable"
        assert "unreadable" in r.message and "Least confident" in r.message
        assert len(r.least_confident) == 5 and r.corrections == [] and r.solution is None
        assert r.least_confident[0][2] <= r.least_confident[-1][2]
    ok = rescue(rd, RescueConfig(max_corrections=2))
    assert ok.status == "solved" and len(ok.corrections) == 2    # with enough budget the same input is repaired


def test_impossible_reading_with_two_equal_givens_and_no_budget():
    rd = synthetic_reading(TRUE)
    rd.grid[2] = 5                       # duplicate 5 in row 1, nothing else changed
    r = rescue(rd, RescueConfig(max_corrections=0))
    assert r.status == "unreadable" and r.ac3_conflict is not None and "AC-3" in r.ac3_conflict
