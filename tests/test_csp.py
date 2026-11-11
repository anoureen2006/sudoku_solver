"""CSP tests. Groups: NORMAL (>=5), EDGE (>=3), IMPOSSIBLE (>=2)."""
import random
import pytest

from sudoku_rescue.csp import (SolverConfig, solve, count_solutions, find_conflicts, propagate_ac3, flatten)
from helpers import PUZZLE, SOLUTION, digits, is_valid_solution

CONFIGS = [SolverConfig(False, False, "static"), SolverConfig(True, False, "static"),
           SolverConfig(False, True, "static"), SolverConfig(True, True, "static"), SolverConfig(True, True, "mrv")]
IDS = [c.label + "/" + c.order for c in CONFIGS]


# ------------------------------------------------------------------------------------- NORMAL --
@pytest.mark.parametrize("cfg", CONFIGS, ids=IDS)
def test_normal_known_puzzle_every_configuration(cfg):          # normal cases 1-5 (one per configuration)
    r = solve(digits(PUZZLE), cfg)
    assert r.status == "solved"
    assert flatten(r.solution) == digits(SOLUTION)


def test_normal_generated_puzzle_has_valid_solution():
    from sudoku_rescue.data import make_puzzle
    pz, sol = make_puzzle(random.Random(5), 28)
    r = solve(pz)
    assert is_valid_solution(r.solution, pz) and flatten(r.solution) == sol


def test_normal_counters_show_pruning_effect():
    stats = {c.label: solve(digits(PUZZLE), SolverConfig(c.forward_checking, c.ac3, "static")).stats
             for c in CONFIGS[:2] + CONFIGS[3:4]}
    assert stats["BT"].assignments > stats["BT+FC"].assignments > stats["BT+FC+AC3"].assignments
    assert stats["BT"].fc_prunings == 0 and stats["BT+FC"].fc_prunings > 0
    assert stats["BT+FC+AC3"].ac3_prunings > 0 and stats["BT+FC"].ac3_prunings == 0
    assert stats["BT"].backtracks <= stats["BT"].assignments


def test_normal_ac3_alone_shrinks_domains_without_search():
    ok, cell, doms, st = propagate_ac3(digits(PUZZLE))
    assert ok and st.ac3_prunings > 0
    assert sum(bin(d).count("1") for d in doms) < 81 * 9


# --------------------------------------------------------------------------------------- EDGE ---
def test_edge_empty_grid_is_solvable_and_has_many_solutions():
    assert solve([0] * 81).status == "solved"
    assert count_solutions([0] * 81, limit=2) == 2


def test_edge_complete_grid_needs_no_search():
    r = solve(digits(SOLUTION))
    assert r.status == "solved" and r.stats.assignments == 0 and flatten(r.solution) == digits(SOLUTION)


def test_edge_single_blank_is_filled():
    g = digits(SOLUTION); g[40] = 0
    assert solve(g).solution[4][4] == digits(SOLUTION)[40]


def test_edge_multiple_solutions_are_counted_not_hidden():
    g = digits(SOLUTION)
    for c in (0, 1, 9, 10):      # a 2x2 "deadly rectangle" pattern often leaves two solutions; at least >=1
        g[c] = 0
    assert count_solutions(g, 5) >= 1
    assert count_solutions([0] * 81, 3) == 3


@pytest.mark.parametrize("bad", [[0] * 80, [10] + [0] * 80, [-1] + [0] * 80, [[0] * 9] * 8])
def test_edge_malformed_input_is_rejected(bad):
    with pytest.raises(ValueError):
        solve(bad)


def test_edge_budget_exceeded_is_reported_not_crashed():
    r = solve(digits(PUZZLE), SolverConfig(False, False, "static", max_assignments=10))
    assert r.status == "budget_exceeded" and "budget" in r.message


# ---------------------------------------------------------------------------------- IMPOSSIBLE ---
@pytest.mark.parametrize("cfg", CONFIGS, ids=IDS)
def test_impossible_duplicate_givens_give_clear_message(cfg):   # impossible case 1
    g = digits(PUZZLE); g[2] = 5       # row 1 now has two 5s (r1c1 and r1c3)
    r = solve(g, cfg)
    assert r.status == "no_solution" and r.message.startswith("No solution")
    assert find_conflicts(g)


@pytest.mark.parametrize("cfg", CONFIGS, ids=IDS)
def test_impossible_cell_with_no_legal_digit(cfg):              # impossible case 2
    g = [0] * 81
    g[0:8] = [1, 2, 3, 4, 5, 6, 7, 8]  # row 1 holds 1-8, so r1c9 needs 9 ...
    g[17] = 9                           # ... but column 9 already has a 9 (r2c9)
    r = solve(g, cfg)
    assert r.status == "no_solution" and "No solution" in r.message
    assert not find_conflicts(g)        # no duplicate digit: the contradiction is only visible via domains/search


def test_impossible_ac3_detects_it_without_search():
    g = [0] * 81
    g[0:8] = [1, 2, 3, 4, 5, 6, 7, 8]; g[17] = 9
    r = solve(g, SolverConfig(True, True, "static"))
    assert r.stats.assignments == 0 and "AC-3" in r.message
