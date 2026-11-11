"""Integration of the three parts.

    NN reads the grid -> AC-3 looks for a conflict -> search proposes the cheapest corrections ->
    the CSP solver tests each corrected puzzle; the first solvable one (UCS: the cheapest) wins.
    If nothing is found: report "unreadable" and list the least confident cells.

Search formulation
  state    : tuple of (cell, new_digit) corrections, listed in candidate order (so each *set* of
             corrections is generated exactly once)
  action   : change one more candidate cell to one of its `n_alternatives` next-most-likely digits
  step cost: log(p_read / p_new)  >= 0  (how much less likely the new digit is; read digit = argmax)
  goal     : the corrected grid is solvable (or has a unique solution, `goal="unique"`)
  limit    : at most `max_corrections` corrections
  A* h     : every duplicate-digit conflict needs at least one of its cells changed. Take a
             cell-disjoint set of conflicts; each needs its own change, so
             h = sum over those conflicts of (cheapest change available among its cells).
             Admissible (never over-estimates). h = inf when a conflict can no longer be repaired.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .csp import (CSPStats, SolverConfig, solve, find_conflicts, propagate_ac3, flatten, cell_name,
                  to_grid, count_solutions)
from .search import best_first_search, SearchStats

EPS = 1e-12


def probs_without_nn(probs, grid):
    """'NN off' ablation: keep the read digits but discard all confidence information (every
    non-blank cell becomes 50% read digit, the other eight digits share the rest equally)."""
    out = np.zeros_like(probs)
    for i, d in enumerate(grid):
        if d:
            out[i] = 0.5 / 8
            out[i, d - 1] = 0.5
    return out


class CorrectionProblem:
    def __init__(self, read_grid, probs, max_corrections=3, n_alternatives=3, max_candidates=24,
                 goal="solvable", csp_config: Optional[SolverConfig] = None, use_confidence_order=True):
        self.read = flatten(read_grid)
        self.probs = np.clip(np.asarray(probs, dtype=float), EPS, 1.0)
        self.max_corrections = max_corrections
        self.goal = goal
        self.csp_config = csp_config or SolverConfig(True, True, "mrv")
        self.csp_stats = CSPStats()
        self.csp_calls = 0
        self._cache = {}
        givens = [i for i in range(81) if self.read[i]]
        if use_confidence_order:      # least confident first, then only the first max_candidates
            givens.sort(key=lambda i: (self.probs[i].max(), i))
            givens = givens[:max_candidates]
        self.candidates = givens
        self.pos = {c: k for k, c in enumerate(self.candidates)}
        self.alts, self.alt_cost = {}, {}
        for c in self.candidates:
            order = [int(d) for d in np.argsort(-self.probs[c], kind="stable")]
            order = [d for d in order if d + 1 != self.read[c]][:n_alternatives]
            p_read = self.probs[c, self.read[c] - 1]
            self.alts[c] = [d + 1 for d in order]
            self.alt_cost[c] = {d + 1: max(0.0, math.log(p_read / self.probs[c, d])) for d in order}

    # -- helpers --------------------------------------------------------------------------------
    def apply(self, state):
        g = self.read[:]
        for cell, digit in state:
            g[cell] = digit
        return g

    def cost_of(self, cell, digit):
        return self.alt_cost[cell][digit]

    # -- search interface -----------------------------------------------------------------------
    def initial_state(self):
        return ()

    def successors(self, state):
        if len(state) >= self.max_corrections:
            return
        last = self.pos[state[-1][0]] if state else -1
        for k in range(last + 1, len(self.candidates)):
            cell = self.candidates[k]
            for digit in self.alts[cell]:
                yield ((cell, digit), state + ((cell, digit),), self.cost_of(cell, digit))

    def is_goal(self, state):
        if state in self._cache:
            return self._cache[state][0]
        grid = self.apply(state)
        ok, sol = False, None
        if not find_conflicts(grid):                    # duplicates => certainly unsolvable (cheap)
            self.csp_calls += 1
            if self.goal == "unique":
                res = solve(grid, self.csp_config, count_limit=2)
                ok = res.solutions_found == 1
            else:
                res = solve(grid, self.csp_config)
                ok = res.solved
            self.csp_stats.add(res.stats)
            sol = res.solution if ok else None
        self._cache[state] = (ok, sol)
        return ok

    def solution_for(self, state):
        self.is_goal(state)
        return self._cache[state][1]

    def heuristic(self, state):
        conflicts = find_conflicts(self.apply(state))
        if not conflicts:
            return 0.0
        remaining = self.max_corrections - len(state)
        last = self.pos[state[-1][0]] if state else -1
        mincost = {}
        for c in self.candidates[last + 1:]:
            if self.alts[c]:
                mincost[c] = min(self.alt_cost[c].values())
        used, total, count = set(), 0.0, 0
        for cf in sorted(conflicts, key=lambda x: len(x.cells)):
            cells = set(cf.cells)
            if cells & used:
                continue
            costs = [mincost[c] for c in cells if c in mincost]
            if not costs:
                return math.inf
            used |= cells
            total += min(costs)
            count += 1
        return math.inf if count > remaining else total


@dataclass
class Correction:
    row: int
    col: int
    read_digit: int
    new_digit: int
    cost: float
    p_read: float
    p_new: float


@dataclass
class RescueConfig:
    strategy: str = "ucs"
    max_corrections: int = 3
    goal: str = "solvable"             # or "unique"
    use_nn_costs: bool = True          # False = 'NN off' ablation (uniform costs, no confidence ordering)
    n_alternatives: int = 3
    max_candidates: int = 24
    max_expansions: int = 3000
    csp: SolverConfig = field(default_factory=lambda: SolverConfig(True, True, "mrv"))


@dataclass
class RescueResult:
    status: str                        # "solved" | "unreadable"
    message: str
    read_grid: list
    final_grid: Optional[list]
    solution: Optional[list]
    corrections: list
    total_cost: float
    ac3_conflict: Optional[str]
    conflicts: list
    search: SearchStats
    csp: CSPStats
    csp_calls: int
    least_confident: list
    trace: list


def least_confident_cells(probs, grid, k=5):
    cells = [i for i in range(81) if grid[i]]
    cells.sort(key=lambda i: probs[i].max())
    return [(i // 9 + 1, i % 9 + 1, float(probs[i].max())) for i in cells[:k]]


def rescue(reading, config: Optional[RescueConfig] = None, trace=False) -> RescueResult:
    """Run the whole pipeline on a Reading (grid + probabilities). See module docstring."""
    cfg = config or RescueConfig()
    grid, probs = reading.grid, reading.probs
    if not cfg.use_nn_costs:
        probs = probs_without_nn(probs, grid)
    ok, cell, _, _ = propagate_ac3(grid)
    conflicts = find_conflicts(grid)
    ac3_msg = None if ok else f"AC-3 wiped out the domain of {cell_name(cell)}"
    problem = CorrectionProblem(grid, probs, cfg.max_corrections, cfg.n_alternatives, cfg.max_candidates,
                                cfg.goal, cfg.csp, use_confidence_order=cfg.use_nn_costs)
    res = best_first_search(problem, cfg.strategy, cfg.max_expansions, trace=trace)
    weak = least_confident_cells(reading.probs, grid)
    if res.found:
        corrs = []
        for cell_i, new in res.state:
            corrs.append(Correction(cell_i // 9 + 1, cell_i % 9 + 1, grid[cell_i], new, problem.cost_of(cell_i, new),
                                    float(problem.probs[cell_i, grid[cell_i] - 1]), float(problem.probs[cell_i, new - 1])))
        final = problem.apply(res.state)
        return RescueResult("solved", f"Solved with {len(corrs)} correction(s)", grid, to_grid(final),
                            problem.solution_for(res.state), corrs, res.cost, ac3_msg, conflicts, res.stats,
                            problem.csp_stats, problem.csp_calls, weak, res.trace)
    why = ("search budget reached" if res.reason == "budget"
           else f"no solvable puzzle within {cfg.max_corrections} corrections")
    weakest = ", ".join(f"r{r}c{c} ({p:.2f})" for r, c, p in weak)
    return RescueResult("unreadable", f"Puzzle unreadable: {why}. Least confident cells: {weakest}", grid, None,
                        None, [], math.inf, ac3_msg, conflicts, res.stats, problem.csp_stats, problem.csp_calls,
                        weak, res.trace)


# ------------------------------------------------------------------------------------ baselines --
def no_correction(reading, csp: Optional[SolverConfig] = None):
    """Solve the first reading as is. Returns (solution or None, CSPResult)."""
    res = solve(reading.grid, csp or SolverConfig(True, True, "mrv"))
    return res.solution, res


def random_correction(reading, rng: random.Random, max_trials=1000, max_corrections=3, n_alternatives=3,
                      csp: Optional[SolverConfig] = None):
    """Random baseline: try random sets of 1..max_corrections cells changed to one of their top
    alternatives (ignoring costs and confidence). Returns (solution or None, trials_used)."""
    csp = csp or SolverConfig(True, True, "mrv")
    probs = np.asarray(reading.probs)
    givens = [i for i in range(81) if reading.grid[i]]
    sol, res = no_correction(reading, csp)
    if sol:
        return sol, 0
    for t in range(1, max_trials + 1):
        k = rng.randint(1, max_corrections)
        g = reading.grid[:]
        for c in rng.sample(givens, k):
            order = [int(d) + 1 for d in np.argsort(-probs[c], kind="stable") if int(d) + 1 != reading.grid[c]]
            g[c] = rng.choice(order[:n_alternatives])
        if find_conflicts(g):
            continue
        r = solve(g, csp)
        if r.solved:
            return r.solution, t
    return None, max_trials
