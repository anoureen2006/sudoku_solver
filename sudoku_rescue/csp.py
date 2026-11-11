"""Sudoku as a constraint satisfaction problem (CSP).

Variables : 81 cells.  Domains : digits 1-9 (stored as 9-bit masks).
Constraints: all-different on every row, column and 3x3 box (binary != arcs).
Givens are cells whose domain starts as a single value.

Three independently switchable components (see SolverConfig):
  * plain backtracking (always on),
  * forward checking (FC): after each assignment, remove the digit from peers' domains,
  * AC-3: arc consistency run once before search (detects conflicts, shrinks domains).

Every variant records counters in CSPStats. Definitions used in all reports:
  nodes        recursive calls of the search function
  assignments  every (cell := digit) tried during search (givens are not counted)
  backtracks   every assignment that is retracted without reaching the requested solution(s)
  fc_prunings  digits removed from a domain by forward checking
  ac3_revisions / ac3_prunings   arcs processed / digits removed by AC-3

Plain BT (no AC-3) still checks the givens for duplicates once before searching; with AC-3 on, the
same conflict is found as an empty domain instead.
"""
from __future__ import annotations

import random
import time
from collections import deque
from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np

ALL = 0x1FF  # bits 0..8  <->  digits 1..9


def _build_units():
    rows = [[r * 9 + c for c in range(9)] for r in range(9)]
    cols = [[r * 9 + c for r in range(9)] for c in range(9)]
    boxes = [[(br * 3 + r) * 9 + (bc * 3 + c) for r in range(3) for c in range(3)]
             for br in range(3) for bc in range(3)]
    return rows, cols, boxes


ROWS, COLS, BOXES = _build_units()
UNITS = ROWS + COLS + BOXES
UNIT_NAMES = ([f"row {i + 1}" for i in range(9)] + [f"column {i + 1}" for i in range(9)]
              + [f"box {i + 1}" for i in range(9)])
PEERS = []
for _i in range(81):
    _s = set()
    for _u in UNITS:
        if _i in _u:
            _s.update(_u)
    _s.discard(_i)
    PEERS.append(tuple(sorted(_s)))


def cell_name(i: int) -> str:
    return f"r{i // 9 + 1}c{i % 9 + 1}"


def bits_to_digits(mask: int):
    return [d + 1 for d in range(9) if (mask >> d) & 1]


def flatten(grid):
    """Accept a 9x9 nested list/array or a flat sequence of 81 ints; return list[int]."""
    arr = np.asarray(grid)
    if arr.shape == (9, 9):
        arr = arr.reshape(81)
    if arr.shape != (81,):
        raise ValueError(f"Sudoku grid must be 9x9 (or 81 values), got shape {arr.shape}")
    flat = [int(v) for v in arr]
    if any(v != a for v, a in zip(flat, arr)) or any(v < 0 or v > 9 for v in flat):
        raise ValueError("Sudoku cells must be integers 0-9 (0 = blank)")
    return flat


def to_grid(flat):
    return [list(flat[r * 9:(r + 1) * 9]) for r in range(9)]


def format_grid(grid) -> str:
    flat = flatten(grid)
    lines = []
    for r in range(9):
        if r and r % 3 == 0:
            lines.append("------+-------+------")
        row = flat[r * 9:(r + 1) * 9]
        parts = []
        for c in range(9):
            if c and c % 3 == 0:
                parts.append("|")
            parts.append(str(row[c]) if row[c] else ".")
        lines.append(" ".join(parts))
    return "\n".join(lines)


@dataclass(frozen=True)
class Conflict:
    unit: str
    digit: int
    cells: tuple

    def __str__(self):
        cells = ", ".join(cell_name(c) for c in self.cells)
        return f"digit {self.digit} appears {len(self.cells)}x in {self.unit} ({cells})"


def find_conflicts(grid):
    """Duplicate digits inside any row/column/box (ignores blanks)."""
    flat = flatten(grid)
    out = []
    for name, unit in zip(UNIT_NAMES, UNITS):
        seen = {}
        for c in unit:
            if flat[c]:
                seen.setdefault(flat[c], []).append(c)
        for d, cells in seen.items():
            if len(cells) > 1:
                out.append(Conflict(name, d, tuple(cells)))
    return out


@dataclass(frozen=True)
class SolverConfig:
    forward_checking: bool = True
    ac3: bool = True
    order: str = "mrv"                 # "static" (row-major) or "mrv" (fewest legal digits first)
    max_assignments: Optional[int] = None

    @property
    def label(self) -> str:
        parts = ["BT"]
        if self.forward_checking:
            parts.append("FC")
        if self.ac3:
            parts.append("AC3")
        return "+".join(parts)


@dataclass
class CSPStats:
    nodes: int = 0
    assignments: int = 0
    backtracks: int = 0
    fc_prunings: int = 0
    ac3_revisions: int = 0
    ac3_prunings: int = 0
    max_depth: int = 0
    seconds: float = 0.0

    def as_dict(self):
        return asdict(self)

    def add(self, other: "CSPStats"):
        for k in ("nodes", "assignments", "backtracks", "fc_prunings",
                  "ac3_revisions", "ac3_prunings", "seconds"):
            setattr(self, k, getattr(self, k) + getattr(other, k))
        self.max_depth = max(self.max_depth, other.max_depth)


@dataclass
class CSPResult:
    status: str                    # "solved" | "no_solution" | "budget_exceeded"
    solution: Optional[list]       # 9x9 list or None
    solutions_found: int
    message: str
    stats: CSPStats
    conflicts: list

    @property
    def solved(self) -> bool:
        return self.status == "solved"


def ac3(domains, stats: CSPStats):
    """AC-3 on the not-equal constraints between peers. Mutates `domains`.

    Arc (i, j) is revised by deleting from D(i) every digit with no support in D(j).
    For a != constraint a digit v has no support exactly when D(j) == {v}.
    Returns (consistent, failing_cell). A wiped-out domain => no solution.
    """
    queue = deque((i, j) for i in range(81) for j in PEERS[i])
    while queue:
        i, j = queue.popleft()
        stats.ac3_revisions += 1
        dj = domains[j]
        if dj and (dj & (dj - 1)) == 0 and (domains[i] & dj):
            domains[i] &= ~dj
            stats.ac3_prunings += 1
            if domains[i] == 0:
                return False, i
            for k in PEERS[i]:
                if k != j:
                    queue.append((k, i))
    return True, None


def propagate_ac3(grid):
    """Run only AC-3 on a grid. Returns (consistent, failing_cell, domains, stats)."""
    flat = flatten(grid)
    stats = CSPStats()
    domains = [(1 << (v - 1)) if v else ALL for v in flat]
    t0 = time.perf_counter()
    ok, cell = ac3(domains, stats)
    stats.seconds = time.perf_counter() - t0
    return ok, cell, domains, stats


class _Budget(Exception):
    pass


class _Search:
    def __init__(self, flat, cfg, limit, rng, trace):
        self.flat, self.cfg, self.limit, self.rng, self.trace = flat, cfg, limit, rng, trace
        self.stats = CSPStats()
        self.count = 0
        self.solution = None

    def _log(self, ev, cell, val):
        if self.trace is not None and len(self.trace) < 20000:
            self.trace.append((ev, cell, val))

    def _cands(self, i, domains, vals):
        if self.cfg.forward_checking:
            return domains[i]
        used = 0
        for p in PEERS[i]:
            if vals[p]:
                used |= 1 << (vals[p] - 1)
        return domains[i] & ~used

    def run(self) -> CSPResult:
        t0 = time.perf_counter()
        flat, cfg, st = self.flat, self.cfg, self.stats
        conflicts = find_conflicts(flat)
        domains = [(1 << (v - 1)) if v else ALL for v in flat]
        fail = None
        if cfg.ac3:
            ok, cell = ac3(domains, st)
            if not ok:
                fail = f"No solution: AC-3 emptied the domain of {cell_name(cell)}"
                if conflicts:
                    fail += f" ({conflicts[0]})"
        elif conflicts:
            fail = f"No solution: the givens already conflict ({conflicts[0]})"
        vals = flat[:]
        if fail is None and cfg.forward_checking:
            for i, v in enumerate(flat):
                if not v:
                    continue
                bit = 1 << (v - 1)
                for p in PEERS[i]:
                    if vals[p] == 0 and domains[p] & bit:
                        domains[p] &= ~bit
                        st.fc_prunings += 1
                        if domains[p] == 0:
                            fail = f"No solution: {cell_name(p)} has no legal digit left"
                            break
                if fail:
                    break
        status = None
        if fail is None:
            try:
                self._search(domains, vals, 0)
            except _Budget:
                status = "budget_exceeded"
        st.seconds = time.perf_counter() - t0
        if fail:
            return CSPResult("no_solution", None, 0, fail, st, conflicts)
        if status:
            return CSPResult(status, to_grid(self.solution) if self.solution else None, self.count,
                             f"Search budget of {cfg.max_assignments} assignments exceeded", st, conflicts)
        if self.count:
            return CSPResult("solved", to_grid(self.solution), self.count, "Solved", st, conflicts)
        return CSPResult("no_solution", None, 0, "No solution: search space exhausted", st, conflicts)

    def _search(self, domains, vals, depth):
        st, cfg = self.stats, self.cfg
        st.nodes += 1
        st.max_depth = max(st.max_depth, depth)
        best, best_c = -1, 0
        if cfg.order == "static":
            for i in range(81):
                if vals[i] == 0:
                    best, best_c = i, self._cands(i, domains, vals)
                    break
        else:
            best_n = 10
            for i in range(81):
                if vals[i] == 0:
                    c = self._cands(i, domains, vals)
                    n = c.bit_count()
                    if n < best_n:
                        best, best_c, best_n = i, c, n
                        if n == 0:
                            break
        if best < 0:
            self.count += 1
            if self.solution is None:
                self.solution = vals[:]
            return self.count >= self.limit
        if best_c == 0:
            return False
        values = bits_to_digits(best_c)
        if self.rng is not None:
            self.rng.shuffle(values)
        for v in values:
            st.assignments += 1
            if cfg.max_assignments is not None and st.assignments > cfg.max_assignments:
                raise _Budget()
            vals[best] = v
            self._log("assign", best, v)
            if cfg.forward_checking:
                nd = domains[:]
                b = 1 << (v - 1)
                nd[best] = b
                ok = True
                for p in PEERS[best]:
                    if vals[p] == 0 and nd[p] & b:
                        nd[p] &= ~b
                        st.fc_prunings += 1
                        if nd[p] == 0:
                            ok = False
                            break
                if ok and self._search(nd, vals, depth + 1):
                    return True
            elif self._search(domains, vals, depth + 1):
                return True
            vals[best] = 0
            st.backtracks += 1
            self._log("backtrack", best, v)
        return False


def solve(grid, config: Optional[SolverConfig] = None, count_limit: int = 1,
          rng: Optional[random.Random] = None, trace: Optional[list] = None) -> CSPResult:
    """Solve a Sudoku. `count_limit` > 1 keeps searching to count solutions (stops at the limit)."""
    return _Search(flatten(grid), config or SolverConfig(), count_limit, rng, trace).run()


def count_solutions(grid, limit: int = 2) -> int:
    return solve(grid, SolverConfig(True, True, "mrv"), count_limit=limit).solutions_found


def is_solvable(grid, config: Optional[SolverConfig] = None) -> bool:
    return solve(grid, config).solved
