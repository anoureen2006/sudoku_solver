"""Generic best-first graph search: Uniform-Cost Search, Greedy best-first and A*.

A problem is any object with
    initial_state()            -> hashable state
    is_goal(state)             -> bool          (goal test is applied when a node is *popped*)
    successors(state)          -> iterable of (action, next_state, step_cost>=0)
    heuristic(state)           -> float >= 0, math.inf = "dead end, prune"   (used by greedy/astar)

Strategies:  ucs: f = g     greedy: f = h     astar: f = g + h

Counters (SearchStats):
    nodes_generated  every successor produced (plus the root); includes ones pruned by h = inf
    nodes_expanded   every node popped, goal-tested and (if not goal) expanded
    nodes_pruned     successors discarded because h = inf
    goal_tests       number of is_goal calls
    max_frontier     largest size of the priority queue
Stale queue entries (a cheaper path to the same state was found later) are skipped and not counted
as expanded.
"""
from __future__ import annotations

import heapq
import itertools
import math
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Optional

STRATEGIES = ("ucs", "greedy", "astar")


@dataclass
class SearchStats:
    nodes_generated: int = 0
    nodes_expanded: int = 0
    nodes_pruned: int = 0
    goal_tests: int = 0
    max_frontier: int = 0
    seconds: float = 0.0

    def as_dict(self):
        return asdict(self)


@dataclass
class SearchResult:
    found: bool
    state: Any
    actions: list
    cost: float
    stats: SearchStats
    reason: str                    # "goal" | "exhausted" | "budget"
    trace: list = field(default_factory=list)   # (expansion#, state, g, h) for each expanded node


def best_first_search(problem, strategy: str = "ucs", max_expansions: Optional[int] = None,
                      trace: bool = False, trace_limit: int = 500) -> SearchResult:
    if strategy not in STRATEGIES:
        raise ValueError(f"strategy must be one of {STRATEGIES}")
    t0 = time.perf_counter()
    stats = SearchStats()
    tie = itertools.count()
    log = []

    def f_value(g, h):
        return {"ucs": g, "greedy": h, "astar": g + h}[strategy]

    start = problem.initial_state()
    h0 = problem.heuristic(start) if strategy != "ucs" else 0.0
    stats.nodes_generated = 1
    if math.isinf(h0):
        stats.seconds = time.perf_counter() - t0
        return SearchResult(False, None, [], math.inf, stats, "exhausted", log)
    frontier = [(f_value(0.0, h0), next(tie), 0.0, start, [], h0)]
    best_g = {start: 0.0}
    stats.max_frontier = 1

    while frontier:
        _, _, g, state, actions, h = heapq.heappop(frontier)
        if g > best_g.get(state, math.inf):
            continue                       # stale entry
        stats.goal_tests += 1
        stats.nodes_expanded += 1
        if trace and len(log) < trace_limit:
            log.append((stats.nodes_expanded, state, g, h))
        if problem.is_goal(state):
            stats.seconds = time.perf_counter() - t0
            return SearchResult(True, state, actions, g, stats, "goal", log)
        if max_expansions is not None and stats.nodes_expanded >= max_expansions:
            stats.seconds = time.perf_counter() - t0
            return SearchResult(False, None, [], math.inf, stats, "budget", log)
        for action, nxt, cost in problem.successors(state):
            if cost < 0:
                raise ValueError("step costs must be non-negative")
            stats.nodes_generated += 1
            g2 = g + cost
            if g2 >= best_g.get(nxt, math.inf):
                continue
            h2 = problem.heuristic(nxt) if strategy != "ucs" else 0.0
            if math.isinf(h2):
                stats.nodes_pruned += 1
                continue
            best_g[nxt] = g2
            heapq.heappush(frontier, (f_value(g2, h2), next(tie), g2, nxt, actions + [action], h2))
        stats.max_frontier = max(stats.max_frontier, len(frontier))
    stats.seconds = time.perf_counter() - t0
    return SearchResult(False, None, [], math.inf, stats, "exhausted", log)


class GraphProblem:
    """Small explicit weighted graph; used for tests and teaching."""

    def __init__(self, edges, start, goal, h=None):
        self.adj = {}
        for a, b, w in edges:
            self.adj.setdefault(a, []).append((b, w))
            self.adj.setdefault(b, []).append((a, w))
        self.start, self.goal, self.h = start, goal, h or {}

    def initial_state(self):
        return self.start

    def is_goal(self, s):
        return s == self.goal

    def successors(self, s):
        for nxt, w in self.adj.get(s, []):
            yield (f"{s}->{nxt}", nxt, w)

    def heuristic(self, s):
        return self.h.get(s, 0.0)
