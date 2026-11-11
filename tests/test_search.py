import math
from sudoku_rescue.search import best_first_search, GraphProblem

EDGES = [("S", "A", 1), ("S", "B", 4), ("A", "B", 2), ("A", "C", 5), ("B", "C", 1), ("C", "G", 3)]
H = {"S": 6, "A": 6, "B": 4, "C": 3, "G": 0}      # admissible and consistent


def test_ucs_known_answer_and_counters():
    r = best_first_search(GraphProblem(EDGES, "S", "G"), "ucs")
    assert r.found and r.cost == 7
    assert r.actions == ["S->A", "A->B", "B->C", "C->G"]
    # Hand trace (undirected graph, goal test on pop):
    #   pop S(0)  -> generates A, B                      (+3 incl. root)
    #   pop A(1)  -> generates S, B(3), C(6)             (+3)
    #   pop B(3)  -> generates S, A, C(4)                (+3)
    #   pop C(4)  -> generates A, B, G(7)                (+3)
    #   stale B(4) and C(6) are skipped; pop G(7) = goal
    # => 5 nodes popped/goal-tested (S, A, B, C, G) and 1 (root) + 2 + 3 + 3 + 3 = 12 generated
    assert r.stats.nodes_expanded == 5 and r.stats.nodes_generated == 12 and r.stats.goal_tests == 5


def test_astar_same_optimal_cost_fewer_expansions():
    ucs = best_first_search(GraphProblem(EDGES, "S", "G"), "ucs")
    astar = best_first_search(GraphProblem(EDGES, "S", "G", H), "astar")
    assert astar.cost == ucs.cost == 7
    assert astar.stats.nodes_expanded <= ucs.stats.nodes_expanded


def test_greedy_finds_a_path_but_not_necessarily_optimal():
    g = {"S": 3, "A": 5, "B": 1, "C": 1, "G": 0}
    r = best_first_search(GraphProblem(EDGES, "S", "G", g), "greedy")
    assert r.found and r.cost >= 7


def test_unreachable_goal_exhausts():
    r = best_first_search(GraphProblem([("S", "A", 1)], "S", "Z"), "ucs")
    assert not r.found and r.reason == "exhausted" and math.isinf(r.cost)


def test_budget_stops_search():
    r = best_first_search(GraphProblem(EDGES, "S", "G"), "ucs", max_expansions=2)
    assert not r.found and r.reason == "budget"
