"""One integrated demo: input -> NN reading -> AC-3 -> correction search -> CSP solution, with plots.

    python run_demo.py                          # 2 injected misreads, UCS, writes results/demo.png
    python run_demo.py --seed 5 --misreads 3 --strategy astar
    python run_demo.py --no-nn                  # ablation: the search gets no confidence information
    python run_demo.py --no-fc --no-ac3         # CSP stage with plain backtracking only
    python run_demo.py --trace --show           # print the search trace / open the figure window
"""
import argparse
import os
import random

import matplotlib
import numpy as np

from sudoku_rescue.csp import (SolverConfig, solve, format_grid, propagate_ac3, find_conflicts, to_grid, cell_name,
                               flatten)
from sudoku_rescue.data import load_dataset, make_instance, make_puzzle, read_mosaic, build_pool, TILE
from sudoku_rescue.integration import RescueConfig, rescue
from sudoku_rescue.nn import MLP


def pick_instance(ctx, seed, misreads, n_givens):
    """Find a puzzle where the injected hard digits really get misread and the reading looks broken."""
    rng = random.Random(seed)
    best = None
    for attempt in range(200):
        pz, sol = make_puzzle(rng, n_givens)
        inst = make_instance(pz, sol, ctx.x_test, ctx.pool, misreads, rng)
        rd = read_mosaic(inst.mosaic, ctx.model)
        wrong = [i for i in range(81) if pz[i] and rd.grid[i] != pz[i]]
        if misreads == 0 or (len(wrong) == misreads and (find_conflicts(rd.grid) or not solve(rd.grid).solved)):
            return inst, rd, wrong
        best = best or (inst, rd, wrong)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/dataset.npz")
    ap.add_argument("--model", default="results/mlp.npz")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--misreads", type=int, default=2)
    ap.add_argument("--givens", type=int, default=30)
    ap.add_argument("--strategy", choices=["ucs", "greedy", "astar"], default="ucs")
    ap.add_argument("--goal", choices=["solvable", "unique"], default="solvable")
    ap.add_argument("--no-nn", action="store_true", help="discard confidence information (NN off ablation)")
    ap.add_argument("--no-fc", action="store_true")
    ap.add_argument("--no-ac3", action="store_true")
    ap.add_argument("--trace", action="store_true")
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--out", default="results/demo.png")
    a = ap.parse_args()
    if a.show:
        matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from sudoku_rescue import eval as ev

    ctx = ev.Context.build(MLP.load(a.model), load_dataset(a.data))
    inst, rd, wrong = pick_instance(ctx, a.seed, a.misreads, a.givens)
    csp_cfg = SolverConfig(not a.no_fc, not a.no_ac3, "mrv")

    print("=" * 60, "\nINPUT: the true puzzle (what the 81 images are supposed to show)\n" + "=" * 60)
    print(format_grid(inst.puzzle))
    print(f"\nNN READING (temperature {ctx.model.temperature:.2f}); misread cells: "
          + (", ".join(f"{cell_name(i)} true {inst.puzzle[i]} read {rd.grid[i]} (conf {rd.probs[i].max():.2f})" for i in wrong) or "none"))
    print(format_grid(rd.grid))
    conflicts = find_conflicts(rd.grid)
    ok, cell, doms_before, ac_stats = propagate_ac3(rd.grid)
    print("\nAC-3 on the reading:", "consistent" if ok else f"CONFLICT, domain of {cell_name(cell)} became empty",
          f"({ac_stats.ac3_revisions} arcs revised, {ac_stats.ac3_prunings} digits pruned)")
    for c in conflicts:
        print("  duplicate:", c)

    cfg = RescueConfig(strategy=a.strategy, goal=a.goal, use_nn_costs=not a.no_nn, csp=csp_cfg)
    res = rescue(rd, cfg, trace=True)
    print("\n" + "=" * 60, f"\nSEARCH ({a.strategy}{', NN OFF' if a.no_nn else ''}, goal={a.goal})\n" + "=" * 60)
    s = res.search
    print(f"nodes expanded {s.nodes_expanded}, generated {s.nodes_generated}, pruned {s.nodes_pruned}, "
          f"max frontier {s.max_frontier}, CSP calls {res.csp_calls}, {s.seconds:.3f}s")
    if a.trace:
        for n, state, g, h in res.trace[:40]:
            print(f"  #{n:<3} g={g:6.3f} h={h:6.3f}  corrections={[(cell_name(c), d) for c, d in state]}")
    print("\n" + res.message)
    for c in res.corrections:
        print(f"  correction: r{c.row}c{c.col}  {c.read_digit} -> {c.new_digit}   cost {c.cost:.3f} "
              f"(p_read {c.p_read:.2f}, p_new {c.p_new:.2f})")
    if res.status == "solved":
        print("\nSOLUTION\n" + format_grid(res.solution))
        print("matches the true solution:", res.solution == to_grid(inst.solution))

    # ---- CSP pruning with / without FC and AC-3 on the corrected puzzle --------------------------
    target = res.final_grid or rd.grid
    rows, variants = [], [SolverConfig(False, False, "static"), SolverConfig(True, False, "static"),
                          SolverConfig(True, True, "static")]
    print("\nCSP stage on the corrected puzzle (static order, to expose the pruning effect):")
    for v in variants:
        r = solve(target, v)
        rows.append((v.label, r.stats))
        print(f"  {v.label:10s} assignments {r.stats.assignments:6d}  backtracks {r.stats.backtracks:6d}  "
              f"fc_prunings {r.stats.fc_prunings:5d}  ac3_prunings {r.stats.ac3_prunings:4d}  -> {r.status}")
    res_off = rescue(rd, RescueConfig(strategy=a.strategy, goal=a.goal, use_nn_costs=a.no_nn, csp=csp_cfg))
    print(f"\nSystem with NN {'ON' if a.no_nn else 'OFF'} for comparison: {res_off.status}, nodes expanded "
          f"{res_off.search.nodes_expanded}, CSP calls {res_off.csp_calls}")

    # ---- figure -------------------------------------------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    ax = axes[0, 0]
    ax.imshow(inst.mosaic, cmap="gray_r")
    for i in range(81):
        r_, c_ = divmod(i, 9)
        col = "red" if i in wrong else None
        if any(cell_name(i) == f"r{x.row}c{x.col}" for x in res.corrections):
            col = "limegreen"
        if col:
            ax.add_patch(Rectangle((c_ * TILE, r_ * TILE), TILE, TILE, fill=False, ec=col, lw=2.5))
    for k in range(10):
        ax.axhline(k * TILE - 0.5, color="lightgray", lw=0.5); ax.axvline(k * TILE - 0.5, color="lightgray", lw=0.5)
    ax.set_title("Input mosaic  (red = misread by NN, green = corrected)"); ax.axis("off")

    ax = axes[0, 1]
    ax.imshow(np.array([[rd.probs[r_ * 9 + c_].max() if not rd.blank[r_ * 9 + c_] else np.nan for c_ in range(9)] for r_ in range(9)]),
              cmap="RdYlGn", vmin=0.3, vmax=1.0)
    for i in range(81):
        if rd.grid[i]:
            ax.text(i % 9, i // 9, rd.grid[i], ha="center", va="center", fontsize=11, fontweight="bold" if i in wrong else None)
    ax.set_title("NN reading: digit and confidence (green = sure)"); ax.set_xticks([]); ax.set_yticks([])

    ax = axes[0, 2]
    for strat, col in (("ucs", "C0"), ("greedy", "C1"), ("astar", "C2")):
        r_ = rescue(rd, RescueConfig(strategy=strat, goal=a.goal, use_nn_costs=not a.no_nn, csp=csp_cfg), trace=True)
        gs = [g for _, _, g, _ in r_.trace]
        ax.plot(range(1, len(gs) + 1), gs, "o-", ms=3, color=col, label=f"{strat} ({r_.search.nodes_expanded} expanded)")
    ax.set_xlabel("expansion order"); ax.set_ylabel("path cost g of expanded node"); ax.legend(); ax.set_title("Search trace (first 500 expansions)")

    def dom_map(domains, ax, title):
        sizes = np.array([bin(d).count("1") for d in domains]).reshape(9, 9)
        ax.imshow(sizes, cmap="viridis_r", vmin=1, vmax=9)
        for i, d in enumerate(domains):
            ax.text(i % 9, i // 9, bin(d).count("1"), ha="center", va="center", fontsize=8, color="white")
        ax.set_title(title); ax.set_xticks([]); ax.set_yticks([])

    _, _, doms_after, st = propagate_ac3(target)
    dom_map([(1 << (v - 1)) if v else 0x1FF for v in flatten(target)], axes[1, 0], "Domain sizes: corrected puzzle, before AC-3")
    dom_map(doms_after, axes[1, 1], f"After AC-3 ({st.ac3_prunings} digits pruned)")

    ax = axes[1, 2]
    labels = [l for l, _ in rows]
    x = np.arange(len(rows))
    ax.bar(x - 0.2, [max(s_.assignments, 1) for _, s_ in rows], 0.4, label="assignments")
    ax.bar(x + 0.2, [max(s_.backtracks, 1) for _, s_ in rows], 0.4, label="backtracks")
    ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_yscale("log"); ax.legend(); ax.set_title("CSP effort on the corrected puzzle")
    fig.suptitle(f"Sudoku Rescue demo  |  seed {a.seed}, {a.misreads} injected hard digits, {a.strategy}, "
                 f"{'NN off' if a.no_nn else 'NN on'}  |  {res.message}", fontsize=11)
    fig.tight_layout()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    fig.savefig(a.out, dpi=130)
    print("\nfigure saved to", a.out)
    if a.show:
        plt.show()


if __name__ == "__main__":
    main()
