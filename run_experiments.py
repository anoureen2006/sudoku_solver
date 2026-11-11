"""Run the three experiments (tables as CSV + Markdown, plots as PNG) into results/.

    python run_experiments.py                              # 5 seeds x 20 puzzles (= 100 per condition)
    python run_experiments.py --quick                      # 2 seeds x 4 puzzles, smoke test only
    python run_experiments.py --only 1 --goal unique       # a single experiment / the uniqueness goal
"""
import argparse
import os
import time

from sudoku_rescue import eval as ev
from sudoku_rescue.data import load_dataset
from sudoku_rescue.nn import MLP

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/dataset.npz")
    ap.add_argument("--model", default="results/mlp.npz")
    ap.add_argument("--out", default="results")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--puzzles-per-seed", type=int, default=20)
    ap.add_argument("--goal", choices=["solvable", "unique"], default="solvable")
    ap.add_argument("--only", type=int, choices=[1, 2, 3])
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    if a.quick:
        a.seeds, a.puzzles_per_seed = 2, 4
    os.makedirs(a.out, exist_ok=True)
    ctx = ev.Context.build(MLP.load(a.model), load_dataset(a.data))
    tag = a.goal
    t0 = time.time()
    if a.only in (None, 1):
        df = ev.experiment1(ctx, a.seeds, a.puzzles_per_seed, goal=a.goal)
        df.to_csv(f"{a.out}/exp1_{tag}_raw.csv", index=False)
        agg = ev.plot_exp1(df, f"{a.out}/exp1_{tag}.png")
        agg.to_csv(f"{a.out}/exp1_{tag}.csv", index=False)
        md = ev.to_markdown(agg, ["misreads", "strategy"], ["recovered", "nodes_expanded", "nodes_generated", "csp_calls"],
                            {"recovered": "{:.3f}", "nodes_expanded": "{:.1f}", "nodes_generated": "{:.1f}", "csp_calls": "{:.1f}"})
        open(f"{a.out}/exp1_{tag}.md", "w").write(md + "\n"); print("\nEXPERIMENT 1 (goal=%s)\n" % a.goal + md)
    if a.only in (None, 2):
        df = ev.experiment2(ctx, a.seeds, max(a.puzzles_per_seed // 2, 2))
        df.to_csv(f"{a.out}/exp2_raw.csv", index=False)
        agg = ev.plot_exp2(df, f"{a.out}/exp2.png")
        agg.to_csv(f"{a.out}/exp2.csv", index=False)
        md = ev.to_markdown(agg, ["givens", "variant"], ["assignments", "backtracks", "seconds", "finished"],
                            {"assignments": "{:.0f}", "backtracks": "{:.0f}", "seconds": "{:.4f}", "finished": "{:.2f}"})
        open(f"{a.out}/exp2.md", "w").write(md + "\n"); print("\nEXPERIMENT 2\n" + md)
    if a.only in (None, 3):
        df = ev.experiment3(ctx, a.seeds, a.puzzles_per_seed, goal=a.goal)
        df.to_csv(f"{a.out}/exp3_{tag}_raw.csv", index=False)
        agg = ev.plot_exp3(df, f"{a.out}/exp3_{tag}.png")
        agg.to_csv(f"{a.out}/exp3_{tag}.csv", index=False)
        md = ev.to_markdown(agg, ["misreads", "method"], ["recovered", "wrong_answer"],
                            {"recovered": "{:.3f}", "wrong_answer": "{:.3f}"})
        open(f"{a.out}/exp3_{tag}.md", "w").write(md + "\n"); print("\nEXPERIMENT 3 (goal=%s)\n" % a.goal + md)
    print(f"\nfinished in {time.time() - t0:.0f}s; files in {a.out}/")
