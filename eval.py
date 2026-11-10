"""Evaluation: NN report (accuracy, confusion matrix, calibration vs baselines) and the three experiments.

Experiment protocol (all three): `seeds` independent runs, each with its own generated puzzles
(seeded), `puzzles_per_seed` puzzles per condition. Per-seed values are averaged first; tables give the
mean and sample standard deviation (ddof=1) ACROSS SEEDS. Within one seed every method sees exactly the
same puzzles and the same mosaics, so differences are due to the method only.
"""
from __future__ import annotations

import os
import random
from dataclasses import dataclass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .csp import SolverConfig, solve, to_grid
from .data import build_pool, make_instance, make_puzzle, read_mosaic
from .integration import RescueConfig, rescue, no_correction, random_correction
from .nn import MLP, reliability_curve, expected_calibration_error


# ------------------------------------------------------------------------------------- NN report --
def nn_table_markdown(summary):
    rows = ["| model | accuracy | NLL | Brier | ECE |", "|---|---|---|---|---|"]
    for name, m in summary["test_metrics"].items():
        rows.append(f"| {name} | {m['accuracy']:.4f} | {m['nll']:.4f} | {m['brier']:.4f} | {m['ece']:.4f} |")
    return "\n".join(rows)


def write_nn_report(summary, probs, y_test0, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    cm = np.array(summary["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(9)); ax.set_xticklabels(range(1, 10)); ax.set_yticks(range(9)); ax.set_yticklabels(range(1, 10))
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title("Confusion matrix (test, temperature-scaled MLP)")
    for i in range(9):
        for j in range(9):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=7, color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "nn_confusion.png"), dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect")
    for key, label in (("raw", "MLP raw"), ("cal", "MLP temp-scaled"), ("logistic regression", "logistic regression")):
        bc, ba, bn = reliability_curve(probs[key], y_test0)
        ok = bn > 0
        ax.plot(bc[ok], ba[ok], "o-", ms=3, label=f"{label} (ECE {expected_calibration_error(probs[key], y_test0):.3f})")
    ax.set_xlabel("confidence"); ax.set_ylabel("accuracy"); ax.set_title("Reliability diagram (test)"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "nn_reliability.png"), dpi=150); plt.close(fig)

    h = pd.DataFrame(summary["history"])
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.plot(h.epoch, h.train_loss, label="train loss"); ax.plot(h.epoch, h.val_loss, label="validation loss")
    ax.set_xlabel("epoch"); ax.legend(); ax.set_title("Training curve")
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "nn_training.png"), dpi=150); plt.close(fig)
    with open(os.path.join(out_dir, "nn_table.md"), "w") as f:
        f.write(nn_table_markdown(summary) + "\n")


# ------------------------------------------------------------------------------------ utilities --
@dataclass
class Context:
    model: MLP
    x_test: np.ndarray
    y_test: np.ndarray
    pool: object

    @classmethod
    def build(cls, model, dataset):
        P = model.predict_proba(dataset["x_test"])
        return cls(model, dataset["x_test"], dataset["y_test"], build_pool(P, dataset["y_test"]))


def _instances(ctx, seed, n, n_givens, level, tag="main"):
    """Puzzles depend only on (seed, n_givens, i); mosaics on (seed, n_givens, i, level)."""
    out = []
    prng = random.Random(f"puzzles-{seed}-{n_givens}")
    for i in range(n):
        pz, sol = make_puzzle(prng, n_givens)
        inst = make_instance(pz, sol, ctx.x_test, ctx.pool, level, random.Random(f"mosaic-{seed}-{n_givens}-{i}-{level}"))
        out.append((inst, read_mosaic(inst.mosaic, ctx.model)))
    return out


def aggregate(df, keys, metrics):
    """mean and sample std across seeds for each metric, grouped by `keys`."""
    g = df.groupby(keys + ["seed"], as_index=False)[metrics].mean()
    mean = g.groupby(keys)[metrics].mean()
    std = g.groupby(keys)[metrics].std(ddof=1).fillna(0.0)
    mean.columns = [f"{c}_mean" for c in metrics]
    std.columns = [f"{c}_std" for c in metrics]
    return pd.concat([mean, std], axis=1).reset_index()


def to_markdown(agg, keys, metrics, fmt=None):
    fmt = fmt or {}
    head = keys + metrics
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for _, r in agg.iterrows():
        cells = [str(r[k]) for k in keys]
        for m in metrics:
            f = fmt.get(m, "{:.2f}")
            cells.append(f.format(r[f"{m}_mean"]) + " ± " + f.format(r[f"{m}_std"]))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


# ----------------------------------------------------------------------------- experiment 1 ------
def experiment1(ctx, seeds=5, puzzles_per_seed=20, levels=(0, 1, 2, 3, 4), strategies=("ucs", "greedy", "astar"),
                goal="solvable", n_givens=30, max_corrections=3, max_expansions=3000, progress=print):
    rows = []
    for s in range(seeds):
        for k in levels:
            data = _instances(ctx, s, puzzles_per_seed, n_givens, k)
            for strat in strategies:
                for inst, rd in data:
                    cfg = RescueConfig(strategy=strat, goal=goal, max_corrections=max_corrections, max_expansions=max_expansions)
                    r = rescue(rd, cfg)
                    exact = r.status == "solved" and r.solution == to_grid(inst.solution)
                    rows.append(dict(seed=s, misreads=k, strategy=strat, recovered=float(exact), accepted=float(r.status == "solved"),
                                     nodes_expanded=r.search.nodes_expanded, nodes_generated=r.search.nodes_generated,
                                     csp_calls=r.csp_calls, seconds=r.search.seconds,
                                     cost=(r.total_cost if r.status == "solved" else np.nan),
                                     actual_misreads=sum(1 for i in range(81) if inst.puzzle[i] and rd.grid[i] != inst.puzzle[i])))
        progress(f"[exp1 goal={goal}] seed {s + 1}/{seeds} done")
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- experiment 2 ------
VARIANTS = [SolverConfig(False, False, "static"), SolverConfig(True, False, "static"), SolverConfig(True, True, "static")]


def experiment2(ctx, seeds=5, puzzles_per_seed=10, givens=(25, 30, 35), budget=200_000, order="static", progress=print):
    rows = []
    for s in range(seeds):
        for ng in givens:
            data = _instances(ctx, s, puzzles_per_seed, ng, 0)
            for base in VARIANTS:
                cfg = SolverConfig(base.forward_checking, base.ac3, order, budget)
                for inst, _ in data:
                    r = solve(inst.puzzle, cfg)
                    rows.append(dict(seed=s, givens=ng, variant=cfg.label, assignments=r.stats.assignments,
                                     backtracks=r.stats.backtracks, nodes=r.stats.nodes, seconds=r.stats.seconds,
                                     finished=float(r.status != "budget_exceeded")))
        progress(f"[exp2] seed {s + 1}/{seeds} done")
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- experiment 3 ------
def experiment3(ctx, seeds=5, puzzles_per_seed=20, levels=(0, 1, 2, 3), goal="solvable", n_givens=30,
                max_corrections=3, random_trials=1000, progress=print):
    rows = []
    for s in range(seeds):
        for k in levels:
            data = _instances(ctx, s, puzzles_per_seed, n_givens, k)
            for i, (inst, rd) in enumerate(data):
                truth = to_grid(inst.solution)
                r = rescue(rd, RescueConfig(strategy="ucs", goal=goal, max_corrections=max_corrections))
                outcomes = {"NN-guided (UCS)": r.solution if r.status == "solved" else None}
                sol, _ = no_correction(rd)
                outcomes["no correction"] = sol
                rsol, _ = random_correction(rd, random.Random(f"rand-{s}-{i}-{k}"), random_trials, max_corrections)
                outcomes["random correction"] = rsol
                for name, sol in outcomes.items():
                    rows.append(dict(seed=s, misreads=k, method=name, recovered=float(sol == truth if sol else 0.0),
                                     accepted=float(sol is not None), wrong_answer=float(sol is not None and sol != truth)))
        progress(f"[exp3 goal={goal}] seed {s + 1}/{seeds} done")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------- plots --
def plot_exp1(df, out):
    keys, metrics = ["misreads", "strategy"], ["recovered", "nodes_expanded", "nodes_generated", "csp_calls"]
    agg = aggregate(df, keys, metrics)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for strat, g in agg.groupby("strategy"):
        axes[0].errorbar(g.misreads, g.recovered_mean, g.recovered_std, marker="o", capsize=3, label=strat)
        axes[1].errorbar(g.misreads, g.nodes_expanded_mean + 1, g.nodes_expanded_std, marker="o", capsize=3, label=strat)
    axes[0].set_xlabel("injected hard digits"); axes[0].set_ylabel("exact-solution recovery rate"); axes[0].set_ylim(-0.02, 1.02)
    axes[1].set_xlabel("injected hard digits"); axes[1].set_ylabel("nodes expanded (+1, log scale)"); axes[1].set_yscale("log")
    axes[0].legend(); axes[0].set_title("Recovery (mean ± std over seeds)"); axes[1].set_title("Search effort")
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    return agg


def plot_exp2(df, out):
    keys, metrics = ["givens", "variant"], ["assignments", "backtracks", "seconds", "finished"]
    agg = aggregate(df, keys, metrics)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    variants = list(dict.fromkeys(df.variant))
    for ax, m, lab in ((axes[0], "assignments", "assignments (log)"), (axes[1], "backtracks", "backtracks (log)")):
        w = 0.8 / len(variants)
        for j, v in enumerate(variants):
            g = agg[agg.variant == v].sort_values("givens")
            ax.bar(np.arange(len(g)) + j * w, g[f"{m}_mean"] + 1, w, yerr=g[f"{m}_std"], capsize=3, label=v)
        ax.set_xticks(np.arange(len(g)) + w); ax.set_xticklabels(sorted(set(df.givens))); ax.set_yscale("log")
        ax.set_xlabel("givens"); ax.set_ylabel(lab)
    axes[0].legend(); axes[0].set_title("CSP effort by technique (mean ± std over seeds)")
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    return agg


def plot_exp3(df, out):
    keys, metrics = ["misreads", "method"], ["recovered", "wrong_answer"]
    agg = aggregate(df, keys, metrics)
    methods = list(dict.fromkeys(df.method))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, m, lab in ((axes[0], "recovered", "exact-solution recovery rate"), (axes[1], "wrong_answer", "confidently wrong answers")):
        w = 0.8 / len(methods)
        for j, name in enumerate(methods):
            g = agg[agg.method == name].sort_values("misreads")
            ax.bar(np.arange(len(g)) + j * w, g[f"{m}_mean"], w, yerr=g[f"{m}_std"], capsize=3, label=name)
        ax.set_xticks(np.arange(len(g)) + w); ax.set_xticklabels(sorted(set(df.misreads)))
        ax.set_xlabel("injected hard digits"); ax.set_ylabel(lab); ax.set_ylim(0, 1.05)
    axes[0].legend(fontsize=8); axes[0].set_title("NN-guided vs no vs random correction")
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    return agg
