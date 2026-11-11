# Sudoku Rescue

A handwritten-digit Sudoku reader that **repairs its own misreads**.
A neural network reads 81 digit images and gives probabilities; if a digit is misread the puzzle breaks
(e.g. two 1s in a row). **Search** proposes the cheapest, most likely corrections; a **CSP solver**
(backtracking + forward checking + AC-3) tests each corrected puzzle. The first solvable one wins.
If none exists the system says *unreadable* and lists the least confident cells.

```
mosaic (81 images) -> NN probabilities -> AC-3 finds a conflict -> UCS/Greedy/A* try corrections
                   -> CSP solves each candidate -> first solvable puzzle = answer (+ correction log)
```

> This project started from a GitHub Sudoku solver (`legacy/`: OpenCV + Keras on 100×100 images, no
> probabilities, no CSP, hard-coded paths). Only its idea is kept; the code was rewritten to meet the course
> checklist. Its camera/photo pipeline (`legacy/solver.py`) is **not** ported: the system reads generated
> mosaics of real MNIST digits, not photographs.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate      # Python 3.10+ (tested on 3.12)
pip install -r requirements.txt
```

## Exact run commands

```bash
python make_data.py                 # download MNIST (3 mirrors incl. yann.lecun.com), split with seed 0 -> data/dataset.npz
python train_nn.py                  # train MLP, temperature-scale, compare with baselines -> results/nn_*.{json,png,md}
python run_demo.py                  # integrated demo -> console + results/demo.png
python run_experiments.py           # 3 experiments, 5 seeds x 20 puzzles -> results/exp*.{csv,md,png}
python run_experiments.py --goal unique    # same with the uniqueness goal (optional extension)
python -m pytest                    # all tests (no dataset needed)
```

Useful demo switches: `--seed 5 --misreads 3`, `--strategy astar|greedy|ucs`, `--trace`
(print the search trace), `--no-nn` (system with the NN's confidence switched off),
`--no-fc --no-ac3` (CSP with plain backtracking), `--goal unique`, `--show`.
`python run_experiments.py --quick` is a 10-second smoke test (2 seeds × 4 puzzles).

## Layout

| path | role |
|---|---|
| `sudoku_rescue/data.py` | MNIST download + split, puzzle generator, mosaics, hard-digit rule, `read_mosaic` |
| `sudoku_rescue/nn.py` | NumPy MLP (784-128-64-9, Adam), ECE / reliability / confusion matrix, temperature scaling, baselines |
| `sudoku_rescue/search.py` | generic UCS / Greedy / A* with counters (nodes generated / expanded / pruned) |
| `sudoku_rescue/csp.py` | Sudoku CSP: BT, forward checking, AC-3 as separate switches; counters |
| `sudoku_rescue/integration.py` | correction problem, A* heuristic, `rescue()` pipeline, baselines |
| `sudoku_rescue/eval.py` | NN report, the 3 experiments, tables and plots |
| `tests/` | normal / edge / impossible cases, known-answer search test, A* admissibility check |
| `docs/` | data dictionary, report outline, presentation + demo script, contribution template |

## Design choices worth knowing (put these in the report)

* **Search state** = set of corrections (≤ 3), generated in a fixed candidate order so each set appears once.
  **Cost** = `log(p_read / p_new)` ≥ 0. Only the 24 least-confident givens are candidates (`max_candidates`),
  3 alternatives per cell, budget 3000 expansions: these are the "limits" to discuss.
* **A\* heuristic**: every duplicate-digit conflict needs ≥ 1 changed cell. We pick cell-disjoint conflicts and sum
  their cheapest available change. Admissible (tested against UCS on random cases); `inf` when a conflict can no longer be repaired.
* **Goal test**: "solvable" (the minimum version) or "unique" (`--goal unique`). *Known weakness of "solvable"*: a
  misread can leave a puzzle that is still solvable (with a wrong answer), so the search stops with 0 corrections.
  Experiments 1 and 3 report `recovered` (exact true solution) separately from `accepted`/`wrong_answer` for this reason.
* **CSP**: AC-3 runs once before search (root only). Plain BT also checks the givens for duplicates once; with AC-3 that
  same conflict shows up as an empty domain. Experiment 2 uses fixed row-major variable order and an assignment budget
  (200 000) so that BT vs FC vs AC-3 is not confounded by MRV ordering; runs that hit the budget are flagged (`finished`).
* **"NN off"** (demo `--no-nn`, `use_nn_costs=False`): the digits are still read, but confidences are thrown away,
  so every change costs the same and candidate cells are not ranked.
* **Hard digits**: real test images the network gets wrong but ranks the truth in its top-3 (see `docs/data_dictionary.md`).
  Because the pool is built from the same network, "k injected hard digits" ≈ k actual misreads (`actual_misreads` column records the truth).
* The NN is NumPy, not TensorFlow/PyTorch: tiny, no GPU, fully seeded, and keeps requirements small.

## Checklist coverage

| # | item | where |
|---|---|---|
| 1 | modules | `sudoku_rescue/` + `tests/` |
| 2 | reproducible data script | `make_data.py` (seeded; hashes in `data/manifest.json`) |
| 3 | data dictionary | `docs/data_dictionary.md` |
| 4 | NN + baselines | `train_nn.py` (logistic regression, majority class; accuracy, confusion matrix, ECE) |
| 5 | search + counters + known-answer test | `search.py`, `tests/test_search.py` |
| 6 | CSP BT / FC / AC-3 + counters | `csp.py`, `tests/test_csp.py` |
| 7 | integrated demo | `run_demo.py` |
| 8 | README, pinned requirements | this file, `requirements.txt` |
| 9 | tests (≥5 normal, ≥3 edge, ≥2 impossible) | `tests/` |
| 10 | experiments | `run_experiments.py` -> `results/` |


