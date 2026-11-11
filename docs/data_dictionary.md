# Data dictionary

"Real" = comes from MNIST unchanged. "Generated" = produced by our code from a seed.
`make_data.py` writes `data/dataset.npz` and `data/manifest.json` (source URL used, SHA-256 of the four raw
files, seed, split sizes) so a run can be reproduced and audited.

## 1. `data/dataset.npz` (output of `make_data.py`)

| array | shape / dtype | real or generated | meaning |
|---|---|---|---|
| `x_train` | (N≈45 000, 28, 28) uint8 | real | MNIST training images, 0 = black background, 255 = ink; digit-0 images removed |
| `y_train` | (N,) int64 | real | digit label 1-9 |
| `x_val`, `y_val` | (N≈9 000, …) | real | validation split: 10 000 of the 60 000 MNIST training images chosen by a seeded permutation (then digit 0 removed) |
| `x_test`, `y_test` | (N≈9 000, …) | real | the official 10 000 test images (different writers than training), untouched, digit 0 removed |

*Generated part:* only the split (seeded permutation, default seed 0). `load_dataset()` rescales pixels to float32 in [0, 1] and flattens to 784 values.
Offline dev fallback `--source digits` (scikit-learn 8×8 digits upsampled to 28×28, 60/20/20 split) is for testing the code only; never report results from it.

## 2. Generated Sudoku data (in memory, rebuilt from seeds)

| object / field | type | generated how |
|---|---|---|
| `puzzle` | list of 81 ints, 0 = blank | full grid by randomised backtracking (`random_full_grid`), then cells removed in random order while the solution stays unique (`count_solutions == 1`) until exactly *n* givens remain |
| `solution` | list of 81 ints 1-9 | the full grid the puzzle was cut from (the unique solution) |
| `Instance.mosaic` | (252, 252) float32 | 9×9 tiles of 28×28. A given cell = a real MNIST **test** image of that digit; a blank cell = uniform noise in [0, 0.05] |
| `Instance.injected` | list of cell indices 0-80 | cells chosen at random among the givens to receive a **hard digit** |
| `Instance.tile_source` | list of 81 ints | index into the test set of the image used (−1 = blank) |
| hard-digit rule | | a test image is *hard* if the trained network misclassifies it but the true digit is still in its top-3 probabilities; *easy* = correctly classified. Ordinary cells draw a random easy image of their digit. (If a digit has no hard image, the 20 lowest-true-probability images are used instead.) |
| blank rule | | tile brightest pixel < 0.3 → blank (no network involved) |
| `Reading.grid` | 81 ints | network argmax per non-blank tile (+1), 0 for blanks |
| `Reading.probs` | (81, 9) float | softmax probabilities (temperature-scaled); rows of blank cells are 0 |
| `Reading.blank` | (81,) bool | brightness rule result |

Cell index = 9·row + col (0-based); reports print `r{row}c{col}` 1-based.

## 3. Result files written by the scripts (`results/`)

`nn_metrics.json` – test accuracy / NLL / Brier / ECE for MLP raw, MLP temperature-scaled, logistic regression, majority class; fitted temperature; training history; confusion matrix.

`exp1_<goal>_raw.csv` (one row per puzzle × strategy × misread level × seed)

| column | meaning |
|---|---|
| seed, misreads, strategy | run seed, number of injected hard digits (0-4), ucs / greedy / astar |
| recovered | 1 if the returned solution equals the true solution (0/1) |
| accepted | 1 if any solution was returned (it may be wrong) |
| nodes_expanded, nodes_generated | search counters (see `search.py` docstring) |
| csp_calls | number of corrected puzzles passed to the CSP goal test |
| seconds | wall-clock time of the search incl. CSP goal tests |
| cost | total correction cost of the answer (NaN if unreadable) |
| actual_misreads | cells where the network's argmax really differs from the truth (injected hard digits do not always get misread) |

`exp2_raw.csv` – seed, givens (25/30/35), variant (BT, BT+FC, BT+FC+AC3), assignments, backtracks, nodes, seconds, finished (0 = hit the assignment budget; counts are then a lower bound).

`exp3_<goal>_raw.csv` – seed, misreads, method (NN-guided (UCS) / no correction / random correction), recovered, accepted, wrong_answer (a solution was returned but it is not the true one).

`*.csv` / `*.md` without `_raw` = mean and sample standard deviation (ddof = 1) across seeds.
