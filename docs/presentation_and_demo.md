# 10-minute talk + 5-minute live demo

## Talk (10 min)
| min | slide |
|---|---|
| 0-1 | Problem + the 7-vs-1 example |
| 1-2 | Pipeline diagram (NN -> AC-3 -> search -> CSP) |
| 2-3.5 | NN: accuracy vs baselines, calibration (reliability diagram), why probabilities matter (costs) |
| 3.5-5 | Search: state/action/cost, UCS vs Greedy vs A*, heuristic idea |
| 5-6.5 | CSP: BT vs FC vs AC-3, counters |
| 6.5-8.5 | Experiments 1-3 (one plot each) |
| 8.5-9.5 | Limits (goal "solvable" vs "unique", candidate cap, synthetic mosaics) |
| 9.5-10 | Conclusions |

## Live demo (5 min) - commands in order
```bash
python run_demo.py --seed 3 --misreads 2 --trace                 # 1. search trace + corrections + solution (1.5 min)
python run_demo.py --seed 3 --misreads 2 --no-fc --no-ac3        # 2. CSP without FC/AC-3: same answer, many more assignments (1 min)
python run_demo.py --seed 3 --misreads 2                         #    ... and with FC + AC-3 (compare the counters / heat maps)
python run_demo.py --seed 3 --misreads 2 --no-nn                 # 3. system with the NN's confidence off -> usually "unreadable" (1 min)
python run_demo.py --seed 5 --misreads 3 --strategy astar        # 4. A* vs UCS on a harder case (1 min)
```
Rehearse with your real MNIST model first: pick seeds where the story is clean and write them down.
Open `results/demo.png` as backup in case the live run fails.
