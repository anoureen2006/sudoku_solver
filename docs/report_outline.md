# Report outline (max 12 pages)

1. **Problem** (0.75 p): handwritten Sudoku; why one misread breaks the puzzle; example (7 read as 1, 52% vs 45%).
2. **System and design choices** (2 p): pipeline figure; search formulation (state, action, cost, goal, ≤3 corrections);
   CSP formulation (81 variables, domains, all-different as != arcs); why costs are −log probability ratios.
3. **Data** (1 p): MNIST (real) vs generated puzzles/mosaics/hard digits; splits; seeds. Point to `data_dictionary.md`.
4. **Neural network** (1.5 p): architecture, training, test table vs baselines (`results/nn_table.md`), confusion matrix,
   reliability diagram, effect of temperature scaling on ECE. Say whether the NN beats logistic regression *and by how much*.
5. **Search** (1.5 p): UCS/Greedy/A*; heuristic and the proof sketch of admissibility; counters; known-answer test.
6. **CSP** (1.5 p): BT, FC, AC-3; counters; what AC-3 detects; budget and fixed ordering in Experiment 2.
7. **Integration** (0.75 p): how the parts cooperate; "NN on vs off" ablation; unreadable handling.
8. **Experiments** (2.5 p): the three experiments - tables with mean ± std (5 seeds), plots, honest reading of results
   (e.g. recovery at 4 misreads is capped by the 3-correction limit; "solvable" vs "unique" goal).
9. **Limits** (0.5 p): synthetic mosaics, not photos; hard-digit pool built from the same network; candidate cap; Python speed;
   misreads that keep the puzzle solvable.
10. **Conclusions** (0.25 p).
