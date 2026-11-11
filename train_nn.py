"""Train the digit classifier, calibrate it, compare with baselines, save plots + metrics.

    python train_nn.py                 # defaults: 12 epochs, seed 0
"""
import argparse

from sudoku_rescue.data import load_dataset
from sudoku_rescue.nn import train_and_evaluate
from sudoku_rescue import eval as ev

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/dataset.npz")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    ds = load_dataset(a.data)
    net, summary, probs = train_and_evaluate(ds, a.out, f"{a.out}/mlp.npz", a.epochs, a.seed)
    ev.write_nn_report(summary, probs, ds["y_test"] - 1, a.out)
    print(f"\ntemperature = {summary['temperature']:.3f}")
    print(ev.nn_table_markdown(summary))
