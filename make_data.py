"""Download and preprocess the dataset reproducibly.

    python make_data.py                    # MNIST -> data/dataset.npz  (seed 0)
    python make_data.py --seed 7           # different train/validation split
    python make_data.py --source digits    # OFFLINE DEV ONLY (sklearn 8x8 digits); never report results from it
"""
import argparse
import json

from sudoku_rescue.data import prepare_dataset

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["mnist", "digits"], default="mnist")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/dataset.npz")
    ap.add_argument("--raw-dir", default="data/raw")
    a = ap.parse_args()
    m = prepare_dataset(a.out, a.raw_dir, a.seed, a.source)
    print(json.dumps({k: v for k, v in m.items() if k != "sha256"}, indent=2))
    print("saved", a.out, "(+ manifest.json with file hashes)")
