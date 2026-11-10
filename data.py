"""Data: MNIST download/preprocessing, Sudoku puzzle generator, 9x9 image mosaics, hard-digit injection.

Real  : digit images and their labels (MNIST).
Generated: Sudoku puzzles (randomised backtracking + uniqueness-preserving removal), the 252x252
          mosaics, blank tiles (low-amplitude noise), and *which* digits are swapped for hard ones.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import random
import struct
import urllib.request
from dataclasses import dataclass, field

import numpy as np

from .csp import SolverConfig, solve, count_solutions

TILE = 28
BLANK_MAX_INTENSITY = 0.3        # brightness rule: a tile whose brightest pixel is < 0.3 is blank

MNIST_FILES = {
    "train-images-idx3-ubyte.gz": 60000,
    "train-labels-idx1-ubyte.gz": 60000,
    "t10k-images-idx3-ubyte.gz": 10000,
    "t10k-labels-idx1-ubyte.gz": 10000,
}
MNIST_MIRRORS = [
    "https://storage.googleapis.com/cvdf-datasets/mnist/",
    "https://ossci-datasets.s3.amazonaws.com/mnist/",
    "http://yann.lecun.com/exdb/mnist/",
]


# ---------------------------------------------------------------------------------------- MNIST --
def _download(raw_dir):
    os.makedirs(raw_dir, exist_ok=True)
    used = {}
    for fname in MNIST_FILES:
        path = os.path.join(raw_dir, fname)
        if not os.path.exists(path):
            last = None
            for base in MNIST_MIRRORS:
                try:
                    print(f"downloading {base}{fname}")
                    with urllib.request.urlopen(base + fname, timeout=60) as r, open(path, "wb") as f:
                        f.write(r.read())
                    used[fname] = base
                    break
                except Exception as e:           # try the next mirror
                    last = e
                    if os.path.exists(path):
                        os.remove(path)
            else:
                raise RuntimeError(f"could not download {fname} from any mirror: {last}. "
                                   f"Download the four MNIST .gz files manually into {raw_dir}/")
        else:
            used[fname] = "(already present)"
    return used


def _read_idx(path):
    with gzip.open(path, "rb") as f:
        magic = struct.unpack(">I", f.read(4))[0]
        if magic == 2051:
            n, r, c = struct.unpack(">III", f.read(12))
            return np.frombuffer(f.read(), dtype=np.uint8).reshape(n, r, c)
        if magic == 2049:
            n = struct.unpack(">I", f.read(4))[0]
            return np.frombuffer(f.read(), dtype=np.uint8).reshape(n)
    raise ValueError(f"{path}: not an IDX file (magic {magic})")


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def _digits_dev_source():
    """OFFLINE DEV ONLY: scikit-learn's 8x8 digits, upsampled to 28x28. Not MNIST."""
    from scipy.ndimage import zoom
    from sklearn.datasets import load_digits
    d = load_digits()
    imgs = np.stack([np.clip(zoom(im / 16.0, 28 / 8, order=1), 0, 1) for im in d.images])
    return (imgs * 255).astype(np.uint8), d.target.astype(np.int64)


def prepare_dataset(out_path="data/dataset.npz", raw_dir="data/raw", seed=0, source="mnist"):
    """Download (if needed), split, drop digit 0, save a single .npz and a manifest.json.

    MNIST: the 60,000 training images are shuffled with `seed` and split 50,000 train / 10,000
    validation; the 10,000 test images are left untouched. Digit-0 images are then removed from every
    split (blank cells are detected by the brightness rule, not by the network). Labels are 1..9.
    """
    rng = np.random.default_rng(seed)
    manifest = {"source": source, "seed": seed}
    if source == "mnist":
        manifest["mirrors_used"] = _download(raw_dir)
        manifest["sha256"] = {f: _sha256(os.path.join(raw_dir, f)) for f in MNIST_FILES}
        xtr = _read_idx(os.path.join(raw_dir, "train-images-idx3-ubyte.gz"))
        ytr = _read_idx(os.path.join(raw_dir, "train-labels-idx1-ubyte.gz")).astype(np.int64)
        xte = _read_idx(os.path.join(raw_dir, "t10k-images-idx3-ubyte.gz"))
        yte = _read_idx(os.path.join(raw_dir, "t10k-labels-idx1-ubyte.gz")).astype(np.int64)
        for name, arr, n in (("train", xtr, 60000), ("test", xte, 10000)):
            if len(arr) != n:
                raise ValueError(f"unexpected MNIST {name} size {len(arr)}")
        perm = rng.permutation(len(xtr))
        tr_idx, va_idx = perm[:50000], perm[50000:]
        splits = {"train": (xtr[tr_idx], ytr[tr_idx]), "val": (xtr[va_idx], ytr[va_idx]), "test": (xte, yte)}
    elif source == "digits":
        x, y = _digits_dev_source()
        manifest["warning"] = "DEV ONLY: sklearn 8x8 digits upsampled, NOT MNIST. Do not report results from it."
        perm = rng.permutation(len(x))
        a, b = int(0.6 * len(x)), int(0.8 * len(x))
        splits = {"train": (x[perm[:a]], y[perm[:a]]), "val": (x[perm[a:b]], y[perm[a:b]]),
                  "test": (x[perm[b:]], y[perm[b:]])}
    else:
        raise ValueError("source must be 'mnist' or 'digits'")
    arrays = {}
    for name, (x, y) in splits.items():
        keep = y != 0
        arrays[f"x_{name}"], arrays[f"y_{name}"] = x[keep], y[keep]
        manifest[f"n_{name}"] = int(keep.sum())
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    np.savez_compressed(out_path, **arrays)
    with open(os.path.join(os.path.dirname(out_path) or ".", "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def load_dataset(path="data/dataset.npz"):
    """Images as float32 in [0,1], shape (N, 784); labels 1..9."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found - run `python make_data.py` first")
    d = np.load(path)
    out = {}
    for s in ("train", "val", "test"):
        out[f"x_{s}"] = (d[f"x_{s}"].reshape(len(d[f"x_{s}"]), -1) / 255.0).astype(np.float32)
        out[f"y_{s}"] = d[f"y_{s}"].astype(np.int64)
    return out


# ------------------------------------------------------------------------------ puzzle generator --
def random_full_grid(rng: random.Random):
    res = solve([0] * 81, SolverConfig(True, False, "mrv"), rng=rng)
    return [v for row in res.solution for v in row]


def make_puzzle(rng: random.Random, n_givens: int, max_attempts: int = 200):
    """Return (puzzle, solution) as flat lists with a UNIQUE solution and exactly n_givens givens."""
    if not 17 <= n_givens <= 81:
        raise ValueError("n_givens must be in 17..81 (17 is the known minimum for a unique 9x9 puzzle)")
    solution = random_full_grid(rng)
    for attempt in range(max_attempts):
        if attempt % 10 == 9:
            solution = random_full_grid(rng)
        puzzle, givens = solution[:], 81
        order = list(range(81))
        rng.shuffle(order)
        for c in order:
            if givens <= n_givens:
                break
            saved, puzzle[c] = puzzle[c], 0
            if count_solutions(puzzle, 2) == 1:
                givens -= 1
            else:
                puzzle[c] = saved
        if givens == n_givens:
            return puzzle, solution
    raise RuntimeError(f"could not reach {n_givens} givens with a unique solution")


# ------------------------------------------------------------------------- mosaics & hard digits --
@dataclass
class DigitPool:
    easy: dict           # digit -> indices of correctly classified test images
    hard: dict           # digit -> indices of hard test images (see build_pool)


def build_pool(P_test, y_test):
    """Hard-digit rule (documented): a test image is HARD if the trained network misclassifies it but the
    true digit is still among its top-3 probabilities. EASY = correctly classified. If a digit has no hard
    image, the 20 images with the lowest true-class probability are used instead."""
    pred = P_test.argmax(1) + 1
    p_true = P_test[np.arange(len(y_test)), y_test - 1]
    rank = (P_test > p_true[:, None]).sum(1)
    easy, hard = {}, {}
    for d in range(1, 10):
        cls = y_test == d
        easy[d] = np.where(cls & (pred == d))[0]
        hard[d] = np.where(cls & (pred != d) & (rank <= 2))[0]
        if len(hard[d]) == 0:
            idx = np.where(cls)[0]
            hard[d] = idx[np.argsort(p_true[idx])[:20]]
    return DigitPool(easy, hard)


@dataclass
class Instance:
    puzzle: list
    solution: list
    mosaic: np.ndarray                 # (252, 252) float32
    injected: list                     # cells where a hard digit was placed
    tile_source: list = field(default_factory=list)   # test-image index per cell (-1 = blank)


def make_instance(puzzle, solution, x_test, pool: DigitPool, n_hard: int, rng: random.Random):
    givens = [i for i in range(81) if puzzle[i]]
    injected = sorted(rng.sample(givens, min(n_hard, len(givens))))
    nprng = np.random.default_rng(rng.getrandbits(32))
    mosaic = np.zeros((9 * TILE, 9 * TILE), dtype=np.float32)
    source = []
    for i in range(81):
        r, c = divmod(i, 9)
        if puzzle[i] == 0:
            tile = nprng.uniform(0.0, 0.05, (TILE, TILE)).astype(np.float32)   # blank + sensor noise
            source.append(-1)
        else:
            idx = int(rng.choice(pool.hard[puzzle[i]] if i in injected else pool.easy[puzzle[i]]))
            tile = x_test[idx].reshape(TILE, TILE)
            source.append(idx)
        mosaic[r * TILE:(r + 1) * TILE, c * TILE:(c + 1) * TILE] = tile
    return Instance(list(puzzle), list(solution), mosaic, injected, source)


@dataclass
class Reading:
    grid: list                  # 81 ints, 0 = blank
    probs: np.ndarray           # (81, 9); rows of blank cells are all zero
    blank: np.ndarray           # (81,) bool


def read_mosaic(mosaic, model) -> Reading:
    """Slice the mosaic into 81 tiles, mark blanks with the brightness rule, classify the rest."""
    tiles = mosaic.reshape(9, TILE, 9, TILE).transpose(0, 2, 1, 3).reshape(81, TILE * TILE)
    blank = tiles.max(axis=1) < BLANK_MAX_INTENSITY
    probs = np.zeros((81, 9), dtype=np.float64)
    if (~blank).any():
        probs[~blank] = model.predict_proba(tiles[~blank])
    grid = [0 if blank[i] else int(probs[i].argmax()) + 1 for i in range(81)]
    return Reading(grid, probs, blank)
