"""NN and data tests (synthetic data; no download required)."""
import random

import numpy as np

from sudoku_rescue.csp import count_solutions, flatten
from sudoku_rescue.data import (make_puzzle, make_instance, read_mosaic, build_pool, DigitPool, TILE)
from sudoku_rescue.nn import (MLP, softmax, expected_calibration_error, fit_temperature, evaluate, nll,
                              confusion_matrix)


def blobs(n_per_class, seed):
    protos = np.random.default_rng(123).uniform(0, 1, (9, 784))   # same 9 class prototypes for every split
    rng = np.random.default_rng(seed)
    X = np.concatenate([np.clip(protos[k] + rng.normal(0, 0.6, (n_per_class, 784)), 0, 1) for k in range(9)]).astype(np.float32)
    y = np.repeat(np.arange(9), n_per_class)
    p = rng.permutation(len(y))
    return X[p], y[p]


def test_mlp_learns_and_roundtrips():
    import os, tempfile
    Xtr, ytr = blobs(60, 0); Xva, yva = blobs(30, 1); Xte, yte = blobs(30, 2)
    net = MLP(seed=0)
    net.fit(Xtr, ytr, Xva, yva, epochs=8, verbose=False)
    P = net.predict_proba(Xte)
    assert np.allclose(P.sum(1), 1) and (P >= 0).all()
    assert (P.argmax(1) == yte).mean() > 0.9
    path = os.path.join(tempfile.mkdtemp(), "m.npz")
    net.save(path)
    assert np.allclose(MLP.load(path).predict_proba(Xte), P, atol=1e-6)


def test_temperature_scaling_does_not_hurt_validation_nll():
    Xtr, ytr = blobs(40, 3); Xva, yva = blobs(40, 4)
    net = MLP(seed=1); net.fit(Xtr, ytr, Xva, yva, epochs=10, verbose=False)
    z = net.logits(Xva)
    T = fit_temperature(z, yva)
    assert nll(softmax(z / T), yva) <= nll(softmax(z), yva) + 1e-9


def test_metrics_basic():
    P = np.eye(9)[[0, 1, 2]] * 0.9 + 0.1 / 9
    y = np.array([0, 1, 0])
    m = evaluate(P, y)
    assert abs(m["accuracy"] - 2 / 3) < 1e-9 and 0 <= m["ece"] <= 1
    assert confusion_matrix(y, P.argmax(1)).sum() == 3 and expected_calibration_error(P, y) > 0


def test_generated_puzzles_are_unique_with_requested_givens_and_reproducible():
    a = make_puzzle(random.Random(11), 30)
    b = make_puzzle(random.Random(11), 30)
    assert a == b                                    # fixed seed -> identical puzzle
    pz, sol = a
    assert sum(1 for v in pz if v) == 30 and count_solutions(pz, 2) == 1
    assert all(p == 0 or p == s for p, s in zip(pz, sol))


class StubModel:
    """digit d is drawn as a constant tile of value 0.35 + 0.05 d, so reading is exact and testable."""
    temperature = 1.0
    def predict_proba(self, X):
        d = np.rint((X.max(axis=1) - 0.35) / 0.05).astype(int)
        return np.eye(9)[d - 1]


def test_mosaic_layout_blank_rule_and_reproducibility():
    x_test = np.stack([np.full(TILE * TILE, 0.35 + 0.05 * d, dtype=np.float32) for d in range(1, 10)])
    pool = DigitPool({d: np.array([d - 1]) for d in range(1, 10)}, {d: np.array([d - 1]) for d in range(1, 10)})
    pz, sol = make_puzzle(random.Random(2), 30)
    i1 = make_instance(pz, sol, x_test, pool, 2, random.Random(9))
    i2 = make_instance(pz, sol, x_test, pool, 2, random.Random(9))
    assert np.array_equal(i1.mosaic, i2.mosaic) and i1.injected == i2.injected and len(i1.injected) == 2
    rd = read_mosaic(i1.mosaic, StubModel())
    assert rd.grid == pz                                        # blanks detected by brightness, digits placed in the right cell
    assert rd.blank.sum() == 51 and np.allclose(rd.probs[rd.blank], 0)


def test_hard_pool_rule():
    rng = np.random.default_rng(0)
    y = np.repeat(np.arange(1, 10), 20)
    P = np.full((len(y), 9), 0.01); P[np.arange(len(y)), y - 1] = 0.9
    P[0] = 0.01; P[0, 4] = 0.6; P[0, 0] = 0.3      # image 0: true digit 1, read as 5, true digit is 2nd choice => hard
    P[1] = 0.01; P[1, 4] = 0.9                      # image 1: true digit 1, read as 5 confidently, true prob 0.01 => rank > top-3 maybe
    pool = build_pool(P / P.sum(1, keepdims=True), y)
    assert 0 in pool.hard[1] and 0 not in pool.easy[1] and 2 in pool.easy[1]
