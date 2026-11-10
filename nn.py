"""Digit classifier (pure NumPy MLP) + calibration tools + baselines.

Classes are digits 1-9, stored as indices 0-8 internally. Inputs are 784-vectors scaled to 0-1.
The probabilities from predict_proba() are what the search stage turns into correction costs, so we
(a) measure calibration (ECE, reliability curve) and (b) optionally fit a temperature on the
validation set (temperature scaling, the optional extension).
"""
from __future__ import annotations

import json
import os
import time

import numpy as np

K = 9


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class MLP:
    def __init__(self, layer_sizes=(784, 128, 64, K), seed: int = 0):
        self.layer_sizes = tuple(layer_sizes)
        rng = np.random.default_rng(seed)
        self.W = [rng.normal(0.0, np.sqrt(2.0 / a), (a, b)).astype(np.float32)
                  for a, b in zip(layer_sizes[:-1], layer_sizes[1:])]
        self.b = [np.zeros(b, dtype=np.float32) for b in layer_sizes[1:]]
        self.temperature = 1.0

    # ---- inference -------------------------------------------------------------------------
    def logits(self, X):
        h = np.asarray(X, dtype=np.float32)
        for l, (W, b) in enumerate(zip(self.W, self.b)):
            h = h @ W + b
            if l < len(self.W) - 1:
                h = np.maximum(h, 0.0)
        return h

    def predict_proba(self, X, temperature=None):
        T = self.temperature if temperature is None else temperature
        return softmax(self.logits(X) / T)

    def predict(self, X):
        return self.logits(X).argmax(axis=1)

    # ---- training (Adam, mini-batch, L2) ---------------------------------------------------
    def fit(self, X, y, Xval, yval, epochs=12, lr=1e-3, batch=128, l2=1e-5, seed=0, verbose=True):
        rng = np.random.default_rng(seed)
        X = np.asarray(X, dtype=np.float32)
        n, L = len(X), len(self.W)
        m = [np.zeros_like(p) for p in self.W + self.b]
        v = [np.zeros_like(p) for p in self.W + self.b]
        t, history = 0, []
        best = (np.inf, None)
        for epoch in range(1, epochs + 1):
            perm = rng.permutation(n)
            tot = 0.0
            for s in range(0, n, batch):
                idx = perm[s:s + batch]
                xb, yb = X[idx], y[idx]
                acts, pre, h = [xb], [], xb
                for l in range(L):
                    z = h @ self.W[l] + self.b[l]
                    pre.append(z)
                    h = np.maximum(z, 0.0) if l < L - 1 else z
                    acts.append(h)
                p = softmax(pre[-1])
                tot += -np.log(p[np.arange(len(yb)), yb] + 1e-12).sum()
                delta = p
                delta[np.arange(len(yb)), yb] -= 1.0
                delta /= len(yb)
                gW, gb = [None] * L, [None] * L
                for l in reversed(range(L)):
                    gW[l] = acts[l].T @ delta + l2 * self.W[l]
                    gb[l] = delta.sum(axis=0)
                    if l > 0:
                        delta = (delta @ self.W[l].T) * (pre[l - 1] > 0)
                t += 1
                params, grads = self.W + self.b, gW + gb
                for k, (P, G) in enumerate(zip(params, grads)):
                    m[k] = 0.9 * m[k] + 0.1 * G
                    v[k] = 0.999 * v[k] + 0.001 * G * G
                    mh, vh = m[k] / (1 - 0.9 ** t), v[k] / (1 - 0.999 ** t)
                    P -= lr * mh / (np.sqrt(vh) + 1e-8)
            pv = softmax(self.logits(Xval))
            vl, va = nll(pv, yval), accuracy(pv, yval)
            history.append({"epoch": epoch, "train_loss": float(tot / n), "val_loss": float(vl), "val_acc": float(va)})
            if verbose:
                print(f"epoch {epoch:2d}  train_loss {tot / n:.4f}  val_loss {vl:.4f}  val_acc {va:.4f}")
            if vl < best[0]:
                best = (vl, ([w.copy() for w in self.W], [b.copy() for b in self.b]))
        self.W, self.b = best[1]          # keep the epoch with the lowest validation loss
        return history

    # ---- persistence -----------------------------------------------------------------------
    def save(self, path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        arrays = {f"W{i}": w for i, w in enumerate(self.W)}
        arrays.update({f"b{i}": b for i, b in enumerate(self.b)})
        np.savez_compressed(path, sizes=np.array(self.layer_sizes), temperature=np.array(self.temperature),
                            **arrays)

    @classmethod
    def load(cls, path):
        d = np.load(path)
        net = cls(tuple(int(s) for s in d["sizes"]))
        net.W = [d[f"W{i}"] for i in range(len(net.W))]
        net.b = [d[f"b{i}"] for i in range(len(net.b))]
        net.temperature = float(d["temperature"])
        return net


# ---- metrics -----------------------------------------------------------------------------------
def accuracy(P, y):
    return float((P.argmax(axis=1) == y).mean())


def nll(P, y):
    return float(-np.log(P[np.arange(len(y)), y] + 1e-12).mean())


def brier(P, y):
    onehot = np.eye(P.shape[1])[y]
    return float(((P - onehot) ** 2).sum(axis=1).mean())


def reliability_curve(P, y, bins=15):
    """Return (bin_confidence, bin_accuracy, bin_count) using top-label confidence."""
    conf, pred = P.max(axis=1), P.argmax(axis=1)
    correct = (pred == y).astype(float)
    edges = np.linspace(0, 1, bins + 1)
    bc, ba, bn = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf > lo) & (conf <= hi)
        bn.append(int(mask.sum()))
        bc.append(float(conf[mask].mean()) if mask.any() else np.nan)
        ba.append(float(correct[mask].mean()) if mask.any() else np.nan)
    return np.array(bc), np.array(ba), np.array(bn)


def expected_calibration_error(P, y, bins=15):
    bc, ba, bn = reliability_curve(P, y, bins)
    ok = bn > 0
    return float((bn[ok] / bn.sum() * np.abs(bc[ok] - ba[ok])).sum())


def confusion_matrix(y, pred, k=K):
    cm = np.zeros((k, k), dtype=int)
    for a, b in zip(y, pred):
        cm[a, b] += 1
    return cm


def evaluate(P, y):
    return {"accuracy": accuracy(P, y), "nll": nll(P, y), "brier": brier(P, y),
            "ece": expected_calibration_error(P, y)}


def fit_temperature(logits, y):
    """Temperature scaling: pick T > 0 minimising validation NLL (grid search, then local refine)."""
    grid = np.exp(np.linspace(np.log(0.2), np.log(10.0), 400))
    losses = [nll(softmax(logits / T), y) for T in grid]
    return float(grid[int(np.argmin(losses))])


# ---- baselines --------------------------------------------------------------------------------
def train_baselines(Xtr, ytr, Xte, seed=0):
    """Return {name: test-probabilities} for the baselines."""
    from sklearn.linear_model import LogisticRegression
    out = {}
    counts = np.bincount(ytr, minlength=K).astype(float)
    out["majority class"] = np.tile(counts / counts.sum(), (len(Xte), 1))
    lr = LogisticRegression(max_iter=500, random_state=seed)
    lr.fit(Xtr, ytr)
    out["logistic regression"] = lr.predict_proba(Xte)
    return out


def train_and_evaluate(dataset: dict, out_dir="results", model_path="results/mlp.npz", epochs=12, seed=0,
                       verbose=True):
    """Train the MLP, fit temperature on validation, compare with baselines on the untouched test set."""
    Xtr, ytr = dataset["x_train"], dataset["y_train"] - 1
    Xva, yva = dataset["x_val"], dataset["y_val"] - 1
    Xte, yte = dataset["x_test"], dataset["y_test"] - 1
    net = MLP(seed=seed)
    t0 = time.time()
    history = net.fit(Xtr, ytr, Xva, yva, epochs=epochs, seed=seed, verbose=verbose)
    train_seconds = time.time() - t0
    net.temperature = fit_temperature(net.logits(Xva), yva)
    net.save(model_path)
    P_raw = net.predict_proba(Xte, temperature=1.0)
    P_cal = net.predict_proba(Xte)
    baselines = train_baselines(Xtr, ytr, Xte, seed)
    results = {"MLP (raw)": evaluate(P_raw, yte), "MLP (temperature-scaled)": evaluate(P_cal, yte)}
    for name, P in baselines.items():
        results[name] = evaluate(P, yte)
    summary = {"temperature": net.temperature, "train_seconds": train_seconds, "epochs": epochs,
               "seed": seed, "n_train": int(len(ytr)), "n_val": int(len(yva)), "n_test": int(len(yte)),
               "test_metrics": results, "history": history,
               "confusion_matrix": confusion_matrix(yte, P_cal.argmax(1)).tolist()}
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "nn_metrics.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return net, summary, {"raw": P_raw, "cal": P_cal, **{k: v for k, v in baselines.items()}}
