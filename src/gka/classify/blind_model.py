"""Invariant-vector blind classification with probability calibration."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BlindModelMetrics:
    accuracy: float
    macro_auroc: float | None
    ece: float


class InvariantBlindClassifier:
    def __init__(
        self,
        max_iter: int = 400,
        lr: float = 0.08,
        reg: float = 1e-3,
        random_state: int = 0,
    ) -> None:
        self.max_iter = int(max_iter)
        self.lr = float(lr)
        self.reg = float(reg)
        self.rng = np.random.default_rng(int(random_state))
        self.classes_: np.ndarray | None = None
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None
        self.W_: np.ndarray | None = None
        self.b_: np.ndarray | None = None
        self.temperature_: float = 1.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "InvariantBlindClassifier":
        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y)
        if X_arr.ndim != 2:
            raise ValueError("X must be 2D")
        if X_arr.shape[0] != y_arr.shape[0]:
            raise ValueError("X and y must have same number of rows")
        classes = np.unique(y_arr)
        if classes.size < 2:
            raise ValueError("Need at least 2 classes to train blind classifier")

        self.classes_ = classes
        self.mean_ = np.nanmean(X_arr, axis=0)
        self.std_ = np.nanstd(X_arr, axis=0)
        self.std_[self.std_ <= 1e-12] = 1.0
        Xn = self._normalize(X_arr)
        K = classes.size
        D = Xn.shape[1]
        self.W_ = self.rng.normal(0.0, 0.05, size=(D, K))
        self.b_ = np.zeros(K, dtype=float)
        y_idx = np.array([np.where(classes == yi)[0][0] for yi in y_arr], dtype=int)
        Y = np.eye(K)[y_idx]

        for _ in range(self.max_iter):
            logits = Xn @ self.W_ + self.b_
            probs = _softmax(logits)
            grad_logits = (probs - Y) / Xn.shape[0]
            grad_W = Xn.T @ grad_logits + self.reg * self.W_
            grad_b = np.sum(grad_logits, axis=0)
            self.W_ -= self.lr * grad_W
            self.b_ -= self.lr * grad_b

        self.temperature_ = self._fit_temperature(Xn, y_idx)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        Xn = self._normalize(np.asarray(X, dtype=float))
        logits = (Xn @ self.W_ + self.b_) / max(self.temperature_, 1e-6)
        return _softmax(logits)

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X)
        idx = np.argmax(probs, axis=1)
        return self.classes_[idx]

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> BlindModelMetrics:
        y_arr = np.asarray(y)
        probs = self.predict_proba(X)
        pred = self.classes_[np.argmax(probs, axis=1)]
        accuracy = float(np.mean(pred == y_arr))
        auroc = macro_ovr_auroc(y_true=y_arr, y_prob=probs, classes=self.classes_)
        ece = expected_calibration_error(y_true=y_arr, y_prob=probs, classes=self.classes_)
        return BlindModelMetrics(accuracy=accuracy, macro_auroc=auroc, ece=ece)

    def _fit_temperature(self, Xn: np.ndarray, y_idx: np.ndarray) -> float:
        logits = Xn @ self.W_ + self.b_
        best_t = 1.0
        best_loss = np.inf
        for t in np.linspace(0.6, 3.0, 33):
            probs = _softmax(logits / t)
            p_true = probs[np.arange(probs.shape[0]), y_idx]
            loss = -float(np.mean(np.log(np.clip(p_true, 1e-12, 1.0))))
            if loss < best_loss:
                best_loss = loss
                best_t = float(t)
        return best_t

    def _normalize(self, X: np.ndarray) -> np.ndarray:
        Xv = np.asarray(X, dtype=float)
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("Classifier not fitted")
        Xv = np.where(np.isfinite(Xv), Xv, self.mean_)
        return (Xv - self.mean_) / self.std_

    def _check_fitted(self) -> None:
        if self.classes_ is None or self.W_ is None or self.b_ is None:
            raise RuntimeError("Classifier is not fitted")


def macro_ovr_auroc(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    classes: np.ndarray,
) -> float | None:
    y_arr = np.asarray(y_true)
    probs = np.asarray(y_prob, dtype=float)
    aucs: list[float] = []
    for i, cls in enumerate(classes):
        y_bin = (y_arr == cls).astype(int)
        score = probs[:, i]
        auc = _binary_auc(y_bin, score)
        if auc is not None:
            aucs.append(float(auc))
    if not aucs:
        return None
    return float(np.mean(aucs))


def expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    classes: np.ndarray,
    n_bins: int = 10,
) -> float:
    y_arr = np.asarray(y_true)
    probs = np.asarray(y_prob, dtype=float)
    pred_idx = np.argmax(probs, axis=1)
    conf = np.max(probs, axis=1)
    pred = classes[pred_idx]
    correct = (pred == y_arr).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = max(len(conf), 1)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i == n_bins - 1:
            mask = (conf >= lo) & (conf <= hi)
        else:
            mask = (conf >= lo) & (conf < hi)
        if not np.any(mask):
            continue
        acc = float(np.mean(correct[mask]))
        c = float(np.mean(conf[mask]))
        ece += abs(acc - c) * (np.sum(mask) / n)
    return float(ece)


def _binary_auc(y_bin: np.ndarray, score: np.ndarray) -> float | None:
    y = np.asarray(y_bin, dtype=int)
    s = np.asarray(score, dtype=float)
    pos = int(np.sum(y == 1))
    neg = int(np.sum(y == 0))
    if pos == 0 or neg == 0:
        return None
    order = np.argsort(s)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(s) + 1, dtype=float)
    sum_pos = float(np.sum(ranks[y == 1]))
    auc = (sum_pos - pos * (pos + 1) / 2.0) / (pos * neg)
    return float(auc)


def _softmax(logits: np.ndarray) -> np.ndarray:
    x = np.asarray(logits, dtype=float)
    x = x - np.max(x, axis=1, keepdims=True)
    exp = np.exp(x)
    return exp / np.sum(exp, axis=1, keepdims=True)
