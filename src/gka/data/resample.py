"""Resampling utilities shared by null models and bootstrap routines."""

from __future__ import annotations

import numpy as np
import pandas as pd


def resample_blocks(values: np.ndarray, block_size: int, rng: np.random.Generator) -> np.ndarray:
    arr = np.asarray(values)
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    n = arr.shape[0]
    starts = rng.integers(0, max(1, n - block_size + 1), size=max(1, n // block_size + 1))
    blocks = [arr[s : s + block_size] for s in starts]
    return np.concatenate(blocks, axis=0)[:n]


def groupwise_shuffle(df: pd.DataFrame, group_col: str, value_col: str, rng: np.random.Generator) -> pd.Series:
    out = df[value_col].copy()
    for _, idx in df.groupby(group_col).groups.items():
        vals = out.loc[idx].to_numpy(copy=True)
        rng.shuffle(vals)
        out.loc[idx] = vals
    return out
