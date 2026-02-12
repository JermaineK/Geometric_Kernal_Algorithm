"""Null model implementations for robustness checks."""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from gka.data.resample import resample_blocks


def time_shuffle(
    df: pd.DataFrame,
    value_col: str,
    group_col: str,
    rng: np.random.Generator,
) -> pd.DataFrame:
    out = df.copy()
    for _, idx in out.groupby(group_col).groups.items():
        vals = out.loc[idx, value_col].to_numpy(copy=True)
        rng.shuffle(vals)
        out.loc[idx, value_col] = vals
    return out


def mirror_randomization(df: pd.DataFrame, hand_col: str, rng: np.random.Generator) -> pd.DataFrame:
    out = df.copy()
    hands = out[hand_col].to_numpy(copy=True)
    swap_mask = rng.random(hands.size) < 0.5
    swapped = hands.copy()
    swapped[(hands == "L") & swap_mask] = "R"
    swapped[(hands == "R") & swap_mask] = "L"
    out[hand_col] = swapped
    return out


def phase_scramble(series: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    arr = np.asarray(series, dtype=float)
    fft = np.fft.rfft(arr)
    mag = np.abs(fft)
    phase = np.angle(fft)
    random_phase = rng.uniform(-np.pi, np.pi, size=phase.shape)
    random_phase[0] = phase[0]
    if random_phase.size > 1:
        random_phase[-1] = phase[-1]
    scrambled = mag * np.exp(1j * random_phase)
    return np.fft.irfft(scrambled, n=arr.size)


def sign_flip(series: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    arr = np.asarray(series, dtype=float)
    signs = rng.choice([-1.0, 1.0], size=arr.shape[0])
    return arr * signs


def block_bootstrap(series: np.ndarray, block_size: int, rng: np.random.Generator) -> np.ndarray:
    arr = np.asarray(series, dtype=float)
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    return resample_blocks(arr, block_size=block_size, rng=rng)


def apply_null_model(
    name: str,
    df: pd.DataFrame,
    value_col: str,
    group_col: str,
    hand_col: str,
    rng: np.random.Generator,
) -> pd.DataFrame:
    model = name.strip().lower()
    if model == "time_shuffle":
        return time_shuffle(df, value_col=value_col, group_col=group_col, rng=rng)
    if model == "mirror_swap":
        return mirror_randomization(df, hand_col=hand_col, rng=rng)
    if model == "phase_scramble":
        out = df.copy()
        out[value_col] = phase_scramble(out[value_col].to_numpy(dtype=float), rng=rng)
        return out
    if model == "sign_flip":
        out = df.copy()
        out[value_col] = sign_flip(out[value_col].to_numpy(dtype=float), rng=rng)
        return out
    if model == "block_bootstrap":
        out = df.copy()
        block_size = max(2, int(np.sqrt(len(out))))
        out[value_col] = block_bootstrap(out[value_col].to_numpy(dtype=float), block_size, rng=rng)
        return out

    raise ValueError(
        f"Unsupported null model '{name}'. Supported: "
        "time_shuffle|mirror_swap|phase_scramble|sign_flip|block_bootstrap"
    )


def available_null_models() -> list[str]:
    return ["time_shuffle", "mirror_swap", "phase_scramble", "sign_flip", "block_bootstrap"]


def model_dispatch() -> dict[str, Callable]:
    return {
        "time_shuffle": time_shuffle,
        "mirror_swap": mirror_randomization,
        "phase_scramble": phase_scramble,
        "sign_flip": sign_flip,
        "block_bootstrap": block_bootstrap,
    }


def parity_significance_pvalues(
    pair_df: pd.DataFrame,
    n_perm: int,
    rng: np.random.Generator,
    alpha: float = 0.05,
) -> dict[str, float | bool]:
    """Estimate parity significance against permutation and direction-randomization nulls."""

    if pair_df.empty:
        return {
            "mirror_stat": 0.0,
            "p_perm": 1.0,
            "p_dir": 1.0,
            "signal_pass": False,
        }

    observed_eta = _pair_eta(pair_df["O_L"].to_numpy(dtype=float), pair_df["O_R"].to_numpy(dtype=float))
    mirror_stat = float(np.median(observed_eta))

    # Null 1: shuffle O_R within each L bin.
    by_l = [idx.to_numpy(dtype=int) for _, idx in pair_df.groupby("L").groups.items()]
    perm_stats = np.empty(n_perm, dtype=float)
    left_vals = pair_df["O_L"].to_numpy(dtype=float)
    right_vals = pair_df["O_R"].to_numpy(dtype=float)
    for i in range(n_perm):
        shuffled = right_vals.copy()
        for idx in by_l:
            shuffled[idx] = shuffled[idx][rng.permutation(idx.size)]
        perm_stats[i] = float(np.median(_pair_eta(left_vals, shuffled)))

    # Null 2: random L/R swaps by pair.
    dir_stats = np.empty(n_perm, dtype=float)
    for i in range(n_perm):
        swap = rng.random(left_vals.size) < 0.5
        ll = left_vals.copy()
        rr = right_vals.copy()
        ll[swap], rr[swap] = rr[swap], ll[swap]
        signed = _pair_signed(ll, rr)
        dir_stats[i] = float(np.abs(np.mean(np.sign(signed))))

    obs_dir = float(np.abs(np.mean(np.sign(_pair_signed(left_vals, right_vals)))))
    p_perm = float((np.sum(perm_stats >= mirror_stat) + 1) / (n_perm + 1))
    p_dir = float((np.sum(dir_stats >= obs_dir) + 1) / (n_perm + 1))
    signal_pass = bool(p_perm < alpha)
    return {
        "mirror_stat": mirror_stat,
        "p_perm": p_perm,
        "p_dir": p_dir,
        "signal_pass": signal_pass,
    }


def _pair_eta(left: np.ndarray, right: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    denom = np.maximum((left + right) / 2.0, eps)
    return np.abs(left - right) / denom


def _pair_signed(left: np.ndarray, right: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    denom = np.maximum((left + right) / 2.0, eps)
    return (left - right) / denom
