"""General plotting helpers."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_eta_vs_L(df: pd.DataFrame, out_path: str | Path) -> Path:
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(df["L"], df["eta"], marker="o", linewidth=1.5)
    ax.set_xlabel("L")
    ax.set_ylabel("eta")
    ax.set_title("Mirror-odd contrast by size")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(p, dpi=140)
    plt.close(fig)
    return p


def plot_null_distribution(null_df: pd.DataFrame, out_path: str | Path) -> Path:
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 4))
    if "gamma" in null_df.columns:
        ax.hist(null_df["gamma"].dropna(), bins=20, alpha=0.8, edgecolor="black")
        ax.set_xlabel("gamma")
    else:
        ax.text(0.5, 0.5, "No null gamma values", ha="center", va="center")
    ax.set_ylabel("count")
    ax.set_title("Null distribution")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(p, dpi=140)
    plt.close(fig)
    return p


def plot_stability_curve(mu: list[float], s_vals: list[float], out_path: str | Path) -> Path:
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 4))
    if len(mu) > 0 and len(s_vals) == len(mu):
        mu_arr = pd.Series(mu, dtype=float).to_numpy()
        s_arr = pd.Series(s_vals, dtype=float).to_numpy()
        ax.plot(mu_arr, s_arr, linewidth=1.6)
        ax.axhline(1.0, color="black", linestyle="--", linewidth=1.0)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("mu")
        ax.set_ylabel("S(b;mu)")
        ax.set_title("Compact stability inequality")
    else:
        ax.text(0.5, 0.5, "No stability curve data", ha="center", va="center")
        ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(p, dpi=140)
    plt.close(fig)
    return p
