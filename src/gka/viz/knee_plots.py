"""Knee-specific plotting helpers."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_knee(df: pd.DataFrame, out_path: str | Path) -> Path:
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    knee = float(df["L_k"].iloc[0]) if "L_k" in df.columns and not df.empty else None
    f_lo = float(df["forbidden_lo"].iloc[0]) if "forbidden_lo" in df.columns and not df.empty else None
    f_hi = float(df["forbidden_hi"].iloc[0]) if "forbidden_hi" in df.columns and not df.empty else None

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(df["L"], df["eta"], marker="o", linewidth=1.5, label="eta(L)")
    if knee is not None:
        ax.axvline(knee, color="red", linestyle="--", label=f"L_k={knee:.3g}")
    if f_lo is not None and f_hi is not None:
        ax.axvspan(f_lo, f_hi, color="orange", alpha=0.2, label="forbidden band")

    ax.set_xlabel("L")
    ax.set_ylabel("eta")
    ax.set_title("Knee and forbidden band")
    ax.grid(alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(p, dpi=140)
    plt.close(fig)
    return p
