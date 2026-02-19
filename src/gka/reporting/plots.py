"""Figure bundle writer for report artifacts."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from gka.viz.knee_plots import plot_knee
from gka.viz.plots import plot_eta_vs_L


def write_report_figures(results_dir: str | Path, out_dir: str | Path) -> dict[str, str]:
    root = Path(results_dir)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    results_path = root / "results.parquet"
    if not results_path.exists():
        raise FileNotFoundError(f"Missing results file: {results_path}")
    df = pd.read_parquet(results_path)
    if df.empty:
        raise ValueError(f"Empty results file: {results_path}")

    eta_path = plot_eta_vs_L(df, out / "eta_vs_L.png")
    knee_path = plot_knee(df, out / "knee.png")
    return {
        "eta_vs_L": str(eta_path.resolve()),
        "knee": str(knee_path.resolve()),
    }
