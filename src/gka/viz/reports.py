"""HTML report generation."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from gka.viz.knee_plots import plot_knee
from gka.viz.plots import plot_eta_vs_L, plot_null_distribution, plot_stability_curve


def generate_html_report(results_dir: Path, out_html: Path) -> None:
    results_path = results_dir / "results.parquet"
    metadata_path = results_dir / "run_metadata.json"
    null_path = results_dir / "nulls" / "null_distributions.parquet"

    if not results_path.exists():
        raise FileNotFoundError(f"Missing results file: {results_path}")

    results_df = pd.read_parquet(results_path)
    metadata = {}
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    assets_dir = out_html.parent / "report_assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    eta_plot = plot_eta_vs_L(results_df, assets_dir / "eta_vs_L.png")
    knee_plot = plot_knee(results_df, assets_dir / "knee.png")
    stage_context = metadata.get("stage_context", {}) if isinstance(metadata, dict) else {}
    s_curve_plot = plot_stability_curve(
        mu=stage_context.get("S_curve_mu", []),
        s_vals=stage_context.get("S_curve_values", []),
        out_path=assets_dir / "stability_curve.png",
    )

    null_plot_rel = ""
    null_table_html = "<p>No null output found.</p>"
    if null_path.exists():
        null_df = pd.read_parquet(null_path)
        null_plot = plot_null_distribution(null_df, assets_dir / "null_dist.png")
        null_plot_rel = _relpath(null_plot, out_html.parent)
        null_table_html = null_df.head(50).to_html(index=False, classes="table")

    summary_html = results_df.to_html(index=False, classes="table")
    diag_cols = [
        "gamma",
        "Delta_hat",
        "omega_k_hat",
        "tau_s_hat",
        "S_at_mu_k",
        "W_mu",
        "W_L",
        "R_align",
        "M_Z",
        "band_hit_rate",
        "band_class_hat",
        "eigen_band",
        "stability_margin",
        "forbidden_middle_width",
        "forbidden_middle_center",
        "forbidden_middle_reason_codes",
        "knee_rejected_because",
    ]
    present = [c for c in diag_cols if c in results_df.columns]
    diag_html = (
        results_df[present].head(1).to_html(index=False, classes="table")
        if present
        else "<p>No diagnostic columns found.</p>"
    )
    metadata_html = (
        f"<pre>{json.dumps(metadata, indent=2, sort_keys=True)}</pre>" if metadata else "<p>None</p>"
    )

    html = f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <title>GKA Report</title>
  <style>
    body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 2rem; color: #1a1a1a; }}
    h1, h2 {{ margin-top: 1.2rem; }}
    img {{ max-width: 780px; border: 1px solid #ddd; margin: 0.75rem 0; }}
    .table {{ border-collapse: collapse; width: 100%; margin: 1rem 0; font-size: 0.9rem; }}
    .table th, .table td {{ border: 1px solid #ddd; padding: 0.4rem 0.5rem; text-align: left; }}
    .table th {{ background: #f5f5f5; }}
  </style>
</head>
<body>
  <h1>GKA Run Report</h1>

  <h2>Diagnostics</h2>
  <img src=\"{_relpath(eta_plot, out_html.parent)}\" alt=\"eta plot\" />
  <img src=\"{_relpath(knee_plot, out_html.parent)}\" alt=\"knee plot\" />
  <img src=\"{_relpath(s_curve_plot, out_html.parent)}\" alt=\"stability curve\" />
  {'<img src="' + null_plot_rel + '" alt="null plot" />' if null_plot_rel else ''}

  <h2>Results Table</h2>
  {summary_html}

  <h2>Operational Diagnostics</h2>
  {diag_html}

  <h2>Null Summary</h2>
  {null_table_html}

  <h2>Run Metadata</h2>
  {metadata_html}
</body>
</html>
"""

    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(html, encoding="utf-8")


def _relpath(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()
