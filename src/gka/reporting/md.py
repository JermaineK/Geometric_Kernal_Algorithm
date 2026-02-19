"""Markdown report writer."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def write_report_md(payload: dict[str, Any], out_path: str | Path) -> None:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    rep = payload.get("reproducibility", {})
    inv = payload.get("invariants", {})
    text = "\n".join(
        [
            "# GKA Report",
            "",
            "## Reproducibility",
            f"- Dataset hash: `{rep.get('dataset_hash')}`",
            f"- Config hash: `{rep.get('config_hash')}`",
            f"- Git commit: `{rep.get('git_commit')}`",
            f"- RNG seed: `{rep.get('rng_seed')}`",
            f"- Timestamp (UTC): `{rep.get('timestamp_utc')}`",
            "",
            "## Key Diagnostics",
            f"- gamma: `{_fmt(inv.get('gamma'))}`",
            f"- Delta_hat: `{_fmt(inv.get('Delta_hat'))}`",
            f"- omega_k_hat: `{_fmt(inv.get('omega_k_hat'))}`",
            f"- tau_s_hat: `{_fmt(inv.get('tau_s_hat'))}`",
            f"- S_at_mu_k: `{_fmt(inv.get('S_at_mu_k'))}`",
            f"- W_mu: `{_fmt(inv.get('W_mu'))}`",
            f"- band_class_hat: `{inv.get('band_class_hat')}`",
            f"- eigen_band: `{inv.get('eigen_band')}`",
            f"- stability_margin: `{_fmt(inv.get('stability_margin'))}`",
            f"- knee_detected: `{inv.get('knee_detected')}`",
            f"- knee_rejected_because: `{inv.get('knee_rejected_because')}`",
            "",
            "## Artifact Paths",
            f"- results.parquet: `{payload.get('artifacts', {}).get('results_parquet')}`",
            f"- run_metadata.json: `{payload.get('artifacts', {}).get('run_metadata_json')}`",
            f"- config_resolved.yaml: `{payload.get('artifacts', {}).get('config_resolved_yaml')}`",
            "",
        ]
    )
    out.write_text(text, encoding="utf-8")


def _fmt(v: Any) -> str:
    try:
        return f"{float(v):.6g}"
    except Exception:
        return str(v)
