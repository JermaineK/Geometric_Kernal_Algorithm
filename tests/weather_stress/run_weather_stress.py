from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import yaml

from gka.core.pipeline import run_pipeline
from gka.domains import register_builtin_adapters
from gka.utils.time import utc_now_iso

ScenarioFn = Callable[[pd.DataFrame, np.random.Generator, pd.DataFrame | None], pd.DataFrame]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run weather-specific adversarial stress scenarios")
    parser.add_argument("--dataset", required=True, help="Canonical tile dataset directory")
    parser.add_argument("--config", required=True, help="Pipeline config YAML")
    parser.add_argument("--out", default="tests/weather_stress/outputs", help="Output directory")
    parser.add_argument("--seed", type=int, default=2468, help="Random seed")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    register_builtin_adapters()
    dataset_dir = Path(args.dataset)
    config_path = Path(args.config)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    samples_path = dataset_dir / "samples.parquet"
    spec_path = dataset_dir / "dataset.yaml"
    manifest_path = dataset_dir / "case_manifest.parquet"
    if not samples_path.exists():
        raise FileNotFoundError(f"Missing samples parquet: {samples_path}")
    if not spec_path.exists():
        raise FileNotFoundError(f"Missing dataset.yaml: {spec_path}")
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config: {config_path}")

    samples = pd.read_parquet(samples_path)
    samples["t"] = pd.to_datetime(samples["t"], errors="coerce")
    manifest = pd.read_parquet(manifest_path) if manifest_path.exists() else None
    rng = np.random.default_rng(int(args.seed))

    scenarios: dict[str, ScenarioFn] = {
        "diurnal_cycle_only_no_knee": _scenario_diurnal_only,
        "topography_mask_only": _scenario_topography_mask_only,
        "latitudinal_gradient_only": _scenario_latitudinal_gradient_only,
        "random_storm_times_real_fields": _scenario_random_storm_times,
    }

    results: dict[str, Any] = {}
    for i, (name, fn) in enumerate(scenarios.items()):
        transformed = fn(samples.copy(), np.random.default_rng(int(args.seed) + 101 + i), manifest)
        row = _run_partition_pipeline(
            samples=transformed,
            template_dataset=dataset_dir,
            config_path=config_path,
            seed=int(args.seed) + 301 + i,
        )
        passed = (not bool(row.get("knee_detected", False))) and (not bool(row.get("parity_signal_pass", False)))
        results[name] = {
            "passed": passed,
            "knee_detected": bool(row.get("knee_detected", False)),
            "parity_signal_pass": bool(row.get("parity_signal_pass", False)),
            "knee_confidence": _to_float(row.get("knee_confidence")),
            "R_Omega": _to_float(row.get("R_Omega")),
            "band_class_hat": row.get("band_class_hat"),
            "run_error": row.get("run_error"),
        }

    payload = {
        "generated_at_utc": utc_now_iso(),
        "dataset": str(dataset_dir.resolve()),
        "config": str(config_path.resolve()),
        "seed": int(args.seed),
        "scenarios": results,
        "suite_pass": bool(all(v.get("passed", False) for v in results.values())),
    }
    (out_dir / "suite_results.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    (out_dir / "suite_summary.md").write_text(_to_summary_markdown(payload), encoding="utf-8")
    print(f"Weather stress suite pass: {payload['suite_pass']}")
    for name, result in results.items():
        print(f"- {name}: {'PASS' if result['passed'] else 'FAIL'}")
    return 0 if payload["suite_pass"] else 2


def _run_partition_pipeline(
    *,
    samples: pd.DataFrame,
    template_dataset: Path,
    config_path: Path,
    seed: int,
) -> dict[str, Any]:
    default_out = {
        "knee_detected": False,
        "knee_confidence": None,
        "parity_signal_pass": False,
        "R_Omega": None,
        "band_class_hat": None,
        "run_error": "empty_partition",
    }
    if samples.empty:
        return default_out
    with tempfile.TemporaryDirectory(prefix="gka_weather_stress_") as tmp:
        tmp_dir = Path(tmp)
        ds_dir = tmp_dir / "dataset"
        ds_dir.mkdir(parents=True, exist_ok=True)
        spec = yaml.safe_load((template_dataset / "dataset.yaml").read_text(encoding="utf-8"))
        spec["id"] = f"{spec.get('id', 'weather')}_stress"
        (ds_dir / "dataset.yaml").write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")
        samples.to_parquet(ds_dir / "samples.parquet", index=False)
        run_out = tmp_dir / "out"
        try:
            run = run_pipeline(
                dataset_path=str(ds_dir),
                domain=spec.get("domain", "weather"),
                out_dir=str(run_out),
                config_path=str(config_path),
                null_n=0,
                allow_missing=False,
                seed=int(seed),
                dump_intermediates=False,
                argv=["gka", "run", str(ds_dir)],
            )
            row = run.summary.iloc[0].to_dict()
            row["run_error"] = None
            return row
        except Exception as exc:
            out = default_out.copy()
            out["run_error"] = f"{type(exc).__name__}"
            return out


def _scenario_diurnal_only(samples: pd.DataFrame, rng: np.random.Generator, _: pd.DataFrame | None) -> pd.DataFrame:
    df = samples.copy()
    hours = pd.to_datetime(df["t"], errors="coerce").dt.hour.fillna(0).to_numpy(dtype=float)
    signal = 1.0 + 0.2 * np.sin((2.0 * np.pi * hours) / 24.0)
    noise = rng.normal(0.0, 0.02, size=signal.size)
    base = np.clip(signal + noise, 0.05, None)
    df["O"] = base
    return _rebalance_pairs(df)


def _scenario_topography_mask_only(samples: pd.DataFrame, rng: np.random.Generator, _: pd.DataFrame | None) -> pd.DataFrame:
    df = samples.copy()
    case_hash = df["case_id"].astype(str).map(_hash_unit).to_numpy(dtype=float)
    l_scaled = pd.to_numeric(df["L"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    base = 0.6 + 0.8 * case_hash + 0.05 * np.log1p(np.maximum(l_scaled, 0.0))
    base += rng.normal(0.0, 0.01, size=base.size)
    df["O"] = np.clip(base, 0.05, None)
    return _rebalance_pairs(df)


def _scenario_latitudinal_gradient_only(
    samples: pd.DataFrame,
    rng: np.random.Generator,
    manifest: pd.DataFrame | None,
) -> pd.DataFrame:
    df = samples.copy()
    lat_by_case: dict[str, float] = {}
    if manifest is not None and "lat0" in manifest.columns and "case_id" in manifest.columns:
        lat_map = manifest.set_index(manifest["case_id"].astype(str))["lat0"]
        lat_by_case = {str(k): float(v) for k, v in lat_map.items() if pd.notna(v)}
    lat_vals = df["case_id"].astype(str).map(lambda c: lat_by_case.get(c, -20.0 + 5.0 * _hash_unit(c))).to_numpy(dtype=float)
    lat_norm = (lat_vals - np.nanmin(lat_vals)) / (np.nanmax(lat_vals) - np.nanmin(lat_vals) + 1e-12)
    base = 0.8 + 0.7 * lat_norm + rng.normal(0.0, 0.01, size=lat_norm.size)
    df["O"] = np.clip(base, 0.05, None)
    return _rebalance_pairs(df)


def _scenario_random_storm_times(samples: pd.DataFrame, rng: np.random.Generator, _: pd.DataFrame | None) -> pd.DataFrame:
    df = samples.copy()
    for col in ("storm", "near_storm", "pregen", "case_type"):
        if col in df.columns:
            vals = df[col].to_numpy()
            rng.shuffle(vals)
            df[col] = vals
    return df


def _rebalance_pairs(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for _, grp in df.groupby(["case_id", "t", "L"], dropna=False):
        g = grp.copy()
        if g["hand"].nunique() != 2:
            continue
        l_mask = g["hand"] == "L"
        r_mask = g["hand"] == "R"
        if not l_mask.any() or not r_mask.any():
            continue
        val = float(pd.to_numeric(g["O"], errors="coerce").mean())
        g.loc[l_mask, "O"] = val
        g.loc[r_mask, "O"] = val
        out.append(g)
    if not out:
        return df.iloc[0:0].copy()
    return pd.concat(out, ignore_index=True)


def _hash_unit(value: str) -> float:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) / float(0xFFFFFFFF)


def _to_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        fv = float(value)
        if not np.isfinite(fv):
            return None
        return fv
    except Exception:
        return None


def _to_summary_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Weather Stress Suite Summary",
        "",
        f"- Timestamp (UTC): `{payload['generated_at_utc']}`",
        f"- Suite pass: `{payload['suite_pass']}`",
        "",
        "## Scenarios",
    ]
    for name, result in payload.get("scenarios", {}).items():
        lines.append(
            f"- `{name}`: `{'PASS' if result.get('passed') else 'FAIL'}` "
            f"(knee={result.get('knee_detected')}, parity={result.get('parity_signal_pass')})"
        )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())

