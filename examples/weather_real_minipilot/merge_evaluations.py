from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import chi2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge per-shard weather evaluation JSON artifacts.")
    parser.add_argument("--inputs", nargs="+", required=True, help="Input JSON files or glob patterns")
    parser.add_argument("--out", required=True, help="Output merged JSON path")
    parser.add_argument(
        "--require-claim-valid-shards",
        action="store_true",
        help="Only merge shards with strict coverage pass for claim metrics",
    )
    return parser.parse_args()


def _expand_inputs(patterns: list[str]) -> list[Path]:
    out: list[Path] = []
    for p in patterns:
        matches = sorted(glob.glob(str(p)))
        if matches:
            out.extend(Path(m) for m in matches)
        else:
            cand = Path(p)
            if cand.exists():
                out.append(cand)
    uniq = sorted({str(p.resolve()): p for p in out}.values(), key=lambda x: str(x))
    return uniq


def _to_float(v: Any) -> float | None:
    try:
        if v is None:
            return None
        f = float(v)
        if not np.isfinite(f):
            return None
        return f
    except Exception:
        return None


def _weighted_mean(rows: list[dict[str, Any]], key: str, weight_key: str = "weight") -> float | None:
    vals: list[float] = []
    wts: list[float] = []
    for r in rows:
        v = _to_float(r.get(key))
        w = _to_float(r.get(weight_key))
        if v is None or w is None or w <= 0:
            continue
        vals.append(v)
        wts.append(w)
    if not vals:
        return None
    return float(np.average(np.array(vals, dtype=float), weights=np.array(wts, dtype=float)))


def _fisher_p_value(p_values: list[float]) -> float | None:
    vals = [float(p) for p in p_values if p is not None and np.isfinite(p) and p > 0.0 and p <= 1.0]
    if not vals:
        return None
    stat = float(-2.0 * np.sum(np.log(np.array(vals, dtype=float))))
    df = int(2 * len(vals))
    return float(1.0 - chi2.cdf(stat, df=df))


def main() -> int:
    args = parse_args()
    paths = _expand_inputs([str(v) for v in (args.inputs or [])])
    if not paths:
        raise ValueError("no_input_evaluation_files")

    shard_rows: list[dict[str, Any]] = []
    for p in paths:
        obj = json.loads(p.read_text(encoding="utf-8"))
        parity = obj.get("parity_confound_gate", {}) or {}
        anomaly = obj.get("anomaly_mode_gate", {}) or {}
        geom = obj.get("geometry_null_collapse", {}) or {}
        angular = (obj.get("angular_witness", {}) or {}).get("gate", {}) or {}
        ib = obj.get("ibtracs_alignment_gate", {}) or {}
        strict_cov = obj.get("strict_coverage_gate", {}) or {}
        subset = obj.get("case_subset", {}) or {}
        interp = obj.get("interpretability", {}) or {}
        shard_rows.append(
            {
                "path": str(p.resolve()),
                "split_date": obj.get("split_date"),
                "shard_index": int(subset.get("case_shard_index", 1)),
                "case_shards": int(subset.get("case_shards", 1)),
                "weight": int(obj.get("n_cases_test", 0)),
                "n_cases_test": int(obj.get("n_cases_test", 0)),
                "strict_coverage_pass": bool(strict_cov.get("passed", True)),
                "diagnostic_only": bool(interp.get("diagnostic_only", True)),
                "parity_pass": bool(parity.get("passed", False)),
                "anomaly_pass": bool(anomaly.get("passed", False)),
                "geometry_pass": bool(geom.get("passed", False)),
                "angular_pass": bool(angular.get("passed", False)),
                "ibtracs_pass": bool(ib.get("passed", False)),
                "parity_event_minus_far": _to_float(parity.get("event_minus_far_rate")),
                "parity_far_rate": _to_float(parity.get("far_nonstorm_parity_rate")),
                "parity_confound_rate": _to_float(parity.get("confound_rate")),
                "angular_margin": _to_float((obj.get("angular_witness", {}) or {}).get("observed", {}).get("margin")),
                "angular_d": _to_float((obj.get("angular_witness", {}) or {}).get("observed", {}).get("effect_size_d")),
                "angular_p": _to_float((obj.get("angular_witness", {}) or {}).get("observed", {}).get("p_value_perm")),
                "ib_event_minus_far": _to_float(ib.get("event_minus_far_parity_rate")),
                "ib_confound_rate": _to_float(ib.get("confound_rate")),
                "all_gates_pass": bool(
                    parity.get("passed", False)
                    and anomaly.get("passed", False)
                    and geom.get("passed", False)
                    and angular.get("passed", False)
                    and ib.get("passed", False)
                    and (not bool(interp.get("diagnostic_only", True)))
                ),
            }
        )

    if bool(args.require_claim_valid_shards):
        valid_rows = [r for r in shard_rows if bool(r.get("strict_coverage_pass", False))]
    else:
        valid_rows = list(shard_rows)

    merged: dict[str, Any] = {
        "n_inputs": int(len(shard_rows)),
        "n_valid_shards": int(len(valid_rows)),
        "require_claim_valid_shards": bool(args.require_claim_valid_shards),
        "shards": shard_rows,
        "pooled": {},
        "gates": {},
    }
    if not valid_rows:
        merged["status"] = "no_valid_shards"
        merged["reason_codes"] = ["no_valid_shards_after_filter"]
    else:
        merged["pooled"] = {
            "n_cases_test_weighted": int(sum(int(r.get("weight", 0)) for r in valid_rows)),
            "parity_event_minus_far": _weighted_mean(valid_rows, "parity_event_minus_far"),
            "parity_far_rate": _weighted_mean(valid_rows, "parity_far_rate"),
            "parity_confound_rate": _weighted_mean(valid_rows, "parity_confound_rate"),
            "angular_margin": _weighted_mean(valid_rows, "angular_margin"),
            "angular_d": _weighted_mean(valid_rows, "angular_d"),
            "angular_p_fisher": _fisher_p_value([_to_float(r.get("angular_p")) for r in valid_rows]),
            "ib_event_minus_far": _weighted_mean(valid_rows, "ib_event_minus_far"),
            "ib_confound_rate": _weighted_mean(valid_rows, "ib_confound_rate"),
        }
        merged["gates"] = {
            "parity_confound_gate": bool(all(bool(r.get("parity_pass", False)) for r in valid_rows)),
            "anomaly_mode_gate": bool(all(bool(r.get("anomaly_pass", False)) for r in valid_rows)),
            "geometry_null_collapse": bool(all(bool(r.get("geometry_pass", False)) for r in valid_rows)),
            "angular_witness": bool(all(bool(r.get("angular_pass", False)) for r in valid_rows)),
            "ibtracs_alignment_gate": bool(all(bool(r.get("ibtracs_pass", False)) for r in valid_rows)),
            "all_gates_pass": bool(all(bool(r.get("all_gates_pass", False)) for r in valid_rows)),
        }
        merged["status"] = "ok"
        merged["reason_codes"] = []
        if not merged["gates"]["all_gates_pass"]:
            merged["reason_codes"].append("one_or_more_gates_failed_in_valid_shards")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(merged, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote merged evaluation to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
