from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd


def _load_eval_module():
    mod_path = Path("examples/weather_real_minipilot/evaluate_minipilot.py")
    spec = importlib.util.spec_from_file_location("eval_minipilot_mod", mod_path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_blocked_time_split_keeps_storm_id_on_one_side() -> None:
    mod = _load_eval_module()
    case_meta = pd.DataFrame(
        {
            "case_id": ["a", "b", "c"],
            "anchor_time": pd.to_datetime(
                [
                    "2025-03-01 00:00:00",
                    "2025-05-01 00:00:00",
                    "2025-02-10 00:00:00",
                ]
            ),
            "storm_id": ["S1", "S1", None],
        }
    )
    split = mod._blocked_time_split(
        case_meta=case_meta,
        split_date=pd.Timestamp("2025-04-01"),
        buffer_hours=72.0,
    )
    # Storm S1 spans both sides and must be buffered, never split train/test.
    assert "a" in split["buffer_cases"]
    assert "b" in split["buffer_cases"]
    assert "a" not in split["train_cases"]
    assert "a" not in split["test_cases"]
    assert "b" not in split["train_cases"]
    assert "b" not in split["test_cases"]
    # Non-storm case still follows time split.
    assert "c" in split["train_cases"]


def test_parity_confound_gate_uses_margin_and_far_cap() -> None:
    mod = _load_eval_module()
    gate_pass = mod._build_parity_confound_gate(
        strata_test={
            "events": {"parity_signal_rate": 0.70},
            "far_nonstorm": {"parity_signal_rate": 0.20},
        },
        confound_max_ratio=0.55,
        event_minus_far_min=0.15,
        far_nonstorm_max=0.35,
    )
    assert gate_pass["passed"] is True

    gate_fail_far = mod._build_parity_confound_gate(
        strata_test={
            "events": {"parity_signal_rate": 0.70},
            "far_nonstorm": {"parity_signal_rate": 0.50},
        },
        confound_max_ratio=0.80,
        event_minus_far_min=0.15,
        far_nonstorm_max=0.35,
    )
    assert gate_fail_far["passed"] is False


def test_rank_anomaly_modes_prefers_margin_then_low_far() -> None:
    mod = _load_eval_module()
    ranked = mod._rank_anomaly_modes_for_canonical(
        {
            "modes": {
                "none": {
                    "available": True,
                    "event_parity_signal_rate": 0.80,
                    "far_nonstorm_parity_signal_rate": 0.50,
                    "event_minus_far_parity_rate": 0.30,
                },
                "lat_hour": {
                    "available": True,
                    "event_parity_signal_rate": 0.70,
                    "far_nonstorm_parity_signal_rate": 0.20,
                    "event_minus_far_parity_rate": 0.50,
                },
                "lat_day": {
                    "available": True,
                    "event_parity_signal_rate": 0.75,
                    "far_nonstorm_parity_signal_rate": 0.30,
                    "event_minus_far_parity_rate": 0.45,
                },
            }
        },
        mode_agreement_min=0.0,
    )
    assert ranked[0] == "lat_hour"
    assert ranked[1] == "lat_day"


def test_casewise_anomaly_mode_selection_majority_and_priority() -> None:
    mod = _load_eval_module()
    mode_case_metrics = {
        "none": pd.DataFrame(
            {
                "case_id": ["c1", "c2"],
                "case_type": ["storm", "control"],
                "parity_signal_pass_effective": [True, False],
            }
        ),
        "lat_hour": pd.DataFrame(
            {
                "case_id": ["c1", "c2"],
                "case_type": ["storm", "control"],
                "parity_signal_pass_effective": [True, True],
            }
        ),
        "lat_day": pd.DataFrame(
            {
                "case_id": ["c1", "c2"],
                "case_type": ["storm", "control"],
                "parity_signal_pass_effective": [False, True],
            }
        ),
    }
    out = mod._build_casewise_anomaly_mode_selection(
        mode_case_metrics=mode_case_metrics,
        mode_priority=["lat_hour", "lat_day", "none"],
        stability_min=0.67,
    )
    per_case = out["per_case"].set_index("case_id")
    # c1 votes: True, True, False -> majority True, canonical picks highest-priority mode with True.
    assert bool(per_case.loc["c1", "canonical_parity_signal_pass"]) is True
    assert per_case.loc["c1", "canonical_mode"] == "lat_hour"
    # c2 votes: False, True, True -> majority True, canonical picks highest-priority mode with True.
    assert bool(per_case.loc["c2", "canonical_parity_signal_pass"]) is True
    assert per_case.loc["c2", "canonical_mode"] == "lat_hour"


def test_resolve_storm_identity_falls_back_to_track_id() -> None:
    mod = _load_eval_module()
    samples = pd.DataFrame({"case_id": ["a"], "track_id": ["trk_1"]})
    assert mod._resolve_storm_id_col(samples, requested="auto") == "track_id"


def test_geometry_null_collapse_gate_requires_drop() -> None:
    mod = _load_eval_module()
    gate = mod._build_geometry_null_collapse_gate(
        observed_strata={"events": {"parity_signal_rate": 0.8}},
        null_strata={
            "theta_roll": {"events": {"parity_signal_rate": 0.2}},
            "center_jitter": {"events": {"parity_signal_rate": 0.3}},
        },
        min_rel_drop=0.5,
        min_abs_drop=0.15,
    )
    assert gate["passed"] is True

    gate_fail = mod._build_geometry_null_collapse_gate(
        observed_strata={"events": {"parity_signal_rate": 0.8}},
        null_strata={
            "theta_roll": {"events": {"parity_signal_rate": 0.7}},
            "center_jitter": {"events": {"parity_signal_rate": 0.3}},
        },
        min_rel_drop=0.5,
        min_abs_drop=0.15,
    )
    assert gate_fail["passed"] is False


def test_mode_invariant_claim_gate_uses_all_modes_margin() -> None:
    mod = _load_eval_module()
    sel = pd.DataFrame(
        {
            "case_id": ["e1", "e2", "c1", "c2"],
            "case_type": ["storm", "storm", "control", "control"],
            "parity_pass_all_modes": [True, True, False, False],
            "parity_pass_any_modes": [True, True, True, False],
        }
    )
    gate = mod._build_mode_invariant_claim_gate(
        selection_df=sel,
        event_minus_far_min=0.15,
        far_max=0.35,
    )
    assert gate["passed"] is True
    assert gate["margin_all_modes"] > 0.15


def test_mode_invariant_claim_gate_prefers_storm_flags_over_case_type() -> None:
    mod = _load_eval_module()
    sel = pd.DataFrame(
        {
            "case_id": ["e1", "e2", "f1", "f2"],
            # Intentionally uninformative labels to ensure storm flags drive event/far split.
            "case_type": ["storm", "storm", "storm", "storm"],
            "storm_max": [1, 1, 0, 0],
            "near_storm_max": [0, 0, 0, 0],
            "pregen_max": [0, 0, 0, 0],
            "parity_pass_all_modes": [True, True, False, True],
            "parity_pass_any_modes": [True, True, False, True],
        }
    )
    gate = mod._build_mode_invariant_claim_gate(
        selection_df=sel,
        event_minus_far_min=0.15,
        far_max=0.60,
    )
    assert gate["event_rate_all_modes"] == 1.0
    assert gate["far_rate_all_modes"] == 0.5
    assert gate["passed"] is True


def test_anomaly_mode_gate_can_pass_on_casewise_stability() -> None:
    mod = _load_eval_module()
    gate = mod._build_anomaly_mode_gate(
        anomaly_ablation={
            "decision_stability": {
                "agreement_vs_none_mean": 0.50,
                "agreement_vs_none_min": 0.40,
                "rank_agreement_vs_none_mean": 0.10,
                "rank_agreement_vs_none_min": 0.00,
            }
        },
        anomaly_selection_summary={
            "selected_all_leads": True,
            "robust_far_all_leads": True,
        },
        agreement_mean_min=0.80,
        agreement_min_min=0.70,
    )
    assert gate["pass_decision_agreement_diagnostic"] is False
    assert gate["canonical_mode_selected"] is True
    assert gate["canonical_mode_robust_far"] is True
    assert gate["passed"] is True


def test_anomaly_mode_distribution_audit_reports_js() -> None:
    mod = _load_eval_module()
    sel = pd.DataFrame(
        {
            "case_id": ["e1", "e2", "c1", "c2"],
            "case_type": ["storm", "storm", "control", "control"],
            "canonical_mode": ["none", "lat_hour", "none", "none"],
        }
    )
    audit = mod._build_anomaly_mode_distribution_audit(sel)
    assert audit["n_cases"] == 4
    assert "event_far_mode_js_divergence" in audit


def test_mode_to_column_supports_anomaly_aliases() -> None:
    mod = _load_eval_module()
    assert mod._mode_to_column("none") == "O_raw"
    assert mod._mode_to_column("lat_day") == "O_lat_day"
    assert mod._mode_to_column("lat_hour") == "O_lat_hour"


def test_select_canonical_anomaly_mode_by_lead_returns_mapping() -> None:
    mod = _load_eval_module()
    train = pd.DataFrame(
        {
            "case_id": ["e1", "e1", "f1", "f1", "e2", "e2", "f2", "f2"],
            "lead_bucket": ["24", "24", "24", "24", "120", "120", "120", "120"],
            "t": pd.to_datetime(["2025-03-01 00:00:00", "2025-03-01 01:00:00"] * 4),
            "L": [100.0] * 8,
            "hand": ["L", "R"] * 4,
            "storm": [1, 1, 0, 0, 1, 1, 0, 0],
            "near_storm": [0] * 8,
            "pregen": [0] * 8,
            "case_type": ["storm", "storm", "control", "control", "storm", "storm", "control", "control"],
            "O_raw": [2.0, 1.2, 1.1, 1.0, 1.7, 1.0, 1.1, 1.0],
            "O_lat_day": [2.1, 1.3, 1.1, 1.0, 1.4, 1.0, 1.1, 1.0],
            "O_lat_hour": [2.0, 1.4, 1.1, 1.0, 1.5, 1.0, 1.1, 1.0],
            "O": [2.0, 1.2, 1.1, 1.0, 1.7, 1.0, 1.1, 1.0],
            "O_polar_left": [1.2] * 8,
            "O_polar_right": [1.1] * 8,
            "O_polar_chiral": [0.2] * 8,
            "O_polar_odd_ratio": [0.2] * 8,
            "O_polar_eta": [0.2] * 8,
        }
    )
    sel = mod._select_canonical_anomaly_mode_by_lead(
        train_df=train,
        seed=123,
        far_tolerance=0.10,
    )
    assert sel["selected_all_leads"] is True
    assert sel["robust_far_all_leads"] is True
    assert "24" in sel["canonical_mode_by_lead"]
    assert "120" in sel["canonical_mode_by_lead"]


def test_angular_witness_report_requires_theta_roll_drop() -> None:
    mod = _load_eval_module()
    obs = pd.DataFrame(
        {
            "case_id": ["e1", "e1", "e2", "e2", "c1", "c1", "c2", "c2"],
            "case_type": ["storm", "storm", "storm", "storm", "control", "control", "control", "control"],
            "lead_bucket": ["24", "24", "120", "120", "24", "24", "120", "120"],
            "hand": ["L", "R", "L", "R", "L", "R", "L", "R"],
            "O_polar_left": [2.0, 2.1, 1.8, 1.9, 0.5, 0.5, 0.6, 0.6],
            "O_polar_right": [0.2, 0.2, 0.3, 0.3, 0.5, 0.5, 0.6, 0.6],
            "O_polar_chiral": [1.5, 1.4, 1.2, 1.1, 0.1, 0.1, 0.12, 0.12],
            "O_polar_odd_ratio": [1.2, 1.1, 1.0, 0.9, 0.1, 0.1, 0.1, 0.1],
            "O_polar_eta": [1.3, 1.2, 1.1, 1.0, 0.1, 0.1, 0.1, 0.1],
        }
    )
    theta = obs.copy()
    theta["O_polar_left"] = [0.05, 0.05, 0.04, 0.04, 0.5, 0.5, 0.6, 0.6]
    theta["O_polar_right"] = [0.05, 0.05, 0.04, 0.04, 0.5, 0.5, 0.6, 0.6]
    theta["O_polar_chiral"] = [0.01, 0.01, 0.01, 0.01, 0.1, 0.1, 0.12, 0.12]
    theta["O_polar_odd_ratio"] = [0.01, 0.01, 0.01, 0.01, 0.1, 0.1, 0.1, 0.1]
    theta["O_polar_eta"] = [0.01, 0.01, 0.01, 0.01, 0.1, 0.1, 0.1, 0.1]
    center = obs.copy()
    center["O_polar_left"] = [0.08, 0.08, 0.07, 0.07, 0.5, 0.5, 0.6, 0.6]
    center["O_polar_right"] = [0.08, 0.08, 0.07, 0.07, 0.5, 0.5, 0.6, 0.6]
    center["O_polar_chiral"] = [0.02, 0.02, 0.02, 0.02, 0.1, 0.1, 0.12, 0.12]
    center["O_polar_odd_ratio"] = [0.02, 0.02, 0.02, 0.02, 0.1, 0.1, 0.1, 0.1]
    center["O_polar_eta"] = [0.02, 0.02, 0.02, 0.02, 0.1, 0.1, 0.1, 0.1]
    rep = mod._build_angular_witness_report(
        observed_samples=obs,
        theta_roll_samples=theta,
        center_jitter_samples=center,
        margin_min=0.05,
        d_min=0.10,
        p_max=0.60,
        null_drop_min=0.20,
        null_abs_drop_min=0.05,
        null_margin_max=0.07,
        permutation_n=300,
        required=True,
    )
    assert rep["gate"]["passed"] is True
    assert rep["observed"]["margin"] > 0.0


def test_theta_roll_polar_collapses_pair_asymmetry() -> None:
    mod = _load_eval_module()
    samples = pd.DataFrame(
        {
            "case_id": ["c1", "c1", "c1", "c1"],
            "t": pd.to_datetime(["2025-03-01 00:00:00", "2025-03-01 00:00:00", "2025-03-01 01:00:00", "2025-03-01 01:00:00"]),
            "L": [100.0, 100.0, 100.0, 100.0],
            "hand": ["L", "R", "L", "R"],
            "O": [2.0, 0.2, 2.2, 0.2],
            "O_polar_left": [2.0, 2.1, 2.0, 2.2],
            "O_polar_right": [0.2, 0.2, 0.2, 0.2],
            "O_polar_chiral": [1.2, 1.3, 1.1, 1.2],
            "O_polar_spiral": [1.8, 1.7, 1.9, 1.8],
            "O_polar_eta": [1.1, 1.0, 1.2, 1.1],
            "O_polar_odd_ratio": [1.4, 1.3, 1.5, 1.4],
        }
    )

    def _eta(df: pd.DataFrame) -> float:
        piv = df.pivot_table(index=["t", "L"], columns="hand", values="O", aggfunc="first")
        den = 0.5 * (piv["L"] + piv["R"])
        eta = (piv["L"] - piv["R"]).abs() / den
        return float(eta.mean())

    eta_before = _eta(samples)
    rolled = mod._theta_roll_polar(samples, rng=np.random.default_rng(7))
    eta_after = _eta(rolled)
    assert eta_after < eta_before * 0.5


def test_geometry_null_by_mode_includes_extended_nulls() -> None:
    mod = _load_eval_module()
    samples = pd.DataFrame(
        {
            "case_id": ["e1", "e1", "c1", "c1"],
            "t": pd.to_datetime(["2025-03-01 00:00:00"] * 4),
            "L": [100.0, 100.0, 100.0, 100.0],
            "hand": ["L", "R", "L", "R"],
            "storm": [1, 1, 0, 0],
            "near_storm": [0, 0, 0, 0],
            "pregen": [0, 0, 0, 0],
            "O": [2.0, 1.0, 1.2, 1.0],
            "O_raw": [2.0, 1.0, 1.2, 1.0],
            "O_lat_day": [2.0, 1.0, 1.2, 1.0],
            "O_lat_hour": [2.0, 1.0, 1.2, 1.0],
            "O_polar_spiral": [2.0, 1.9, 1.1, 1.0],
            "O_polar_chiral": [0.8, 0.7, 0.1, 0.1],
            "O_polar_eta": [0.9, 0.8, 0.1, 0.1],
            "O_polar_odd_ratio": [0.9, 0.8, 0.1, 0.1],
            "lon0": [150.0, 150.25, 151.0, 151.25],
        }
    )
    out = mod._build_geometry_null_by_mode(samples=samples, seed=7, modes=["none"])
    rec = out["modes"]["none"]
    assert "mirror_axis_jitter" in rec
    assert "radial_scramble" in rec


def test_claim_contract_validation_accepts_required_schema(tmp_path) -> None:
    mod = _load_eval_module()
    ds = tmp_path / "dataset"
    ds.mkdir(parents=True, exist_ok=True)
    (ds / "dataset.yaml").write_text("schema_version: 1\n", encoding="utf-8")
    pd.DataFrame({"case_id": ["c1"], "t": [1], "L": [1.0], "hand": ["L"], "O": [1.0]}).to_parquet(ds / "samples.parquet", index=False)
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("knee: {method: segmented}\n", encoding="utf-8")
    out = tmp_path / "evaluation.json"
    args = SimpleNamespace(
        claim_contract_schema_version=1,
        seed=42,
        split_date="2025-04-01",
        time_buffer_hours=72.0,
        parity_event_minus_far_min=0.15,
        parity_far_max=0.35,
        parity_confound_max_ratio=0.6,
        anomaly_agreement_mean_min=0.85,
        anomaly_agreement_min_min=0.8,
        null_collapse_min_drop=0.5,
        null_collapse_min_abs_drop=0.15,
        angular_witness_margin_min=0.1,
        angular_witness_d_min=0.4,
        angular_witness_p_max=0.1,
        angular_witness_permutation_n=200,
        angular_witness_null_drop_min=0.3,
        angular_witness_null_abs_drop_min=0.05,
        angular_witness_null_margin_max=0.05,
        dataset=str(ds),
        config=str(cfg),
        out=str(out),
        split_start="2025-02-01",
        split_end="2025-05-31",
    )
    summary = {
        "storm_id_col": "storm_id",
        "claim_mode": "ensemble",
        "claim_mode_applied": "scalars_only",
        "anomaly_mode_selection": {"canonical_mode_by_lead": {"24": "lat_hour"}, "default_mode": "none"},
    }
    contract = mod._build_claim_contract(summary=summary, args=args, dataset_dir=ds, config_path=cfg, out_path=out)
    errors = mod._validate_claim_contract(contract)
    assert errors == []


def test_ibtracs_strict_eval_builds_overlay() -> None:
    mod = _load_eval_module()
    metrics = pd.DataFrame(
        {
            "case_id": ["e1", "e2", "f1", "f2"],
            "case_type": ["storm", "storm", "control", "control"],
            "storm_max": [1, 1, 0, 0],
            "near_storm_max": [0, 0, 0, 0],
            "pregen_max": [0, 0, 0, 0],
            "lead_bucket": ["24", "24", "24", "24"],
            "parity_signal_pass_effective": [True, True, False, False],
            "parity_signal_pass": [True, True, False, False],
            "knee_detected_effective": [False, False, False, False],
            "knee_detected": [False, False, False, False],
            "P_lock": [0.9, 0.8, 0.1, 0.2],
            "run_error": [None, None, None, None],
            "nearest_storm_distance_km": [120.0, 250.0, 1800.0, 2200.0],
        }
    )
    samples = pd.DataFrame(
        {
            "case_id": ["e1", "e1", "e2", "e2", "f1", "f1", "f2", "f2"],
            "nearest_storm_distance_km": [120.0, 120.0, 250.0, 250.0, 1800.0, 1800.0, 2200.0, 2200.0],
            "nearest_storm_time_delta_h": [1.0, 1.0, 2.0, 2.0, 8.0, 8.0, 9.0, 9.0],
            "storm_id_nearest": ["S1", "S1", "S1", "S1", None, None, None, None],
        }
    )
    out = mod._build_ibtracs_strict_eval(
        case_metrics=metrics,
        samples=samples,
        radius_km=300.0,
        far_min_km=1500.0,
        time_hours=3.0,
        p_lock_threshold=0.5,
    )
    assert out["available"] is True
    assert out["n_event_cases"] == 2
    assert out["n_far_cases"] == 2
