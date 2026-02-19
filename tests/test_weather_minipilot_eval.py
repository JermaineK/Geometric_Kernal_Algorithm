from __future__ import annotations

import importlib.util
from pathlib import Path

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
        }
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
