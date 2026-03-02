from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import pandas as pd


class TimeBasis(str, Enum):
    SOURCE = "source"
    VALID = "valid"
    SELECTED = "selected"


@dataclass(frozen=True)
class CoverageContract:
    min_event_total: int = 0
    min_far_total: int = 0
    min_event_per_lead: int = 0
    min_far_per_lead: int = 0


def strict_ib_case_flags(
    *,
    samples: pd.DataFrame,
    time_basis: str,
    use_flags: bool,
    far_quality_tags: tuple[str, ...],
    min_far_quality_fraction: float = 0.0,
    min_far_kinematic_clean_fraction: float = 0.0,
    strict_far_min_any_storm_km: float = 0.0,
    strict_far_min_nearest_storm_km: float = 0.0,
) -> pd.DataFrame:
    sub = samples.copy()
    if sub.empty or ("case_id" not in sub.columns):
        return pd.DataFrame()
    basis = str(time_basis or "selected").strip().lower()
    if basis not in {TimeBasis.SOURCE.value, TimeBasis.VALID.value, TimeBasis.SELECTED.value}:
        basis = TimeBasis.SELECTED.value
    if "lead_bucket" not in sub.columns:
        sub["lead_bucket"] = "none"
    if "ib_far_quality_tag" not in sub.columns:
        sub["ib_far_quality_tag"] = "missing"
    sub["ib_far_quality_tag"] = sub["ib_far_quality_tag"].fillna("missing").astype(str)

    if basis == TimeBasis.SOURCE.value:
        if bool(use_flags):
            event_pref = ("ib_event_strict_source", "ib_event_source", "ib_event_strict", "ib_event")
            far_pref = ("ib_far_strict_source", "ib_far_source", "ib_far_strict", "ib_far")
        else:
            event_pref = ("ib_event_source", "ib_event", "ib_event_strict_source", "ib_event_strict")
            far_pref = ("ib_far_source", "ib_far", "ib_far_strict_source", "ib_far_strict")
        min_any_col = "ib_min_dist_any_storm_km_source"
        nearest_col = "nearest_storm_distance_km_source"
    elif basis == TimeBasis.VALID.value:
        if bool(use_flags):
            event_pref = ("ib_event_strict_valid", "ib_event_valid", "ib_event_strict", "ib_event")
            far_pref = ("ib_far_strict_valid", "ib_far_valid", "ib_far_strict", "ib_far")
        else:
            event_pref = ("ib_event_valid", "ib_event", "ib_event_strict_valid", "ib_event_strict")
            far_pref = ("ib_far_valid", "ib_far", "ib_far_strict_valid", "ib_far_strict")
        min_any_col = "ib_min_dist_any_storm_km_valid"
        nearest_col = "nearest_storm_distance_km_valid"
    else:
        if bool(use_flags):
            event_pref = ("ib_event_strict", "ib_event", "ib_event_strict_max", "ib_event_max")
            far_pref = ("ib_far_strict", "ib_far", "ib_far_strict_max", "ib_far_max")
        else:
            event_pref = ("ib_event", "ib_event_strict", "ib_event_max", "ib_event_strict_max")
            far_pref = ("ib_far", "ib_far_strict", "ib_far_max", "ib_far_strict_max")
        min_any_col = "ib_min_dist_any_storm_km"
        nearest_col = "nearest_storm_distance_km"

    def _first_present(pref: tuple[str, ...]) -> str | None:
        for c in pref:
            if c in sub.columns:
                return c
        return None

    event_col = _first_present(event_pref)
    far_col = _first_present(far_pref)
    if (event_col is None) or (far_col is None):
        return pd.DataFrame()

    agg_spec: dict[str, tuple[str, str]] = {
        "lead_bucket": ("lead_bucket", "first"),
        "event_flag": (event_col, "max"),
        "far_flag": (far_col, "max"),
        "far_quality_tag": ("ib_far_quality_tag", "first"),
    }
    if min_any_col in sub.columns:
        agg_spec["min_dist_any_storm_km_case"] = (min_any_col, "min")
    if nearest_col in sub.columns:
        agg_spec["nearest_storm_dist_km_case"] = (nearest_col, "min")
    case_df = sub.groupby("case_id", as_index=False).agg(**agg_spec)

    case_tags = (
        sub.groupby("case_id")["ib_far_quality_tag"]
        .agg(lambda x: sorted({str(v).strip().lower() for v in x if str(v).strip()}))
        .rename("far_quality_tags_case")
        .reset_index()
    )
    case_df = case_df.merge(case_tags, on="case_id", how="left")
    case_df["far_quality_tags_case"] = case_df["far_quality_tags_case"].apply(
        lambda v: v if isinstance(v, list) else []
    )

    allowed = {str(v).strip().lower() for v in (far_quality_tags or ()) if str(v).strip()}
    if allowed:
        quality_frac = (
            sub.assign(_ok=sub["ib_far_quality_tag"].str.strip().str.lower().isin(allowed))
            .groupby("case_id")["_ok"]
            .mean()
            .rename("far_quality_ok_fraction")
            .reset_index()
        )
        case_df = case_df.merge(quality_frac, on="case_id", how="left")
        case_df["far_quality_ok_fraction"] = pd.to_numeric(
            case_df["far_quality_ok_fraction"], errors="coerce"
        ).fillna(0.0)
    else:
        case_df["far_quality_ok_fraction"] = 1.0

    if "ib_kinematic_clean" in sub.columns:
        kclean = (
            sub.assign(_kc=pd.to_numeric(sub["ib_kinematic_clean"], errors="coerce"))
            .groupby("case_id")["_kc"]
            .mean()
            .rename("far_kinematic_clean_fraction")
            .reset_index()
        )
        case_df = case_df.merge(kclean, on="case_id", how="left")
    else:
        case_df["far_kinematic_clean_fraction"] = 1.0
    case_df["far_kinematic_clean_fraction"] = pd.to_numeric(
        case_df["far_kinematic_clean_fraction"], errors="coerce"
    ).fillna(0.0)

    if allowed:
        tag_pass = case_df["far_quality_tags_case"].apply(lambda tags: bool(set(tags) & allowed))
    else:
        tag_pass = pd.Series(True, index=case_df.index)

    far_pass = pd.to_numeric(case_df["far_flag"], errors="coerce").fillna(0) > 0
    far_pass = far_pass & tag_pass
    far_pass = far_pass & (
        pd.to_numeric(case_df["far_quality_ok_fraction"], errors="coerce").fillna(0.0)
        >= float(max(0.0, min_far_quality_fraction))
    )
    far_pass = far_pass & (
        pd.to_numeric(case_df["far_kinematic_clean_fraction"], errors="coerce").fillna(0.0)
        >= float(max(0.0, min_far_kinematic_clean_fraction))
    )
    if "min_dist_any_storm_km_case" in case_df.columns and float(strict_far_min_any_storm_km) > 0:
        far_pass = far_pass & (
            pd.to_numeric(case_df["min_dist_any_storm_km_case"], errors="coerce").fillna(-1.0)
            >= float(strict_far_min_any_storm_km)
        )
    if "nearest_storm_dist_km_case" in case_df.columns and float(strict_far_min_nearest_storm_km) > 0:
        far_pass = far_pass & (
            pd.to_numeric(case_df["nearest_storm_dist_km_case"], errors="coerce").fillna(-1.0)
            >= float(strict_far_min_nearest_storm_km)
        )

    case_df["far_flag"] = far_pass.astype(bool)
    case_df["event_flag"] = (
        pd.to_numeric(case_df["event_flag"], errors="coerce").fillna(0) > 0
    ).astype(bool)
    case_df["lead_bucket"] = case_df["lead_bucket"].astype(str)
    case_df["time_basis"] = str(basis)
    case_df["event_col_used"] = str(event_col)
    case_df["far_col_used"] = str(far_col)
    return case_df


def strict_ib_coverage(
    *,
    samples: pd.DataFrame,
    case_ids: set[str],
    time_basis: str,
    use_flags: bool,
    far_quality_tags: tuple[str, ...],
    min_far_quality_fraction: float = 0.0,
    min_far_kinematic_clean_fraction: float = 0.0,
    strict_far_min_any_storm_km: float = 0.0,
    strict_far_min_nearest_storm_km: float = 0.0,
) -> dict[str, Any]:
    if samples is None or samples.empty or not case_ids:
        return {
            "event_total": 0,
            "far_total": 0,
            "event_by_lead": {},
            "far_by_lead": {},
            "available": False,
            "reason": "empty",
        }
    sub = samples.loc[samples["case_id"].astype(str).isin(case_ids)].copy()
    if sub.empty:
        return {
            "event_total": 0,
            "far_total": 0,
            "event_by_lead": {},
            "far_by_lead": {},
            "available": False,
            "reason": "no_rows",
        }
    case_df = strict_ib_case_flags(
        samples=sub,
        time_basis=str(time_basis),
        use_flags=bool(use_flags),
        far_quality_tags=tuple(far_quality_tags),
        min_far_quality_fraction=float(min_far_quality_fraction),
        min_far_kinematic_clean_fraction=float(min_far_kinematic_clean_fraction),
        strict_far_min_any_storm_km=float(strict_far_min_any_storm_km),
        strict_far_min_nearest_storm_km=float(strict_far_min_nearest_storm_km),
    )
    if case_df.empty:
        return {
            "event_total": 0,
            "far_total": 0,
            "event_by_lead": {},
            "far_by_lead": {},
            "available": False,
            "reason": "missing_ibtracs_columns",
        }
    ev = pd.to_numeric(case_df["event_flag"], errors="coerce").fillna(0) > 0
    fr = pd.to_numeric(case_df["far_flag"], errors="coerce").fillna(0) > 0
    event_cases = case_df.loc[ev].copy()
    far_cases = case_df.loc[fr].copy()
    test_leads = sorted(case_df["lead_bucket"].dropna().astype(str).unique().tolist())
    basis = str((case_df.get("time_basis", pd.Series(["selected"])).iloc[0])).strip().lower()
    event_col = str((case_df.get("event_col_used", pd.Series([""])).iloc[0]))
    far_col = str((case_df.get("far_col_used", pd.Series([""])).iloc[0]))
    allowed = {str(v).strip().lower() for v in (far_quality_tags or ()) if str(v).strip()}
    return {
        "event_total": int(event_cases["case_id"].nunique()),
        "far_total": int(far_cases["case_id"].nunique()),
        "event_by_lead": {
            str(k): int(v)
            for k, v in event_cases.groupby("lead_bucket")["case_id"].nunique().to_dict().items()
        },
        "far_by_lead": {
            str(k): int(v)
            for k, v in far_cases.groupby("lead_bucket")["case_id"].nunique().to_dict().items()
        },
        "test_leads": test_leads,
        "event_col_used": str(event_col),
        "far_col_used": str(far_col),
        "time_basis": str(basis),
        "use_flags": bool(use_flags),
        "far_quality_tags": sorted(list(allowed)),
        "available": True,
        "reason": "ok",
    }


def strict_coverage_ok(
    *,
    coverage: dict[str, Any],
    min_event_total: int,
    min_far_total: int,
    min_event_per_lead: int,
    min_far_per_lead: int,
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    ev_total = int((coverage or {}).get("event_total", 0))
    fr_total = int((coverage or {}).get("far_total", 0))
    test_leads = [str(v) for v in ((coverage or {}).get("test_leads", []) or [])]
    if int(min_event_total) > 0 and ev_total < int(min_event_total):
        reasons.append(f"event_total<{int(min_event_total)}")
    if int(min_far_total) > 0 and fr_total < int(min_far_total):
        reasons.append(f"far_total<{int(min_far_total)}")
    if int(min_event_per_lead) > 0:
        ev_by = {str(k): int(v) for k, v in ((coverage or {}).get("event_by_lead", {}) or {}).items()}
        for lead in sorted(test_leads):
            if int(ev_by.get(str(lead), 0)) < int(min_event_per_lead):
                reasons.append(f"event_lead:{lead}<{int(min_event_per_lead)}")
    if int(min_far_per_lead) > 0:
        fr_by = {str(k): int(v) for k, v in ((coverage or {}).get("far_by_lead", {}) or {}).items()}
        for lead in sorted(test_leads):
            if int(fr_by.get(str(lead), 0)) < int(min_far_per_lead):
                reasons.append(f"far_lead:{lead}<{int(min_far_per_lead)}")
    return (len(reasons) == 0), reasons
