"""Adapter for generic tabular timeseries sources."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from gka.adapters.base import AdapterOutput


@dataclass
class GenericTimeseriesAdapter:
    """Prepare pair-aware timeseries data from csv/parquet tables."""

    name: str = "generic_timeseries"

    def prepare(
        self,
        source: str,
        *,
        value_col: str = "O",
        case_col: str = "case_id",
        time_col: str = "t",
        size_col: str = "L",
        hand_col: str = "hand",
    ) -> AdapterOutput:
        path = Path(source)
        df = self._load_frame(path)
        required = [value_col, case_col, time_col, size_col, hand_col]
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        df = df.copy()
        df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
        df = df.dropna(subset=[value_col, size_col])
        x = df[value_col].to_numpy(dtype=float)
        if x.size == 0:
            raise ValueError("No finite observable values remain after cleaning")

        pair_index = self._pair_index(df, case_col=case_col, time_col=time_col, size_col=size_col, hand_col=hand_col)
        return AdapterOutput(
            X=x,
            mirror_op={"type": "label_swap", "pair_index": pair_index.tolist()},
            coords={
                "time": pd.to_numeric(df[time_col], errors="coerce").fillna(0.0).to_numpy(dtype=float),
                "L": pd.to_numeric(df[size_col], errors="coerce").fillna(0.0).to_numpy(dtype=float),
            },
            meta={"source": str(path.resolve()), "rows": int(df.shape[0])},
        )

    def _load_frame(self, path: Path) -> pd.DataFrame:
        ext = path.suffix.lower()
        if ext == ".parquet":
            return pd.read_parquet(path)
        if ext in {".csv", ".txt"}:
            return pd.read_csv(path)
        raise ValueError(f"Unsupported table format: {ext}")

    def _pair_index(
        self,
        df: pd.DataFrame,
        *,
        case_col: str,
        time_col: str,
        size_col: str,
        hand_col: str,
    ) -> np.ndarray:
        keys = [case_col, time_col, size_col]
        frame = df.reset_index(drop=True)
        mapping: dict[tuple[Any, ...], int] = {}
        for idx, row in frame.iterrows():
            key = tuple(row[k] for k in keys) + (str(row[hand_col]),)
            mapping[key] = int(idx)

        pair = np.empty(frame.shape[0], dtype=int)
        for idx, row in frame.iterrows():
            h = str(row[hand_col]).upper()
            if h not in {"L", "R"}:
                raise ValueError(f"Invalid hand value at row {idx}: {row[hand_col]!r}")
            partner = "R" if h == "L" else "L"
            key = tuple(row[k] for k in keys) + (partner,)
            if key not in mapping:
                raise ValueError(f"Missing paired row for key={key}")
            pair[idx] = mapping[key]
        return pair
