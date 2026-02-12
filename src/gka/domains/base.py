"""Base domain adapter protocol and reusable implementation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from gka.core.types import (
    DatasetBundle,
    FrequencySeries,
    ImpedanceInputs,
    MirrorSpec,
    Observable,
    SizeSeries,
)
from gka.data.io import load_dataset_spec, load_samples


class DomainAdapter(Protocol):
    name: str

    def load(self, dataset_path: str) -> DatasetBundle:
        """Return canonical in-memory objects (xarray/pandas/np arrays) plus metadata."""

    def mirror_map(self, bundle: DatasetBundle) -> MirrorSpec:
        """Define involution M for parity split. Must be involutive: M(M(x))=x."""

    def observable(self, bundle: DatasetBundle) -> Observable:
        """Return X and any derived fields used by ops modules."""

    def size_proxy(self, bundle: DatasetBundle) -> SizeSeries:
        """Return L values (sizes) tied to samples/records."""

    def frequency_proxy(self, bundle: DatasetBundle) -> FrequencySeries:
        """Return omega axis if spectral analysis is possible; may be None."""

    def impedance_proxy(self, bundle: DatasetBundle) -> ImpedanceInputs:
        """Return pieces needed for impedance check: (omega_k, cm or v, a or L)."""


@dataclass
class TabularAdapterBase:
    """Base adapter for canonical table-driven datasets."""

    name: str

    def load(self, dataset_path: str) -> DatasetBundle:
        root = Path(dataset_path)
        spec = load_dataset_spec(root)
        samples = load_samples(root)
        return DatasetBundle(dataset_path=root, samples=samples, dataset_spec=spec, arrays={}, metadata={})

    def mirror_map(self, bundle: DatasetBundle) -> MirrorSpec:
        spec = bundle.dataset_spec
        cols = spec.get("columns", {})
        gcol = cols.get("group", "case_id")
        tcol = cols.get("time", "t")
        lcol = cols.get("size", "L")
        hcol = cols.get("handedness", "hand")

        df = bundle.samples.reset_index(drop=True)
        key_cols = [gcol, tcol, lcol]
        if "omega" in df.columns:
            key_cols.append("omega")

        key_to_index: dict[tuple[object, ...], int] = {}
        for idx, row in df.iterrows():
            key = tuple(row[c] for c in key_cols) + (str(row[hcol]),)
            key_to_index[key] = int(idx)

        pair_index = np.empty(df.shape[0], dtype=int)
        for idx, row in df.iterrows():
            hand = str(row[hcol])
            partner_hand = "R" if hand == "L" else "L"
            partner_key = tuple(row[c] for c in key_cols) + (partner_hand,)
            if partner_key not in key_to_index:
                raise ValueError(
                    f"Missing mirrored partner for row index {idx} key={partner_key}; "
                    "run gka validate and repair pairing"
                )
            pair_index[idx] = key_to_index[partner_key]

        return MirrorSpec(
            mirror_type=spec.get("mirror", {}).get("type", "label_swap"),
            pair_index=pair_index,
            details=spec.get("mirror", {}).get("details", {}),
        )

    def observable(self, bundle: DatasetBundle) -> Observable:
        cols = bundle.dataset_spec.get("columns", {})
        observable_cols = cols.get("observable", ["O"])
        df = bundle.samples

        for col in observable_cols:
            if col in df.columns:
                arr = df[col].to_numpy(dtype=float)
                return Observable(X=arr, names=[col], metadata={"source": "column"})

        if "O_path" in df.columns:
            values = np.empty(df.shape[0], dtype=float)
            for i, rel_path in enumerate(df["O_path"].astype(str).tolist()):
                path = bundle.dataset_path / rel_path
                if not path.exists():
                    raise ValueError(f"O_path does not exist: {path}")
                if path.suffix.lower() == ".npy":
                    data = np.load(path)
                    values[i] = float(np.mean(data))
                else:
                    raise ValueError(
                        f"Unsupported O_path file format '{path.suffix}'. Only .npy is supported."
                    )
            return Observable(X=values, names=["O_path_mean"], metadata={"source": "O_path"})

        raise ValueError(
            "No observable found. Expected one of columns.observable in samples.parquet or O_path"
        )

    def size_proxy(self, bundle: DatasetBundle) -> SizeSeries:
        size_col = bundle.dataset_spec.get("columns", {}).get("size", "L")
        if size_col not in bundle.samples.columns:
            raise ValueError(f"Missing size proxy column '{size_col}' in samples")
        return SizeSeries(L=bundle.samples[size_col].to_numpy(dtype=float))

    def frequency_proxy(self, bundle: DatasetBundle) -> FrequencySeries:
        if "omega" in bundle.samples.columns:
            return FrequencySeries(omega=bundle.samples["omega"].to_numpy(dtype=float))
        return FrequencySeries(omega=None)

    def impedance_proxy(self, bundle: DatasetBundle) -> ImpedanceInputs:
        df = bundle.samples
        if "omega_k" in df.columns:
            omega = df["omega_k"].to_numpy(dtype=float)
        elif "omega" in df.columns:
            omega = df["omega"].to_numpy(dtype=float)
        else:
            omega = None
        L = df["L"].to_numpy(dtype=float) if "L" in df.columns else None

        cm_or_v = None
        for name in ("cm", "v", "c_m"):
            if name in df.columns:
                cm_or_v = df[name].to_numpy(dtype=float)
                break

        a = None
        for name in ("a", "a0", "L"):
            if name in df.columns:
                a = df[name].to_numpy(dtype=float)
                break

        return ImpedanceInputs(omega_k=omega, L=L, cm_or_v=cm_or_v, a_or_L=a)
