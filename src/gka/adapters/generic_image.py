"""Adapter for generic image stacks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from gka.adapters.base import AdapterOutput


@dataclass
class GenericImageAdapter:
    """Prepare scalar observables from image tensor stacks."""

    name: str = "generic_image"

    def prepare(
        self,
        source: str,
        *,
        mirror_axis: int = -1,
        reduce: str = "mean",
    ) -> AdapterOutput:
        path = Path(source)
        if path.suffix.lower() != ".npy":
            raise ValueError("generic_image adapter expects a .npy tensor stack")
        arr = np.load(path)
        if arr.ndim < 3:
            raise ValueError("Image stack must be at least 3D: [n_samples, ...]")

        if reduce == "mean":
            x = arr.reshape(arr.shape[0], -1).mean(axis=1)
        elif reduce == "std":
            x = arr.reshape(arr.shape[0], -1).std(axis=1)
        else:
            raise ValueError(f"Unsupported reduce mode: {reduce}")

        return AdapterOutput(
            X=np.asarray(x, dtype=float),
            mirror_op={"type": "spatial_reflection", "axis": int(mirror_axis)},
            coords={"sample": np.arange(arr.shape[0], dtype=float)},
            meta={
                "source": str(path.resolve()),
                "shape": list(arr.shape),
                "reduce": reduce,
            },
        )
