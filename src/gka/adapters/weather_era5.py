"""ERA5-style weather adapter (xarray Dataset -> standardized tensors)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from gka.adapters.base import AdapterOutput


@dataclass
class WeatherERA5Adapter:
    """Prepare weather fields with normalization, masking, and windowing."""

    name: str = "weather_era5"

    def prepare(
        self,
        source: str,
        *,
        u_var: str = "u10",
        v_var: str = "v10",
        time_dim: str = "time",
        lat_dim: str = "latitude",
        lon_dim: str = "longitude",
        window: int = 3,
    ) -> AdapterOutput:
        xr = _require_xarray()
        ds = xr.open_dataset(source) if isinstance(source, (str, Path)) else source
        if u_var not in ds.data_vars or v_var not in ds.data_vars:
            raise ValueError(f"Dataset must contain variables {u_var!r} and {v_var!r}")
        for dim in (time_dim, lat_dim, lon_dim):
            if dim not in ds.dims:
                raise ValueError(f"Dataset missing required dimension: {dim}")

        u = ds[u_var].astype("float64")
        v = ds[v_var].astype("float64")
        speed = np.sqrt(np.square(u) + np.square(v))
        speed = speed.where(np.isfinite(speed), drop=False)

        if int(window) > 1:
            speed = speed.rolling({lat_dim: int(window), lon_dim: int(window)}, center=True, min_periods=1).mean()

        # Normalize per-time slice to control nonstationary amplitude drift.
        mean_t = speed.mean(dim=(lat_dim, lon_dim), skipna=True)
        std_t = speed.std(dim=(lat_dim, lon_dim), skipna=True)
        speed_z = (speed - mean_t) / (std_t + 1e-12)

        # Observable is odd-channel proxy: mean signed lat gradient magnitude per frame.
        lat_grad = speed_z.diff(lat_dim).mean(dim=(lat_dim, lon_dim), skipna=True)
        x = np.asarray(lat_grad.to_numpy(), dtype=float).reshape(-1)
        mask = np.isfinite(x)
        x = x[mask]
        if x.size == 0:
            raise ValueError("No finite weather samples remained after masking")

        time_vals = np.asarray(ds[time_dim].to_numpy()).reshape(-1)
        if time_vals.size == mask.size:
            time_vals = time_vals[mask]
        else:
            time_vals = time_vals[: x.size]

        return AdapterOutput(
            X=x,
            mirror_op={"type": "spatial_reflection", "axis": lon_dim},
            coords={
                "time_index": np.arange(x.size, dtype=float),
                "latitude": np.asarray(ds[lat_dim].to_numpy(), dtype=float),
                "longitude": np.asarray(ds[lon_dim].to_numpy(), dtype=float),
            },
            meta={
                "source": str(Path(source).resolve()) if isinstance(source, (str, Path)) else "xarray.Dataset",
                "u_var": u_var,
                "v_var": v_var,
                "window": int(window),
                "n_frames": int(x.size),
                "time_values": [str(v) for v in time_vals[: min(8, time_vals.size)]],
            },
        )

    def mirror_lon_about(
        self,
        frame: pd.DataFrame,
        *,
        lon0: float = 150.0,
        time_col: str = "time",
        lat_col: str = "lat",
        lon_col: str = "lon",
        u_col: str = "u10",
        v_col: str = "v10",
        scalar_cols: list[str] | None = None,
        max_lon_distance: float = 0.13,
    ) -> pd.DataFrame:
        """Mirror rows about a central meridian and add mirrored channels.

        Vector transform for longitude reflection:
        - u -> -u
        - v ->  v
        """

        required = [time_col, lat_col, lon_col, u_col, v_col]
        missing = [c for c in required if c not in frame.columns]
        if missing:
            raise ValueError(f"mirror_lon_about missing columns: {missing}")
        if scalar_cols is None:
            scalar_cols = []
        else:
            scalar_cols = [c for c in scalar_cols if c in frame.columns and c not in {u_col, v_col}]

        df = frame.copy()
        lon_values = np.sort(df[lon_col].dropna().unique().astype(float))
        if lon_values.size < 2:
            raise ValueError("mirror_lon_about requires at least two unique longitudes")

        mirrored = (2.0 * float(lon0)) - pd.to_numeric(df[lon_col], errors="coerce").to_numpy(dtype=float)
        snapped = _snap_to_grid(mirrored, lon_values)
        snap_dist = np.abs(snapped - mirrored)
        df["_mirror_lon"] = snapped
        df["_mirror_lon_dist"] = snap_dist
        df = df.loc[df["_mirror_lon_dist"] <= float(max_lon_distance)].copy()

        # Build right table for mirrored value lookup.
        join_cols = [time_col, lat_col, lon_col, u_col, v_col] + scalar_cols
        right = df[join_cols].rename(columns={lon_col: "_mirror_lon"})
        right = right.rename(columns={u_col: f"{u_col}_mirror_src", v_col: f"{v_col}_mirror_src"})
        for col in scalar_cols:
            right = right.rename(columns={col: f"{col}_mirror"})

        merged = df.merge(
            right,
            on=[time_col, lat_col, "_mirror_lon"],
            how="left",
            validate="many_to_one",
        )

        # Mirror transform for vector components.
        mirror_src_u = pd.to_numeric(merged[f"{u_col}_mirror_src"], errors="coerce")
        mirror_src_v = pd.to_numeric(merged[f"{v_col}_mirror_src"], errors="coerce")
        merged["_mirror_hit"] = np.isfinite(mirror_src_u.to_numpy(dtype=float)) & np.isfinite(
            mirror_src_v.to_numpy(dtype=float)
        )
        merged[f"{u_col}_mirror"] = -pd.to_numeric(merged[f"{u_col}_mirror_src"], errors="coerce")
        merged[f"{v_col}_mirror"] = pd.to_numeric(merged[f"{v_col}_mirror_src"], errors="coerce")
        merged = merged.drop(columns=[f"{u_col}_mirror_src", f"{v_col}_mirror_src"])
        return merged

    def add_parity_channels(
        self,
        frame: pd.DataFrame,
        *,
        u_col: str = "u10",
        v_col: str = "v10",
        u_mirror_col: str = "u10_mirror",
        v_mirror_col: str = "v10_mirror",
        eps: float = 1e-8,
    ) -> pd.DataFrame:
        """Add even/odd channels and bounded parity contrast."""

        required = [u_col, v_col, u_mirror_col, v_mirror_col]
        missing = [c for c in required if c not in frame.columns]
        if missing:
            raise ValueError(f"add_parity_channels missing columns: {missing}")

        df = frame.copy()
        u = pd.to_numeric(df[u_col], errors="coerce").to_numpy(dtype=float)
        v = pd.to_numeric(df[v_col], errors="coerce").to_numpy(dtype=float)
        um = pd.to_numeric(df[u_mirror_col], errors="coerce").to_numpy(dtype=float)
        vm = pd.to_numeric(df[v_mirror_col], errors="coerce").to_numpy(dtype=float)

        u_even = 0.5 * (u + um)
        u_odd = 0.5 * (u - um)
        v_even = 0.5 * (v + vm)
        v_odd = 0.5 * (v - vm)

        odd_mag = np.sqrt(np.square(u_odd) + np.square(v_odd))
        even_mag = np.sqrt(np.square(u_even) + np.square(v_even))

        df["u_even"] = u_even
        df["u_odd"] = u_odd
        df["v_even"] = v_even
        df["v_odd"] = v_odd
        df["speed_l"] = np.sqrt(np.square(u) + np.square(v))
        df["speed_r"] = np.sqrt(np.square(um) + np.square(vm))
        df["eta_parity"] = odd_mag / (odd_mag + even_mag + float(eps))
        return df


def _require_xarray():
    try:
        import xarray as xr
    except Exception as exc:  # pragma: no cover - optional dependency guard
        raise RuntimeError(
            "WeatherERA5Adapter requires optional dependency 'xarray'. "
            "Install with: pip install -e .[weather]"
        ) from exc
    return xr


def _snap_to_grid(values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Snap values to nearest point on a monotonic grid."""

    idx = np.searchsorted(grid, values)
    idx = np.clip(idx, 0, grid.size - 1)
    left = np.clip(idx - 1, 0, grid.size - 1)
    right = idx
    left_dist = np.abs(values - grid[left])
    right_dist = np.abs(values - grid[right])
    choose_left = left_dist <= right_dist
    out = np.where(choose_left, grid[left], grid[right])
    return out
