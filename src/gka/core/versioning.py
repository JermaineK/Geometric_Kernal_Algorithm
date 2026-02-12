"""Version and environment helpers."""

from __future__ import annotations

import importlib.metadata
import subprocess
from pathlib import Path
from typing import Any


def package_version() -> str:
    try:
        return importlib.metadata.version("gka")
    except importlib.metadata.PackageNotFoundError:
        return "0.0.0+local"


def git_commit_hash(cwd: str | Path) -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(cwd),
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return out
    except Exception:
        return None


def package_versions(packages: list[str] | None = None) -> dict[str, str]:
    names = packages or ["numpy", "pandas", "pyarrow", "scipy", "matplotlib", "PyYAML"]
    out: dict[str, str] = {}
    for name in names:
        try:
            out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            out[name] = "not-installed"
    out["gka"] = package_version()
    return out


def invocation_string(argv: list[str]) -> str:
    return " ".join(argv)
