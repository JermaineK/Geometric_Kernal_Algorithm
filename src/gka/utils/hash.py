"""Hashing helpers used for reproducibility metadata."""

from __future__ import annotations

import hashlib
from pathlib import Path


def file_sha256(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    p = Path(path)
    with p.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def dataset_hash(dataset_path: str | Path) -> str:
    p = Path(dataset_path)
    digest = hashlib.sha256()
    files = sorted(
        f for f in p.rglob("*") if f.is_file() and ".git" not in f.parts and "__pycache__" not in f.parts
    )
    for file_path in files:
        rel = file_path.relative_to(p).as_posix().encode("utf-8")
        digest.update(rel)
        digest.update(file_sha256(file_path).encode("ascii"))
    return digest.hexdigest()
