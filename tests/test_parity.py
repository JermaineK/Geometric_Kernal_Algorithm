import numpy as np

from gka.core.types import MirrorSpec
from gka.ops.parity import parity_split


def test_parity_split_basic():
    x = np.array([1.0, 3.0, 2.0, 4.0])
    mirror = MirrorSpec(mirror_type="label_swap", pair_index=np.array([1, 0, 3, 2]))
    out = parity_split(x, mirror)
    assert out.X_plus.shape == x.shape
    assert out.X_minus.shape == x.shape
    assert out.E_plus >= 0
    assert out.E_minus >= 0
    assert np.all(out.eta >= 0)


def test_parity_split_requires_involution():
    x = np.array([1.0, 2.0, 3.0])
    mirror = MirrorSpec(mirror_type="label_swap", pair_index=np.array([1, 2, 0]))
    try:
        parity_split(x, mirror)
        assert False
    except ValueError as exc:
        assert "involutive" in str(exc)
