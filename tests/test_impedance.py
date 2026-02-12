import numpy as np

from gka.ops.impedance import impedance_alignment


def test_impedance_alignment_passes_close_to_one():
    out = impedance_alignment(
        omega_k=2 * np.pi * 10.0,
        L=3.0,
        cm_or_v=30.0,
        a=None,
        tolerance=0.1,
    )
    assert out.ratio is not None
    assert out.passed


def test_impedance_alignment_handles_missing_inputs():
    out = impedance_alignment(None, 3.0, 30.0, None)
    assert out.ratio is None
    assert out.passed is None
