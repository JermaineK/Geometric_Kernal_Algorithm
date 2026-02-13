import pandas as pd

from gka.cli.audit import _build_audit_payload


def test_audit_payload_decisions():
    row = pd.Series(
        {
            "knee_detected": False,
            "knee_confidence": 0.12,
            "knee_delta_bic": 3.1,
            "knee_rejected_because": "bic_weak;knee_at_edge",
            "L_k": float("nan"),
            "parity_signal_pass": False,
            "parity_mirror_stat": 0.03,
            "parity_p_perm": 0.41,
            "parity_p_dir": 0.53,
            "P_lock": 0.11,
            "gamma": 0.22,
            "Delta_hat": 1.78,
            "S_at_mu_k": 0.82,
            "W_mu": 0.4,
            "band_class_hat": "incoherent",
            "stability_class": "stable",
        }
    )

    payload = _build_audit_payload(
        row=row,
        metadata={"dataset_path": "dataset/x", "domain": "weather", "timestamp_utc": "2026-02-13T00:00:00Z"},
    )

    assert payload["knee"]["decision"] == "rejected"
    assert payload["knee"]["reasons"] == ["bic_weak", "knee_at_edge"]
    assert payload["parity"]["decision"] == "null"
    assert payload["stability"]["band_class"] == "incoherent"
