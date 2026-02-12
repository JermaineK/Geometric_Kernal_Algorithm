# Invariants

GKA computes a consistent set of diagnostics across domains:

- `eta`: mirror-odd contrast from paired handedness records.
- `L_k`: knee size where parity signal changes regime.
- `forbidden_band`: `[L_k / rho, rho * L_k]` exclusion range used by scaling fits.
- `gamma`: slope in the post-knee scaling law.
- `Delta_hat = 2 - gamma`.
- `lambda_eff = b ** gamma` and inequality pass/fail checks.
- impedance ratio `omega_k * L / (2*pi*c_m)` (or velocity proxy).
- coherence metrics `A`, `F`, and `P_lock`.

Invariants are produced in `results.parquet` and summarized in report output.
