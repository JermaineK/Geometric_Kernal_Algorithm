# Null Tests

`gka run --null N` executes null replications and writes outputs under `results/nulls/`.
Implemented null models:

- `time_shuffle`: random permutation within each segment/group.
- `mirror_swap`: random swaps of handedness labels.
- `phase_scramble`: preserve spectrum magnitude and randomize Fourier phases.
- `sign_flip`: random +/-1 multipliers where sign symmetry is valid.
- `block_bootstrap`: block resampling preserving local autocorrelation.

Each replication reruns parity, knee, and scaling summaries so users can compare
observed invariants against null distributions.
