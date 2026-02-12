# Theory Mapping

The pipeline is organized into invariant operators:

1. Geometry: local curve descriptors and covariance summaries.
2. Spectral separation: candidate characteristic frequencies (`Omega_candidates`).
3. Parity split: decompose `X` into mirror-even and mirror-odd components.
4. Knee detection: identify transition scale `L_k` from `eta(L)`.
5. Scaling fit: estimate `gamma` with exclusion of the forbidden middle band.
6. Stability class: evaluate coarse-graining consistency.
7. Impedance alignment: compare knee frequency/size against propagation velocity proxy.
8. Coherence: assess parity locking in the post-knee regime.

Domain adapters only provide mapping of raw observables to canonical terms. The core
algorithm and invariants stay unchanged across domains.
