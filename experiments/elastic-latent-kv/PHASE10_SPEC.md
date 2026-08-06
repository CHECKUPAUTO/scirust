# Elastic Latent KV — Phase 10 Online Bases

Phase 10 adds deterministic online basis learning without changing the validated
Phase 8 attention path. `DeterministicBasisLearner` performs scalar Oja-style
updates, scheduled Gram-Schmidt re-orthogonalization, fixed-capacity version
metadata, stable basis fingerprints, and quality-gated epoch commits.

Committed versions apply only to future cache epochs. Existing resident tokens
must remain associated with the basis version used when their coefficients were
encoded; migration is deferred to the lifecycle machinery in Phase 11.

The learner allocates basis, coefficient, residual and version storage only at
construction. `observe` reuses those buffers and follows a fixed scalar update
order for reproducibility.
