//! Shared support for every `scirust-solvers`-backed [`Simulate`] adapter
//! (the [`crate::ode`] and [`crate::quadrature`] modules): encoding `f64`
//! configuration fields for the kernel's float-free canonical encoding, and
//! mapping `scirust-solvers`' error type onto [`SimError`]'s two-variant
//! contract.
//!
//! [`Simulate`]: sos_simulation::Simulate

use scirust_solvers::SolverError;
use sos_core::canonical::{Canonical, CanonicalEncoder};
use sos_simulation::SimError;

/// Encode an `f64` configuration field exactly, as its shortest round-trip
/// decimal string, into the kernel's float-free canonical encoding
/// (`sos_core::canonical` module docs).
///
/// An earlier version of this helper quantized to a fixed-point `i64` at a
/// declared nanoscale precision instead — the docs' own suggested pattern for
/// an `L2` *output*, whose accuracy is inherently bounded by a tolerance
/// anyway. But these are exact, caller-specified *inputs* (`rtol`, `atol`,
/// quadrature `tol`, ...), not approximate measurements, and configs commonly
/// carry values well below nanoscale (`1e-10`, `1e-12`): quantizing them at a
/// fixed nanoscale collapsed distinct tolerances to the same encoded value —
/// a real cache-key collision this crate's own test suite caught. A shortest
/// round-trip string is exact at any magnitude (no scale to choose, so
/// nothing to collide), while still never hashing a raw bit pattern.
/// `-0.0` is normalized to `0.0` first so the two, which compare `==`, also
/// encode identically.
pub(crate) fn encode_f64(enc: &mut CanonicalEncoder, v: f64) {
    let v = if v == 0.0 { 0.0 } else { v };
    enc.str(&v.to_string());
}

/// An `f64` slice, [`Canonical`] via [`encode_f64`] on each element — lets a
/// `Vec<f64>` configuration field (a state vector, `y0`) use the encoder's
/// own [`CanonicalEncoder::seq`] for length-prefixing rather than hand-rolling
/// it.
pub(crate) struct ExactF64Seq<'a>(pub(crate) &'a [f64]);

impl Canonical for ExactF64Seq<'_> {
    fn encode(&self, enc: &mut CanonicalEncoder) {
        enc.seq(&self.0.iter().copied().map(ExactF64).collect::<Vec<_>>());
    }
}

struct ExactF64(f64);

impl Canonical for ExactF64 {
    fn encode(&self, enc: &mut CanonicalEncoder) {
        encode_f64(enc, self.0);
    }
}

/// Map a `scirust-solvers` error to the two-variant `SimError` contract:
/// input rejected before compute began vs. a failure while running.
pub(crate) fn map_solver_error(e: SolverError) -> SimError {
    match e
    {
        SolverError::InvalidInput(msg) => SimError::InvalidConfig(msg),
        other => SimError::Backend(other.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use sos_core::canonical::Canonical;

    use super::*;

    struct F(f64);
    impl Canonical for F {
        fn encode(&self, enc: &mut CanonicalEncoder) {
            encode_f64(enc, self.0);
        }
    }

    #[test]
    fn negative_and_positive_zero_encode_identically() {
        assert_eq!(F(-0.0).canonical_bytes(), F(0.0).canonical_bytes());
    }

    #[test]
    fn tiny_distinct_tolerances_do_not_collide() {
        // The exact bug this helper replaced quantize() to fix: both used to
        // round to the same fixed-point value at a nanoscale (1e9) scale.
        assert_ne!(F(1e-10).canonical_bytes(), F(1e-12).canonical_bytes());
    }

    #[test]
    fn equal_values_encode_identically_and_distinct_values_differ() {
        assert_eq!(F(1.5).canonical_bytes(), F(1.5).canonical_bytes());
        assert_ne!(F(1.5).canonical_bytes(), F(1.500_000_1).canonical_bytes());
    }
}
