#![forbid(unsafe_code)]
// This is fixed-width integer-algebra code. Two Clippy style lints are relaxed
// crate-wide because the "fix" would obscure the math it is meant to make
// auditable (the same posture SciRust's `clippy.toml` documents for numeric
// code): `needless_range_loop` (basis/coefficient indices `0..8` are read
// against the specification's index conventions, often indexing several arrays
// or a routing table at once) and `manual_is_multiple_of` (`x & 1 == 0` /
// `% 2 == 0` parity tests read as the norm/valuation math they are). Algebra
// types intentionally expose `add`/`sub`/`mul`/`neg` inherent methods rather
// than operator traits, so `should_implement_trait` is allowed where they live.
#![allow(clippy::needless_range_loop)]
#![allow(clippy::manual_is_multiple_of)]
//! # `scirust-hypercrypto` — EXPERIMENTAL structural falsification harness
//!
//! ```text
//! EXPERIMENTAL RESEARCH CONSTRUCTION
//!
//! This crate implements Phase 1 of the SciRust-HyperCrypto falsification
//! program. Its FIRST-ORDER GOAL IS TO BREAK the v0.1 construction as early as
//! possible, not to make it look promising.
//!
//! This is NOT a cipher, hash, KEM, signature, or post-quantum scheme.
//! It has NOT received independent cryptanalysis.
//! It MUST NOT be used to protect real data, credentials, financial records,
//! health information, production secrets, or communication systems.
//!
//! Use established, standardized cryptographic primitives for production.
//! Nothing in this crate is secure, hardened, novel, or suitable for real data.
//! ```
//!
//! The authoritative definition of every constant, table, and transformation
//! implemented here is the merged specification
//! `docs/research/SCIRUST_HYPERCRYPTO_SPEC_V0_1.md`. Where this code and the
//! specification disagree, the specification wins and the disagreement is a bug.
//!
//! ## Scope (Phase 1)
//!
//! - exact scalar octonion arithmetic over `Z/2^k` (k in {2,4,8,16,64});
//! - the exact v0.1 round function `F-PROG` and reduced balanced Feistel shell;
//! - deliberately weakened *control* variants used to validate the analysis;
//! - matrix-lifting, linearity/affinity, algebraic-degree, norm/conjugation
//!   invariant, zero-divisor, and subspace experiments;
//! - a deterministic machine-readable research report.
//!
//! Everything is pure Rust, integer-only, `#![forbid(unsafe_code)]`, scalar
//! (no SIMD), deterministic, and platform-independent.

pub mod algebra;
pub mod analysis;
pub mod fixtures;
pub mod permutation;

/// Specification version this harness targets. Bump only with the spec.
pub const SPEC_VERSION: &str = "SCIRUST-HYPERCRYPTO-V0.1";

/// One-line experimental banner reused by binaries and reports.
pub const EXPERIMENTAL_BANNER: &str =
    "EXPERIMENTAL research-only falsification harness — NOT secure, NOT for production data.";
