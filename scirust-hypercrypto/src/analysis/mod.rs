//! Adversarial structural-analysis experiments (spec §17, Phase-1 Experiments
//! 1–7). Every experiment is a concrete attempt to *break* v0.1 early. Passing
//! (i.e. "no break found by this experiment") is never evidence of security.

pub mod battery;
pub mod degree;
pub mod invariants;
pub mod linearity;
pub mod matrix_lifting;
pub mod modmatrix;
pub mod report;
pub mod subspace;
pub mod util;
pub mod zero_divisors;
