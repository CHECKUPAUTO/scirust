//! Exact integer algebra for the falsification harness (spec §7–§8, §16.1):
//! the `Z/2^k` coefficient rings, the octonion type with its authoritative
//! 64-term multiplication oracle, and an associative quaternion for Control D.

pub mod octonion;
pub mod quaternion;
pub mod table;
pub mod word;

pub use octonion::{LAMBDA, Oct, PI};
pub use quaternion::Quat;
pub use word::{W2, W4, W8, W16, W64, WidthTag, Word};
