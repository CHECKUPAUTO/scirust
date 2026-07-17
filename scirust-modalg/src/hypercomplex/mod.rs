//! Exact integer hypercomplex numbers over a ring `Z/2^k`: octonions (with the
//! authoritative 64-term multiplication oracle and its independent Fano-triple
//! cross-check) and associative quaternions.

pub mod octonion;
pub mod quaternion;
pub mod table;

pub use octonion::Oct;
pub use quaternion::Quat;
