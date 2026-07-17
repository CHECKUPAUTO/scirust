#![forbid(unsafe_code)]
// Fixed-width integer-algebra code: two Clippy style lints are relaxed
// crate-wide because the "fix" obscures the math (the same posture SciRust's
// `clippy.toml` documents for numeric code): `needless_range_loop` (basis and
// coefficient indices are read against fixed conventions, often indexing several
// arrays or a routing table at once) and `manual_is_multiple_of` (`x & 1 == 0` /
// `% 2 == 0` parity tests read as the norm/valuation math they are).
#![allow(clippy::needless_range_loop)]
#![allow(clippy::manual_is_multiple_of)]
//! # `scirust-modalg` — exact deterministic modular integer algebra
//!
//! A small, dependency-free, `#![forbid(unsafe_code)]` toolbox of **exact**
//! integer algebra over the rings `Z/2^k`, built for SciRust's bit-exact,
//! platform-independent, no-floating-point discipline. It contains capabilities
//! that are individually useful and, together, unusual to find packaged:
//!
//! - [`ring`] — the finite rings `Z/2^k` as sealed [`ring::Word`] types
//!   (`W2, W4, W8, W16, W64`). Only explicit wrapping arithmetic is exposed, so
//!   accidental overflowing `+`/`*` does not compile. Includes 2-adic valuation,
//!   unit test, and modular inverse of odd elements (Newton iteration).
//! - [`linalg`] — dense matrices over any `Word`, with the rare exact operations
//!   over `Z/2^k`: determinant **mod `2^k`**, rank over **`GF(2)`** (kept strictly
//!   distinct from ring rank), the **2-adic Smith normal form** (elementary-divisor
//!   valuations) and hence exact **kernel / image sizes**, and matrix inverse when
//!   the determinant is a unit.
//! - [`hypercomplex`] — exact integer **octonions** and **quaternions** over any
//!   `Word`, with an authoritative 64-term multiplication oracle cross-checked
//!   against an independent Fano-triple generator, conjugation, the modular norm,
//!   and (octonion) little-endian serialization.
//! - [`boolean`] — the fast Möbius transform and exact **algebraic-normal-form
//!   degree** of a Boolean vector function on up to a few dozen input bits.
//! - [`numtheory`] — deterministic integer number theory: extended GCD, modular
//!   inverse and exponentiation, the CRT, a *deterministic* Miller–Rabin
//!   primality test exact for every `u64`, deterministic Pollard–Brent
//!   factorization, Euler's totient, divisors, and the Jacobi symbol.
//! - [`gf2`] — carryless `GF(2)[x]` multiplication and finite fields `GF(2^n)`
//!   (add/multiply/power/invert) with the Rijndael `GF(2^8)` and a primitive
//!   `GF(2^16)` built in — the exact kernel behind CRCs, LFSRs, Reed–Solomon
//!   and AES-style diffusion.
//!
//! Everything is deterministic and reproducible bit-for-bit on every platform.
//!
//! ## Examples
//!
//! ```
//! use scirust_modalg::ring::{Word, W8};
//! use scirust_modalg::linalg::ModMatrix;
//!
//! // A matrix over Z/2^8 is invertible iff its determinant is odd.
//! let mut m = ModMatrix::<W8>::identity(3);
//! m.set(0, 1, W8::from_u64(2));
//! assert!(m.is_unit());                 // det = 1 (odd)
//! assert_eq!(m.kernel_log2(), 0);       // trivial kernel
//! let inv = m.inverse().unwrap();
//! assert!(m.matmul(&inv).is_identity());
//! ```
//!
//! ```
//! use scirust_modalg::numtheory::{is_prime, factor, crt};
//!
//! // Deterministic primality and factorization over the machine integers.
//! assert!(is_prime((1u64 << 61) - 1));            // Mersenne prime M61
//! assert_eq!(factor(1_000_000u64), vec![(2, 6), (5, 6)]);
//! // Chinese Remainder Theorem: the unique residue mod 105.
//! assert_eq!(crt(&[(2, 3), (3, 5), (2, 7)]), Some((23, 105)));
//! ```
//!
//! ```
//! use scirust_modalg::gf2::Gf2Field;
//!
//! // The AES/Rijndael field GF(2^8): FIPS-197's worked example {57}·{83}={c1}.
//! let f = Gf2Field::rijndael8();
//! assert_eq!(f.mul(0x57, 0x83), 0xc1);
//! assert_eq!(f.mul(0x53, f.inv(0x53).unwrap()), 1);
//! ```

pub mod boolean;
pub mod gf2;
pub mod hypercomplex;
pub mod linalg;
pub mod numtheory;
pub mod ring;

pub use gf2::Gf2Field;
pub use hypercomplex::{Oct, Quat};
pub use linalg::ModMatrix;
pub use ring::Word;
