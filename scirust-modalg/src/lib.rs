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
//!   degree**, plus the fast **Walsh–Hadamard transform** and its spectral
//!   metrics (nonlinearity, balancedness, the bent property, correlation
//!   immunity) for Boolean functions on up to a few dozen input bits.
//! - [`numtheory`] — deterministic integer number theory: extended GCD, modular
//!   inverse and exponentiation, the CRT, a *deterministic* Miller–Rabin
//!   primality test exact for every `u64`, deterministic Pollard–Brent
//!   factorization, Euler's totient, divisors, and the Jacobi symbol.
//! - [`gf2`] — carryless `GF(2)[x]` multiplication and finite fields `GF(2^n)`
//!   (add/multiply/power/invert) with the Rijndael `GF(2^8)` and a primitive
//!   `GF(2^16)` built in — the exact kernel behind CRCs, LFSRs, Reed–Solomon
//!   and AES-style diffusion.
//! - [`codes`] — systematic **Reed–Solomon** codes over `GF(2^n)`: a syndrome /
//!   Berlekamp–Massey / Chien-search decoder that corrects up to `⌊nsym/2⌋`
//!   symbol errors, plus **erasure** and combined **errors-and-erasures**
//!   decoding (up to `nsym` known-position losses — the RAID-6 path), composing
//!   the `gf2` field with `numtheory` (to verify the primitive element).
//! - [`crc`] — parameterised **cyclic redundancy checks** (the Rocksoft model)
//!   with a streaming digest and named presets (CRC-32, CRC-32C, CRC-16
//!   variants, CRC-8, CRC-64/XZ) that reproduce the published check values.
//! - [`ntt`] — the exact **number-theoretic transform** over `Z/p` (an integer
//!   FFT) and the `O(n log n)` exact integer **convolution** / polynomial
//!   multiplication it enables, composing `numtheory` to validate the prime and
//!   its primitive root.
//! - [`sbox`] — exact **S-box analysis**: difference distribution table and
//!   differential uniformity, linear approximation table and nonlinearity (via
//!   the Walsh transform), algebraic degree, and the strict-avalanche matrix —
//!   composing `boolean` for cryptographic S-box design and audit.
//! - [`negacyclic`] — exact **negacyclic convolution** (multiplication in
//!   `Z_q[x]/(x^n + 1)`, the core ring operation of lattice cryptography) built
//!   on `ntt`, plus **Montgomery** reduction — reference building blocks, not a
//!   hardened cryptosystem.
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
//!
//! ```
//! use scirust_modalg::codes::ReedSolomon;
//!
//! // RS(255, 251): 4 parity bytes correct up to 2 corrupted symbols.
//! let rs = ReedSolomon::qr(4);
//! let msg: Vec<u8> = (1..=20).collect();
//! let mut received = rs.encode_bytes(&msg);
//! received[3] ^= 0x5a; // corrupt two symbols in transit
//! received[11] ^= 0xff;
//! let (recovered, corrected) = rs.decode_bytes(&received).unwrap();
//! assert_eq!(recovered, msg);
//! assert_eq!(corrected, 2);
//!
//! // Erasure decoding (RAID-6 style): nsym parity symbols recover nsym
//! // known-position losses — twice the error-correction capacity.
//! let codeword: Vec<u64> = rs.encode_bytes(&msg).iter().map(|&b| b as u64).collect();
//! let mut lossy = codeword.clone();
//! lossy[2] = 0; lossy[5] = 0; lossy[9] = 0; lossy[14] = 0; // 4 lost symbols
//! let filled = rs.decode_erasures(&lossy, &[2, 5, 9, 14]).unwrap();
//! assert_eq!(filled, codeword);
//! ```
//!
//! ```
//! use scirust_modalg::crc::Crc;
//!
//! // The canonical CRC-32 (zlib) check value of "123456789".
//! assert_eq!(Crc::crc32_iso_hdlc().checksum(b"123456789"), 0xCBF4_3926);
//! ```
//!
//! ```
//! use scirust_modalg::boolean::{is_bent, nonlinearity};
//!
//! // The 2-bit AND function is bent: maximal nonlinearity 2^1 − 2^0 = 1.
//! let and = [0u8, 0, 0, 1]; // truth table of x0 ∧ x1
//! assert!(is_bent(&and, 2));
//! assert_eq!(nonlinearity(&and, 2), 1);
//! ```
//!
//! ```
//! use scirust_modalg::ntt::Ntt;
//!
//! // Exact O(n log n) polynomial multiplication: (1 + 2x + 3x²)(1 + x) over Z.
//! let ntt = Ntt::new_default();
//! assert_eq!(ntt.convolve(&[1, 2, 3], &[1, 1]), vec![1, 3, 5, 3]);
//! ```
//!
//! ```
//! use scirust_modalg::sbox::Sbox;
//!
//! // An S-box's differential uniformity — a small linear box has the worst
//! // possible value (every difference propagates deterministically).
//! let identity = Sbox::from_fn(4, 4, |x| x);
//! assert_eq!(identity.differential_uniformity(), 16); // 2^4
//! assert_eq!(identity.nonlinearity(), 0);             // affine
//! ```
//!
//! ```
//! use scirust_modalg::negacyclic::NegacyclicNtt;
//!
//! // Multiply in Z_q[x]/(x^n + 1): x^{n-1}·x = x^n ≡ −1 = q−1.
//! let ring = NegacyclicNtt::falcon(8);
//! let q = ring.modulus();
//! let mut hi = vec![0; 8]; hi[7] = 1; // x^7
//! let mut x = vec![0; 8]; x[1] = 1;   // x
//! let mut minus_one = vec![0; 8]; minus_one[0] = q - 1;
//! assert_eq!(ring.mul(&hi, &x), minus_one);
//! ```

pub mod boolean;
pub mod codes;
pub mod crc;
pub mod gf2;
pub mod hypercomplex;
pub mod linalg;
pub mod negacyclic;
pub mod ntt;
pub mod numtheory;
pub mod ring;
pub mod sbox;

pub use codes::ReedSolomon;
pub use crc::Crc;
pub use gf2::Gf2Field;
pub use hypercomplex::{Oct, Quat};
pub use linalg::ModMatrix;
pub use negacyclic::{Montgomery, NegacyclicNtt};
pub use ntt::Ntt;
pub use ring::Word;
pub use sbox::Sbox;
