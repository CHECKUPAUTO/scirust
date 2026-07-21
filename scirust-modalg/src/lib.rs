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
//! - [`poly`] — the univariate polynomial ring **`GF(p)[x]`** over any prime
//!   field: long division, monic (extended) GCD, modular exponentiation,
//!   Lagrange interpolation, the formal derivative, an exact **Rabin
//!   irreducibility test**, and full **factorization into irreducibles**
//!   (deterministic Cantor–Zassenhaus) — the field-generic companion to `gf2`
//!   (the `p = 2` case), composing `numtheory`.
//! - [`extfield`] — finite **extension fields `GF(p^k)`** as `GF(p)[x]/(m)` for
//!   a monic irreducible `m`: add/sub/mul/pow/inverse and the Frobenius map,
//!   with an automatic modulus search. Generalises `gf2` — `GF(2^8)` with the
//!   AES modulus reproduces `gf2::Gf2Field::rijndael8` exactly. Composes `poly`.
//! - [`sbox`] — exact **S-box analysis**: difference distribution table and
//!   differential uniformity, linear approximation table and nonlinearity (via
//!   the Walsh transform), algebraic degree, and the strict-avalanche matrix —
//!   composing `boolean` for cryptographic S-box design and audit.
//! - [`negacyclic`] — exact **negacyclic convolution** (multiplication in
//!   `Z_q[x]/(x^n + 1)`, the core ring operation of lattice cryptography) built
//!   on `ntt`, plus **Montgomery** reduction — reference building blocks, not a
//!   hardened cryptosystem.
//! - [`rational`] — exact **rational** arithmetic (`Fraction`) and rounding-free
//!   linear algebra over `ℚ` (solve, determinant, inverse, rank), plus the
//!   integer **Hermite normal form** with a unimodular certificate — certified
//!   linear algebra for verification and computer-algebra settings.
//! - [`bigint`] — arbitrary-precision signed integers (`BigInt`): decimal I/O,
//!   comparison, `+ − ×`, truncated `divmod`, `pow`, and `gcd`, lifting the
//!   crate's exactness above the `i128` ceiling.
//! - [`bigrational`] — exact rationals over `BigInt` (`BigRational`) and
//!   **overflow-free** exact linear algebra (`solve`, `determinant`), the
//!   scalable counterpart of `rational` (e.g. an exact Hilbert-matrix solve).
//! - [`lll`] — exact **LLL lattice-basis reduction** with rational Gram–Schmidt
//!   (no floating point, no overflow) and a unimodular certificate `U` proving
//!   the reduced basis spans the same lattice.
//! - [`smith`] — the exact **Smith normal form** of an integer matrix over
//!   `BigInt` (**no overflow ceiling**): the invariant factors `d₁ | d₂ | …`
//!   plus unimodular certificates `U`, `V` with `U · A · V = D` — the
//!   overflow-free companion to the `rational` Hermite normal form.
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
//! use scirust_modalg::poly::Poly;
//!
//! // In GF(2)[x], the AES reduction polynomial x⁸+x⁴+x³+x+1 is irreducible —
//! // exactly why GF(2)[x] modulo it is the field GF(2^8).
//! let m = Poly::from_coeffs(2, &[1, 1, 0, 1, 1, 0, 0, 0, 1]);
//! assert!(m.is_irreducible());
//! // Long division is exact: (x²+1) = (x+1)·(x+1) over GF(2).
//! let x2p1 = Poly::from_coeffs(2, &[1, 0, 1]);
//! let xp1 = Poly::from_coeffs(2, &[1, 1]);
//! let (q, r) = x2p1.divmod(&xp1);
//! assert_eq!(q, xp1);
//! assert!(r.is_zero());
//! // Factorization: x²+1 = (x+1)² over GF(2) (one factor, multiplicity 2).
//! assert_eq!(x2p1.factor(), vec![(xp1, 2)]);
//! ```
//!
//! ```
//! use scirust_modalg::extfield::ExtField;
//!
//! // GF(2^8) via the AES modulus: FIPS-197's worked example {57}·{83} = {c1}.
//! let f = ExtField::new(scirust_modalg::poly::Poly::from_coeffs(
//!     2, &[1, 1, 0, 1, 1, 0, 0, 0, 1],
//! ));
//! let byte = |b: u8| f.element(&(0..8).map(|i| ((b >> i) & 1) as u64).collect::<Vec<_>>());
//! let prod = f.mul(&byte(0x57), &byte(0x83));
//! let as_byte = (0..8).fold(0u8, |acc, i| acc | ((prod.coeff(i) as u8) << i));
//! assert_eq!(as_byte, 0xc1);
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
//!
//! ```
//! use scirust_modalg::rational::{Fraction, RatMatrix};
//!
//! // Exact solve of 2x + y = 3, x + 3y = 5 → x = 4/5, y = 7/5 (no rounding).
//! let a = RatMatrix::from_int_rows(&[vec![2, 1], vec![1, 3]]);
//! let b = [Fraction::from_int(3), Fraction::from_int(5)];
//! let x = a.solve(&b).unwrap();
//! assert_eq!(x, vec![Fraction::new(4, 5), Fraction::new(7, 5)]);
//! assert_eq!(a.matvec(&x), b.to_vec()); // A·x == b exactly
//! ```
//!
//! ```
//! use scirust_modalg::bigint::BigInt;
//!
//! // Arbitrary precision: 2^128 is exact, far beyond i128.
//! assert_eq!(
//!     BigInt::from_i128(2).pow(128).to_decimal(),
//!     "340282366920938463463374607431768211456"
//! );
//! ```
//!
//! ```
//! use scirust_modalg::bigrational::BigRational;
//!
//! // Exact rationals with no overflow ceiling: 1/2 + 1/3 = 5/6.
//! let sum = BigRational::from_i128(1).div(&BigRational::from_i128(2))
//!     .add(&BigRational::from_i128(1).div(&BigRational::from_i128(3)));
//! assert_eq!(sum.to_string_frac(), "5/6");
//! ```
//!
//! ```
//! use scirust_modalg::lll;
//!
//! // The basis (100,1),(99,1) generates Z²; exact LLL finds a unit-vector basis.
//! let res = lll::reduce_i128(&[vec![100, 1], vec![99, 1]]);
//! for row in &res.basis {
//!     let n2 = row[0].mul(&row[0]).add(&row[1].mul(&row[1]));
//!     assert_eq!(n2.to_decimal(), "1"); // ‖·‖² == 1
//! }
//! ```
//!
//! ```
//! use scirust_modalg::bigint::BigInt;
//! use scirust_modalg::smith::smith_normal_form;
//!
//! // Smith normal form of diag(2,3): coprime, so the CRT merges them into
//! // diag(1,6) — the invariant factors, with 1 | 6.
//! let big = |v: i128| BigInt::from_i128(v);
//! let a = vec![vec![big(2), big(0)], vec![big(0), big(3)]];
//! let snf = smith_normal_form(&a);
//! assert_eq!(snf.invariants, vec![big(1), big(6)]);
//! ```

pub mod bigint;
pub mod bigrational;
pub mod boolean;
pub mod codes;
pub mod crc;
pub mod extfield;
pub mod gf2;
pub mod hypercomplex;
pub mod linalg;
pub mod lll;
pub mod negacyclic;
pub mod ntt;
pub mod numtheory;
pub mod poly;
pub mod rational;
pub mod ring;
pub mod sbox;
pub mod smith;

pub use bigint::BigInt;
pub use bigrational::BigRational;
pub use codes::ReedSolomon;
pub use crc::Crc;
pub use extfield::ExtField;
pub use gf2::Gf2Field;
pub use hypercomplex::{Oct, Quat};
pub use linalg::ModMatrix;
pub use lll::LllResult;
pub use negacyclic::{Montgomery, NegacyclicNtt};
pub use ntt::Ntt;
pub use poly::Poly;
pub use rational::{Fraction, RatMatrix};
pub use ring::Word;
pub use sbox::Sbox;
pub use smith::{SmithNormalForm, smith_normal_form};
