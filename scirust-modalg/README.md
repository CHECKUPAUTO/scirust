# scirust-modalg

**Exact, deterministic, dependency-free algebra for SciRust.**

`scirust-modalg` is a toolbox of *exact* integer/modular/finite-field algebra and
the higher-level primitives built on it. Everything here shares one discipline:

- **exact** — no floating point, no rounding; results are mathematically exact;
- **deterministic** — no OS entropy, no wall-clock, no threads in the library;
  identical inputs give identical outputs, **bit-for-bit on every platform**;
- **dependency-free** — no external crates;
- **`#![forbid(unsafe_code)]`** — no `unsafe`, anywhere;
- **MSRV 1.89**, `rustfmt` (style edition 2024), `clippy -D warnings` clean.

The differentiator is that discipline, not raw speed. These are correct
**reference** implementations for contexts where bit-exact reproducibility,
auditability and zero-dependency/zero-unsafe matter more than throughput
(safety-critical systems, deterministic replay / consensus, reproducible builds,
formal verification, teaching). Where you need maximum speed, reach for an
optimized SIMD/hardware library instead — see [Positioning](#positioning).

> **EXPERIMENTAL research code.** Nothing here is a security, post-quantum, or
> production claim. The cryptography-adjacent modules are *analysis tools and
> exact references*, not hardened or constant-time implementations.

## Modules at a glance

### Foundation

| Module | What it provides |
|--------|------------------|
| [`ring`] | The finite rings `Z/2^k` as sealed `Word` types (`W2 … W64`), explicit wrapping arithmetic only, 2-adic valuation, unit test, modular inverse of odd elements. |
| [`numtheory`] | Extended GCD, modular inverse/exponentiation, CRT, integer sqrt, **deterministic** Miller–Rabin primality (exact for every `u64`), **deterministic** Pollard–Brent factorization, Euler's totient, divisors, Jacobi symbol. |
| [`dlog`] | Exact **discrete logarithms** in `(ℤ/pℤ)*`: baby-step giant-step and **Pohlig–Hellman** (fast for smooth group orders), plus the multiplicative order of an element. |
| [`gf2`] | Carryless `GF(2)[x]` multiply / divide / gcd, and finite fields `GF(2^n)` (add/mul/pow/inv) with the AES/Rijndael `GF(2^8)`, a primitive `GF(2^8)` and a `GF(2^16)`. |
| [`poly`] | The univariate polynomial ring **`GF(p)[x]`** over any prime field: long division, monic (extended) GCD, modular exponentiation, Lagrange interpolation, the formal derivative, an exact **Rabin irreducibility test**, and full **factorization into irreducibles** (deterministic Cantor–Zassenhaus) — the field-generic companion to `gf2` (the `p = 2` case). |
| [`extfield`] | Finite **extension fields `GF(p^k)`** as `GF(p)[x]/(m)`: add/sub/mul/pow/inverse and the Frobenius map, with an automatic irreducible-modulus search. Generalises `gf2` — `GF(2^8)` with the AES modulus reproduces `gf2::Gf2Field::rijndael8` exactly. |
| [`boolean`] | Fast Möbius transform + exact **algebraic-normal-form degree**, and the fast **Walsh–Hadamard transform** with nonlinearity / balancedness / bent / correlation-immunity. |
| [`linalg`] | Dense `ModMatrix` over `Z/2^k`: determinant mod `2^k`, `GF(2)` rank (kept distinct from ring rank), **2-adic Smith normal form** → exact kernel/image sizes, inverse, solve, matrix power, 2-adic pivot rank. |
| [`hypercomplex`] | Exact integer **octonions** and **quaternions** over any `Word`, with an authoritative multiplication oracle cross-checked against a Fano-triple generator. |
| [`bigint`] | Arbitrary-precision signed integers: decimal I/O, comparison, `+ − ×`, truncated `divmod`, `pow`, `gcd`, and an **NTT-accelerated `mul_ntt`** (multi-prime + CRT). |
| [`bigrational`] | Exact rationals over `BigInt` and **overflow-free** exact linear algebra (`solve`, `determinant`). |
| [`smith`] | The exact **Smith normal form** of an integer matrix over `BigInt` (**no overflow ceiling**): invariant factors `d₁ \| d₂ \| …` plus unimodular certificates `U`, `V` with `U · A · V = D` — the overflow-free companion to the Hermite normal form. |

### The four outlets (*débouchés*)

Each foundation brick composes into a higher-level capability targeting a distinct use case:

| Outlet | Modules | What you get |
|--------|---------|--------------|
| **Storage / erasure coding** (RAID-6, edge) | [`codes`], [`crc`] | Systematic **Reed–Solomon** with error **and erasure** decoding (up to `nsym` known-position losses), and parameterised **CRCs** (Rocksoft model, named presets reproducing the published check values). |
| **Symmetric-crypto audit** (S-box design) | [`sbox`], `boolean`, `gf2` | Exact **DDT / LAT**, differential uniformity, nonlinearity, algebraic degree, strict-avalanche matrix — the AES S-box reproduces its textbook metrics (DU 4, NL 112, degree 7). |
| **Lattice / post-quantum reference** | [`ntt`], [`negacyclic`], [`lll`] | Exact **number-theoretic transform** (integer FFT + convolution), **negacyclic** multiplication in `Z_q[x]/(x^n+1)` (the ring op of Kyber/Dilithium/Falcon) + Montgomery reduction, and exact **LLL** lattice reduction with a unimodular certificate. |
| **Exact / formal linear algebra** | [`rational`], `bigrational`, `bigint`, `linalg`, `numtheory` | Rounding-free linear algebra over `ℚ` (solve/det/inverse/rank), the integer **Hermite normal form** with a unimodular certificate, and the same over `BigInt`/`BigRational` with **no overflow ceiling** (e.g. an exact Hilbert-matrix solve). |

## How the pieces compose

```
        ring          numtheory ─────────────┐
          │               │  │               │
        linalg           gf2 │             bigint ── mul_ntt ─┐
                          │  │               │                │
   boolean ── sbox ◄──────┘  │           bigrational          │
                    ▲        ntt ──► negacyclic (PQC)  │       │
   codes (RS+erasure) ◄─ gf2 │        │                ▼       │
        (storage)      + numtheory    └──► bigint::mul_ntt ◄───┘
                                              (via ntt + CRT)
   rational ── HNF (formal) ;  lll ◄── bigint + bigrational
```

Concretely: `codes` builds on `gf2` + `numtheory`; `ntt` composes `numtheory`;
`poly` (the field-generic `GF(p)[x]`) composes `numtheory`; `negacyclic` and
`bigint::mul_ntt` build on `ntt`; `sbox` builds on `boolean`; `bigrational` and
`lll` build on `bigint`.

## Testing discipline

Every capability is **cross-checked against an independent method or a canonical
vector**, not just self-consistency:

- determinant vs the Leibniz formula; kernel size vs brute force;
- primality vs a sieve + Carmichael numbers; factorization vs product reconstruction;
- CRCs vs the published `"123456789"` catalogue check values;
- the AES S-box vs its textbook DU/nonlinearity/degree; LAT vs a brute-force LAT;
- negacyclic multiply vs an `O(n²)` reference; `mul_ntt` vs schoolbook `mul`;
- `GF(p)[x]` division reconstructed (`a = q·b + r`), Bézout (`u·a + v·b = g`), the product rule (`(fg)' = f'g + fg'`), Rabin irreducibility vs an exhaustive root search (plus the AES modulus `x⁸+x⁴+x³+x+1`), and factorization reconstructed (`∏ qᵢ^eᵢ = f`, each `qᵢ` irreducible; `x^p − x` splits into all `p` linear factors);
- `GF(p^k)` field axioms on random elements, `a^(p^k−1) = 1` and `a^(p^k) = a` (Lagrange + Frobenius), and `GF(2^8)` products/inverses matching the packed `gf2::Gf2Field::rijndael8`;
- LLL output verified LLL-reduced with a unimodular certificate (`U·A = reduced`, `det U = ±1`) and preserved lattice volume;
- `BigInt` arithmetic vs `i128` over thousands of random cases;
- Smith normal form reconstructed (`U·A·V = D`), `U`, `V` unimodular (`det = ±1`), the divisibility chain `dᵢ | dᵢ₊₁`, and `∏ dᵢ = |det A|` for square matrices;
- discrete logs vs brute force and BSGS vs Pohlig–Hellman agreement (`gˣ = h`), including recovery of a known exponent modulo the Fermat prime `65537` (a fully smooth group order).

## Usage

```rust
use scirust_modalg::codes::ReedSolomon;
use scirust_modalg::sbox::Sbox;
use scirust_modalg::bigint::BigInt;

// Reed–Solomon: 4 parity symbols recover 4 known-position erasures (RAID-6).
let rs = ReedSolomon::qr(4);
let msg: Vec<u8> = (1..=20).collect();
let codeword: Vec<u64> = rs.encode_bytes(&msg).iter().map(|&b| b as u64).collect();
let mut lossy = codeword.clone();
for &p in &[2, 5, 9, 14] { lossy[p] = 0; }
assert_eq!(rs.decode_erasures(&lossy, &[2, 5, 9, 14]).unwrap(), codeword);

// S-box metrics (a tiny illustrative box; feed a real 2^n-entry lookup table).
let sbox = Sbox::from_fn(4, 4, |x| (x.wrapping_mul(3) ^ (x >> 1)) & 0xF);
let _du = sbox.differential_uniformity();

// Arbitrary precision: 2^128 exactly, far beyond i128.
assert_eq!(
    BigInt::from_i128(2).pow(128).to_decimal(),
    "340282366920938463463374607431768211456"
);
```

## Positioning

These are single-threaded scalar reference implementations. They are typically
**1–2 orders of magnitude** slower than optimized libraries (GMP for bignum,
hardware CRC32 / PCLMULQDQ, SIMD Reed–Solomon / NTT). The `modalg-bench`
binary measures the crate's own operations and prints an honest comparison:

```
cargo run --release -p scirust-modalg --bin modalg-bench
```

For example, `BigInt::mul_ntt` only beats the schoolbook `mul` past ~2k–4k
limbs (~10⁵ bits), because of its constant factor (three NTT primes, `u128`
modular arithmetic) — the benchmark reports the real crossover rather than
pretending the NTT path is always faster.

Choose `scirust-modalg` when bit-exact reproducibility and zero unsafe/deps
matter more than speed; choose an optimized library otherwise.

[`ring`]: src/ring.rs
[`numtheory`]: src/numtheory.rs
[`dlog`]: src/dlog.rs
[`gf2`]: src/gf2.rs
[`boolean`]: src/boolean.rs
[`linalg`]: src/linalg.rs
[`hypercomplex`]: src/hypercomplex/mod.rs
[`bigint`]: src/bigint.rs
[`bigrational`]: src/bigrational.rs
[`smith`]: src/smith.rs
[`codes`]: src/codes.rs
[`crc`]: src/crc.rs
[`sbox`]: src/sbox.rs
[`ntt`]: src/ntt.rs
[`poly`]: src/poly.rs
[`extfield`]: src/extfield.rs
[`negacyclic`]: src/negacyclic.rs
[`lll`]: src/lll.rs
[`rational`]: src/rational.rs
