# SCIRUST_HYPERCRYPTO_SPEC_V0_1

**A deterministic, pure-Rust, hypercomplex diagrammatic keyed-permutation candidate — research specification.**

- Specification version: `0.1` (draft)
- Domain-separation namespace: `SCIRUST-HYPERCRYPTO-V0.1`
- Status: **experimental research construction — NOT a cipher, hash, KEM, signature, or PQC scheme**
- Scope: mathematical + cryptographic specification only. No implementation is authorised by this document.

---

> ## ⚠ EXPERIMENTAL RESEARCH CONSTRUCTION
>
> ```text
> EXPERIMENTAL RESEARCH CONSTRUCTION
>
> This design has not received independent cryptanalysis.
> It must not be used to protect real data, credentials, financial records,
> health information, production secrets, or communication systems.
>
> Use established and standardized cryptographic primitives for production.
> ```
>
> This specification defines a **falsification target**, not a security claim. Its
> purpose is to be precise enough that two independent implementations produce
> identical bytes, and structured enough that the design can be *attacked* early
> and cheaply. Nothing in this document should be read as evidence that the
> construction is secure, one-way, collision-resistant, or post-quantum.

---

## Table of contents

1. [Executive summary](#1-executive-summary)
2. [Research motivation](#2-research-motivation)
3. [Terminology](#3-terminology)
4. [Explicit non-claims](#4-explicit-non-claims)
5. [Repository findings](#5-repository-findings)
6. [Threat model](#6-threat-model)
7. [Algebraic-domain comparison](#7-algebraic-domain-comparison)
8. [Exact octonion convention](#8-exact-octonion-convention)
9. [Graph model](#9-graph-model)
10. [Key and graph derivation](#10-key-and-graph-derivation)
11. [Candidate reversible construction](#11-candidate-reversible-construction)
12. [Mathematical forward definition](#12-mathematical-forward-definition)
13. [Mathematical inverse definition](#13-mathematical-inverse-definition)
14. [Canonical serialization](#14-canonical-serialization)
15. [Test-vector format](#15-test-vector-format)
16. [Reduced models](#16-reduced-models)
17. [Cryptanalysis plan](#17-cryptanalysis-plan)
18. [Constant-time implementation requirements](#18-constant-time-implementation-requirements)
19. [Proposed crate architecture](#19-proposed-crate-architecture)
20. [Open research questions](#20-open-research-questions)
21. [Go / No-go criteria for implementation](#21-go--no-go-criteria-for-implementation)
22. [Experimental-use warning](#22-experimental-use-warning)
23. [References](#23-references)

---

## 1. Executive summary

This document specifies **SciRust-HyperCrypto v0.1**, an experimental, deterministic,
pure-Rust *keyed permutation candidate*

```text
P_K : {0,1}^1024 -> {0,1}^1024
```

built from an **explicitly parenthesized, ordered, typed computation graph** ("diagrammatic
computation graph") over an **exact eight-component hypercomplex type** whose coefficients
live in the finite ring `Z / 2^64 Z`. The graph is the round function of a **balanced
two-branch Feistel network**; each Feistel branch is exactly one octonion (`8 x u64 = 512
bits`), so the full state is `2 x 512 = 1024 bits`.

The selected design points for v0.1 are:

| Decision | v0.1 choice | Why (conservative rationale) |
|---|---|---|
| State width | **1024 bits** = two octonion branches | Each Feistel branch is a *whole* octonion; largest generic-attack margin of the three candidates; cleanest round function. |
| Coefficient domain | **`R = Z/2^64`** (`u64::wrapping_*`) | Exact, native to Rust, trivially constant-time (no data-dependent reduction), ARX-compatible, maximally portable. Non-invertible multipliers are *acceptable inside a Feistel F*. |
| Reversible outer structure | **Balanced 2-branch Feistel** + input/output whitening (Even–Mansour-style) | Invertibility is *structural*, independent of whether the round function is a bijection. Cleanest possible inverse proof. |
| Round function nonlinearity | Octonion multiplication (`MUL`) over `Z/2^64`, sandwiched by wrapping key-add, per-lane bit rotation, coefficient permutation, and round-constant XOR | Mixes two incompatible algebraic structures (`GF(2)^n` XOR/rotation vs. `Z/2^64` add/multiply) so no single clean linear model describes it. |
| Topology secrecy | **Graph topology is PUBLIC in v0.1** (derived only from public parameters) | Non-negotiable constraints 10–11 (no secret-dependent control flow / addressing). A key-derived topology *is* secret-dependent control flow; it is deferred to a clearly-flagged non-constant-time experimental mode. |
| Derivation primitive | **HKDF-SHA-256 (RFC 5869)** + an HMAC-SHA-256 counter-mode XOF | Reuses the *already-in-tree, RFC-KAT-tested* pure SHA-256 / HMAC-SHA-256 code (§5); zero new crypto dependencies; standardized construction (not invented). |

The construction is offered as a **research platform for algebraic cryptanalysis of
non-commutative / non-associative round functions**, not as a usable primitive. Section 17
is a falsification plan whose objective is to *break* it.

**Headline caveat (see §7).** Real octonions form a normed division algebra only over
subfields of the reals. Over *any* finite ring or finite field — including both candidate
domains here — the octonion algebra is **split**: it has **zero divisors**, and **arbitrary
elements are not invertible**. The design therefore must never rely on octonion inversion for
reversibility; the Feistel structure (§11–§13) is what guarantees the permutation is a
bijection.

---

## 2. Research motivation

The open research question is narrow and falsifiable:

> *Can an ordered, non-commutative, non-associative, exactly-deterministic algebraic
> computation graph be assembled into a keyed permutation whose diffusion and algebraic
> structure resist the standard cryptanalytic toolkit — under Kerckhoffs' principle, with
> only the key secret?*

We are explicitly **not** claiming the answer is "yes". We are building the object precisely
enough that the answer can be pursued by measurement and attack.

Three properties motivate the exploration:

1. **Ordering is semantically significant** because octonion multiplication is
   non-commutative: `x ⊗ y ≠ y ⊗ x` in general. The left/right position of operands in a
   `MUL` instruction changes the result.
2. **Parenthesization is semantically significant** because octonion multiplication is
   non-associative: `(x ⊗ y) ⊗ z ≠ x ⊗ (y ⊗ z)` in general. The bracketing chosen in the
   computation graph changes the result.
3. **A computation graph is a natural, auditable representation** of an exactly-specified,
   deterministic evaluation program. SciRust already contains deterministic expression-tree
   and fused-operator evaluators (§5); the same discipline (integer-only, bit-exact,
   canonical encoding) applies here.

The analogy to Feynman diagrams is **representational only**: we borrow the idea of an
ordered diagram whose vertices are operations and whose edges carry values. This system is
not physics, computes no amplitudes, and must never be described as a physical
Feynman-diagram computation. Preferred terms: *diagrammatic computation graph*, *typed
algebraic graph*, *ordered evaluation tree*, *explicitly parenthesized computation graph*.

**Why this is worth specifying even if it is likely broken.** Negative results are cheap and
valuable here: a precise object lets us run bijection, avalanche, differential, linear,
degree, interpolation, Gröbner, and invariant-subspace probes on *reduced* variants (§16) and
document exactly which structural property collapses, rather than hand-waving about
"chaos".

---

## 3. Terminology

| Term | Meaning in this document |
|---|---|
| **Word** | A `u64` (64-bit unsigned integer). |
| **Coefficient domain `R`** | The finite commutative ring `Z/2^64 Z`, with arithmetic = `u64` wrapping. |
| **Octonion / hypercomplex element** | An 8-tuple `x = (x0, …, x7) ∈ R^8`; the reference type name is `Oct8` (§8). `x0` is the *scalar/real* coefficient; `x1…x7` are *imaginary* coefficients. |
| **Basis units** | `e0 = (1,0,…,0)` (multiplicative identity), `e1 … e7` the imaginary units. |
| `⊗` | Octonion multiplication over `R` (§8), non-commutative and non-associative. |
| `⊞`, `⊟` | Component-wise wrapping addition / subtraction in `R` (i.e. `u64::wrapping_add/ sub` per coefficient). |
| `⊕` | Bitwise XOR over the 512-bit encoding of an octonion (8 independent `u64` XORs). |
| `ROT_λ` | Per-lane left bit-rotation of the 8 words by the fixed vector `λ`. |
| `PERM_π` | Fixed permutation of the 8 coefficient slots by `π`. |
| `CONJ` | Octonion conjugation: negate `x1…x7`, keep `x0`. |
| **Branch** | One octonion half of the Feistel state; `L` (left) and `R` (right). |
| **State** | The pair `(L, R) ∈ R^8 × R^8`, 1024 bits total. |
| **Round function `F_r`** | The per-round map `R^8 → R^8` defined by the round program plus round keys/constants. |
| **Program / graph** | A canonical postfix instruction stream evaluated on an octonion stack (§9). "Graph" and "program" are used interchangeably; the program *is* the linearized ordered evaluation tree. |
| **XOF** | Extendable-output function; here HKDF-Expand over HMAC-SHA-256 (§10). |
| **KDF** | Key-derivation function; here HKDF-SHA-256 (RFC 5869). |
| **Unit** | An element of `R` (or of `Oct8`) that has a two-sided multiplicative inverse. In `Z/2^64`, the units are exactly the odd words. |
| **Zero divisor** | A nonzero element `a` with a nonzero `b` such that `a·b = 0`. |
| **KAT** | Known-answer test (a fixed input → fixed expected output vector). |
| **v0.1 reference** | The exact, constant-time, public-topology instantiation defined by §8–§14. |

**Notation conventions.** All integers are unsigned 64-bit unless stated. `LE64(n)` is the
8-byte little-endian encoding of a `u64`; `LE32(n)` is the 4-byte little-endian encoding of a
`u32`. Rotation amounts are taken `mod 64`. Coefficient indices run `0..=7`. Round index `r`
runs `0..=R_rounds-1`.

---

## 4. Explicit non-claims

This construction, as of v0.1, does **not** provide and does **not** claim:

- confidentiality, indistinguishability (IND-CPA/CCA), or PRP/PRF security;
- authenticated encryption, integrity, or misuse resistance;
- collision resistance, preimage resistance, or second-preimage resistance;
- one-wayness (lack of an obvious inverse is *not* one-wayness);
- post-quantum security of any kind;
- side-channel resistance beyond the constant-time *coding rules* of §18 (which are necessary,
  not sufficient);
- production suitability or regulatory compliance.

### 4.1 Fallacies this document explicitly rejects

The following arguments are **invalid** and are not used anywhere as security justification:

1. *"Non-commutativity implies security."* False. Many non-commutative structures
   (e.g. matrix rings) are trivially linear and easy to solve.
2. *"Non-associativity implies post-quantum security."* False. Non-associativity is a
   structural property; it says nothing about quantum or classical hardness. No reduction to
   a hard problem is claimed.
3. *"A large number of parenthesizations (Catalan growth) proves resistance."* False. Counting
   possible programs/topologies is not a security argument (see §6.3). The attacker does not
   need to enumerate programs.
4. *"An attacker must enumerate every graph."* False. An attacker works against the *fixed*
   published algorithm and the *unknown key*, using algebraic and statistical structure — not
   brute enumeration of topologies.
5. *"Chaotic-looking output proves pseudorandomness."* False. Passing statistical tests is
   necessary, never sufficient (§17). Linear/algebraic structure can hide behind
   statistically-random output.
6. *"Octonions with floating-point coefficients are suitable for deterministic crypto."*
   False, and specifically forbidden here: floats are non-deterministic across platforms,
   non-exact, and not usable in constant-time integer pipelines. (SciRust's existing octonion
   type is `f32`-based — see §5 — and is **not** reused for this reason.)
7. *"Graph secrecy alone provides Kerckhoffs-compliant security."* False. Kerckhoffs requires
   security to rest on the key. In v0.1 the topology is *public* (§6, §10).
8. *"No obvious inverse ⇒ cryptographically one-way."* False. One-wayness is an average-case
   hardness statement requiring evidence; "I don't see how to invert it" is not evidence.

### 4.2 What an attacker may exploit instead

An adversary is expected to try — and this design must be measured against — at least:
algebraic identities (alternativity, flexibility, Moufang, the eight-square norm identity);
linear and matrix representations of the round map; polynomial-system solving; differential
and truncated/impossible/higher-order differentials; invariant subspaces and invariant
ideals; low-rank structure; graph/program equivalences; meet-in-the-middle decompositions;
SAT/SMT modelling; Gröbner-basis methods; interpolation attacks; related-key and
related-branch structure; side channels; and implementation errors. Section 17 turns each of
these into a concrete experiment.

---

## 5. Repository findings

Phase-0 inspection of the SciRust workspace (root `Cargo.toml`, 130+ member crates). No
repository files were modified during inspection. All findings below are load-bearing for the
design choices in this document.

### 5.1 Workspace policy (MSRV, lints, determinism)

| Item | Value | Source |
|---|---|---|
| Package edition | `2021` | root `Cargo.toml` |
| **MSRV** | **`1.89.0`** (CI job "Check (MSRV 1.89.0)" runs `cargo +1.89.0 check --workspace --all-targets --locked`) | root `Cargo.toml` `rust-version = "1.89"`, `.github/workflows/ci.yml` |
| Pinned toolchain | `nightly-2026-07-02` (+ `rustfmt`, `clippy`, `llvm-tools-preview`); nightly is required only for `portable_simd` in `scirust-simd` | `rust-toolchain.toml` |
| Formatting | `rustfmt.toml`: `style_edition = "2024"`, `max_width = 100`, block indent, 4 spaces, `control_brace_style = "AlwaysNextLine"`, `match_block_trailing_comma = true`, `unstable_features = true`. CI: `cargo +nightly-2026-07-02 fmt --all -- --check`. | `rustfmt.toml`, CI `fmt` job |
| Clippy | `cargo +nightly-2026-07-02 clippy --workspace --all-targets --locked -- -D warnings`; workspace `RUSTFLAGS: "-D warnings"`. | CI `clippy` job |
| Tests | `cargo +nightly-2026-07-02 test --workspace --locked` and a `+stable` mirror. | CI `build-test`, `build-test-stable` |
| Miri | Run on targeted `unsafe`-containing modules only (`scirust-core` views, `scirust-arena`), with `-Zmiri-isolation-error=warn`. | CI `miri` job |
| Cross-platform determinism | Explicit x86 ↔ `aarch64` **bit-exactness** legs (`cross-check-aarch64`, native ARM tests of `portable_f32`, `lowprec`, `tree_allreduce`, `formal_proof`). | CI, `docs/TEST_PROTOCOL.md` |
| Supply chain | `cargo deny check` (advisories, licenses, sources); `deny.toml`. Licenses are PolyForm-Noncommercial for workspace crates, permissive for third-party. `unknown-registry`/`unknown-git = "deny"`. | `deny.toml`, CI `deny` job |
| `#![forbid(unsafe_code)]` | Present in ~20 leaf crate roots (`scirust-stats`, `scirust-units`, `scirust-sigma`, `scirust-fractional`, `scirust-relativity`, …). **Absent** in `scirust-core` and `scirust-simd` (both use `unsafe` + `std::simd`). | crate `lib.rs` headers |
| `#![no_std]` | **Not used anywhere** in the active workspace; everything is `std`. | workspace grep |
| SBOM | CycloneDX SBOM emitted to `docs/sbom/scirust.cdx.json`. | CI |

**Design consequences:** the new crate targets MSRV `1.89` (stable), `#![forbid(unsafe_code)]`,
`std`, rustfmt/clippy-clean under the pinned nightly, integer-only arithmetic (no floats), and
bit-exact cross-platform output. A `no_std` variant is possible but would be the workspace's
first (open question §20).

### 5.2 Existing crypto / algebra surface (reuse vs. avoid)

The workspace's third-party crypto surface is deliberately tiny: the only crypto-adjacent
external crate is **`sha2 = "0.10"` (RustCrypto)**, plus `num-bigint`/`num-traits` and
`rand 0.8`. There is **no** `sha3`/Keccak/SHAKE, **no** `hmac`/`hkdf` crate, **no** `blake*`,
**no** `subtle`, **no** direct `getrandom`, **no** cipher.

| Category | In-tree asset | Reuse decision for this crate |
|---|---|---|
| Pure SHA-256 (no deps) | `scirust-sciagent/src/sha256.rs` — `sha256`, `sha256_hex`, FIPS-180-4 / RFC 6234 KATs inline, big-endian per spec. | **Reuse** as the hash core of the KDF/XOF (§10). |
| Pure HMAC-SHA-256 (no deps) | `scirust-discovery/src/hmac.rs` — `hmac_sha256`, RFC 4231 KATs inline. | **Reuse** as HKDF's HMAC and the XOF's PRF (§10). |
| HMAC + constant-time compare demo | `src/main.rs` (`openclaw-u`), `scirust-discovery/src/scope.rs`, `scirust-trader/src/wallet.rs` — the `ct_eq` XOR-fold pattern. | **Reuse the `ct_eq` pattern** for tag/vector comparison (§18); no `subtle` dependency added. |
| Hypercomplex algebra | `scirust-simd/src/hypercomplex/{quat,octonion,sedenion,dual,scalar}.rs` — Cayley–Dickson `(a,b)(c,d) = (a·c − d̄·b, d·a + b·c̄)`, documents non-associativity, zero divisors, inverse `s̄/‖s‖²`. **`f32`-based, nightly `portable_simd`.** | **Do NOT reuse for arithmetic** — it is floating-point (violates the no-float, determinism, and constant-time constraints). Retained only as *prior art* and as a differential-test oracle for the **sign structure** of the multiplication table (§8, §17). This crate defines its own exact integer octonion. |
| Finite-field / modular | `scirust-core/src/homomorphic.rs` (`mod_inverse`, `modpow`, Miller–Rabin over `num-bigint`; docstring: modexp **not** constant-time); `scirust-signal/src/radar/crt_prf.rs` (`i64` `mod_inverse`, CRT). No `F_p`/Montgomery/Barrett type, no `GF(2^k)`. | **Not needed** for Candidate A (`Z/2^64`). Relevant only if a future `F_p` variant (§7, §20) is pursued; even then, `homomorphic.rs` is not constant-time and would not be reused directly. |
| Deterministic RNG | `scirust-core/src/philox.rs` (Philox4x32-10, Random123 KATs), `scirust-stats/src/rng.rs` (SplitMix64), `scirust-core/src/nn/rng.rs` (PCG). `OsRng` appears only in `homomorphic.rs` keygen; deterministic library paths never call `thread_rng`. | **Not used** for derivation (we use the KDF/XOF). Confirms the workspace rule: *never* call OS entropy in deterministic paths (§18). Note: `scirust-som/crates/pcg` is a *Place Capability Graph*, unrelated to RNG. |
| Graph / evaluator substrate | `scirust-graph/src/{lib,dag}.rs` (`CausalDag`, cycle-rejecting, "bit-for-bit deterministic `usize`"); `scirust-symbolic` `enum Expr` tree with `eval`; `scirust-tensor-compile` `ElementwiseOp` fused evaluator. No bytecode VM. | **Not reused directly** (all are float/tensor-oriented), but they establish the house pattern: `enum`-`match` interpreters over an op vector, deterministic, cycle-rejecting. Our postfix evaluator (§9) follows the same discipline. |
| Permutation primitive | Only Fisher–Yates shuffles inside data loaders; no standalone permutation type. | Nothing to reuse; we define the coefficient permutation `π` explicitly (§8). |
| Serialization conventions | Two documented byte orders: **little-endian for determinism fingerprints** (`scirust-runtime/src/hash.rs`: `u64.to_le_bytes`, f32-bits LE; `scirust-core/src/portable_f32.rs` FNV) and **big-endian inside the SHA-256 core** (per FIPS). Hex is uniform lowercase `{:02x}`. `serde` is optional/feature-gated. | **Adopt little-endian** for octonion/state serialization (matches the determinism-fingerprint convention). The SHA-256 core keeps its internal big-endian per FIPS; that is an implementation detail of the primitive, not of our wire format. Hex output is lowercase, no prefix, no whitespace. |
| KAT / test-vector convention | No standalone `.kat`/vector files; all KATs are inline `#[cfg(test)]` modules with hex literals, plus stored fingerprint contracts (e.g. Philox `0xf96c6b6a_eca699f5`). | **Adopt** inline hex KAT modules + a stored fingerprint contract; publish the machine-readable vector format of §15 alongside. |

**Do-not-duplicate summary.** We reuse the pure SHA-256 + HMAC-SHA-256 code and the `ct_eq`
pattern. We do **not** duplicate or reuse the `f32` hypercomplex algebra (wrong domain), and we
do not add `sha3`, `hmac`, `hkdf`, or `subtle` crates (the in-tree primitives suffice and keep
the dependency set minimal, matching the workspace's zero-consumed-FFI, tight-supply-chain
posture in `SECURITY.md` / `deny.toml`).

---

## 6. Threat model

### 6.1 What v0.1 attempts to provide (targets, not guarantees)

- a deterministic, invertible keyed permutation `P_K` on 1024-bit states;
- structural sensitivity to operand ordering and to parenthesization;
- a platform for measuring diffusion (avalanche, branch number) by round;
- a reference for algebraic cryptanalysis of non-commutative/non-associative round functions;
- a simple scalar oracle against which any future optimized/SIMD implementation must match
  bit-for-bit.

### 6.2 What v0.1 does not provide

See §4. In short: no proven confidentiality, integrity, collision/preimage resistance,
post-quantum security, side-channel resistance, misuse resistance, production suitability, or
compliance.

### 6.3 Adversary model (Kerckhoffs, worst-case knowledge)

The adversary is assumed to know **everything except the secret key**:

- the full algorithm and this specification;
- the coefficient domain and octonion multiplication table;
- the graph-generation process and the resulting **public** round programs (topology is not
  secret in v0.1 — see below);
- all constants (`λ`, `π`, round constants, domain strings);
- the reference source code and the intended implementation strategy;
- **arbitrary chosen-plaintext / chosen-ciphertext access**: the adversary may query `P_K` and
  `P_K^{-1}` on inputs of its choosing and observe outputs;
- in the strongest setting, **related-key / related-tweak** access.

Only the master key (and, if used, a secret tweak) is unknown.

**Topology is public, by necessity.** Non-negotiable constraints 10–11 forbid secret-dependent
control flow and secret-dependent memory addressing. A *key-derived program* would select
opcodes and operand indices from secret material — i.e. it *is* secret-dependent control flow
and addressing. Therefore v0.1 derives the round programs from **public** parameters only
(version + round index), so the instruction sequence and memory-access pattern are fixed and
key-independent. Consequences:

- Security must rest entirely on the **round keys, whitening, and constants** (data), exactly
  as in a conventional cipher. This is the Kerckhoffs-compliant posture.
- "Number of possible topologies" and Catalan/Fano counting are **explicitly excluded** as
  security arguments (§4.1). The topology is not a secret and contributes zero secret entropy.
- A key-derived topology is documented as a **future experimental mode** that *breaks*
  constant-time and is therefore out of scope for the v0.1 constant-time reference (§20).

### 6.4 Non-goals of the model

Physical fault injection, invasive hardware attacks, and micro-architectural leakage beyond
the coding rules of §18 are acknowledged (§17, implementation attacks) but not defended
against by construction in v0.1.

---

## 7. Algebraic-domain comparison

We study two coefficient domains for the octonion coefficients. **v0.1 selects Candidate A.**

### 7.1 Candidate A — `R = Z/2^64 Z` (`u64` wrapping)

Arithmetic is `u64::wrapping_add`, `wrapping_sub`, `wrapping_mul` (exact two's-complement mod
`2^64`).

- **Exact semantics:** fully defined by Rust's wrapping ops; no undefined behavior; identical
  on every platform (no `u128` needed — each partial product uses `wrapping_mul`, and the
  octonion product is a wrapping sum of such products). Portable and reproducible.
- **Units:** exactly the **odd** words (`gcd(a, 2^64) = 1 ⇔ a odd`). Half of all words are
  non-units.
- **Zero divisors:** abundant — every even word is a zero divisor (`2^63 · 2 = 0`).
- **Invertibility of multiplication:** multiplication by an odd word is a bijection on `R`;
  multiplication by an even word is **not** injective (it collapses information). Hence
  `x ↦ a·x` is a permutation **iff** `a` is odd.
- **Cost of reduction:** none — wrapping *is* the reduction. This is the cheapest possible and
  the reason it is trivially constant-time.
- **Constant-time:** trivial. No conditional subtraction, no data-dependent branch, no table.
- **ARX compatibility:** excellent. Wrapping add + bit-rotation + XOR is the classic ARX
  toolkit; mixing it with wrapping multiplication gives strong `GF(2)` non-linearity via
  carries.
- **Algebraic-simplification risk:** carries make `Z/2^64` addition **non-linear over
  `GF(2)`**, which is good against `GF(2)`-linear attacks — but the ring is `GF(2)`-friendly for
  *differential* analysis of addition, and multiplication by fixed constants is affine over
  `Z/2^64`. Low-degree structure per multiply (degree 2) must be blown up by round count and
  cross-structure mixing.
- **Portability / testing:** best of the two; pure integer, easy property tests, easy exact
  reduced models (`Z/2^k`).

### 7.2 Candidate B — a prime field `F_p`

Pick a reduction-friendly prime, e.g. the Goldilocks prime `p = 2^64 − 2^32 + 1` (fast
reduction) or a pseudo-Mersenne `p = 2^64 − 59`. Canonical reduction returns the unique
representative in `[0, p)`; encodings `≥ p` are rejected.

- **Exact semantics:** exact, but reduction must be specified to the bit (Solinas reduction for
  Goldilocks; conditional final subtraction).
- **Units / zero divisors:** every nonzero element is a unit; **no zero divisors in the field**.
  (But see the octonion-level caveat in §7.4 — the *octonion algebra over* `F_p` still has
  zero divisors.)
- **Invertibility of multiplication:** `x ↦ a·x` is a bijection for every `a ≠ 0`.
- **Cost of reduction:** non-trivial. Multiplication produces a 128-bit product needing
  reduction; a constant-time conditional subtract (or Montgomery form) is required.
- **Constant-time:** achievable but *harder* — the final conditional subtraction and any
  canonicalization must be branch-free.
- **Cayley–Dickson suitability:** the octonion construction works over any commutative ring, so
  `F_p` is fine structurally.
- **Algebraic-simplification risk:** **higher** in one important sense — a clean field is the
  natural home for Gröbner-basis and interpolation attacks; a low-degree round map over `F_p`
  is exactly what those attacks target. The absence of carry-nonlinearity removes a cheap
  source of `GF(2)`-hardness.
- **Portability / testing:** good, but reduced models are less clean (need small primes; the
  structure changes qualitatively between `F_p` and `Z/2^k`).

### 7.3 Selection

**v0.1 = Candidate A (`Z/2^64`).** Rationale, in priority order:

1. **Constant-time is trivial** (constraint 10–11, 18): no data-dependent reduction.
2. **The field's one advantage — every element invertible — is irrelevant here.** The Feistel
   round function `F` need **not** be a bijection (§11), so zero divisors and non-invertible
   multipliers inside `F` are harmless to *correctness*. We deliberately do **not** buy field
   invertibility we do not use.
3. **Best portability and cheapest exact reduced models** (`Z/2^4`, `Z/2^8`, `Z/2^16`) for §16.
4. **ARX cross-structure** (`GF(2)` XOR/rotation vs. `Z/2^64` add/multiply) is a documented,
   studied source of non-linearity that resists a single clean algebraic model.

`F_p` is recorded as an **alternative for a future "arithmetization-friendly" variant** (§20),
with the explicit warning that its cleaner structure may *help* algebraic attackers.

### 7.4 Which octonion properties survive the move to a finite domain?

This is the most important algebraic section. Real octonions `O_R` are a **normed division
algebra**; almost none of the "division-algebra magic" survives over a finite domain. Each
property below is classified as **holds**, **holds but is an attack surface**, or **fails**.

| Property | Over `R` (reals) | Over `Z/2^64` (Candidate A) | Over `F_p` (Candidate B) | Notes |
|---|---|---|---|---|
| Non-commutativity | holds | **holds** | holds | Structure constants are `±1`; `e_i e_j = −e_j e_i` for distinct imaginary units. A ring-level property. |
| Non-associativity | holds | **holds** (attack surface) | holds | The associator of a non-Fano-line triple is `±2·e_m`; over `Z/2^64`, `2·1 = 2 ≠ 0`, so it survives. **The factor of 2 matters for differential analysis** (the associator lives in the even sublattice). |
| Alternativity `(xx)y=x(xy)` | holds | **holds** (attack surface) | holds | A polynomial identity with integer (`±1`) coefficients ⇒ valid over any commutative ring. **This is exploitable, not protective.** |
| Flexibility `(xy)x=x(yx)` | holds | **holds** (attack surface) | holds | Same reasoning. |
| Moufang identities | holds | **holds** (attack surface) | holds | Same reasoning; strong identities an attacker can use to build equations. |
| Conjugation `x̄` | holds | **holds** | holds | `x̄ = (x0, −x1, …, −x7)`; `x + x̄ = 2x0` (trace); `−` is wrapping negation. |
| Norm `N(x)=x x̄=Σ x_i²` | positive-definite | **holds as a form** (attack surface) | holds as a form | Well-defined scalar, but **not** positive/anisotropic. |
| Norm multiplicativity `N(xy)=N(x)N(y)` | holds (composition) | **holds** (attack surface) | holds | Degen's eight-square identity — a polynomial identity with integer coefficients ⇒ valid over any commutative ring. **A multiplicative invariant an attacker can track. The round function must break norm-tracking.** |
| Zero divisors | **none** | **many** | **exist** | Over `Z/2^64`: any octonion with even norm (and more). Over `F_p`: the 8-variable norm form is **isotropic** (a form in ≥ 3 variables over a finite field is isotropic, Chevalley–Warning), so the octonion algebra is **split** and has zero divisors. |
| Every nonzero element invertible | **yes** (division algebra) | **no** | **no** | Over `Z/2^64`, `x^{-1} = x̄ · N(x)^{-1}` exists **iff** `N(x)` is odd (a unit). Over `F_p`, exists iff `N(x) ≠ 0`, and isotropy guarantees nonzero non-invertible elements. |

**Conclusions carried into the design:**

- **Neither domain preserves the division-algebra property.** Raw octonion multiplication is
  therefore **not** a permutation over either domain (§11). This is the single most important
  correction relative to naive "octonion cipher" folklore.
- **The surviving identities (alternativity, flexibility, Moufang, norm multiplicativity) are
  attack surface, not features.** They give the adversary free algebraic relations. The design
  must actively *break* them across rounds (via XOR of round constants, per-lane rotations that
  are not algebra automorphisms, and the Feistel XOR combiner, which respects none of these
  identities). Their persistence is flagged for §17 (invariant/norm experiments).
- Any property whose validity over the finite domain was uncertain has been resolved above; no
  property is left "assumed". The one subtlety worth restating: the associator carries a factor
  of 2, so on `Z/2^64` non-associativity "loses one bit" per associator — a concrete lead for
  differential cryptanalysis.

---

## 8. Exact octonion convention

### 8.1 Type

```rust
// Reference type. Illustrative signature; final naming follows crate conventions (§19).
// #![forbid(unsafe_code)]  — no unsafe anywhere in the reference.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct Oct8 {
    /// Coefficients c[0..=7]; c[0] is the scalar/real unit e0, c[1..=7] are e1..e7.
    pub c: [u64; 8],
}
```

- **Component ordering:** index `i` is the coefficient of basis unit `e_i`. `c[0]` is the
  scalar. This ordering is fixed and authoritative.
- **Additive identity (zero):** `ZERO = Oct8 { c: [0;8] }`.
- **Multiplicative identity (one):** `ONE = Oct8 { c: [1,0,0,0,0,0,0,0] }` (`= e0`).
- **Equality:** exact per-coefficient `u64` equality (derive `Eq`). There is exactly one
  representation per algebra element (no reduction ambiguity in `Z/2^64`).

### 8.2 Component-wise operations (all in `Z/2^64`)

```text
(x ⊞ y)_i = x_i.wrapping_add(y_i)        for i in 0..=7      # addition
(x ⊟ y)_i = x_i.wrapping_sub(y_i)        for i in 0..=7      # subtraction
(neg x)_i = 0u64.wrapping_sub(x_i)       for i in 0..=7      # negation (two's complement)
CONJ(x)   = (x0, neg x1, neg x2, …, neg x7)                  # conjugation
(x ⊕ y)_i = x_i ^ y_i                     for i in 0..=7      # bitwise XOR (per lane)
```

### 8.3 Multiplication — authoritative convention

We fix **one** convention and give it three redundant, mutually-consistent forms: (a) the
generating rule, (b) the full signed basis table, (c) the general bilinear formula. **Do not
mix this with any other octonion sign convention** (including the `f32` Cayley–Dickson
convention in `scirust-simd`, which is retained only as prior art — §5).

**(a) Generating rule (cyclic / Fano).** For imaginary indices taken on `{1,…,7}` with
arithmetic **mod 7** (representing `0` as `7`):

```text
e_i · e_{i+1} = e_{i+3}          for i = 1..7 (indices mod 7 on {1..7})
e_i · e_i     = −e0              (imaginary units square to −1)
e0 · e_i = e_i · e0 = e_i        (e0 is the identity)
```

This yields the seven "quaternionic triples" (Fano lines) `(i, i+1, i+3)`:

```text
(1,2,4)  (2,3,5)  (3,4,6)  (4,5,7)  (5,6,1)  (6,7,2)  (7,1,3)
```

Each ordered triple `(a,b,c)` means the **cyclic** relations

```text
a·b = c,   b·c = a,   c·a = b        (and reversing any product negates it:)
b·a = −c,  c·b = −a,   a·c = −b
```

These seven triples partition all `21 = C(7,2)` unordered imaginary pairs exactly once, so the
table below is total and unambiguous.

**(b) Full signed multiplication table.** Entry at row `e_i`, column `e_j` is `e_i · e_j`.
`ê_k` denotes `−e_k` (coefficient `2^64 − 1` at slot `k`, i.e. `0xffffffffffffffff`).

```text
   ×  | e0    e1    e2    e3    e4    e5    e6    e7
  -----+-------------------------------------------------
   e0 | e0    e1    e2    e3    e4    e5    e6    e7
   e1 | e1   −e0    e4    e7    ê2    e6    ê5    ê3
   e2 | e2    ê4   −e0    e5    e1    ê3    e7    ê6
   e3 | e3    ê7    ê5   −e0    e6    e2    ê4    e1
   e4 | e4    e2    ê1    ê6   −e0    e7    e3    ê5
   e5 | e5    ê6    e3    ê2    ê7   −e0    e1    e4
   e6 | e6    e5    ê7    e4    ê3    ê1   −e0    e2
   e7 | e7    e3    e6    ê1    e5    ê4    ê2   −e0
```

(Read, e.g., row `e1`: `e1·e0=e1`, `e1·e1=−e0`, `e1·e2=e4`, `e1·e3=e7`, `e1·e4=−e2`,
`e1·e5=e6`, `e1·e6=−e5`, `e1·e7=−e3`. Every off-diagonal imaginary entry is antisymmetric:
`e_i·e_j = −(e_j·e_i)`, visible as the table's skew symmetry.)

**(c) General bilinear product.** For `x, y ∈ R^8`, the product `z = x ⊗ y` is

```text
z_k = Σ_{ (i,j) : e_i·e_j = s_{ij}·e_k }  s_{ij} · x_i · y_j     (all ops wrapping, mod 2^64)
```

where `s_{ij} ∈ {+1, −1}` and the target index are read from the table (b). Equivalently, with
a precomputed structure `SIGN[i][j] ∈ {+1,−1}` and `IDX[i][j] ∈ {0..7}`:

```text
z = [0u64; 8]
for i in 0..=7:
  for j in 0..=7:
    let p = x[i].wrapping_mul(y[j])           # partial product in Z/2^64
    let k = IDX[i][j]
    if SIGN[i][j] == +1 { z[k] = z[k].wrapping_add(p) }
    else                { z[k] = z[k].wrapping_sub(p) }
z
```

This double loop (64 wrapping multiplies, 64 wrapping add/sub) is the **authoritative scalar
reference**. No `u128`, no float, no branch on secret data (the loop bounds and `SIGN`/`IDX`
tables are public constants; see §18 for the constant-time note on the `if`). Any SIMD or
Cayley–Dickson-recursive implementation MUST match this bit-for-bit (§14, §17).

### 8.4 Worked examples (all exact in `Z/2^64`)

Let `ê_k` again denote the octonion with coefficient `0xffffffffffffffff` at slot `k`, zero
elsewhere (`= −e_k`).

**(1) Non-commutativity — `e1·e2` vs `e2·e1`:**

```text
e1 ⊗ e2 = e4            # c = (0,0,0,0,1,0,0,0)
e2 ⊗ e1 = −e4 = ê4      # c = (0,0,0,0, 0xffffffffffffffff, 0,0,0)
=> e1 ⊗ e2 ≠ e2 ⊗ e1 . (Non-commutative.)
```

**(2) Associative triple (Fano line `{1,2,4}`) — `(e1·e2)·e4` vs `e1·(e2·e4)`:**

```text
(e1 ⊗ e2) ⊗ e4 = e4 ⊗ e4 = −e0 = ê0
 e1 ⊗ (e2 ⊗ e4) = e1 ⊗ e1 = −e0 = ê0        # since e2·e4 = e1 (triple (1,2,4): 2·4=1)
=> equal. On a Fano line, association holds; the associator is 0. (Sanity/consistency check.)
```

**(3) Non-associativity (non-line triple `{1,2,3}`) — `(e1·e2)·e3` vs `e1·(e2·e3)`:**

```text
(e1 ⊗ e2) ⊗ e3 = e4 ⊗ e3 = −e6 = ê6         # (3,4,6): 4·3 = −6
 e1 ⊗ (e2 ⊗ e3) = e1 ⊗ e5 =  e6             # (2,3,5): 2·3 = 5 ; (5,6,1): 1·5 = 6
=> (e1⊗e2)⊗e3 = ê6  ≠  e6 = e1⊗(e2⊗e3).  Associator = ê6 ⊟ e6 = −2·e6. (Non-associative;
   note the factor 2, per §7.4.)
```

**(4) A general product (multiplication KAT) — `x = e1 ⊞ e2`, `y = e2 ⊞ e3`:**

```text
x ⊗ y = (e1+e2)(e2+e3)
      = e1⊗e2 + e1⊗e3 + e2⊗e2 + e2⊗e3
      = e4     + e7     + (−e0) + e5
      = (−e0) ⊞ e4 ⊞ e5 ⊞ e7
c = ( 0xffffffffffffffff, 0, 0, 0, 1, 1, 0, 1 )
```

**(5) Reverse product (order matters at the algebra level) — `y ⊗ x`:**

```text
y ⊗ x = (e2+e3)(e1+e2)
      = e2⊗e1 + e2⊗e2 + e3⊗e1 + e3⊗e2
      = (−e4)  + (−e0)  + (−e7)  + (−e5)
      = (−e0) ⊞ (−e4) ⊞ (−e5) ⊞ (−e7)
c = ( 0xffffffffffffffff, 0, 0, 0,
      0xffffffffffffffff, 0xffffffffffffffff, 0, 0xffffffffffffffff )
=> x ⊗ y ≠ y ⊗ x (the e4,e5,e7 signs flip). Confirms non-commutativity on non-basis elements.
```

These five vectors are hand-verifiable and serve as the first-tier multiplication KATs (§15).

---

## 9. Graph model

The round function is expressed as a **typed, ordered, explicitly-parenthesized computation
graph**, linearized to a **canonical postfix (stack-machine) program**. This is not a generic
unordered graph: operand order is fixed by stack order, and bracketing is fixed by program
order — which is exactly what makes ordering and parenthesization semantically significant.

### 9.1 Values, stack, arities

- The only value type on the stack is `Oct8` (§8). There is one typed value class ⇒ the type
  system is trivial but total (no ill-typed program is representable).
- The evaluator maintains a bounded stack of `Oct8`. Root semantics: after `END`, the stack
  must contain **exactly one** element — the round-function output. Leaf semantics: the `PUSH_*`
  instructions are the leaves (they read fixed inputs — the branch octonion, subkeys,
  constants).
- Evaluation order is strictly left-to-right over the instruction stream (postfix / reverse
  Polish). A binary op consumes the top two elements (`b = pop()`, `a = pop()`), pushing the
  result; **`a` is the left operand, `b` is the right operand** — this fixes both ordering and
  bracketing.

### 9.2 Instruction set

Each instruction has exact semantics and a fixed stack effect (`in → out`).

| Opcode | Mnemonic | Operand | Stack effect | Semantics | Algebraic character |
|---:|---|---|---|---|---|
| `0x00` | `END` | — | `[x] → [x]` | Terminate; final stack size must be 1. | — |
| `0x01` | `PUSH_INPUT` | `idx: u8` | `[] → [v]` | Push input octonion `idx` (v0.1: only `idx=0`, the branch `R`). | leaf |
| `0x02` | `PUSH_KEY` | `idx: u8` | `[] → [K_idx]` | Push round subkey octonion `idx`. | leaf (secret data) |
| `0x03` | `PUSH_CONST` | `idx: u8` | `[] → [C_idx]` | Push derived constant octonion `idx`. | leaf (public data) |
| `0x04` | `ADD` | — | `[a,b] → [a⊞b]` | Wrapping component add. | **linear** over `Z/2^64` |
| `0x05` | `SUB` | — | `[a,b] → [a⊟b]` | Wrapping component sub. | **linear** over `Z/2^64` |
| `0x06` | `MUL` | — | `[a,b] → [a⊗b]` | Octonion product, `a` left / `b` right. | **bilinear** (degree-2 nonlinearity; the only ring-nonlinear op) |
| `0x07` | `CONJ` | — | `[a] → [ā]` | Conjugation. | **linear** (affine involution) |
| `0x08` | `ROT` | `lid: u8` | `[a] → [ROT_{λ(lid)}(a)]` | Per-lane left bit-rotation by fixed vector `λ(lid)`. | `GF(2)`-linear, `Z/2^64`-**non**linear |
| `0x09` | `PERM` | `pid: u8` | `[a] → [PERM_{π(pid)}(a)]` | Permute the 8 coefficient slots by fixed `π(pid)`. | **linear** (slot permutation) |
| `0x0A` | `XORC` | `idx: u8` | `[a] → [a ⊕ RC_idx]` | XOR the 512-bit round constant `RC_idx`. | affine over `GF(2)` |

**Curation rationale (which instructions are useful / safe / dangerous):**

- **`MUL` is the sole source of ring non-linearity** and the carrier of non-commutativity /
  non-associativity. It is *not* invertible in general (multipliers may be non-units / zero
  divisors) — **safe inside a Feistel `F`**, unsafe as a standalone permutation (§11).
- **`ADD`/`SUB`/`CONJ`/`PERM` are linear** (and `CONJ`/`PERM` are `GF(2)`-linear too). They
  provide diffusion but, used alone, expose a clean linear model. They must be *sandwiched*
  between `MUL`s and key injections; a program consisting only of linear ops is a red flag the
  validator does not forbid but the analysis plan (§17) must catch.
- **`ROT` and `XORC` provide the `GF(2)`/ring cross-structure** (ARX flavor). `ROT` breaks
  `Z/2^64`-linearity and — importantly — breaks the norm-multiplicative and algebra-automorphism
  structure that `MUL`/`CONJ`/`PERM` alone would preserve. `XORC` injects round-dependent
  asymmetry (anti-slide, anti-fixed-point) and carries domain separation.
- **Dropped as redundant/dangerous:** a separate `RotateComponents` (cyclic slot rotation) is a
  special case of `PERM` and is omitted. A raw `Invert`/`Reciprocal` instruction is **forbidden**
  — it would depend on octonion invertibility, which does not hold in general (§7.4) and would
  introduce a data-dependent branch. A `PushCounter`-with-secret is forbidden (would be
  secret-dependent addressing).

### 9.3 Canonical encoding

A program is a byte string: a sequence of `(opcode [operand])` items terminated by a single
`END`. Operand presence is fixed by opcode (table above). **Canonicalization rules — a program
is valid iff all hold; there is exactly one valid encoding per abstract program:**

1. The byte stream parses exactly into a sequence of well-formed instructions ending with
   `END`, with **no trailing bytes** after `END`.
2. Every operand index is in range: `PUSH_INPUT idx < N_inputs`; `PUSH_KEY idx < N_keys`;
   `PUSH_CONST idx < N_consts`; `XORC idx < N_rc`; `ROT lid < N_lambda`; `PERM pid < N_pi`.
   (v0.1 parameters in §12.)
3. **Stack discipline:** simulate the stack effects; the stack never underflows (a binary op
   requires ≥ 2 elements, a unary op ≥ 1) and never exceeds `MAX_STACK`.
4. **Final stack size is exactly 1** at `END`.
5. Resource bounds: instruction count `≤ MAX_NODES`; the implied evaluation-tree depth
   `≤ MAX_DEPTH`.
6. No `PUSH_KEY`/`PUSH_CONST`/`PUSH_INPUT` index may reference a slot not provided for the
   current round (checked against the round's key/constant vector lengths).

**Cycle policy:** postfix programs are inherently acyclic (a linear stream over a stack) — there
is no back-reference instruction, so cycles are unrepresentable by construction. **Max depth /
max nodes** are fixed constants (§12) enforced by the validator. The validator is total and
returns a typed error (never panics) on any violation.

**Canonical graph identifier:** `GraphId = SHA256("SCIRUST-HYPERCRYPTO-V0.1/GRAPH-ID" ||
encoding)` (32 bytes), used to name a program in test vectors and to detect encoding drift.
Because the encoding is canonical, `GraphId` is a faithful identity for the abstract program.

### 9.4 Evaluator rejection list

The evaluator/validator MUST reject (typed error, no panic, no allocation on the reject path):
stack underflow; stack overflow (`> MAX_STACK`); final stack size `≠ 1`; any out-of-range
operand index; instruction count `> MAX_NODES`; depth `> MAX_DEPTH`; unknown opcode; truncated
operand; trailing bytes after `END`; a program with no `END`. Malformed input is a first-class
tested surface (fuzzing, §17–§18).

---

## 10. Key and graph derivation

All secret- and public-dependent material is derived by a **standardized KDF/XOF**, never by an
ad-hoc construction and never from OS entropy. We reuse the in-tree pure SHA-256 and
HMAC-SHA-256 (§5), so the derivation adds **zero new crypto dependencies**.

### 10.1 Primitive: HKDF-SHA-256 + HMAC-SHA-256 XOF

- **KDF:** HKDF (RFC 5869) instantiated with HMAC-SHA-256.
  - `PRK = HKDF-Extract(salt, IKM)` with
    `salt = ASCII("SCIRUST-HYPERCRYPTO-V0.1/EXTRACT")` and `IKM = master_key || tweak`.
- **XOF (squeeze):** `Squeeze(PRK, info, L) = HKDF-Expand(PRK, info, L)`, i.e. the RFC 5869
  counter-mode HMAC expansion. Per RFC 5869, `L ≤ 255 · 32 = 8160` bytes per call. When more is
  needed, the `info` string carries an explicit 4-byte block counter (`|| LE32(block)`), and
  blocks are concatenated; this is documented so two implementations agree exactly.
- **Domain separation** is carried entirely by the `info` argument (§8 domains). Distinct
  purposes get distinct `info` prefixes; **no derivation stream is ever reused across
  purposes.**

`Alternative:` SHAKE256 (FIPS 202) is the theoretically cleaner XOF and is recorded as an
open-question swap (§20). It is **not** chosen for v0.1 only because it would add a `sha3`
dependency where the in-tree, RFC-KAT-tested HMAC-SHA-256 already suffices. (The abstract
interface — "a domain-separated XOF" — is identical either way.)

### 10.2 Inputs

| Input | Size | Notes |
|---|---|---|
| `master_key` | **32 bytes (256-bit)** in v0.1 | Fixed for v0.1. Invalid length ⇒ typed error (no panic). |
| `tweak` | 16 bytes (128-bit), default all-zero | Public domain-separator / nonce; may be public. A *secret* tweak (if ever used) must be handled with the same constant-time care as the key. |
| `version` | ASCII `"V0.1"` | Baked into every domain string. |
| `state_width` | `1024` | Encoded `LE64` in `info`. |
| `round_index r` | `0..=R_rounds-1` | Encoded `LE32`. |
| `branch/slot index t` | `u32` | Encoded `LE32`; disambiguates the several keys/constants a round needs. |

### 10.3 Derivation streams (distinct `info` per purpose)

Let `H = "SCIRUST-HYPERCRYPTO-V0.1"`. Each stream's `info` is
`domain || LE64(1024) || LE32(r) || LE32(t) [ || LE32(block) ]`.

| Purpose | `domain` string | What is squeezed |
|---|---|---|
| Round subkeys | `H + "/ROUNDKEY"` | For round `r`, three octonions `K_{r,0}, K_{r,1}, K_{r,2}` (each `8·8 = 64` bytes; `t = 0,1,2`). Words are read little-endian (§14). |
| Round constants | `H + "/CONSTANT"` | Round-constant octonions `C_{r,*}` if used, and the 512-bit XOR constant `RC_r` (`64` bytes, `t = 0`). |
| Whitening | `H + "/WHITENING"` | Input whitening `(W_inL, W_inR)` at `r = 0xFFFFFFFF, t = 0,1`; output whitening `(W_outL, W_outR)` at `r = 0xFFFFFFFE, t = 0,1`. (Reserved round indices avoid collision with real rounds.) |
| Graph / program schedule | `H + "/GRAPH"` | v0.1: **public** program schedule from `version + r` only (no key material — see §10.5). Squeezed bytes seed the bounded program generator. |

**Preventing related-round / related-branch symmetry:** every stream binds `r` and `t`
explicitly in `info`, so `K_{r,*}` and `K_{r',*}` are independent for `r ≠ r'`, and the four
whitening octonions are mutually independent. The reserved-index trick keeps whitening streams
disjoint from round-key streams. This is the design's defense against slide and related-key
symmetry at the *schedule* level (round constants `RC_r` add per-round asymmetry at the
*state* level).

### 10.4 Round-subkey derivation (pseudocode)

```text
fn round_keys(PRK, r) -> (K0, K1, K2):
    for t in 0..3:
        buf = Squeeze(PRK, "SCIRUST-HYPERCRYPTO-V0.1/ROUNDKEY" || LE64(1024) || LE32(r) || LE32(t), 64)
        K_t = octonion_from_le_bytes(buf)          # 8 u64 little-endian, no rejection (any 512-bit
                                                    # value is a valid Oct8)
    return (K0, K1, K2)
```

No rejection sampling is needed for keys/constants: **every** 512-bit string is a valid `Oct8`
(there is no canonical-form constraint in `Z/2^64`). Rejection sampling is only relevant to
*program generation* (§10.5) and to any future biased-range sampling; the general rule (bias
avoidance, bounded attempts, deterministic retry) is stated there.

### 10.5 Program (graph) derivation — v0.1 vs. experimental

- **v0.1 reference (public topology, constant-time).** The program schedule is derived from
  **public** parameters only (`version`, `r`). In fact, for maximal auditability, **v0.1 uses a
  single fixed program `F-PROG` for every round** (defined in §12), so the "generation" is
  degenerate: the same public program each round, with per-round variation supplied only by the
  derived keys/constants. This guarantees a fixed instruction sequence and fixed memory-access
  pattern (constant-time, constraints 10–11). The `/GRAPH` stream is reserved for the
  public-derived multi-program variant below.
- **Public-derived multi-program variant (still constant-time, optional).** Derive a *distinct*
  program per round from `H + "/GRAPH" || LE32(r)` (public only) via bounded rejection sampling:
  draw opcodes/operands from the XOF, simulate the stack, and **accept** the first program that
  passes §9.3 validation with `≤ MAX_NODES` and final stack size 1; **reject and redraw**
  otherwise. Bound: `MAX_GEN_ATTEMPTS` (e.g. 64); on exhaustion, fall back to `F-PROG`
  deterministically. Because inputs are public, the resulting per-round program is public, so
  evaluation remains constant-time.
- **Key-derived topology (EXPERIMENTAL, NOT constant-time, out of scope for v0.1).** Deriving
  the program from the *key* makes the opcode/operand stream secret ⇒ secret-dependent control
  flow and addressing ⇒ violates constraints 10–11. This mode is documented as a research option
  only (§20) and is **explicitly excluded** from the v0.1 constant-time reference and from any
  security claim. Its "topology entropy" is **not** counted toward security (§4.1, §6.3).

**Bias avoidance & determinism (general rule for any sampling):** map XOF bytes to bounded
ranges by rejection (reject values in the biased tail of a modulo reduction; redraw from the
next XOF bytes), never by naive `% range`. All retries consume fresh XOF output deterministically,
so the process is reproducible bit-for-bit on every platform. `MAX_GEN_ATTEMPTS` bounds work
and makes failure deterministic (fallback to `F-PROG`).

---

## 11. Candidate reversible construction

### 11.1 Why raw octonion multiplication is not a permutation

Consider the naive "reduce three octonions with a bracketing", e.g. `O1 ⊗ (O2 ⊗ O3)`:

- **Dimensional compression / information loss.** `⊗` is a bilinear map `R^8 × R^8 → R^8`. Fixing
  one operand, `x ↦ a ⊗ x` is linear; it is a bijection **iff** the corresponding `8×8` matrix
  over `Z/2^64` is invertible, i.e. iff its determinant is a **unit (odd)**. For a random or
  even-normed `a` it is singular or non-injective, collapsing distinct inputs to equal outputs
  (**collisions**).
- **Zero divisors.** Because both domains are split (§7.4), there exist nonzero `a, b` with
  `a ⊗ b = 0`. Any map that ends in a multiply by such an `a` cannot be inverted on the affected
  fibers.
- **Non-invertible multipliers.** Even when a specific bracketing is fully known to the attacker,
  knowledge of the *parenthesization does not grant an inverse*: inverting `x ↦ a ⊗ x` requires
  `a` to be a unit octonion (`N(a)` a unit and the left-multiplication matrix invertible), which
  is not guaranteed.
- **Conclusion:** a permutation must obtain invertibility from a **reversible state-update
  structure**, not from octonion arithmetic. The octonion multiply is confined to the *inside* of
  a round function that need not be invertible.

### 11.2 Candidate reversible structures (evaluated)

| Structure | Which part must be invertible | Must `F`/mix be a bijection? | Inverse | Algebraic risk | Cost |
|---|---|---|---|---|---|
| **Balanced Feistel** (chosen) | only the combiner (`⊕`, an involution) | **No** | reuse the same `F` in reverse round order | slow diffusion (needs enough rounds); classic differential/linear on 2-branch | 1× `F` per round |
| Generalized (Type-II) Feistel | combiner only | No | analogous | more branches ⇒ faster diffusion but larger state / more subkeys | 1× `F` per sub-block |
| ARX-augmented reversible layer | each ARX layer (add/rotate/xor are invertible) | the *whole layer* must be invertible | invert each layer | if any layer embeds a non-invertible multiply, reversibility breaks | medium |
| Lai–Massey | the half-difference map + an orthomorphism | needs an orthomorphism | invert the round map | the multiply must be arranged so the round map stays invertible — harder to guarantee with zero divisors | medium |
| Reversible triangular map | each triangular update (invertible if diagonal fixed) | each component update invertible | back-substitution | forces a specific structured `F`; easy to linearize if not careful | low |
| Even–Mansour composition | the inner permutation | inner map must be a permutation | invert the inner permutation | reduces to "build a permutation" — used here as *whitening around* the Feistel, not as the whole thing | 2 XORs |

### 11.3 Selection for v0.1

**Balanced two-branch Feistel, with Even–Mansour-style input/output whitening.** Reasons,
conservative:

1. **Invertibility is structural and independent of `F`.** The combiner is XOR (an involution);
   the Feistel is a bijection for *any* function `F` (§12–§13). This is the cleanest possible
   invertibility argument and directly satisfies the constraint "every operation intended for a
   permutation must be demonstrably invertible" — the *permutation* is invertible even though its
   round function is not.
2. **It lets us use the octonion multiply exactly where non-invertibility is harmless** — inside
   `F`.
3. **It is the most studied structure** (Luby–Rackoff; decades of differential/linear analysis),
   giving the falsification plan (§17) a rich, well-understood baseline.
4. **Whitening (Even–Mansour)** adds key material at the boundary, frustrating some
   structural/slide attacks and giving a second, independent keying channel.

Each Feistel branch is one full octonion (8 words), so `F : R^8 → R^8` naturally consumes and
produces an octonion — no sub-octonion padding, and octonion non-associativity is exercised on a
genuine octonion.

---

## 12. Mathematical forward definition

### 12.1 Parameters (v0.1)

```text
STATE_WIDTH  = 1024 bits         # two octonion branches L, R (each 8 u64)
R_rounds     = 24                # conservative margin; exact count is an open question (§20)
N_inputs     = 1                 # only input 0 = the branch octonion R
N_keys       = 3                 # K0, K1, K2 per round
N_consts     = 0                 # (v0.1 F uses no PUSH_CONST)
N_rc         = 1                 # one 512-bit round constant RC per round
MAX_NODES    = 64                # per-program instruction bound
MAX_DEPTH    = 32                # evaluation-tree depth bound
MAX_STACK    = 32                # stack depth bound
key size     = 256 bits ; tweak = 128 bits (default 0)
```

**Fixed public diffusion constants** (placeholder values; final selection is a cryptanalytic
open question, §20):

```text
λ  (per-lane left-rotation amounts, lid = 0):  [ 7, 19, 31, 47, 11, 23, 37, 53 ]   # lane j rotated by λ[j] mod 64
π  (coefficient-slot permutation, pid = 0):    [ 3,  6,  1,  4,  7,  2,  5,  0 ]   # (PERM_π(x))_i = x_{π[i]}; a derangement
```

### 12.2 The v0.1 round program `F-PROG`

`F_r(R)` computes, as explicit algebra:

```text
a  = R ⊞ K_{r,0}                       # inject subkey 0 (wrapping add)
p  = K_{r,1} ⊗ a                       # left-multiply by subkey 1   (order matters: K1 on the LEFT)
q  = p ⊗ K_{r,2}                       # right-multiply by subkey 2   (bracketing fixed as (K1⊗a)⊗K2)
u  = ROT_λ(q)                          # per-lane bit rotation  (ARX cross-structure)
w  = PERM_π(u)                         # coefficient-slot permutation
F_r(R) = w ⊕ RC_r                      # round-constant XOR (asymmetry, domain sep)
```

The bracketing `q = (K_{r,1} ⊗ a) ⊗ K_{r,2}` is **explicitly chosen and semantically
significant**: because `⊗` is non-associative, `K_{r,1} ⊗ (a ⊗ K_{r,2})` would give a different
result. Because `⊗` is non-commutative, placing `K_{r,1}` on the left and `K_{r,2}` on the right
are distinct operations. This is the concrete realization of "ordering and parenthesization are
semantically significant."

**Equivalent canonical postfix program** (`F-PROG`; opcodes per §9.2, this is the exact byte
sequence up to the operand values):

```text
PUSH_KEY 1        ; [K1]
PUSH_INPUT 0      ; [K1, R]
PUSH_KEY 0        ; [K1, R, K0]
ADD               ; [K1, a]              (a = R ⊞ K0 ; a is on top, K1 below)
MUL               ; [K1 ⊗ a]            (a_left=K1, b_right=a  ⇒  p = K1 ⊗ a)
PUSH_KEY 2        ; [p, K2]
MUL               ; [p ⊗ K2]            (q = p ⊗ K2 = (K1⊗a)⊗K2)
ROT 0             ; [ROT_λ(q)]
PERM 0            ; [PERM_π(ROT_λ(q))]
XORC 0            ; [ ... ⊕ RC_r ]      (= F_r(R))
END               ; final stack size = 1  ✓
```

Byte encoding (hex, opcodes+operands): `02 01  01 00  02 00  04  06  02 02  06  08 00  09 00
0a 00  00`. `GraphId = SHA256("SCIRUST-HYPERCRYPTO-V0.1/GRAPH-ID" || <these bytes>)`
(computed by the reference implementation; see §15).

**Nonlinearity accounting.** `F` contains two `MUL`s (each degree-2 over `Z/2^64`), so a single
round contributes algebraic degree up to 4 in the input coefficients, interleaved with the
`GF(2)`-linear `ROT`/`XORC`. Over `R_rounds = 24` this compounds; §17 measures the actual degree
growth on reduced variants.

### 12.3 Full forward permutation

```text
fn P_K(state1024) -> state1024:
    # 1. parse state into two octonion branches (little-endian, §14)
    (L, R) = split_1024(state1024)               # L = words[0..8], R = words[8..16]

    # 2. input whitening (Even–Mansour)
    L = L ⊕ W_inL
    R = R ⊕ W_inR

    # 3. balanced Feistel, R_rounds rounds
    for r in 0 .. R_rounds:
        (K0, K1, K2) = round_keys(PRK, r)
        RC           = round_constant(PRK, r)
        Fout         = F_r(R)                     # uses K0,K1,K2,RC and fixed λ,π (§12.2)
        (L, R)       = (R, L ⊕ Fout)              # standard Feistel swap-and-combine

    # 4. output whitening
    L = L ⊕ W_outL
    R = R ⊕ W_outR

    return join_1024(L, R)
```

The combiner in step 3 is XOR (`⊕`) over 512-bit branches (8 lane-wise `u64` XORs). Whitening
octonions and round keys/constants come from §10. All arithmetic is `u64` wrapping; no float, no
`u128`, no branch on secret data.

---

## 13. Mathematical inverse definition

### 13.1 Feistel inverse — derived, not asserted

Let one forward round be the map `ρ_r(L, R) = (R, L ⊕ F_r(R))`. We derive its inverse
explicitly for the exact octonion state.

Write the output as `(L', R') = ρ_r(L, R)`. Then

```text
L' = R                       ...(i)
R' = L ⊕ F_r(R)              ...(ii)
```

From (i), `R = L'`. Substitute into (ii): `R' = L ⊕ F_r(L')`, hence (since `⊕` is an involution:
`x ⊕ y ⊕ y = x`)

```text
L = R' ⊕ F_r(L')             ...(iii)
```

Therefore the inverse round is the **total, explicit** map

```text
ρ_r^{-1}(L', R') = ( R' ⊕ F_r(L') ,  L' ) = (L, R).
```

Key facts this derivation makes precise for the exact state:

- **`F_r` is used, never inverted.** The inverse round recomputes `F_r(L')` and XORs it out. `F_r`
  may be non-injective, contain non-invertible octonion multiplies, or hit zero divisors — none of
  that matters, because we never solve `F_r(y) = z`.
- **The only inverted operation is `⊕`**, whose inverse is itself. This is the entire reason the
  permutation is invertible.
- **Reproducibility of the inverse** requires only that `round_keys`, `round_constant`, and the
  program (`F-PROG`, `λ`, `π`) are the *same* on the decrypt side — guaranteed because they are
  deterministic functions of `(PRK, r)` and public constants (§10, §12).

This is stronger than "Feistel networks are reversible": it is the reversal specialized to this
state representation and this `F`, showing exactly which sub-operation carries invertibility (the
XOR combiner) and which does not need it (the octonion-multiplying `F`).

### 13.2 Full inverse permutation

```text
fn P_K_inv(state1024) -> state1024:
    (L, R) = split_1024(state1024)

    # 1. undo output whitening
    L = L ⊕ W_outL
    R = R ⊕ W_outR

    # 2. reverse Feistel: run rounds in reverse order, applying ρ_r^{-1}
    #    (state after forward round r is (L,R); we invert from r = R_rounds-1 down to 0)
    for r in (R_rounds-1) ..= 0 step -1:
        (K0, K1, K2) = round_keys(PRK, r)
        RC           = round_constant(PRK, r)
        # current (L, R) plays the role of (L', R') = ρ_r(L_prev, R_prev):
        Lprev = R ⊕ F_r(L)         # (iii): R' ⊕ F_r(L')  with L'=L (current), R'=R (current)
        Rprev = L                  # (i):   R = L'
        (L, R) = (Lprev, Rprev)

    # 3. undo input whitening
    L = L ⊕ W_inL
    R = R ⊕ W_inR

    return join_1024(L, R)
```

**Correctness (round-trip) invariant.** For all keys/tweaks and all `state`,
`P_K_inv(P_K(state)) = state` and `P_K(P_K_inv(state)) = state`. This holds by §13.1 composed
`R_rounds` times plus the whitening XORs (each undone by re-XOR). The reference test suite (§15,
§17) asserts this exhaustively on reduced models and by property test on the full size.

**A note on bijection tests.** Because invertibility is *structural*, a passing bijection test on
a reduced model validates the **implementation** (that `F` is applied identically forward and
back, that endianness and constant derivation match), not a design property. The design-relevant
reduced-model work is differential/linear/algebraic (§16–§17), not bijection confirmation.

---

## 14. Canonical serialization

Serialization is **little-endian** (matching SciRust's determinism-fingerprint convention, §5),
fixed on every platform regardless of host endianness (big-endian hosts MUST byte-swap so output
is host-independent).

### 14.1 Octonion

- An `Oct8` serializes to **64 bytes**: coefficients in index order `c[0], c[1], …, c[7]`, each
  as `LE64` (little-endian `u64`).
- Parsing reads 64 bytes into 8 `u64` little-endian. **No rejection is possible** — every 64-byte
  string is a valid `Oct8` (no canonical-form constraint in `Z/2^64`). (This differs from an `F_p`
  variant, which would reject any coefficient `≥ p`.)

### 14.2 State (1024 bits)

- The state serializes to **128 bytes** = `L (64 bytes) || R (64 bytes)`, each an octonion as
  above. `split_1024` / `join_1024` are the inverse maps.

### 14.3 Program

- As in §9.3: a byte stream of `(opcode [operand])` ending in a single `END`, validated to
  canonical form. `GraphId = SHA256(domain || bytes)`.

### 14.4 Hex formatting (test vectors, logs)

- **Lowercase** hex, `{:02x}` per byte (matches the workspace-wide convention, §5).
- **No `0x` prefix**, **no whitespace**, **no separators** — a contiguous string.
- Byte order within a field follows §14.1–§14.2 (little-endian words). A 64-byte octonion is 128
  hex chars; a 128-byte state is 256 hex chars.
- Keys, tweaks, and round constants are serialized as raw byte strings (the bytes squeezed from
  the XOF, in order) and hex-encoded the same way.

### 14.5 Versioning

- Every domain string embeds `V0.1`. A change to the multiplication table, `λ`/`π`, `R_rounds`,
  the derivation primitive, byte order, or the program schedule is a **breaking** change and MUST
  bump the version (e.g. `V0.2`) and the namespace `SCIRUST-HYPERCRYPTO-V0.2`, invalidating prior
  vectors. `GraphId` and the stored KAT fingerprint (§15) detect accidental drift.

---

## 15. Test-vector format

### 15.1 Record schema

Each vector is a record with these fields (machine-readable form: one JSON object per vector, or
an inline Rust `#[cfg(test)]` table per the workspace convention, §5). Field order is fixed;
hex per §14.4.

```text
algorithm_version        # "SCIRUST-HYPERCRYPTO-V0.1"
coefficient_domain       # "Z/2^64"
state_width              # 1024
round_count              # 24
master_key               # 64 hex chars (32 bytes)
tweak                    # 32 hex chars (16 bytes)
input                    # 256 hex chars (128-byte state, L||R, little-endian)
derived_graph_encoding   # hex of F-PROG bytes  +  GraphId (32-byte SHA-256)
round_keys               # per round r: K_{r,0} || K_{r,1} || K_{r,2}  (3×64 bytes hex)
round_states             # optional trace: (L,R) after each round (for debugging/avalanche)
output                   # 256 hex chars = P_K(input)
inverse_output           # 256 hex chars = P_K_inv(output)  (MUST equal input)
```

### 15.2 Required categories

The reference implementation MUST publish at least one vector in each category:

1. all-zero key and all-zero input;
2. all-one key (`0xff…`) with all-zero input;
3. single-bit input (one bit set) with zero key;
4. single-bit key (one bit set) with zero input;
5. incrementing bytes (`input[i] = i mod 256`; key = `00 01 … 1f`);
6. alternating bit patterns (`0x55…`, `0xaa…`) for input and key;
7. maximum component values (every coefficient `= 0xffffffffffffffff`);
8. inputs chosen to exercise wraparound in `⊞`/`⊗` (e.g. coefficients near `2^64−1`);
9. graph-generation rejection cases (malformed program encodings that MUST be rejected, with the
   expected typed error — see §9.4);
10. forward/inverse round-trip cases (every vector asserts `inverse_output == input`).

### 15.3 What this document computes vs. defers

- **Computed here (hand-verifiable, first-tier multiplication KATs):** the five octonion products
  of §8.4. These validate the multiplication table independently of any implementation and are
  the anchor KATs for the algebra layer.
- **Deferred to the reference implementation:** full `P_K` vectors (categories 1–8, 10) and the
  concrete `GraphId`. These require running HKDF/HMAC-SHA-256 (24 rounds × several squeezes) and
  the 24-round Feistel; they **cannot be produced by hand without risk of error** and are
  therefore intentionally **not** fabricated in this document (per the mission: *do not invent
  numerical vectors without calculating them*). The implementation MUST generate them and store a
  fingerprint (e.g. `SHA256` over the concatenated official vectors), analogous to the Philox
  `0xf96c6b6a_eca699f5` fingerprint contract in `scirust-core` (§5), so cross-implementation drift
  fails a test.

---

## 16. Reduced models

Reduced variants exist to make exhaustive or near-exhaustive analysis feasible while preserving
the structure under study. **Full-size security must never be inferred from a reduced model
surviving an attack; reduced models are for *discovering* weaknesses, not certifying strength.**

### 16.1 Coefficient-width reductions (`Z/2^k`)

Replace `u64` coefficients with `Z/2^k` for `k ∈ {4, 8, 16}` (`wrapping` mod `2^k`). The
multiplication table, `π`, and the Feistel structure are unchanged; `λ` rotation amounts are
taken `mod k`. These preserve: non-commutativity, non-associativity, zero divisors, the
norm/conjugation structure, and the ARX cross-structure — while shrinking the state.

| Variant | coeff width `k` | branch bits | state bits | Feasibility |
|---|---|---|---|---|
| `NANO-2` | 2 | 16 | 32 | Full `2^32` bijection / inverse sweep feasible (~4·10⁹). |
| `NANO-4` | 4 | 32 | 64 | Exhaustive infeasible; large-sample differential/linear; full algebraic (ANF/Gröbner) on 1–2 rounds. |
| `MINI-8` | 8 | 64 | 128 | Sampling-based statistics; SAT/SMT on few rounds. |
| `MINI-16` | 16 | 128 | 256 | Sampling; interpolation-degree probes. |

### 16.2 Structural reductions

- **Fewer rounds:** `R_rounds ∈ {1, 2, 4, 6, 8}` for differential/linear trail search and degree
  growth measurement; compare against the full 24.
- **Shallow graphs:** restrict `F` to one `MUL` (or zero — a purely linear `F`, to *confirm* the
  attack tooling finds the linearity it must).
- **Fewer components:** a quaternion (4-component, associative) drop-in as a control — since
  quaternions are associative, comparing octonion vs. quaternion isolates the contribution (if
  any) of non-associativity to diffusion/resistance.
- **Reduced key size:** 32/64-bit keys to enable key-recovery experiments and meet-in-the-middle.

### 16.3 Discipline

Every reduced-model experiment records: the exact variant parameters, whether it was exhaustive
or sampled (and sample size), the attack applied, and the *observed structural fact* (e.g. "a
1-round degree-2 relation over `NANO-4` links input bit 3 and output bit 17 with probability 1").
Silent truncation of coverage is forbidden — if an experiment samples rather than exhausts, it
says so.

---

## 17. Cryptanalysis plan

The objective is **falsification**: to break v0.1, not to confirm it. Each experiment below is a
concrete attempt to find a distinguisher, an invariant, a low-degree relation, a trail, or a
key-recovery shortcut. **Passing a statistical test is necessary but never sufficient** — a
construction can be statistically random and algebraically broken.

### 17.1 Structural tests

- Bijection tests on `NANO-2` (full `2^32` sweep): confirm `P_K` and `P_K_inv` are exact inverses
  and that `P_K` is a permutation (validates the *implementation*, per §13.2).
- Exhaustive inverse tests on `NANO-2`; sampled round-trip on all larger variants and full size.
- Graph canonicalization tests: every valid program has exactly one encoding; every malformed
  encoding (§9.4) is rejected with the expected error.
- Equivalence-class discovery: search for distinct programs computing the same `F` (program
  equivalences) and for equivalent keys (distinct keys inducing the same `P_K`).
- Repeated-subtree / symmetry detection in generated programs (if the multi-program variant is
  used); detect accidental structure (e.g. rounds that collapse to linear).

### 17.2 Statistical tests (necessary, not sufficient)

- Strict avalanche criterion (SAC) over input, key, and round-constant bit flips.
- Bit-independence criterion (BIC).
- Output-bit balance; input/output correlation.
- Branch-number measurement of `F` and per-round diffusion; **diffusion-by-round** curves.
- Key-bit avalanche and constant-bit avalanche.
- (If the multi-program variant is used) program/graph-bit avalanche.

State explicitly in every report: *these measure diffusion, not security*.

### 17.3 Differential analysis

- XOR-difference distribution over `F` (single round) and full permutation; find high-probability
  differentials.
- Additive (`⊞`) differentials, since half the mixing is over `Z/2^64` (ARX-style).
- Truncated, impossible, and higher-order differentials; boomerang/rectangle where applicable.
- Rotational and word-oriented differentials (the `ROT`/lane structure invites these).
- Related-key and (multi-program variant) graph-related differentials.
- **Associator-driven differentials:** exploit the §7.4 fact that non-associativity carries a
  factor of 2 — search for differentials living in the even sublattice.

### 17.4 Linear and algebraic analysis

- Linear approximations (Matsui); affine-subspace and invariant-subspace search.
- Algebraic degree growth by round (measure ANF degree on reduced variants); interpolation-attack
  degree probes (Jakobsen–Knudsen) — is `P_K` a low-degree polynomial in the key over `Z/2^64`?
- ANF experiments on `NANO-4`/`MINI-8`; Gröbner-basis attacks on 1–3-round systems.
- SAT/SMT modelling of few-round key recovery.
- **Matrix-lifting:** attempt to represent the octonion multiply as a linear/matrix action and
  linearize the round map (the single most likely structural break — octonion left/right
  multiplication *are* `8×8` matrices over `Z/2^64`, so each round is "linear-with-key-dependent-
  matrices"; the security depends entirely on the `GF(2)` cross-structure (`ROT`/`XORC`) defeating
  a global linear model).
- **Invariant experiments:** track the norm `N` and conjugation-based invariants across rounds;
  test whether `ROT`/`XORC` actually destroy norm-multiplicativity (they must — this is a primary
  falsification target). Search for invariant ideals.
- Zero-divisor exploitation: seek inputs steering intermediate `F` values into zero-divisor
  fibers to force detectable structure.

### 17.5 Generic attacks

- Brute force (reduced key sizes); meet-in-the-middle across the Feistel; time-memory tradeoffs.
- Slide attacks (the round constants `RC_r` and per-round subkeys are the defense — test them);
  rotational attacks; biclique-like decompositions.
- Structural graph/program recovery (multi-program variant); equivalent-key search.

### 17.6 Implementation attacks

- Timing / cache / branch-prediction leakage: verify the constant-time coding rules (§18) with
  `dudect`-style statistical timing tests and `cargo asm` inspection of the hot loop.
- Secret-dependent allocation: confirm none (core round path is allocation-free).
- Serialization ambiguity and malformed-graph inputs: fuzz the parser/validator/evaluator (the
  workspace already has a nightly `fuzz/` sub-workspace, §5).
- Fault injection, panic behavior on untrusted input (must be `Result`, never panic), denial of
  service (resource bounds §9.3/§12), and **compiler-optimization differences** (the scalar oracle
  must match optimized builds bit-for-bit across `-O` levels and targets — §5 cross-platform legs).

### 17.7 Kill criteria (a finding that ends v0.1 as-is)

Any of: a practical distinguisher on the full 24 rounds; a global linear/affine representation of
`P_K` (matrix-lifting succeeds); an invariant subspace or norm invariant surviving all rounds; a
key-recovery faster than brute force on full size; a differential/linear trail with usable
probability across 24 rounds; or a demonstrated cross-platform non-determinism. Any such finding
is recorded and triggers a `NO-GO`/redesign (§21).

---

## 18. Constant-time implementation requirements

These are **necessary coding rules**, not a proof of side-channel security (§4). They target
constraints 9–12.

1. **No secret-dependent branches.** No `if`/`match`/loop bound may depend on key, tweak, or
   plaintext. The octonion product's inner `if SIGN[i][j]` (§8.3) branches on **public** table
   entries only — but to be safe it MUST be written branch-free (e.g. select via
   `p.wrapping_mul(sign_as_0_or_neg1_mask)` or add/sub chosen by a public mask), so no branch
   exists at all on that path.
2. **No secret-dependent memory addressing.** No array index may be computed from secret data.
   `PUSH_KEY idx`/`PUSH_CONST idx`/`XORC idx`/`ROT lid`/`PERM pid` indices are **public** (part of
   the public program) — this is the reason v0.1 topology is public (§6.3, §10.5). A key-derived
   program is therefore excluded from the constant-time reference.
3. **Constant-time comparisons.** Any equality check on secret-derived bytes (tags, vector
   compares) uses the `ct_eq` XOR-fold pattern already in the workspace (§5); no early-exit
   `==` on secrets. No `subtle` dependency is added.
4. **No floating point anywhere.** No `f32`/`f64`, no float intrinsics. All arithmetic is `u64`
   (and small `u32` for encodings) wrapping. (The existing `f32` octonion in `scirust-simd` is
   explicitly *not* reused, §5.)
5. **No OS RNG in deterministic paths / tests.** All randomness (for keys/graphs) comes from the
   KDF/XOF over provided key material; tests use fixed inputs. Never call `thread_rng`/`OsRng`/
   `getrandom` in the library (workspace rule, §5).
6. **No `unsafe`.** `#![forbid(unsafe_code)]` in the crate root (compatible with workspace policy;
   the crate does not need `std::simd`'s `unsafe` for the *scalar reference* — a future SIMD
   optimization may live behind a feature and must still avoid `unsafe` or be confined and Miri-
   clean, matching `scirust-simd`'s posture).
7. **No panics on untrusted input.** Parsing/validation/evaluation return typed `Result` errors;
   the core round path is panic-free and allocation-free.
8. **Determinism / portability.** Little-endian serialization enforced regardless of host
   endianness; no platform-dependent behavior; scalar reference semantics (§8.3, §12–§13) are
   authoritative and **every** future SIMD implementation must match them bit-for-bit (validated by
   a differential test between the scalar oracle and any optimized path, and by the x86 ↔ aarch64
   CI legs, §5).
9. **Fixed resource limits** (`MAX_NODES`, `MAX_DEPTH`, `MAX_STACK`, `MAX_GEN_ATTEMPTS`, XOF block
   caps) enforced by the validator; no unbounded work on any input.

---

## 19. Proposed crate architecture

Adapted to SciRust conventions (leaf crate, `publish = false`, MSRV 1.89, `edition 2021`,
`#![forbid(unsafe_code)]`, `std`, rustfmt/clippy-clean). **This crate is NOT to be created by this
phase** — the architecture is proposed for review only.

```text
scirust-hypercrypto/
├── Cargo.toml                 # publish=false; edition 2021; rust-version 1.89; deps: (reuse in-tree
│                              #   sha256/hmac via path deps, or vendor the two files) — no sha3/subtle
├── README.md                  # points to this spec; repeats the experimental-use warning
├── SECURITY.md                # "no production use before independent cryptanalysis" (§22)
├── specs/
│   └── SCIRUST_HYPERCRYPTO_SPEC_V0_1.md   # (this document, if colocated with the crate later)
└── src/
    ├── lib.rs                 # #![forbid(unsafe_code)] ; crate docs restate non-claims (§4)
    ├── algebra/
    │   ├── mod.rs
    │   ├── domain.rs          # Z/2^64 wrapping ops (overflow-explicit helpers)
    │   ├── octonion.rs        # Oct8 type, add/sub/neg/conj/eq/serialize (§8, §14)
    │   └── multiplication_table.rs   # SIGN[8][8], IDX[8][8]; the authoritative ⊗ (§8.3)
    ├── graph/
    │   ├── mod.rs
    │   ├── instruction.rs     # opcodes, operands (§9.2)
    │   ├── program.rs         # canonical encoding, GraphId (§9.3, §14.3)
    │   ├── validator.rs       # total, panic-free validation + typed errors (§9.4)
    │   └── evaluator.rs       # postfix stack machine (§9.1)
    ├── derivation/
    │   ├── mod.rs
    │   ├── domains.rs         # domain strings incl. version (§8 domains, §10.3)
    │   ├── graph.rs           # public program schedule / bounded generator (§10.5)
    │   └── round_keys.rs      # HKDF-SHA-256 + HMAC XOF; subkeys, constants, whitening (§10)
    ├── permutation/
    │   ├── mod.rs
    │   ├── state.rs           # 1024-bit state, split/join_1024 (§14.2)
    │   ├── round.rs           # F_r (F-PROG), λ, π, RC (§12.2)
    │   ├── forward.rs         # P_K (§12.3)
    │   └── inverse.rs         # P_K_inv (§13.2)
    ├── analysis/
    │   ├── mod.rs
    │   ├── avalanche.rs       # SAC/BIC/diffusion-by-round (§17.2)
    │   ├── differential.rs    # difference distributions (§17.3)
    │   ├── algebraic.rs       # ANF/degree/norm-invariant probes (§17.4)
    │   └── reduced.rs         # Z/2^k reduced models (§16)
    └── test_vectors.rs        # inline KAT tables (§15) + stored fingerprint contract
```

**Dependency note:** prefer path-depending on (or vendoring) the two existing pure files —
`scirust-sciagent/src/sha256.rs` and `scirust-discovery/src/hmac.rs` — over adding `sha2`/`sha3`/
`hmac`/`hkdf`/`subtle`, to keep the supply chain minimal (`deny.toml`, `SECURITY.md`). The exact
reuse mechanism (extract a shared `scirust-hash` leaf crate vs. vendor) is an open question (§20).

---

## 20. Open research questions

1. **Round count.** Is 24 conservative, excessive, or insufficient? Determine minimal secure
   rounds from measured degree growth and best trails on reduced models; `R_rounds` is a
   placeholder until then.
2. **Matrix-lifting resistance (the central risk).** Octonion left/right multiplication are `8×8`
   matrices over `Z/2^64`; each round is "linear with key-dependent matrices" plus `ROT`/`XORC`.
   Does the `GF(2)`/`Z/2^64` cross-structure actually prevent a global linear/affine model? If not,
   v0.1 collapses.
3. **Norm and invariant survival.** Do `ROT`/`XORC` provably destroy norm-multiplicativity and all
   conjugation-based invariants across rounds? Quantify.
4. **Diffusion constants `λ`, `π`.** The current values are placeholders. Optimize for branch
   number / diffusion; should they vary per round (from public data) rather than being fixed?
5. **Multi-program vs. single program.** Does a public-derived per-round program improve resistance
   over the fixed `F-PROG`, or only complicate analysis?
6. **Coefficient domain.** Would `F_p` (Goldilocks) give a cleaner security argument despite the
   higher algebraic-attack risk and harder constant-time reduction? (Recorded alternative, §7.)
7. **Derivation primitive.** Swap HKDF/HMAC-SHA-256 for SHAKE256 (FIPS 202) if a reviewed pure-Rust
   Keccak becomes a workspace dependency; the abstract interface is unchanged.
8. **Key-derived topology.** Can a key-derived program ever be reconciled with constant-time (e.g.
   fully-oblivious evaluation), or is it fundamentally a non-constant-time research-only mode?
9. **Width.** Is 512-bit (single octonion, non-Feistel reversible structure) or a `u32`-coefficient
   512-bit two-branch Feistel a better cost/security point than 1024-bit?
10. **State-width vs. key size.** A 256-bit key on a 1024-bit permutation — is a larger key
    warranted, or does it invite related-key structure?
11. **Reuse mechanism** for the in-tree SHA-256/HMAC (shared leaf crate vs. vendor) without
    duplicating code or widening the dependency graph.
12. **`no_std`.** Should the crate be the workspace's first `no_std` member (embedded targets)?

---

## 21. Go / No-go criteria for implementation

### GO for a reference implementation — allowed only if **all** hold:

- [x] **All arithmetic semantics are unambiguous** — `Z/2^64` wrapping; the full signed
  multiplication table and general formula (§8.3) fix every product to the bit.
- [x] **The round transformation has a valid inverse argument** — the Feistel inverse is *derived*
  (§13.1), and it uses `F` without inverting it; the only inverted op is the XOR involution.
- [x] **Graph encoding is canonical** — one encoding per program; total validator; `GraphId`
  (§9.3, §14.3).
- [x] **Graph generation is deterministic and bounded** — v0.1 uses a fixed public program;
  optional generation is public-input-only, rejection-sampled, bounded by `MAX_GEN_ATTEMPTS` with
  deterministic fallback (§10.5).
- [x] **Test-vector format is complete** — schema and all 10 categories specified (§15); anchor
  multiplication KATs computed exactly (§8.4).
- [x] **No operation relies on floating point** — integer-only throughout (§18).
- [x] **No known immediate structural collapse identified** — the leading risk (matrix-lifting) is
  named and made a first-order falsification target, but no break is known at spec time.
- [x] **Implementable without secret-dependent control flow** — public topology (§6.3), branch-free
  product (§18), public operand indices.
- [x] **Reduced variants can be exhaustively analyzed** — `NANO-2` gives a full `2^32` sweep;
  `Z/2^k` family defined (§16).

All GO criteria are met **for producing a reference implementation whose purpose is
cryptanalysis** — not for any security claim or use.

### NO-GO (redesign required) if **any** hold:

- [ ] The construction compresses state on a supposedly reversible path — *avoided*: the reversible
  path is XOR-only; compression is confined to the non-inverted `F` (§11, §13).
- [ ] Invertibility depends on arbitrary octonion elements being invertible — *avoided*: no octonion
  is ever inverted (§13.1).
- [ ] Coefficient-domain properties are incorrectly inherited from real octonions — *avoided*: §7.4
  enumerates exactly what fails (division-algebra property, invertibility, anisotropy) and what
  survives (and is treated as attack surface).
- [ ] Graph generation is biased or unbounded — *avoided*: fixed public program in v0.1; any
  sampling is rejection-based and bounded (§10.5).
- [ ] Multiple encodings represent one program without canonicalization — *avoided*: §9.3.
- [ ] The security argument reduces to topology counting — *avoided*: topology is public and
  contributes **zero** claimed entropy (§4.1, §6.3).
- [ ] An obvious linearization or invariant breaks diffusion — **open, must be tested first**
  (§17.4, §20 Q2–Q3). *This is the one criterion the specification cannot discharge on paper;* the
  first implementation milestone is precisely the matrix-lifting / invariant probe on reduced
  models. If it succeeds, v0.1 is NO-GO and must be redesigned before any further use.
- [ ] Deterministic cross-platform semantics cannot be guaranteed — *avoided*: integer-only, fixed
  little-endian, no floats, CI cross-platform legs (§5, §18).

**Net recommendation:** **GO to build the reference implementation and the reduced-model analysis
harness**, with the explicit, gating milestone that the matrix-lifting and norm-invariant
experiments (§17.4) run *first*; a positive break there is a hard NO-GO for the v0.1 design.

---

## 22. Experimental-use warning

```text
EXPERIMENTAL RESEARCH CONSTRUCTION

This design has not received independent cryptanalysis.
It must not be used to protect real data, credentials, financial records,
health information, production secrets, or communication systems.

Use established and standardized cryptographic primitives for production.
```

- **No custom construction may replace established cryptography in SciRust production workflows.**
  For real protection, use standardized, widely-reviewed primitives (e.g. AES-GCM/ChaCha20-Poly1305
  for AEAD, SHA-2/SHA-3 for hashing, HKDF for key derivation, and the NIST PQC standards ML-KEM /
  ML-DSA / SLH-DSA for post-quantum needs). SciRust already confines real integrity to
  HMAC-SHA-256 (§5); this experiment does not change that.
- The hypercomplex permutation may **later** be studied — never described as secure until
  supported by analysis — as: an internal permutation candidate; a keyed mixing function; a
  domain-separated transcript mixer; an anti-tamper research primitive; or a white-box /
  obfuscation experiment. Each such use is gated on independent cryptanalysis.
- **No production use before independent cryptanalysis. No benchmark-based security claims.**

---

## 23. References

Primary/credible sources, grouped as requested. Entries marked **[verify]** are cited from memory
and should be confirmed (exact title/venue/year/authors) before the reference implementation
publishes them; they are **not** fabricated placeholders and none is cited as evidence of this
construction's security.

### 23.1 Octonions and non-associative algebra
- J. C. Baez, "The Octonions," *Bulletin of the American Mathematical Society* 39(2), 145–205, 2002.
- J. H. Conway and D. A. Smith, *On Quaternions and Octonions*, A K Peters, 2003.
- R. D. Schafer, *An Introduction to Nonassociative Algebras*, Academic Press, 1966. **[verify]**
- T. Y. Lam, *Introduction to Quadratic Forms over Fields*, AMS GSM 67, 2005 (isotropy of forms over finite fields; split composition algebras). **[verify]**
- L. E. Dickson, "On Quaternions and Their Generalization and the History of the Eight Square Theorem," *Annals of Mathematics*, 1919 (Cayley–Dickson doubling, Degen's eight-square identity). **[verify]**

### 23.2 Reversible constructions and Feistel networks
- H. Feistel, "Cryptography and Computer Privacy," *Scientific American* 228(5), 1973.
- M. Luby and C. Rackoff, "How to Construct Pseudorandom Permutations from Pseudorandom Functions," *SIAM Journal on Computing* 17(2), 1988.
- S. Even and Y. Mansour, "A Construction of a Cipher from a Single Pseudorandom Permutation," *Journal of Cryptology* 10(3), 1997.
- X. Lai and J. L. Massey, "A Proposal for a New Block Encryption Standard" (IDEA / Lai–Massey), *EUROCRYPT* 1990.
- K. Nyberg, "Generalized Feistel Networks," *ASIACRYPT* 1996. **[verify]**
- Y. Zheng, T. Matsumoto, H. Imai, "On the Construction of Block Ciphers Provably Secure and Not Relying on Any Unproved Hypotheses," *CRYPTO* 1989 (generalized Feistel types). **[verify]**

### 23.3 Modern cryptographic design methodology
- J. Daemen and V. Rijmen, *The Design of Rijndael*, Springer, 2002 (wide-trail strategy).
- E. Biham and A. Shamir, *Differential Cryptanalysis of the Data Encryption Standard*, Springer, 1993.
- M. Matsui, "Linear Cryptanalysis Method for DES Cipher," *EUROCRYPT* 1993.
- L. R. Knudsen, "Truncated and Higher Order Differentials," *FSE* 1994.
- T. Jakobsen and L. R. Knudsen, "The Interpolation Attack on Block Ciphers," *FSE* 1997.
- G. Leander, M. A. Abdelraheem, H. AlKhzaimi, E. Zenner, "A Cryptanalysis of PRINTcipher: The Invariant Subspace Attack," *CRYPTO* 2011.
- N. Courtois and J. Pieprzyk, "Cryptanalysis of Block Ciphers with Overdefined Systems of Equations," *ASIACRYPT* 2002.

### 23.4 Constant-time implementation
- P. Kocher, "Timing Attacks on Implementations of Diffie-Hellman, RSA, DSS, and Other Systems," *CRYPTO* 1996.
- D. J. Bernstein, "Cache-Timing Attacks on AES," technical report, 2005.
- J. B. Almeida, M. Barbosa, G. Barthe, F. Dupressoir, M. Emmi, "Verifying Constant-Time Implementations," *USENIX Security* 2016.
- O. Reparaz, J. Balasch, I. Verbauwhede, "Dude, is my code constant time?" (dudect), *DATE* 2017. **[verify]**
- NIST FIPS 202, *SHA-3 Standard: Permutation-Based Hash and Extendable-Output Functions*, 2015 (SHAKE, the recorded XOF alternative).
- H. Krawczyk and P. Eronen, RFC 5869, *HMAC-based Extract-and-Expand Key Derivation Function (HKDF)*, 2010.
- NIST FIPS 198-1 (HMAC) and FIPS 180-4 (SHA-256), 2008 / 2015.

### 23.5 Non-commutative and non-associative cryptanalysis
- I. Anshel, M. Anshel, D. Goldfeld, "An Algebraic Method for Public-Key Cryptography," *Mathematical Research Letters*, 1999 (non-commutative/braid-group proposal). **[verify]**
- A. Myasnikov, V. Shpilrain, A. Ushakov, *Group-Based Cryptography*, Birkhäuser, 2008 (surveys attacks — length-based, linear representation — that broke several non-commutative proposals). **[verify]**
- V. Shpilrain and A. Ushakov, "The Conjugacy Search Problem in Public Key Cryptography: Unnecessary and Insufficient," *Applicable Algebra in Engineering, Communication and Computing*, 2006. **[verify]**
- (General caution) Multiple proposed non-commutative/non-associative schemes have been broken by linear/matrix representation and algebraic methods; cited here as evidence that these structural properties are **not** self-justifying (§4.1), not as support for this design.

### 23.6 Post-quantum standardization
- P. W. Shor, "Polynomial-Time Algorithms for Prime Factorization and Discrete Logarithms on a Quantum Computer," *SIAM Journal on Computing* 26(5), 1997.
- L. K. Grover, "A Fast Quantum Mechanical Algorithm for Database Search," *STOC* 1996.
- NIST FIPS 203, *Module-Lattice-Based Key-Encapsulation Mechanism (ML-KEM)*, 2024.
- NIST FIPS 204, *Module-Lattice-Based Digital Signature Standard (ML-DSA)*, 2024.
- NIST FIPS 205, *Stateless Hash-Based Digital Signature Standard (SLH-DSA)*, 2024.

*No informal or marketing claims are cited as evidence of security. No standard, author, date, or
URL has been fabricated; uncertain bibliographic details are flagged **[verify]** for confirmation
prior to any publication of the reference implementation.*

---

*End of SCIRUST_HYPERCRYPTO_SPEC_V0_1 (draft).*
