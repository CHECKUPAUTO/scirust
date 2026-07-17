# SciRust-HyperCrypto — Phase 1 Structural Falsification Report

> ```text
> EXPERIMENTAL RESEARCH CONSTRUCTION
>
> This report documents an adversarial attempt to BREAK the SciRust-HyperCrypto
> v0.1 keyed-permutation candidate. It is not a cipher, hash, KEM, or signature,
> has not received independent cryptanalysis, and must not protect real data.
> A "CONTINUE" verdict is NOT evidence of security.
> ```

**Phase-1 gating question.** *Can the v0.1 round function or reduced permutation
be represented, approximated, or structurally decomposed as an unexpectedly
simple linear, affine, matrix, norm-preserving, invariant-subspace, or
zero-divisor-driven system?*

The authoritative definition of every constant and transformation exercised here
is the merged specification
[`SCIRUST_HYPERCRYPTO_SPEC_V0_1.md`](./SCIRUST_HYPERCRYPTO_SPEC_V0_1.md). Where
code and spec disagree, the spec wins and the disagreement is a bug. No
internal contradiction in the specification was found while implementing Phase 1.

---

## 1. Exact commit and environment

| Item | Value |
|---|---|
| Crate | `scirust-hypercrypto` (workspace member; `publish = false`) |
| Base commit (spec merge) | `16245e1ef3ddf58cb5c514e2a6bea8adc963c7d5` |
| Phase-1 code commit | the head commit of this PR branch (`claude/scirust-hypercrypto-spec-jh6ht8`) |
| Toolchain (pinned) | `nightly-2026-07-02` → `rustc 1.98.0-nightly (4c9d2bfe4 2026-07-01)` |
| MSRV verified | `rustc 1.89.0 (29483883e 2025-08-04)` |
| Platform | Linux `6.18.5` x86_64 |
| Dependencies | `sha2 = "0.10"` (result-file fingerprints and the spec's `GraphId` only — not used in any keyed construction) |
| `unsafe` | none (`#![forbid(unsafe_code)]`) |
| Floating point | none (integer-only over `Z/2^k`) |
| OS entropy | none (all randomness is deterministic SplitMix64 over provided seeds) |

The reference implementation is pure Rust, integer-only, scalar (no SIMD),
deterministic, and platform-independent. Machine-readable results are written to
`target/hypercrypto-falsification/` (git-ignored) with a SHA-256 fingerprint per
file.

---

## 2. Code architecture

```text
scirust-hypercrypto/
├── src/
│   ├── lib.rs                      # #![forbid(unsafe_code)] + experimental banner
│   ├── algebra/
│   │   ├── word.rs                 # sealed Word trait; W2/W4/W8/W16/W64 over Z/2^k
│   │   ├── table.rs                # authoritative IDX/SIGN + independent triple oracle
│   │   ├── octonion.rs             # Oct<W>: 64-term MUL oracle, conj, norm, ROT_λ, PERM_π
│   │   └── quaternion.rs           # Quat<W> (associative) — Control D
│   ├── fixtures.rs                 # deterministic experimental round material (NOT keys)
│   ├── permutation/
│   │   ├── round.rs                # exact v0.1 F-PROG + pre-rotation map G
│   │   ├── feistel.rs              # balanced Feistel forward/inverse + trace
│   │   └── controls.rs             # Controls A/B/C (octonion) + D (quaternion)
│   └── analysis/
│       ├── modmatrix.rs            # 8×8 over Z/2^k: det mod 2^k, GF(2) rank, 2-adic SNF
│       ├── matrix_lifting.rs       # Experiment 1
│       ├── linearity.rs            # Experiments 2–3 (ring-affine / GF(2) / bit-affine)
│       ├── degree.rs               # Experiment 4 (exact ANF via Möbius)
│       ├── invariants.rs           # Experiment 5 (norm / conjugation / associator)
│       ├── zero_divisors.rs        # Experiment 6
│       ├── subspace.rs             # Experiment 7
│       ├── battery.rs              # orchestration + verdict
│       ├── report.rs               # ordered-JSON writer + SHA-256 fingerprints
│       └── util.rs                 # deterministic sampling / exhaustive enumeration
├── src/bin/hypercrypto-falsify.rs  # research CLI
└── tests/                          # algebra KATs, inverse, controls, reduced properties
```

Two independent multiplication oracles exist: the authoritative hardcoded
`IDX`/`SIGN` table (spec §8.3(b)) and a generator that rebuilds the table from
the seven Fano triples (spec §8.3(a)). A test asserts they agree, so a
transcription typo cannot pass silently.

---

## 3. Authoritative arithmetic checks

All pass (`tests/algebra_kats.rs`, exact over `Z/2^64` unless noted).

- **The five specification KATs (spec §8.4)** reproduced bit-for-bit:
  `e1⊗e2 = e4`, `e2⊗e1 = −e4`; the Fano-line associative case
  `(e1⊗e2)⊗e4 = e1⊗(e2⊗e4) = −e0`; the non-associative case
  `(e1⊗e2)⊗e3 = −e6 ≠ e6 = e1⊗(e2⊗e3)` with associator `−2·e6`; and the two
  general products `(e1+e2)⊗(e2+e3)` and its reverse.
- **All 64 basis products** equal the independent triple-derived oracle.
- **Antisymmetry** `e_i·e_j = −(e_j·e_i)` for distinct imaginary units; **squares**
  `e_i² = −e0`; **identity** `e0`.
- **Conjugation**: `conj(conj(x)) = x`, `x + x̄ = 2x_0`, and `x⊗x̄ = N(x)·e0` (scalar).
- **Alternativity** `(x⊗x)⊗y = x⊗(x⊗y)`, `(y⊗x)⊗x = y⊗(x⊗x)` and **flexibility**
  `(x⊗y)⊗x = x⊗(y⊗x)` hold on random octonions — confirming the table defines a
  genuine (alternative) octonion algebra.
- **Norm multiplicativity** `N(x⊗y) = N(x)·N(y)` (Degen's eight-square identity)
  holds exactly.

**Forward/inverse round-trip** (`tests/inverse.rs`): `P_K^{-1}∘P_K = id` on every
coefficient width `k ∈ {2,4,8,16,64}`, for all six fixtures, at 24 rounds; plus an
exhaustive `2^16` projection sweep at NANO-2. No forward/inverse mismatch was
found. *This establishes implementation correctness, not security.*

---

## 4. Control-variant validation (is the harness trustworthy?)

The analysis harness must break the deliberately-weakened controls, or Phase 1 is
inconclusive. It does:

| Control | Definition | Detector result |
|---|---|---|
| **A** — linear-only | `F_A(R) = PERM_π(R ⊞ K0)` | **recovered** as ring-affine over `Z/2^k` (exhaustive at NANO-2, sampled at MINI-8); exact ANF degree **1** for every key. |
| **B** — ring-linear | `F_B(R) = (K1 ⊗ R) ⊗ K2` | **recovered** as ring-**linear** (zero offset); with odd-norm `K1,K2` the recovered `8×8` matrix is invertible over `Z/2^k`. |
| **C** — one-multiply | `F_C(R) = PERM_π(ROT_λ(K1 ⊗ R)) ⊕ RC` | used for degree comparison (degree 2 at NANO-2). |
| **D** — quaternion | associative 4-component analogue | used for structural comparison (degree 2 at NANO-2). |

Because Controls A and B are recovered at both NANO-2 (exhaustive) and MINI-8
(sampled), the harness is considered trustworthy for the negative results below.
A **negative sanity check** confirms the real round is *not* recovered by either
model.

---

## 5. Left/right multiplication matrix results (Experiment 1)

For a fixed `a`, `L_a(x) = a⊗x` and `R_a(x) = x⊗a` are linear over `Z/2^k`. The
lifted `8×8` matrices **matched the octonion oracle on every tested input**
(exhaustive `2^16` at NANO-2). Invertibility over `Z/2^k` tracks the norm exactly:
`a` is invertible **iff** `N(a)` is odd (a unit).

NANO-2 (`Z/2^2`), `L_a`, exhaustive:

| multiplier `a` | `N(a)` | `det mod 2^k` | invertible (ring) | GF(2) rank | `log2 |ker|` | matrix = oracle |
|---|---|---|---|---|---|---|
| odd-norm `K1` | 3 | 1 | **yes** | 8 | 0 | ✓ |
| even-norm `K1` | 2 | 0 | no | 4 | 4 | ✓ |
| `k1` (default) | 2 | 0 | no | 4 | 4 | ✓ |
| `k2` (default) | 2 | 0 | no | 4 | 4 | ✓ |

The `log2 |ker|` values come from an exact **2-adic Smith normal form** over
`Z/2^k` (validated against brute-force enumeration at NANO-2), *not* from a GF(2)
rank — the two are reported separately per the spec's warning. Even-norm
multipliers are genuine zero-divisor-adjacent maps with large kernels.

---

## 6. Affine decomposition of the pre-rotation map (Experiment 2)

`G(x) = (K1 ⊗ (x ⊞ K0)) ⊗ K2` is recovered as **exactly ring-affine** over `Z/2^k`
— `G(x) = A·x ⊞ b` with `A = M_R(K2)·M_L(K1)` — verified exhaustively at NANO-2
and on large samples at MINI-8. This is expected and is **documented attack
surface, not a break**: everything up to and including the two multiplies and the
key addition is affine over `Z/2^k`. The construction's non-linearity therefore
rests entirely on the layers that follow — `ROT_λ` (bitwise, GF(2)-linear but
`Z/2^k`-nonlinear), `PERM_π`, and `XORC` — interacting with the ring arithmetic.

---

## 7. Full-round affinity results (Experiment 3)

The full v0.1 round `F` is **not** captured by any simple global model:

| model | NANO-2 (exhaustive) | MINI-8 (sampled) |
|---|---|---|
| ring-affine over `Z/2^k` | **no** | **no** |
| GF(2)-affine | **no** | **no** |
| exact GF(2) bit-affine recovery | disagrees on ~86% of inputs | disagrees on ~100% |
| multi-round (`r=2,4`) ring-affine | **no** | **no** |

**FINDING — key-dependent weak-key linearization (documented, NOT a kill).** The
round's GF(2)-nonlinearity is *key-dependent*. A structural scan over 8 fixtures
shows the round is **not** affine for all keys (so no construction-level break),
but it **is** GF(2)-affine for a class of weak keys:

- **high-bit-only multipliers**: an all-`2^{k-1}` multiplier octonion makes the
  product GF(2)-linear, because `2^{k-1}·x ≡ 2^{k-1}·(x mod 2)`. With such `K1,K2`
  the whole round collapses to a GF(2)-affine map at any width.
- **zero multipliers**: degenerate, the round becomes constant/affine.

Realistic pseudo-random, odd-norm, even-norm, and incrementing keys keep the round
nonlinear. This is a genuine lead for **Phase-2 weak-key / related-key analysis**
and a constraint on the eventual key schedule (it must avoid degenerate
multipliers). Because affinity does not hold for all keys, it does **not** trigger
the verdict gate.

---

## 8. Algebraic-degree results (Experiment 4)

Exact ANF via the Möbius transform, at NANO-2 (16 input bits; larger widths are
out of exact range and are reported as such — no `2^64` truth table).

- Single-round `F` (default key): max degree **2**.
- Feistel branch degree by round: **2 → 3 → 6 → 8** for `r = 1,2,3,4` — degree
  grows quickly and saturates the 8-bit branch by round 4.
- v0.1 vs controls (single round, NANO-2): v0.1 = 2, Control A = **1**
  (GF(2)-affine), Control B = 2, Control C = 2, Control D (quaternion) = 2. The
  degree tool cleanly separates the linear control from the nonlinear ones; the
  octonion vs. quaternion degrees are equal here and **no security claim** is drawn
  from any difference.

---

## 9. Norm and conjugation results (Experiment 5)

| relation | result |
|---|---|
| `N(a⊗x) = N(a)·N(x)` | **holds** (multiplicative norm survives every `⊗` — attack surface) |
| `N(conj(x)) = N(x)` | holds |
| `N(PERM_π(x)) = N(x)` | holds |
| `N(ROT_λ(x)) = N(x)` | **fails** — bit-rotation destroys the norm |
| `N(x)` determines `N(F(x))` | **no** (explicit refutation witness found) |
| `N(x)` determines `N(P_8(x))` (8-round branch) | **no** |
| associator `∈ 2·R^8` (all-even coefficients) | **holds** (e.g. `[0,0,2,2,0,2,2,0]` at NANO-2) |

The multiplicative norm and the even-associator are real invariants of the
*algebra* and are flagged as attack surface. Crucially, **no norm-derived
invariant survives the full round or the multi-round permutation** — `ROT_λ` and
`XORC` break norm-tracking, so the norm kill-criterion is **not** triggered. The
associator's factor of 2 (spec §7.4) is confirmed and is a concrete Phase-2
differential lead (the even sublattice).

---

## 10. Zero-divisor examples and fiber sizes (Experiment 6)

The finite octonion algebra is split, so zero divisors exist. Explicit NANO-2
examples (exhaustively found, `a⊗b = 0`, `a,b ≠ 0`):

| `a` | `b` | `N(a)` |
|---|---|---|
| `2·e0` | `2·e0` | 0 (`2·2 = 0 mod 4`) |
| `e0+e1` | `2e0+2e1` | 2 |

Collision-kernel census over sampled multipliers: at MINI-8, **9 of 16** non-unit
multipliers have a non-trivial left/right kernel, with `log2 |ker|` up to **20**
(out of 64 state bits). Differential probe of `F` under `Δ = e1`: the
most-frequent output difference occurs with frequency ~50% at NANO-2 (a
single-round, 2-bit-width artifact) but drops to **150 ppm** at MINI-8 — i.e. the
tiny-width bias is a coverage artifact, **not** a full-round differential. As the
spec notes, zero divisors may bias the *round function* but do not break the outer
Feistel permutation, which remains structurally invertible (Section 3).

---

## 11. Subspace and low-bit findings (Experiment 7)

The round function `F` **breaks** every structured input set tested (exhaustive at
NANO-2 where feasible):

| structured set | preserved by `F`? |
|---|---|
| even sublattice `2·R^8` | no |
| scalar-only inputs | no |
| imaginary-only inputs | no |
| Fano-line subalgebra `{e0,e1,e2,e4}` | no |

Fixed-point / short-cycle census of the 4-round reduced permutation (sampled,
20k states at NANO-2 and MINI-8): **0 fixed points, 0 two-cycles**. No invariant
subspace or low-bit projection survived the round.

---

## 12. Reduced-round findings

At NANO-2, degree growth and round-trip integrity by round count:

| rounds | branch max degree (exact ANF) | forward/inverse round-trip |
|---|---|---|
| 1 | 2 | ✓ |
| 2 | 3 | ✓ |
| 4 | 8 | ✓ |
| 6 | 8 | ✓ |
| 8 | 8 | ✓ |

Degree saturates the branch by ~4 rounds; the 24-round full construction has a
large margin over the point where these exact-ANF probes lose resolution.

---

## 13. Limitations and incomplete experiments

- **Exact ANF is limited to ≤ 18 input bits** — i.e. NANO-2 single-octonion
  functions. Degree at MINI-8+ and the full 1024-bit state is out of exact range
  and is *not* estimated (a sampled degree would be mislabeled science).
- **Full-state exhaustion is only feasible at NANO-2 (`2^32`)** and is gated behind
  the explicit `exhaustive-nano2` command (disabled by default). The report battery
  exhausts the single-octonion domain (`2^16`) and samples the state.
- **No Gröbner-basis, SAT/SMT, interpolation, or higher-order/boomerang differential
  attacks** were run — these are Phase-2+ scope. Phase 1 targets the *most obvious*
  structure (linear/affine/matrix/norm/zero-divisor), not the full cryptanalytic
  toolkit.
- **Differential analysis here is a single-difference bias probe**, not a full
  differential-trail search.
- The `bit-affine` recovery verifies exhaustively only at NANO-2 (16 bits) and
  samples at MINI-8; MINI-16+ are out of range for that detector.
- Statistical randomness was not the objective and is not claimed; SAC/BIC batteries
  are deferred.

None of these gaps is required to answer the Phase-1 gating question, which
concerns simple structural collapse.

---

## 14. Exact reproducibility commands

```bash
# from the workspace root; set the commit for result metadata
export HYPERCRYPTO_GIT_COMMIT=$(git rev-parse HEAD)

# gates
cargo +nightly-2026-07-02 fmt -p scirust-hypercrypto -- --check
cargo +nightly-2026-07-02 clippy -p scirust-hypercrypto --all-targets --locked -- -D warnings
cargo +nightly-2026-07-02 test  -p scirust-hypercrypto --locked
cargo +1.89.0             check -p scirust-hypercrypto --all-targets --locked

# full battery + verdict (writes target/hypercrypto-falsification/phase1-report.json)
cargo +nightly-2026-07-02 run --release -p scirust-hypercrypto \
  --bin hypercrypto-falsify -- report --sample 80000

# focused experiments
cargo run --release -p scirust-hypercrypto --bin hypercrypto-falsify -- controls --width nano2
cargo run --release -p scirust-hypercrypto --bin hypercrypto-falsify -- matrix-lifting --width nano2
cargo run --release -p scirust-hypercrypto --bin hypercrypto-falsify -- affinity --width nano2
cargo run --release -p scirust-hypercrypto --bin hypercrypto-falsify -- degree
cargo run --release -p scirust-hypercrypto --bin hypercrypto-falsify -- invariants --width mini8
cargo run --release -p scirust-hypercrypto --bin hypercrypto-falsify -- zero-divisors --width mini8
cargo run --release -p scirust-hypercrypto --bin hypercrypto-falsify -- reduced-rounds

# EXPLICIT, COSTLY, disabled-by-default full 2^32 NANO-2 sweep (or --limit N for a prefix)
cargo run --release -p scirust-hypercrypto --bin hypercrypto-falsify -- \
  exhaustive-nano2 --rounds 1 --limit 200000     # add --full for the complete 2^32 sweep
```

The battery output is deterministic (no OS entropy, no wall-clock): repeated runs
with the same commit and seed produce byte-identical JSON and identical
fingerprints (`tests/reduced_properties.rs::machine_output_is_deterministic`).

---

## 15. Machine-readable output fingerprints

Result documents live under `target/hypercrypto-falsification/` (git-ignored),
each with a `.json.sha256` sidecar. Representative fingerprints from the
`--sample 80000` run at base commit `16245e1`:

| file | SHA-256 |
|---|---|
| `phase1-report.json` | `d16903323da4fc8e01b3c4cd2309c19f525ddb363a88be13e1c17d2eb3002e6d` |
| `exhaustive-nano2.json` (limit 200000, rounds 1) | fingerprint of `nano2-sweep:200000:<acc>` = `d5f3ed…` |

(The `phase1-report.json` fingerprint embeds `git_commit`; recomputing it on a
different commit yields a different value by design. The per-experiment structural
results are commit-independent.)

The fixed `F-PROG` `GraphId` (spec §12.2),
`SHA256("SCIRUST-HYPERCRYPTO-V0.1/GRAPH-ID" || F-PROG-bytes)`, is computed by
`analysis::report::f_prog_graph_id()` and pinned by a unit test.

---

## 16. Triggered kill criteria

**None.** Specifically, none of the following occurred:

- no exact global affine representation of the full or multi-round permutation;
- no invariant (norm, subspace, low-bit, associator-derived) survived the full
  round or the reduced permutation;
- no norm/valuation relation yielded a deterministic full-round distinguisher;
- no zero-divisor structure produced a practical full-round differential;
- no forward/inverse mismatch, and no deterministic-semantics failure;
- no specification contradiction invalidating the claimed permutation.

The weak-key GF(2)-linearization (Section 7) is a documented *observation*, not a
kill: it does not hold for general keys and is exactly the kind of lead Phase 1
exists to surface.

---

## 17. Final Phase-1 verdict

```text
PHASE-1 VERDICT: CONTINUE — NO GATING BREAK FOUND BY THESE EXPERIMENTS
```

`CONTINUE` means only that the *most obvious* algebraic structure did not collapse
the construction under the Phase-1 experiments. **It is not evidence of security.**
The v0.1 design remains an experimental research construction with no
cryptanalysis and no production use.

### Recommended next actions (Phase 2)

1. **Weak-key / related-key analysis** — characterize the full class of keys that
   linearize the round (beyond high-bit-only multipliers) and constrain the key
   schedule to exclude them.
2. **Differential-trail search** — replace the single-difference bias probe with a
   proper trail search, exploiting the even-associator (factor-of-2) structure and
   the ARX/ring cross-structure.
3. **Algebraic attacks** — Gröbner-basis and SAT/SMT modeling of few-round
   key-recovery at NANO-2/NANO-4; interpolation-degree probes.
4. **Norm/ideal invariants mod 2^t** — push the invariant search to the `2`, `4`,
   `8`-adic filtration rather than the full modulus.
5. **Diffusion constants** — the placeholder `λ`, `π` should be optimized for branch
   number; measure whether the weak-key linearization interacts with them.

---

## 18. Phase-2 increment — weak-key class and differential probing

Two Phase-2 experiments have been implemented and are exposed as the CLI
subcommands `weak-keys` and `differential` (module `analysis::differential`).
Neither changes the Phase-1 verdict; both sharpen the Phase-1 leads.

### 18.1 The GF(2)-linearizing weak-key class is fully characterized

Phase 1 observed that *some* keys linearize the round over `GF(2)`. This is now
pinned down. Let `C = { octonions whose every coefficient is in {0, 2^{k-1}} }`
(the "high-bit-only" multipliers), `|C| = 2^8 = 256`.

- **Every** member of `C` yields a `GF(2)`-linear left-multiplication
  `x ↦ a ⊗ x` (measured exhaustively/sampled at MINI-8: `256/256` linear;
  `all-2^{k-1}` multiplier linear = true).
- **`0` of `512`** random multipliers are `GF(2)`-linear — the class is a
  vanishing-density structured set.
- **Algebraic reason:** `2^{k-1}·b ≡ 2^{k-1}·(b mod 2)` keeps only the low bit,
  and `2^{k-1} + 2^{k-1} ≡ 0 (mod 2^k)`, so every octonion output slot collapses
  to a `GF(2)` **parity** of the routed input bits — a linear map, independent of
  the `±1` structure constants.
- **Consequence (key-schedule constraint, not a break):** the eventual key
  schedule must exclude multipliers with all coefficients in `{0, 2^{k-1}}` (and,
  more conservatively, avoid low-Hamming-weight/degenerate multipliers). Because
  the class has density `≈ 2^{-56}` at `k = 8` (and far lower at `k = 64`), a
  well-distributed HKDF schedule (§next) essentially never hits it, but it must
  be excluded explicitly.

### 18.2 Differential probing shows no high-probability full-round differential

- **Best single-round differential** (MINI-8, sampled input differences, exact
  per-difference output distribution at NANO-2): the most probable output
  difference occurs with probability on the order of `10^-4` (e.g. ~244 ppm for
  the best sampled difference) — a *single-round, tiny-width* figure, expected.
- **Multi-round decay** (empirical, fixed input difference through the Feistel):
  the best output-difference probability drops to the sampling floor by the first
  measured round count and stays there through 12 rounds — i.e. no
  differential of usable probability survives even a few rounds at these widths,
  let alone the full 24. **No full-round differential kill criterion is
  triggered.** This is a bounded empirical probe, not a proven trail bound; a
  branch-and-bound trail search over an exact DDT remains future work.

**Reproduce:**

```bash
cargo run -p scirust-hypercrypto --bin hypercrypto-falsify -- weak-keys --width mini8
cargo run -p scirust-hypercrypto --bin hypercrypto-falsify -- differential --width mini8 --sample 20000
```

---

## 19. Real key schedule and official test vectors

The spec §10 **HKDF-SHA-256** key schedule (RFC 5869; HMAC-SHA-256 counter-mode
expansion) is implemented in `derivation` and validated against the **RFC 4231
HMAC** and **RFC 5869 HKDF** known-answer tests. It derives, from a 32-byte
master key and 16-byte tweak, the per-round subkeys `K0,K1,K2` (`/ROUNDKEY`),
round constants `RC` (`/CONSTANT`), and Even–Mansour whitening (`/WHITENING`),
each via a domain-separated `HKDF-Expand`.

Keyed by this schedule, the full v0.1 permutation (1024-bit state, two `W64`
octonion branches, 24 rounds, whitening) produces the **official test vectors**
(`test_vectors`, spec §15 categories: all-zero, all-one key, single-bit input,
single-bit key, incrementing bytes, alternating bits, maximum components). Every
vector round-trips (`P_K^{-1}(P_K(x)) = x`), and the whole set is pinned by a
SHA-256 contract:

```text
vectors_fingerprint = 07db3cbfa4bab6f8cb68ab349075e2a7000c6327fb5ea77dd8caaed2344ceb4b
```

A committed reference copy lives at
`scirust-hypercrypto/test_vectors/official_v0_1.json`; regenerate with
`hypercrypto-falsify vectors`. This makes the construction "finished and
reproducible" (a second independent implementation can now match bit-for-bit),
without any change to the security posture — it remains an experimental research
permutation.

---

## References

- Merged spec: `docs/research/SCIRUST_HYPERCRYPTO_SPEC_V0_1.md` (authoritative).
- J. C. Baez, "The Octonions," *Bull. AMS* 39 (2002) — octonion structure, Fano plane.
- M. Luby, C. Rackoff (1988); H. Feistel (1973) — Feistel invertibility.
- N. Courtois, J. Pieprzyk, *ASIACRYPT* 2002; T. Jakobsen, L. Knudsen, *FSE* 1997 —
  algebraic / interpolation attacks (Phase-2 tooling).

*This report documents an attempt to break v0.1. Its purpose is not to make the
construction look promising.*
