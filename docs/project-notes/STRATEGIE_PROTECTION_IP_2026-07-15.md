# SciRust — Anti-cloning protection strategy & simulated audit

_Architecture audit and IP-locking strategy — 2026-07-15_
_Method: reconnaissance of the real crates (`scirust-autodiff`, `scirust-simd`, `scirust-transpiler`,
`scirust-gpu`, `scirust-macros`, `scirust-simd-macros`, `scirust-license`), design of 16 mechanisms
across 4 pillars, adversarial red-team per mechanism, then numerical-correctness review, user-harm
review and legal/forensic soundness review._

---

## 0. Executive summary (the uncomfortable truth, then what works)

Your threat model is explicit: **a competitor with access to the source code**, who knows how to do
reverse engineering, and wants to recreate a functionally identical tool by rewriting/obfuscating the
code.

Against **that** specific adversary, the audit's conclusion is clear and must be faced:

> **No mathematical canary, no codegen watermark and no macro obfuscation survives a reimplementation
> from the sources.** Any "client-side" protection you compile into your binary is, by construction,
> source code the adversary edits or simply never emits. The near-totality of the 16 mechanisms designed
> was rated "effort to remove = trivial" by the red-team.

This is not a design failure: it is a fundamental property. A watermark cannot be both **invisible to
the compiler / neutral on the result** (the condition for not breaking a scientific library) **and
carrying enough to survive a rewrite** (the condition for proving theft). These two properties are
contradictory. Mechanisms that touch the output bits (to be detectable in black box) **break the
numerical reproducibility of your honest users** for near-zero deterrent value against the cloner — the
worst of both worlds.

**What, on the other hand, has real value — and where the effort should go:**

| Lever | What it really protects | Against whom |
|---|---|---|
| **Asymmetric Lamport/Merkle signatures** (reuse `scirust-license/hashsig.rs`) on emitted artifacts | *Non-forgeable* legal proof of provenance | The **verbatim** redistributor of artifacts THAT YOU produced (client leak, repackaging) |
| **Similarity of the source code itself** (GEMM tiling scheme, packing layout, constants, comments) | The real copyright/trade-secret evidence | The source copier — that is the proof, not the watermark |
| **Licensing / node-lock with graceful refusal** (the `CoreKernels` gate already exists) | Revenue: prevent unlicensed use by honest clients | The user who did not pay, NOT the cloner |
| **Neutral tripwires** (thread-local execution canary, `#[used]` statics) | Catch the *lazy* verbatim copier at ~zero cost | Someone a simple `diff` already condemns |

**The golden rule of this strategy:** never confuse three distinct things —
1. **Deterrence / tripwire** (~zero cost, catches the lazy, no legal value),
2. **Legal proof** (asymmetric signatures + custody + timestamping),
3. **Licensing** (protects revenue, not the algorithm).

Each mechanism below is classified into one of these categories, **honestly**. Anything sold as
"irrefutable proof" while being a tripwire is a trap that will backfire in court (the legal red-team
details why: fictitious p-values, forgeable residuals, the "independent creation" defense served on a
silver platter by your own tolerance documentation).

---

## 1. Threat model and principles

### 1.1 The adversary spectrum (never mix them up)

1. **Verbatim redistributor**: takes your binaries/artifacts as-is. → The asymmetric signatures
   nail him. Easy.
2. **Artifact launderer**: takes your generated code (WGSL, emitted Rust) and runs it through a
   formatter / minifier / naga round-trip. → Only semantics-bound channels survive, plus the signature
   if the medium is preserved. Medium.
3. **Source cloner** (YOUR declared adversary): reimplements the logic. → **Nothing** on the binary
   side survives. His only vulnerability is the **substantial similarity of the source code** he
   copied, and **licensing** if he wants to use YOUR binary.

This entire strategy consists of: (a) maximizing proof against 1 and 2 at near-zero cost, (b) accepting
that against 3 the defense is **legal and contractual**, not technical, and preparing the ground for
that fight.

### 1.2 Non-negotiable guardrails (from the security reviews)

These constraints take precedence over any protection gain. They are detailed in §7.

- **G1 — Proven numerical neutrality.** SciRust is a scientific/DL library. No watermark may modify
  any user-visible result. Mechanisms that perturb the low-order ULP (reduction-order residuals,
  autodiff "Channel B", WGSL schedule) **break bit-for-bit reproducibility** and are **ruled out**.
- **G2 — Zero user harm.** No corruption, no silent bias, no hardware gate that refuses mid-compute,
  no hidden file I/O on the compute path, no per-licensee fingerprint in numerical outputs the user
  can publish.
- **G3 — Forensic soundness.** Present as *proof* only the asymmetric signatures, with custody of the
  seed + timestamped public root. Everything else is a *tripwire*, never evidence.
- **G4 — Transparency.** Any watermarking is disclosed in the EULA; a "no-watermark" build target
  (`--no-default-features`) is provided for users sensitive to reproducibility.

---

## 2. Verdict by pillar (overview)

| Pillar | Best mechanism designed | Removal effort (red-team) | Verdict |
|---|---|---|---|
| **1. Math canaries** | Neutral execution canary in `chain()`; reduction-order residuals | trivial | **Neutral tripwire = YES. Numerical residuals = NO (harmful).** |
| **2. Codegen watermark** | Lamport/Merkle-signed banner on the emitted artifact (`emit.rs`) | low | **YES as provenance/leak tracing. NO as anti-clone.** |
| **3. Macro obfuscation** | CFF + opaque predicates + MBA at expansion time | trivial | **NO.** Separable from the computation, the most mature obfuscation class to deobfuscate, perf tax. Keep only the *build gating* idea. |
| **4. Environmental binding** | GPU node-lock + `CoreKernels` entitlement gate | trivial (anti-clone) | **YES as honest graceful-refusal licensing. NO as forensics.** |

---

## 3. Pillar 1 — Mathematical canary traps

### 3.1 What was designed (anchored in the real code)

`scirust-autodiff/src/lib.rs` routes **all** non-linear forward-mode tangent propagation through a
single funnel: `fn chain(factor, deriv) -> f64` (lines 50-53), which already handles the
`deriv == 0.0 -> 0.0` case (anti-`0*inf=NaN` guard). This is the ideal anchor point — a single
`#[inline]` covers Div, `f64/Dual` Div, `powi`, `powf`, `sqrt`, `ln`.

Two canary families were designed there:

**(A) EXECUTION canary, strictly neutral — TO KEEP (tripwire).**
The canary never touches the returned `f64`. It reads `factor`/`deriv` and folds a path digest,
derived from a seed tied to your vendor Merkle root, into a `thread_local` accumulator.

```rust
// scirust-autodiff/src/lib.rs — derived offline, engraved as a const (drift-guard test like DEMO_ROOT_HEX)
const PROV_TAG: u64 = 0x_....; // = first 8 bytes of hashsig::hash(b"SRL.canary", vendor_root, b"scirust-autodiff")

#[cfg(feature = "canary")]
thread_local! { static CANARY: core::cell::Cell<u64> = core::cell::Cell::new(0); }

#[inline(always)]
fn chain(factor: f64, deriv: f64) -> f64 {
    #[cfg(feature = "canary")]
    CANARY.with(|c| {
        let h = c.get().rotate_left(7)
            ^ PROV_TAG.wrapping_mul((deriv == 0.0) as u64 + 1)
            ^ (factor.to_bits() >> 52)
            ^ (deriv.to_bits() & 0xFFFF);
        c.set(h);
    });
    // NUMERICAL BRANCH: byte-for-byte identical to the original.
    if deriv == 0.0 { 0.0 } else { factor * deriv }
}
```

A frozen "probe" (closure + fixed inputs) produces a reproducible 64-bit digest **only** from this
source. The neutrality harness (file `tests/canary_neutrality.rs`) proves, via proptest over the whole
domain (inf/NaN/subnormals/signed zeros), that `chain(f,d).to_bits() == chain_ref(f,d).to_bits()`.

**(B) NUMERICAL canary (ULP bias / reduction order) — DO NOT SHIP.**
The idea: exploit IEEE-754 non-associativity as a fingerprint. Either in the autodiff ("Channel B":
choosing between equivalent factorizations `1/x` vs `x/x²` that round differently), or in the SIMD
reductions (`PermFold`: keyed-permutation summation tree in `portable::dot_f32`,
`dispatch::sdot_f32_avx2`, `gemm.rs`).

### 3.2 Red-team verdict

- **Canary A**: *trivial* to remove. In black box it **carries no signal** (by construction it does not
  touch the output). It only catches the verbatim source copier — whom you already condemn with a
  `diff`. Marginal value ~0 beyond the copy proof the diff provides. **Remains useful as a near-free
  tripwire**, never as proof.
- **Canary B**: *trivial* too (rewrite the reduction in canonical order / enable FMA / fast-math /
  change ISA incidentally erases it) **AND actively harmful** (see §3.3). The legal red-team classifies
  it **CRITICAL**: fictitious p-values (the 64 probes are not independent — a single tree choice fixes
  the residual class for all of them, the real sample size ≈ 1), and the residual derives from the
  **public** root hence is **forgeable** — it proves at best "knowledge of a public datum", not access
  to a secret.

### 3.3 Numerical-correctness review (why B is ruled out)

Measured on 99,999 sampled inputs: the `ln` rewrite `1/x -> x/x²` differs on **25 %** of inputs; the
`div` rewrite `f/g² -> (f/g)/g` on **35 %** (~1 ULP each). These are not rare edge cases. In iterated
optimizers or ODE solvers, these per-step ULP errors **amplify** into visibly diverging trajectories.
Moreover the "few-ULP bound" only holds for well-conditioned sums: on strongly compensated dot/GEMM
(result near zero), the gap between two summation orders is **unbounded relative to the result** and
can exceed the 1e-3/1e-2 tolerances of the GEMM tests.

A confirmed tricky detail: even the "Channel A" swap `a-b -> -(b-a)` **is not** bit-exact — it turns
`+0.0` into `-0.0` when the two contributions are equal (e.g. derivative of `x/x`), and this signed
zero propagates (`1.0/deriv -> -inf` vs `+inf`, `copysign`, `atan2`, `signbit`). `assert_eq!` masks it
(`-0.0 == 0.0`). **Keep only the commutative swap of `Mul` addends** (bit-exact for finite/zero
values), and classify it "within tolerance", not "bit-exact".

### 3.4 Pillar 1 recommendation

- **Ship**: the neutral execution canary (A) **behind a non-default `canary` feature**, presented
  honestly as an anti-verbatim-copy tripwire, with its neutrality harness in CI. Cost ~zero.
- **Do not ship**: any numerical residual (B, PermFold, order residuals). If you insist anyway, it is
  **opt-in default-off**, excluded from every deterministic path (`deterministic_fp32_gemm`, KahanSum,
  Zq/INT8/Q15.16/Q31.32/`dequantize_int4`), with a **single key common to all licensees** (never
  per-seat) and disclosure in the reproducibility documentation.
- **Keep intact as integrity anchors**: the bit-exact contract paths
  (`dequantize_int4_simd_matches_scalar_bit_exact`) — never inject a mark there; use them instead as
  tamper detectors.

---

## 4. Pillar 2 — Transpiler / codegen watermarking

### 4.1 The only mechanism with real legal value: signed banner on the emitted artifact

Real anchor: `scirust-transpiler/src/emit.rs::emit_module` (lines 15-25) is **the only** point where
`PRELUDE + join(emit_func)` is concatenated — both front-ends (Python, MATLAB) pass through it. This is
the perfect chokepoint to sign.

The mechanism reuses **as-is** `scirust-license/src/hashsig.rs` (SHA-256 Lamport OTS + Merkle,
deterministic) — no reinvented crypto:

```
//! srl-emit:v1 root=<hex8> leaf=<u32> sig=<MerkleSig::to_hex>
```

- The banner is a Rust comment (`//!`): the lexer discards it, **the compiled artifact is bit-for-bit
  identical** (G1 neutrality respected). The `fmt_f64` field is **never** touched (no literal
  perturbation).
- The signature covers a **canonicalization** of the artifact (`C(src)`: removes the banner, removes
  comments while respecting string literals, collapses whitespace). It therefore survives
  indentation / reformatting / `rustfmt`.
- A competitor **cannot transplant your banner** onto his own code: his canonical digest will not
  match, `hashsig::verify` returns `false`.
- **The master secret seed never leaves the vendor station.** Signing happens in a **separate offline
  pass**, not in the shipped transpiler. The client binary carries only the **32-byte public root**
  (pattern `DEMO_ROOT_HEX` + drift-guard test). `emit_module` takes an `Option<&Provenance>`:
  `None` => output identical to today (snapshot tests and OSS builds unchanged).

A `prov verify artifact.rs` tool (new `src/bin/prov.rs`) recanonicalizes, recomputes the digest, and
calls `scirust_license::hashsig::verify(&EMIT_ROOT, &digest, &sig)` against the **pinned public** root.
A third-party expert, with the public root alone, **reproduces the verdict and cannot forge** — this is
exactly what makes it admissible.

**Covert redundancy channel (survives comment stripping)**: `PRELUDE` (`emit.rs:28-345`) is ~25
mutually independent free `np::` functions (`sum`, `dot`, `prod`, `max`, `zeros`, `linspace`…).
Item order is semantically inert in Rust → emit them in a key-derived permutation (K=8 → 8! = 40,320
orders). Survives `rustfmt` and minification; falls under an AST pass that re-sorts.
**To present as weak corroboration, never as standalone proof** (see §4.3).

### 4.2 Red-team verdict

- Against the **source cloner**: **zero**. He rebuilds from the `None`/OSS path and emits no
  watermark. The watermark protects the **artifact**, not the **tool**.
- Against the **artifact launderer**: a single AST round-trip (`syn` + `prettyplease`, ~40 lines) kills
  both layers (comments discarded → dead banner; items re-sorted → zero-bit permutation).
  "Semantically inert" = "canonicalizable by the attacker".
- Against the **verbatim redistributor**: **strong and clean**. As long as the banner is present, the
  asymmetric signature is a non-forgeable proof, reproducible by a neutral expert. This is the real
  value, and it serves above all to **trace which client leaked** (the OTS `leaf` is a unique serial
  number).

### 4.3 Legal review (the sine qua non conditions)

The legal red-team classifies the signature layer **HIGH** *usable only if* three things, absent from
all designs, are established **before any distribution**:

1. **Custody of the master seed**: generated in an HSM, logged procedure with controlled access.
2. **Timestamped anchoring of the public root**: published in an immutable, timestamped registry
   (Certificate-Transparency, signed and pushed git tag, notarization/anchoring) **before** the
   suspect artifact — otherwise the "vendor planted the signature after the fact" defense remains
   unanswered.
3. **Reproduction by a third-party expert**, not by the vendor.

Other **MEDIUM/LOW** points to incorporate:
- A valid signature proves the **provenance of the bytes, not the act of copying** by that defendant.
  Frame it rather as **removal of rights-management information (DMCA §1202)** / license violation,
  whose real strength is non-forgeability + removal resistance.
- The small-alphabet combinatorial channels (1/40,320 permutation) compute their FP rate against an
  ideal uniform author; real independent authors cluster on conventional orders (alphabetical, call
  graph). **Calibrate the false-positive rate empirically** against a real corpus before citing any
  probability.

### 4.4 Pillar 2 recommendation

- **Ship**: the **offline signing pass** (layer 1) on emitted artifacts, reusing `hashsig`, + the
  `prov verify` tool, + the §4.3 custody/timestamping discipline. This is your only device with real
  courtroom value, and it is cheap and safe (G1 respected).
- **Keep as corroboration**: the `PRELUDE` permutation (covert layer), honestly scoped "only holds
  against rustfmt alone, falls under AST normalization".
- **Abandon**: any claim that this defends against clone-from-source. And **abandon** the equivalent
  WGSL channel (steganography of identifiers/order in shaders) — naga erases CH1/CH2 before even
  execution on non-passthrough backends, so the deployed GPU program carries no mark (detectable only
  on the pre-naga source, which a reimplementer never ships).

---

## 5. Pillar 3 — Procedural macro obfuscation

### 5.1 What was designed

Real anchors: `scirust-macros/src/lib.rs` (the generated `_grad` body, lines 144-179) and
`scirust-simd-macros/src/lib.rs` (the 4 `#block` re-emissions avx2/sse2/neon/scalar, lines 62-100).
Three layers keyed by a 128-bit vendor seed derived via `hashsig::hash(b"SRL.wm", …)`:

1. **Control-flow flattening**: split the body into `S0..Sk`, dispatch via a state machine
   `loop { match __st { … } }` whose labels are a keyed PRP (splitmix64 constants already present in
   `scirust-gpu/deterministic.rs:201-204`). Execution order is preserved.
2. **Opaque predicates** on the transitions: `if OPAQUE(__st) { real } else { dead }` with an
   always-true MBA identity that LLVM cannot fold (`black_box` on one operand).
3. **MBA labels**: the label constants emitted as MBA expressions of the seed.

The `f64`/`Dual`/SIMD arithmetic is **copied token-for-token** into the state arms — nothing numerical
is touched (MBA stays in the `u64` domain).

### 5.2 Red-team + perf verdict (why we abandon)

- **Fatal separability**: by design "nothing numerical is touched". The adversary with the sources
  never ships the fingerprint — he reimplements the two trivial macros (`#[autodiff]` emits
  `Dual::var/primal` seeds + `.grad()`; `#[simd]` emits 4 `target_feature` copies + an
  `is_x86_feature_detected` scale), a few hours.
- **CFF is the MOST mature obfuscation class to deobfuscate**: the predicate `(a|b)==(a^b)+(a&b)` is
  a textbook MBA identity that GAMBA/msynth/SSPAM reduce instantly; `black_box` becomes an identity
  node; angr/miasm relinearizes the `loop{match}`. Worse, the design doc praises the predicate as a
  "rare and distinctive idiom" — it is a **greppable signal that LOCATES** the instrumented functions
  for the attacker.
- **Perf tax on honest users**: `black_box` on the SIMD dispatch and the autodiff hot path **inhibits
  inlining/vectorization** exactly where the library must be fast. The harm review classifies this
  **MEDIUM**: permanent perf cost for the honest, a watermark removed for free.

### 5.3 What we keep from this pillar

Only the **two-key build hygiene** (good practice, not anti-clone):
- Non-default `obf` feature on the macro crates + `proc_macro::tracked_env` to force-off in
  reproducible builds.
- Emitted profile split: clean arm under `#[cfg(debug_assertions)]`, (possibly obfuscated) arm under
  `#[cfg(not(debug_assertions))]`. cfg-stripping happens **before** HIR/MIR lowering → the dev loop /
  `cargo test` remains **byte-for-byte identical to today**, zero cost in debug.

**Recommendation**: do NOT inject CFF/predicates/MBA into the numerical hot paths. Redirect the effort
to the signature crypto (§4) and licensing (§6). Macro obfuscation against an adversary holding the
sources is expensive theater.

---

## 6. Pillar 4 — Environmental binding / attestation (non-destructive)

### 6.1 What already exists and what was designed

Good news: `scirust-license` already provides everything needed — `verify_license_on_node`
(lib.rs:286), `module_gate!` (gate.rs), `node_fingerprint` **salted by license identity**
(license.rs:181, with a "the raw machine_id must not leak" test), and the zero-sized-token pattern
`_sealed:()` unconstructible without a successful `Entitlements::require`. The crate is **clockless and
networkless** (the `now:u64` is provided by the host) → **no phone-home**, 100 % offline verification.

Two bindings were designed:
- **GPU node-lock**: derive a `cap_hash` from `adapter.get_info()` (`wgpu_backend.rs:675`), pass it as
  the `machine_id` to `verify_license_on_node` **before** any `create_shader_module`. A shader
  extracted in a competitor's harness never reaches this handshake → the dispatch refuses (`Err`),
  never corrupts.
- **`CoreKernels` entitlement wall** on `sgemm_tiled`/`dgemm_tiled` (renamed `*_impl` `pub(crate)`,
  bodies **unchanged**), re-exposed as token methods.

### 6.2 Red-team verdict

As anti-clone: **trivial** to remove (the adversary deletes the `verify_*` call, de-`pub`s the kernels,
drops the `scirust-license` dependency). A zero-sized token compiles to **no** machine code — it is a
visibility barrier, and source access dissolves a barrier. Forensic value against the cloner:
**zero** (the real copy signal remains the **copied micro-kernel body**, orthogonal to the gate).

### 6.3 User-harm review (the traps to absolutely avoid)

The review classifies **hard-refusal** gates as **HIGH** — the classic anti-feature that punishes
paying customers:
- `adapter.get_info().name` is an **unstable string** (driver upgrade, Mesa/lavapipe vs real GPU,
  VM/container migration, cloud instance reprogramming, eGPU swap). A legitimate licensee can be
  **refused mid-workload**.

**Mandatory non-destructive equivalents:**
- (a) **soft-fail / warn-and-continue** on an unrecognized node, with a **generous offline grace
  period**, never a computation refusal;
- (b) bind at best to a **stable, coarse capability class** (vendor family / feature set), not the
  exact string;
- (c) stay **offline** (no phone-home);
- (d) provide a documented **manual offline re-activation path**, so an environment change never
  requires contacting the vendor;
- (e) **never** a gate that can `Err` **in or after** a partially executed computation;
- (f) provide an eval/demo license + an env-var license path, so a transient config never locks out an
  honest user.

**Strict prohibitions (G2):** no file I/O on the compute path (the designed "append-only provenance
journal" writes a file from `derivative/gradient/backward` — to remove: fails in read-only/HPC/
container environments, fills the disk, races between processes, constitutes undisclosed logging); no
per-licensee identity in publishable numerical outputs; no destructive/anti-tamper payload.

### 6.4 Pillar 4 recommendation

- **Ship**: the entitlement gate + node-lock **as honest product licensing**, with **graceful
  refusal** (`Result`, documented `Err`, instantly reversible by providing a license), with **all**
  the §6.3 guardrails. This is real value — against the user who did not pay and casual sharing — and
  the salted/privacy-preserving node-lock is well designed.
- **Do not sell** this as anti-clone or as forensics. Against the cloner, it binds nothing.
- **Add** a proper `Module::Gpu`/`Module::Autodiff` variant (rather than over-scoping `Module::Core`)
  to separate entitlements.

---

## 7. The guardrails, in detail (security reviews)

### 7.1 Numerical correctness (G1)
- **Hard exclusion** of any reduction-order canary from the `deterministic_fp32_gemm`, `KahanSum`,
  `Zq`/`INT8`/`Q15.16`/`Q31.32`/`dequantize_int4` paths. Add a **golden test** pinning the output bits
  of `deterministic_fp32_gemm` against a **checked-in** reference vector (cross-build, not just
  run-to-run — the current `…is_bit_reproducible` test only checks run-to-run and would let a fixed
  keyed order pass) + a call-graph assertion that the watermarked reduction is **unreachable** from
  the deterministic inputs.
- Restate the neutrality bound **relative to `sum|terms|`**, not to the result. Make the Kahan-oracle
  self-check **mandatory for all** reduction variants + test on ill-conditioned cases (deliberate
  compensation, large K).
- `to_bits()` harness over **the whole domain** (inf/NaN/subnormals/signed zeros) mandatory for any
  mechanism claimed neutral. Document the NaN both-operands payload exception (order-dependent on SSE
  x86).

### 7.2 User harm (G2)
See §6.3. In summary: graceful refusal never hard, no file I/O on the compute path, no per-licensee
identity in numerical outputs, obfuscation/canary **off** the hot path (bookkeeping once per process
in cold init, never per-op), and **opt-out + EULA disclosure** for everything.

### 7.3 Forensic soundness (G3)
- Only the **asymmetric signature** layers are presentable as proof, and only against verbatim
  redistribution, **and** only with HSM custody + timestamped root + third-party expert reproduction
  (§4.3).
- **Explicitly reclassify as tripwire/deterrence**: all ULP/reduction-order residuals
  (non-independent p-values = fictitious; residuals derived from a public root = forgeable) and all
  neutral execution canaries (no black-box signal; only catch the verbatim copier already condemned
  by diff).
- **Acknowledge the reversal of your own doc**: the crate documents that reductions differ bitwise by
  backend/width and are only guaranteed within tolerance. This very clause that makes the watermark
  "neutral" **destroys its evidentiary value** ("just another valid summation order" defense).
- **Bit-identity of a dual-number autodiff is NOT probative**: forward-mode duals and reverse-mode
  tapes are textbook; IEEE-754 determinism makes identical results **expected** of any correct
  independent implementation. No expert report should claim otherwise.
- Build the real evidence around **substantial similarity of the source** (tiling scheme, packing,
  constants, comments, identifiers) — preserve and document kernel provenance.

---

## 8. Prioritized action plan

**P0 — Legal foundations (to do BEFORE any distribution):**
1. Generate the master Merkle seed in an HSM, logged procedure. Publish the public root in an
   immutable timestamped registry (signed git tag + CT log). Without this, no signature has courtroom
   value.
2. Disclose the watermarking in the EULA; document a `--no-default-features` watermark-free target.

**P1 — What has real value (to implement):**
3. **Offline signing pass** on `emit_module` (reuse `hashsig`) + `prov verify` tool
   (§4.1/§4.4). Domain `b"scirust-emit:v1\0"`, `EMIT_ROOT_HEX` + drift-guard test.
4. **Graceful-refusal licensing**: finalize `CoreKernels` / node-lock with **all** the §6.3
   guardrails (soft-fail, offline grace, capability class, manual re-activation). Add `Module::Gpu`.
5. **Cross-build reproducibility golden tests** on the deterministic paths (§7.1) — protects your
   users AND serves as a tamper anchor.

**P2 — Near-free tripwires (honestly labeled):**
6. Neutral execution canary in `chain()` behind a non-default `canary` feature + neutrality harness.
7. Covert `PRELUDE` permutation as weak corroboration.
8. Two-key build hygiene (§5.3) to isolate any release-profile-sensitive code.

**WHAT NOT TO DO (harmful anti-patterns, summary):**
- ❌ Perturb the output bits (order residuals, Channel B, WGSL schedule) — breaks repro, harmful,
  removed for free.
- ❌ **Per-licensee** watermark key in numerical code — two licensed installs would give different
  results (fatal for a scientific library).
- ❌ Hardware gate with **hard refusal** on `adapter.get_info()` — locks out paying clients.
- ❌ Hidden file I/O / logging on the compute path.
- ❌ `black_box`/CFF/MBA in the numerical hot paths.
- ❌ Presenting a forgeable residual derived from a public root as "keyed proof".

---

## 9. Appendix — Real anchor points (recon map)

| Crate | File:symbol | Role / opportunity |
|---|---|---|
| autodiff | `lib.rs:50-53` `fn chain` | Unique forward-tangent funnel — neutral execution canary |
| autodiff | `lib.rs:18-30` `Dual::var/primal/new` | Tangent bootstrapping |
| autodiff | `lib.rs:507-534` `derivative_1d/gradient_2d/3d` | Driver boundary (probes / gate) |
| simd | `portable.rs:151/170`, `dispatch.rs:324` | Dot reductions (integrity anchors, **no** residual) |
| simd | `gemm.rs` `micro_kernel_8x16`/`sgemm_tiled` | Crown-jewel — source similarity = the real proof; entitlement wall |
| simd | `dequantize_int4…bit_exact` | Bit-exact contract — tamper anchor, **never** a mark |
| transpiler | `emit.rs:15-25` `emit_module` | Unique chokepoint — signature + banner |
| transpiler | `emit.rs:28-345` `PRELUDE` | Covert permutation channel |
| transpiler | `emit.rs:1484` `fmt_f64` | **Forbidden** to touch (f64 repro) |
| gpu | `wgpu_backend.rs:675` `adapter.get_info()` | Node-lock (capability class, soft-fail) |
| gpu | `deterministic.rs` `verify_bit_exact` | Deterministic contract — cross-build golden test |
| macros | `scirust-macros/lib.rs:144-179` | Build gating (no hot-path obfuscation) |
| license | `hashsig.rs` (Lamport/Merkle SHA-256, deterministic) | **Reuse** for all signing |
| license | `lib.rs:286` `verify_license_on_node`, `gate.rs` `module_gate!` | Graceful-refusal licensing |
| license | `license.rs:181` `node_fingerprint` (salted) | Privacy-preserving node-lock |

---

_End of audit. The guiding line fits in one sentence: against a cloner who has your sources, your
defense is legal and contractual, not technical — so invest the engineering effort where it composes
with the law (asymmetric provenance signatures + timestamping + custody, preserved source similarity,
honest licensing), and never ship a "trap" that damages your honest users for a cloner it does not
slow down._
