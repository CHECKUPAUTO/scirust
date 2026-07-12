# Production Gates — `scirust-finmigrate`

**Status: NOT production-ready.** This document is the single, authoritative
exit checklist that must be satisfied **before any `scirust-finmigrate` unit is
deployed against real money**. It consolidates the per-unit gates scattered
across `audit_report.md`, the `cobol/SEMANTICS_*.md` contracts, and each
`tests/compiler-derived/gnucobol-3.1.2/RESULTS.md`.

Nothing here revokes the work already done — it states precisely what the
existing evidence does **and does not** establish, so the remaining decision is
explicit rather than implied.

---

## 0. Verdict in one line

The units are **validated under GnuCOBOL 3.1.2** (numerically equivalent to the
semantic model and the Rust port on the tested datasets, reproducibly and gated
in CI). They are **not** validated under the **production IBM Enterprise COBOL /
z/OS** compiler and runtime, against **real production data**, with **business
sign-off**. The gap between those two statements is this checklist.

---

## 1. What the current evidence proves

| Claim | Status | Where |
|---|---|---|
| No binary floating point in the money path | ✅ enforced | `tests/no_float_guard.rs` |
| Rust port == Python semantic model, exact parity (\|Δ\|<1e-10) | ✅ | `tests/*_equivalence.rs` |
| Model baselines tamper-evident (SHA-256) | ✅ | `finaudit`, `*.sha256` |
| COBOL == model under a **real** compiler (GnuCOBOL 3.1.2) | ✅ tested datasets | `tests/compiler-derived/.../RESULTS.md` |
| Reproducible from sources + documented toolchain | ✅ | `REPRODUCE.md`, `tools/run_baselines.py` |
| Equivalence gated on every push/PR | ✅ | `.github/workflows/ci.yml` (`finmigrate GnuCOBOL Equivalence`) |

This moved the evidence from *model-only* to *validated under a real COBOL
compiler*. It is a genuine, reproducible result — within its scope.

---

## 2. What the current evidence does NOT prove

- Equivalence with **IBM Enterprise COBOL for z/OS** (the production compiler).
- Equivalence with the **z/OS runtime** and its arithmetic options.
- Correctness on **real production data volumes and the full input domain**
  (current datasets are hand-picked edge cases + a small deterministic sample).
- Agreement with the **legacy system's real historical outputs**.
- That the **CURRCVT Gap-R correction** (a deliberate change to legacy behaviour)
  is **approved by business/legal**.

---

## 3. Blocking gates

Each gate is **blocking**: a unit does not ship until its row is `CLOSED` with
evidence attached.

### G1 — Re-run every baseline under the target compiler & runtime
**Why.** All compiler evidence is GnuCOBOL 3.1.2 (libcob/GMP, arbitrary-precision
decimal). Production is IBM Enterprise COBOL for z/OS, whose fixed-point
intermediates are **capped at 30 digits (`ARITH(COMPAT)`) / 31 digits
(`ARITH(EXTEND)`)** — a different arithmetic model. On multi-step chains
(PAYCALC `(1+i)^n`, CURRCVT triangulation, AMORTSCH running balance) this can
shift a result by a minor unit. IBM even shipped a correctness APAR (PH64936) for
compiler-version-dependent arithmetic, which is exactly why a live-compiler
baseline is mandatory and a portable model is not sufficient.
**How to close.** Compile each `cobol/<UNIT>.cbl` with the production z/OS
Enterprise COBOL build under the **production `ARITH` option**, drive it over the
committed scenarios, and re-diff at exact parity against the model baselines
(reuse `run_baselines.py`'s comparison logic; swap the compiler). Record the
compiler version, `ARITH` setting, and any `ROUNDED`/`TRUNC` options.
**Owner.** Migration engineering + z/OS build team.
**Evidence.** A target-compiler baseline set + a green diff, committed alongside
the GnuCOBOL evidence.
**Refs.** `audit_report.md` §Gap-6 / §5; IBM APAR PH64936; IBM Enterprise COBOL
6.4 — intermediate results & arithmetic precision, ROUNDED phrase.

### G2 — Confirm intermediate precision choices against the target
**Why.** Two contract decisions are precision-sensitive and marked GATE in the
semantics:
- **CURRCVT Gap-Q** — the euro intermediate is rounded to exactly 3 dp. The
  regulation permits ≥3 dp; a different target choice can change the final minor
  unit on edge cases.
- **INTACCR Gap-6** — the single-rounding, high-precision intermediate must match
  the target's arithmetic precision.
**How to close.** Confirm the production system's intermediate-euro precision and
arithmetic precision, and assert the chosen values (3 dp; single rounding event)
reproduce the target's results on the boundary scenarios. Fold the confirmation
into the G1 re-run.
**Owner.** Business analyst (regulatory precision) + migration engineering.
**Refs.** `cobol/SEMANTICS_CURR.md` (Gap-Q), `cobol/SEMANTICS.md` (Gap-6).

### G3 — Reconcile against real production data & historical outputs
**Why.** The current scenario sets are designed to *discriminate the contract*
(bounds, negatives, rounding ties), not to represent production traffic. Exact
parity on curated cases does not prove parity on the real input distribution.
**How to close.** Run a representative (ideally exhaustive over a period) sample
of **real production inputs** through both the legacy system and the Rust port and
reconcile at exact parity; investigate every mismatch as a finding. Prefer
comparing against the **legacy system's actual recorded outputs**, not a
re-derivation.
**Owner.** Data/operations + migration engineering.
**Refs.** the "does not prove … exhaustive coverage" caveat in every `RESULTS.md`.

### G4 — Full input-domain & size-error behaviour on the target
**Why.** Overflow / size-error paths, sign handling at field limits, and
COMP-3/DISPLAY exactness are only exercised within PIC ranges here; production
must define and verify behaviour at and beyond the boundaries.
**How to close.** Enumerate each field's PIC limits, verify the size-error /
ON SIZE ERROR branch on the target, and confirm the port's error mapping matches
the legacy operational contract (loud failure, not silent truncation).
**Owner.** Migration engineering + operations.
**Refs.** `audit_report.md` (Gap-5 / size-error notes per unit).

### G5 — Business & legal sign-off on the CURRCVT Gap-R correction
**Why.** The migration **changed legacy behaviour**: the original COBOL stored
CURRCVT results in a fixed 2-dp field; we corrected it to round to the target
currency's minor unit (0 dp for ITL/ESP), per EC 1103/97. This is defensible and
documented (`audit_trail.log`, 2026-07-12), **but a port must not unilaterally
change legacy money behaviour** — even a regulation-compliant change — without an
owner accepting it.
**How to close.** Product/legal sign-off that the corrected (regulation-compliant)
behaviour is the intended production behaviour, or an explicit decision to
preserve bug-for-bug legacy behaviour instead. Record the decision and its owner.
**Owner.** Product owner + legal/compliance.
**Refs.** `tests/compiler-derived/gnucobol-3.1.2/RESULTS.md` (Gap-R reconciliation),
`audit_trail.log` (2026-07-12 decision).

### G6 — Operational readiness (reversibility & error reconstruction)
**Why.** The audit protocol requires a rollback path and the ability to
reconstruct any produced figure. These are process gates, not arithmetic ones.
**How to close.** A documented rollback plan (dual-run / shadow period), a
reconstruction procedure (given inputs, re-derive any output deterministically),
and a monitoring/alerting plan for divergence during the shadow period.
**Owner.** Operations + migration engineering.

---

## 4. Per-unit gate matrix

`M` = mitigated in the port/model · `G#` = blocking gate above. All six units are
GnuCOBOL-validated; none is target-validated.

| Unit | Precision-sensitive gates | Behaviour-change gate | Status |
|---|---|---|---|
| INTACCR  | G1, G2 (Gap-6) | — | GnuCOBOL-validated; target GATE open |
| AMORTSCH | G1 (running-balance chain) | — | GnuCOBOL-validated; target GATE open |
| PAYCALC  | G1 (`(1+i)^n` chain) | — | GnuCOBOL-validated; target GATE open |
| DAYCOUNT | G1 | — | GnuCOBOL-validated; target GATE open |
| BRKTCALC | G1 | — | GnuCOBOL-validated; target GATE open |
| CURRCVT  | G1, G2 (Gap-Q) | **G5** (Gap-R correction) | GnuCOBOL-validated; target GATE + sign-off open |

G3, G4, G6 apply to **all** units.

---

## 5. Sign-off record (to be completed before deployment)

| Gate | Closed? | Date | Owner | Evidence link |
|---|---|---|---|---|
| G1 target compiler/runtime re-run | ☐ | | | |
| G2 intermediate precision confirmed | ☐ | | | |
| G3 real-data reconciliation | ☐ | | | |
| G4 input-domain & size-error | ☐ | | | |
| G5 Gap-R business/legal sign-off | ☐ | | | |
| G6 operational readiness | ☐ | | | |

A unit is production-eligible only when every applicable row is checked with
evidence attached. **Until then, treat all outputs as validated-under-GnuCOBOL,
not production-certified.**

---

## 6. References

- `audit_report.md` — per-unit gaps and production gates (§Gap-6, §5).
- `audit_trail.log` — chronological decision record (incl. the 2026-07-12 Gap-R
  reconciliation).
- `cobol/SEMANTICS_*.md` — the per-unit legacy contracts (Gap-Q, Gap-6, …).
- `tests/compiler-derived/gnucobol-3.1.2/{RESULTS,REPRODUCE}.md` — the GnuCOBOL
  evidence and its explicit scope limitation.
- IBM Enterprise COBOL for z/OS 6.4 — ROUNDED phrase; intermediate results &
  arithmetic precision (30/31 digits, `ARITH(COMPAT|EXTEND)`).
- IBM APAR PH64936 — compiler-version-dependent arithmetic (why a live target
  baseline is mandatory).
- Council Regulation (EC) No 1103/97 — euro conversion & rounding (CURRCVT).
