# Archive — scirust-finmigrate

**Status: ARCHIVED** (2026-07-14)

**Unblocking condition:** Legal z/OS IBM Enterprise COBOL distribution for testing.

---

## What's complete

All three follow-up deliverables are finished and merged to master:

1. **CURRCVT Gap-R reconciliation** (commit 090e96d)
   - Modified `cobol/CURRCVT.cbl` to implement regulation-compliant target minor-unit rounding (variable 0 dp for ITL/ESP, 2 dp otherwise)
   - Applied the same fix to normalized sources and the `-RUN` wrapper
   - Regenerated compiler-derived baseline; verified 0 mismatches vs live GnuCOBOL 3.1.2
   - The legacy bug (hard-coded 2-dp result field) is now closed in both COBOL and Rust

2. **Edge-case scenario coverage expansion** (commit 090e96d)
   - Widened all six units' test scenarios with bounds, negatives, and rounding-limit cases
   - INTACCR 75→79, AMORTSCH 94→105, PAYCALC 8→12, DAYCOUNT 10→15, BRKTCALC 9→14, CURRCVT 10→14 rows
   - Regenerated all model baselines (239 rows total)
   - Regenerated compiler baselines; verified 1347/0 comparisons, 975/0 equivalence checks
   - All Rust equivalence tests pass with updated assertions

3. **Compiler-derived validation gated in CI** (commit 090e96d)
   - Added `.github/workflows/ci.yml` job `finmigrate-compiler` (pinned to ubuntu-22.04, GnuCOBOL 3.1.2)
   - Gate: `sha256sum -c SHA256SUMS`, then `run_baselines.py check` (model equivalence), then `run_baselines.py verify` (compiler reproduction)
   - Blocks CI if any baseline is tampered or compiler-derived parity is lost

4. **Production-readiness consolidated** (commit 090e96d)
   - Created `PRODUCTION_GATES.md` — the single authoritative exit checklist
   - 6 blocking gates (G1–G6): target compiler re-run, intermediate precision confirmation, real-data reconciliation, full input-domain & size-error, business/legal sign-off on CURRCVT Gap-R, operational readiness
   - Per-unit gate matrix + sign-off table
   - Updated README.md and audit_report.md to cross-reference the gates

**Test status:** All 57 tests pass locally (6 units, 239 model scenarios, 975 compiler comparisons).

---

## What's NOT complete (production gates)

The units are **GnuCOBOL-validated**, **not production-certified**. Blocking gates:

| Gate | Blocker | Evidence needed |
|------|---------|-----------------|
| **G1** | Target compiler | Re-run all six units under IBM Enterprise COBOL for z/OS + production `ARITH` option; re-diff against committed baselines |
| **G2** | Intermediate precision | Confirm production system's euro precision (≥3 dp) and arithmetic cap (30/31 digits vs model's 38) |
| **G3** | Real-data reconciliation | Run production input samples through legacy and Rust; reconcile at exact parity |
| **G4** | Full input-domain | Enumerate PIC limits; verify size-error / ON SIZE ERROR on the target for each field |
| **G5** | Business/legal sign-off | Confirm approval of CURRCVT Gap-R behaviour change (regulation-compliant 0-dp target rounding, not legacy fixed-2dp) |
| **G6** | Operational readiness | Document rollback plan, reconstruction procedure, monitoring/alerting |

See `PRODUCTION_GATES.md` §3 for detailed closure criteria for each gate.

---

## How to resume

**When z/OS access becomes available:**

1. **Fetch the z/OS build output** for each `cobol/<UNIT>.cbl` under the production `ARITH` option (or `ARITH(COMPAT)` if legacy binary compatibility is required).

2. **Re-generate compiler baselines** on the z/OS system using `tools/run_baselines.py generate` (or the z/OS equivalent).

3. **Compare against the committed model baselines** using `run_baselines.py check`. Expect 975 comparisons; if any diverge beyond documented tolerances (currently none for Gap-R, since we reconciled the COBOL), investigate as a potential arithmetic precision issue.

4. **Update `.github/workflows/ci.yml`** to run against the z/OS baseline if it differs from GnuCOBOL (or add a separate z/OS gate).

5. **Run G3–G6** gates (real-data reconciliation, input-domain verification, business sign-off, operational readiness).

6. **Update `PRODUCTION_GATES.md` §5** to mark each gate `CLOSED` with the z/OS evidence and owner sign-off.

All code, test harnesses, and audit trails are ready; **no additional development work is required** — only re-validation against the production target.

---

## Files to resume from

- `PRODUCTION_GATES.md` — the unblocking checklist
- `audit_trail.log` — decision history (search `[2026-07-12]` for the three recent deliverables)
- `tests/compiler-derived/gnucobol-3.1.2/tools/run_baselines.py` — the reproducible driver (portable to z/OS or callable from z/OS tooling)
- `tests/compiler-derived/gnucobol-3.1.2/SHA256SUMS` — baseline tamper-check manifest

## References

- **Regulatory:** Council Regulation (EC) No 1103/97 (euro conversion rounding)
- **IBM COBOL:** Enterprise COBOL for z/OS 6.4 — ROUNDED phrase, intermediate results & arithmetic precision (`ARITH(COMPAT|EXTEND)`), APAR PH64936 (compiler-version-dependent arithmetic)
- **Audit:** `audit_report.md` (pre-migration gap analysis), `cobol/SEMANTICS_*.md` (per-unit legacy contracts)
- **GnuCOBOL evidence:** `tests/compiler-derived/gnucobol-3.1.2/RESULTS.md` (current validation scope)
