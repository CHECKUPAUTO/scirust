# scirust-sis

Safety Instrumented Systems (IEC 61511) — the process-safety analogue of
`scirust-func-safety` (ISO 26262/automotive). Builds the SIS "systems and
logic" layer — voting architectures, full SIF loops, cause-and-effect
matrices, proof-test-interval sizing, fault injection, a hash-chained audit
log — on top of the pure quantitative reliability math already in
`scirust-reliability` (`PFDavg`/`PFH`/SIL for the 1oo1/1oo2/2oo2/2oo3/1oo3
MooN family).

## Why this exists

Ranked D1 in [`docs/DOMAIN_ROADMAP.md`](../docs/DOMAIN_ROADMAP.md) as the
fastest path to a differentiated, audit-grade SciRust product: process
safety has a well-documented incident that is the canonical argument for
tamper-evident SIS logic.

**Triton/Trisis (2017)** targeted Schneider Electric Triconex Tricon safety
controllers at a petrochemical plant via a zero-day firmware flaw and the
proprietary TriStation protocol (CISA ICSA-20-205-01). Attackers reached the
SIS engineering workstation and reprogrammed controller logic; a coding
flaw in their tooling triggered an unintended safety shutdown in June 2017,
and a second shutdown in August 2017 exposed the intrusion before physical
harm occurred (Mandiant; MIT Technology Review; US government attribution
to Russia's TsNIIKhM institute, CISA AA22-083A). It is the first known
malware built specifically to manipulate a SIS's safety logic — not just
process control — and it is the reason every module here that touches a
trip decision or a cause-and-effect link writes to a hash-chained audit log
([`audit.rs`]): a modified safety-logic link should be *provable*, not
merely assumed absent.

## What's in it

- **`voting`** — `Architecture { m, n }` ("M out of N channels must vote
  trip"): evaluates per-channel votes into a trip decision, and dispatches
  to the matching `scirust-reliability::pfd_*` formula.
- **`sif_loop`** — a full Safety Instrumented Function loop (sensors → logic
  solver → final elements), each subsystem with its own architecture. Total
  `PFDavg` is the **sum** across subsystems — standard ISA-TR84.00.02
  SIL-verification practice — and `achieved_sil()` reports the resulting
  band.
- **`fault_injection`** — simulate a real process demand against a set of
  channels stuck dangerous-undetected, and classify the outcome (safe trip /
  dangerous failure / spurious trip). Empirically demonstrates what
  `PFDavg` states abstractly: e.g. 2oo3 tolerates one failed channel and
  still trips; 2oo2 does not (2oo2 trades dangerous-failure tolerance for a
  lower spurious-trip rate — confirmed by the crate's tests, not just
  asserted).
- **`cause_effect`** — cause-and-effect matrices: named causes (detected
  conditions) mapped to named effects (safety actions), evaluated
  deterministically against a set of active causes.
- **`proof_test`** — inverts `PFDavg` to size the longest proof-test
  interval meeting a target: "how rarely can we test this loop and still
  claim SIL2?" Uses `scirust-solvers::roots::bisection` since the
  quadratic/cubic 1oo2/2oo3/1oo3 forms have no closed-form inverse — a
  direct reuse of the linear-algebra/numerics work in this same effort
  rather than a hand-derived root formula per architecture.
- **`audit`** — SHA-256 hash-chained log of trip decisions and
  cause-and-effect matrix changes (same principle as
  `scirust-mcp`/`scirust-discovery`'s audit logs).

Exposed as MCP tools (`sis_verify_sif_loop`, `sis_size_proof_test_interval`)
in `scirust-mcp` — see that crate's README.

## Formula provenance

The `PFDavg` formulas themselves live in `scirust-reliability` (IEC 61508-6
Annex B simplified equations); `scirust-reliability`'s test suite validates
1oo1/1oo2/2oo2/2oo3/1oo3 against hand derivations *and* an independently
published worked example (Lundteigen & Rausand, NTNU, ch. 8 slide 27/43: a
2oo3 loop with λDU=1e-6/h, τ=8760h, β=10% → PFDavg≈5.00e-4, matching this
crate's implementation to 4 significant figures, including the slide's
stated ~87%/~13% independent/common-cause split).

## Honest limitations

- `Architecture::pfd_avg` dispatches any valid `M`-out-of-`N` shape: the
  five IEC 61508-6 Annex B closed forms
  (1oo1/1oo2/2oo2/2oo3/1oo3) are used directly, and any other valid shape
  (2oo4, 3oo4, ...) falls back to `scirust_reliability::pfd_moon`'s
  cross-validated general formula — see that crate's module docs for the
  derivation and why it's kept distinct from the five named cases. Only a
  structurally invalid shape (`m == 0` or `m > n`) is refused.
- `fault_injection` models two independent channel fault modes —
  dangerous-undetected (never votes trip) and spurious (always votes trip)
  — via `simulate_demand_with_spurious`; a channel that is *intermittently*
  faulty (neither permanently stuck nor permanently healthy) is not
  modeled.
- The underlying IEC 61508-6 Annex B equations (and their `pfd_moon`
  generalization) are first-order approximations valid for `λ·T1 < 0.1`
  (Brissaud et al., arXiv:1501.06487); this crate does not warn when that
  bound is exceeded — documented here rather than silently assumed away.
