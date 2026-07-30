# TDI upstream scientific record

A verbatim copy of the [TDI](https://github.com/Memorithm/TDI) research
project's `docs/` — every preregistration, every results report, and every
SHA-256 manifest — captured so that SciRust's `scirust-tdi` and
`scirust-tdi-bench` crates carry the evidence for their own claims rather than
pointing at another repository.

**Nothing in this directory is authored by SciRust.** These files are the
upstream record. `UPSTREAM-README.md` and `UPSTREAM-LICENSING.md` are the TDI
repository's own root files, renamed only to avoid colliding with this one.

## Why the copy exists

The `scirust-tdi` crate is a faithful re-homing of TDI's `tdi-core`: the twelve
modules are functionally identical to the upstream frozen crate, differing only
by SciRust's `rustfmt.toml` brace style and by documentation. That means the
upstream confirmatory results describe this crate's behaviour directly — so the
results have to be readable here, not merely cited.

## How to read it

Each experiment has three files:

- `TDI-<n>-<NAME>-PREREGISTRATION.md` — the design, frozen and SHA-256-pinned
  **before** the confirmatory run. Criteria, margins, seeds and output contract
  are all fixed in advance; the freeze rule forbids revisiting them after seeing
  a result.
- `TDI-<n>-<NAME>-RESULTS.md` — the report written from the single, real,
  human-gated confirmatory run.
- `*.sha256` — the manifests the reproduction scripts verify before generating
  anything.

The chain is strictly ordered: each experiment changes exactly **one** factor
relative to a frozen ancestor and verifies that ancestor's hashes before it may
run.

## What the record establishes, and what it does not

The short version, with the honest half first.

**Not established.** No universal law. Nothing outside small synthetic
finite-state branching families at widths 3–5. No transportable calibration —
this is a *measured* failure, not an untested question. No single transportable
effect size: nearly width-invariant (0.61 pp spread) but strongly
generator-dependent (39.8 pp). No superiority over an arbitrarily expressive
learner: degree-2 interaction ridge is the strongest model tested, and kernel
methods and tree ensembles are untested. TDI-1's original signal was fully
subsumed by an orbital baseline, incremental gain exactly `0`.

**Established, within that scope.** The intervention-conditioned overlaps carry
predictive information beyond an entropy-and-topology baseline, beyond exact
contraction descriptors, beyond exact spectral moments `s₂, s₃` *and* a fourth
moment `s₄`, beyond the literal spectral gap `|λ₂|` and the ε-mixing time — and
the advantage **grows** rather than shrinks under a degree-2 nonlinear model.
This replicates across four structurally distinct generator families and three
system widths. The exact descriptor ladder visibly saturates while the signal
does not.

**The most useful negative results.** Cross-domain transfer of the deficit
*level* is broken, and two label-free repairs were preregistered and refuted
with their mechanisms identified: feature re-standardization (TDI-6.6) annihilates
the domain displacement that carried the level, and an additive observable offset
(TDI-6.7) moves only the bias — a *perfect* offset would help in fewer cells
(9/24) than the imperfect one (12/24). Refutations with identified mechanisms
bound the family of possible remedies rather than merely failing to find one.

**One experiment is built but not run.** TDI-6.8 asks whether transferred
*ordering* survives where the level does not — a reading asserted four times
across the series and tested zero times. Its preregistration is frozen, its
evaluator complete, its criteria and output contract fixed. The confirmatory run
is a deliberate human action behind a token no test, commit or CI supplies. As
of this copy it has **not been executed**, and no file here reports its outcome.

## Licensing

TDI is dual-licensed on the same model as SciRust: free for noncommercial and
personal use under the PolyForm Noncommercial License 1.0.0, with commercial use
under separate terms. See `UPSTREAM-LICENSING.md`.
