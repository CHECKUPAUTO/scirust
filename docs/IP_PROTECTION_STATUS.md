# SciRust — IP-protection status & resume sheet

_Snapshot of the anti-plagiarism / IP-protection work: what is shipped, and the
operational steps that remain (they require a human — an HSM, and legal review)._
_Last updated: 2026-07-15._

## TL;DR

The **engineering and EU legal-doc layers are complete and merged.** What remains
is **operational and legal**, and can only be done by the vendor: generate the
master seed in an HSM, pin & timestamp-publish the public root before shipping,
fill the legal `<placeholders>`, and have counsel + a DPO review the templates.

**Guiding principle:** SciRust is protected by the *cryptographic traceability of
its artifacts* and by *law* — not by an in-binary lock — and never at the expense
of honest users. Against a source-level cloner, technical measures deter and trace
but do not prevent; the real defence is forensic (provenance signatures) and
contractual, plus the substantial similarity of the copied source itself.

## Shipped (all merged to `master`)

| PR | Deliverable | Key paths |
|---|---|---|
| #476 | Simulated audit + anti-clone strategy | `STRATEGIE_PROTECTION_IP_2026-07-15.md` |
| #480 | Offline signing/verification of emitted artifacts | `scirust-provenance/` (`prov` binary) |
| #482 | Reproducibility / tamper anchor (bit-exact SIMD) | `scirust-simd/tests/reproducibility_anchor.rs` |
| #485 | Opt-in output-neutral execution-path canary | `scirust-autodiff` (feature `canary`) |
| #490 | Graceful-refusal GPU licensing gate | `scirust-license` (`Module::Gpu`, `activation`), `scirust-gpu/src/license.rs` (feature `license-gate`) |
| #493 | Ops runbook + root-publication script + EULA clause | `docs/PROVENANCE_OPERATIONS.md`, `scripts/publish-provenance-root.sh`, `LICENSING.md` |
| #494 | Release guard: refuse demo roots | `scripts/check-production-root.sh` |
| #497 | EULA clause rewritten for EU law | `LICENSING.md` |
| #498 | French clause (Toubon) + GDPR privacy notice EN/FR | `LICENSING.fr.md`, `docs/PRIVACY_NOTICE.md`, `docs/PRIVACY_NOTICE.fr.md` |

## What each mechanism is (and is not)

- **Provenance signature** (`scirust-provenance`): unforgeable Lamport/Merkle mark
  in emitted artifacts. Strong for **verbatim redistribution / leak attribution**
  (the OTS `leaf` is a per-artifact serial). Does **not** detect a from-source
  reimplementation. Output-neutral (rides in a comment).
- **Reproducibility anchor** (`scirust-simd`): pins bit-exact kernel output —
  protects users *and* proves no watermark silently biases results.
- **Execution-path canary** (`scirust-autodiff`, `canary` feature, default off): a
  tripwire against verbatim source copying; no black-box signal. Not evidence.
- **Licensing gate** (`scirust-gpu`, `license-gate` feature, default off): honest
  product licensing (revenue from unlicensed *use*), graceful/offline, **not**
  anti-clone.

## Remaining — vendor actions (NONE of these are done)

Full detail in **`docs/PROVENANCE_OPERATIONS.md`**.

1. **Generate the master seed in an HSM / air-gapped host**; derive the public
   root with `license-tool keygen --seed <hex> --height 20`. The seed must never
   touch the repo, CI, or a shipped binary.
2. **Pin the production root** in `EMIT_ROOT_HEX` (`scirust-provenance`),
   `GPU_LICENSE_ROOT_HEX` (`scirust-gpu`), and `PROV_TAG` (`scirust-autodiff`),
   updating the drift-guard tests. A `PROV_TAG` re-derivation snippet is in the
   runbook § 2.
3. **Publish the root to a timestamped, immutable venue BEFORE distributing**:
   `scripts/publish-provenance-root.sh --root <hex>` (signed git tag + optional
   OpenTimestamps). This is what defeats the "signature planted after the fact"
   defence.
4. **Wire `scripts/check-production-root.sh` into the release pipeline** so a
   demo-signed build can never ship. (It intentionally fails today because the
   roots are still the demo root.)
5. **Legal**: fill every `<placeholder>` (controller, DPO, governing law,
   retention, recipients) in `LICENSING*.md` and `docs/PRIVACY_NOTICE*.md`;
   publish the privacy notice; have the set reviewed by qualified **EU/French
   IP-IT counsel and a DPO**. Provide the French versions to French
   consumers/data subjects (Loi « Toubon »).

## Optional follow-ups (deferred, not started)

- Auto-wire `license::activate` inside `WgpuContext` — *only worthwhile with
  GPU-capable CI + a test license*; the current explicit arm-time entry point is
  the cleaner design and is what the audit recommends.
- Machine-readable TDM reservation file (`robots.txt` / TDM metadata) alongside
  published distributions (see `LICENSING.md` § 6).

## How to resume

The working branch `claude/scirust-plagiarism-protection-au8bhe` has been merged.
Any follow-up = a **fresh branch from up-to-date `master`**. Start from
`docs/PROVENANCE_OPERATIONS.md`.
