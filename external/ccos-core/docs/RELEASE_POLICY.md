# CCOS Core — Release Policy

## Approval

**No release is created without the explicit approval of ZEKRITI Tarek**
(§45). This file defines the procedure; nothing here automates publication.

## Versioning

- SemVer. Current line: `0.4.0-pre` (post-migration lineage marker from the
  CCOS 0.3 / CCOS_EXTENDED 0.4 lineage; the first numbered release decides
  the final version).
- Schema versions (event payloads, CCPS envelopes, snapshot format) are
  versioned independently and additively; breaking a schema requires a
  migration note and a compatibility test.

## Required artifacts per release

- source archive + SHA-256 checksums;
- SBOM (`cargo sbom`, CI artifact);
- dependency manifest (`Cargo.lock`) + license report (`cargo deny`);
- test report (full CI run on the exact release commit);
- build provenance: exact commit, rustc version, build parameters;
- two-build reproducibility comparison (REPRODUCIBLE_BUILDS.md);
- migration/upgrade documentation when schemas change.

## Prohibited

- moving-branch dependencies;
- self-update executors inside Core (§4.1 — signed-manifest *verification*
  tooling lives in CCOS Enterprise);
- publishing benchmark claims without the data required by
  INTELLIGENCE_CLAIMS_POLICY.md;
- tagging or releasing from a dirty tree, or from any commit that has not
  passed the full CI gate.
