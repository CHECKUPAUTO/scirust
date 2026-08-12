# Archived raw evidence — "determinism as evidence" workstream (2026-07-10)

Provenance pieces for the figures cited by `docs/DEAD_GUARDS_STUDY.md`,
`paper/PAPER_PLAN.md` (claims → evidence table) and the draft
`paper/correctness26/main.tex`. Archived here on 2026-07-11 at the
close of the workstream (Correctness submission postponed — not in 2026).

## `dead-guards/` — mining campaign (Lot 2, NO-GO verdict)

The 22 Markdown reports per repository, produced by
`epsilon-audit --mine <dépôt> --out <rapport>` on 2026-07-10, **as-is**:
each one is sealed by its `Report-SHA256` (hash of the body — any
alteration is detectable) and reproducible bit-for-bit on an identical source
tree. `SHAS.txt` lists, for each repository: name, cloned commit SHA
(`--depth 1`), URL, and the sparse-clone subdirectories if any.
Synthesis and manual review of the candidates: `docs/DEAD_GUARDS_STUDY.md`.

Provenance: generated in the session container (x86-64) at commit
`ecf575b3` of the tool; copied here without modification (the SHA-256 seals
attest to this).

## `o1-bench/` — "cost of determinism" benchmark (protocol O1)

Raw outputs of `bench_reduction_overhead`
(`scirust-core/src/bin/bench_reduction_overhead.rs`):

- `x86-20260710.md` — two runs on the x86-64 host of the session
  container (4 cores). Provenance: terminal output of the session,
  archived by the agent that executed it.
- `jetson-20260710T094509Z.md` and `jetson-20260710T114542Z.md` — the two
  complete protocols on the **NVIDIA Jetson AGX Thor Developer Kit**
  (14 cores, L4T R38.4.0, MAXN), executed by the operator via
  `scripts/bench-o1-jetson.sh` at commits `0c2f1bf` then `014795f`.
  Provenance: **transcription of the terminal output pasted by the
  operator in session** — the original bundles
  (`bench-o1-jetson-<UTC>/`) remain on the Jetson machine (`~/scirust/
  scirust/`), git-ignored by design; in case of doubt, they prevail.

Key result re-verifiable in these files: the 4 fingerprints of the
fixed-order reduction (`0x60daf62cf2cb2c29`, `0x9bf7c3f3e9b18898`,
`0xd5b8e15fc7c028e6`, `0x7e99a9d050da4d55` at 1/2/4/8 threads) are
**bit-identical between x86-64 and aarch64**, on independent runs.
