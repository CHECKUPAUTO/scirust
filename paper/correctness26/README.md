# Correctness submission — ARCHIVED (not submitted in 2026)

> **Status (2026-07-11, user decision)**: submission postponed — the
> draft was not submitted to Correctness '26 and is archived as is,
> ready for a future edition (analogous CFP expected around June-July of
> each year). **Figures frozen at commits `0c2f1bf`/`014795f` of
> 2026-07-10** (raw evidence: `docs/evidence/`); before any future
> submission, re-measure O1 (x86 + Jetson), re-verify the bibliography, and
> take up the TODOs below. See `paper/PAPER_PLAN.md` §7.

Draft of the paper "Determinism as Certification Evidence: A Fully Auditable
Rust Stack for Bit-Reproducible Training and Quantized Edge Inference".

- **Deadline targeted at the time**: 23 July 2026 (notification:
  1 September 2026) — not met by decision, not by delay.
- **Format**: ACM `acmart` option `sigconf`; regular paper = 7 to 8 pages
  of content (all included **except** references); short paper fallback =
  4 pages. CFP: <https://correctness-workshop.github.io/2026/>.
- **Sources**: `main.tex` + `references.bib` (references verified on
  2026-07-10 — do not add anything without verification).

## Compiling

No TeX toolchain is required in the repository; two options:

```bash
# Option 1 — machine locale avec TeX Live :
cd paper/correctness26 && latexmk -pdf main.tex

# Option 2 — Overleaf : importer main.tex + references.bib tels quels
# (la classe acmart est fournie par Overleaf).
```

## Content and discipline

Each claim of the paper is backed by the claims → evidence table of
`paper/PAPER_PLAN.md` (CI tests T1-T4/R1-R4/Q1-Q3/S1-S3/A1, protocols
O1-O2). The figures of the "cost of determinism" section come from the
runs recorded in `docs/archive/LIVESTATE.md` (x86-64 4 cores + Jetson AGX Thor 14
cores, MAXN, pinned clocks, 3 runs × 30 reps per platform).

## TODO before submission (marked `TODO` in main.tex)

1. Exact affiliation of the author.
2. Public artifact link for the reviewers (or an access statement).
3. Check with the CFP whether the submission must be anonymous
   (`[sigconf,review,anonymous]`).
4. Length review after compilation: aim for ≤ 8 pages excluding
   references (cut first in §2 and §7 if over).
