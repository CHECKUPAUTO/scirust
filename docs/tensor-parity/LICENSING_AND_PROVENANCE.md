# Licensing and Provenance

## Rule

SciRust is written from scratch. The Tensor Parity Program **does not copy,
port, translate or reimplement PyTorch source code**, nor code from its
derivatives, under any license.

- PyTorch is BSD-3-Clause licensed. That license permits reading source for
  reference; it does not permit claiming derivative status under another
  license without notice, and it does not make API names or documented
  semantics copyrightable in a way that blocks clean-room reimplementation.
- All kernels in this workspace are original implementations written from the
  mathematical specification and the *documented behavior* of the frozen
  baseline, not from PyTorch source text.
- Operator names and shapes of public APIs are interface facts (names,
  signatures, documented semantics), not creative expression; matching them is
  required for drop-in compatibility and is not a license issue.
- SciRust's own license headers apply to all new files. New code added by this
  program carries the same license as the crate it lands in (SPDX header
  present in every new `.rs` file).

## What we record

- `tensor-operators.toml` rows may cite the PyTorch **documentation** page for
  an operator (specification reference), never a source file excerpt.
- Test fixtures generated from the frozen PyTorch baseline are numeric data
  (input/output/grad tensors) produced by executing PyTorch 2.13.0. Numeric
  fixture data is not copyrightable expression; we nonetheless record the
  generating script and the exact baseline commit for provenance
  (`docs/tensor-parity/provenance/`).
- The only PyTorch artifact vendored into the repo (if any) is the frozen
  operator metadata extracted from the public API stubs, with a recorded
  origin and hash. No `.py` implementation files are copied.

## Provenance ledger

| Artifact | Origin | License | Recorded where |
| --- | --- | --- | --- |
| Baseline identity (version/commit) | pytorch/pytorch tag v2.13.0 | BSD-3-Clause (metadata only) | PYTORCH_BASELINE.md |
| Operator metadata (names, signatures) | public API stubs at pinned commit | interface facts | tensor-operators.toml |
| Differential fixtures | execution of pinned baseline | numeric data | provenance/ + artifact hash |
| All Rust kernels | original work of this workspace | workspace license | each crate's headers |

## Verification

The audit checklist for every PR in this program includes: no file under
`third_party/`-style vendor dirs appears without a provenance record, and no
diff contains verbatim PyTorch source text (enforced by a CI check that scans
for PyTorch file-path headers and BSD attribution boilerplate in new files).
