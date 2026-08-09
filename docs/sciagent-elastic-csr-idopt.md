# SciAgent ElasticTokenizer CSR-SoA ID-OPT evidence

This note records the measurement chain used to select the compact canonical BPE rule-table layout and its first production integration in `TinyScanBpe`.

## Semantic gate

All performance candidates are checked against canonical rank-priority BPE before timing. No result below is accepted when a semantic mismatch is reported. The production implementation keeps the independent reference oracle unchanged.

Compact storage is an execution optimization only:

- token IDs, merge ranks and outputs are packed only after checked `u32` conversion;
- sparse/high-left-ID compact rule sets fall back to a flat packed table when CSR offsets would be disproportionately large;
- values outside the compact domain return control to the complete historical wide path;
- no piece is split or truncated to force compact execution.

## Rule lookup: flat versus CSR

A deterministic canonical tokenizer with vocabulary 2048 and 1,788 merge rules was used. The probe executed 3,576 hit/miss queries per sweep, 100 sweeps, 3 warmups and 11 measured rounds.

| Layout | Median | Relative to flat |
| --- | ---: | ---: |
| Global flat packed binary search | 4,488,604 ns | 1.000000x |
| CSR local binary search | 2,088,021 ns | **2.149693x** |
| CSR local linear scan | 3,051,095 ns | 1.471145x |

The local binary-search CSR layout was selected.

## CSR storage: AoS versus SoA

The same logical CSR algorithm was then measured with two storage layouts.

| Layout | Payload | Median | Speedup |
| --- | ---: | ---: | ---: |
| AoS: `(right_u32, PackedRule)` | 36,804 B | 2,184,279 ns | 1.000000x |
| SoA: `rights: u32` + `rules: u64` | 29,652 B | 2,035,852 ns | **1.072907x** |

SoA removes 7,152 bytes of padding/alignment payload, about 19.4% of the measured CSR payload, and is also faster. Production CSR therefore stores three arrays: `offsets: Vec<u32>`, `rights: Vec<u32>`, and `rules: Vec<PackedRule>`.

## TinyScan end-to-end A/B

The production CSR-SoA patch was measured end-to-end against master `d65ea3dbe6328d5eee657e5fc291a0c58a3d8056`.

To eliminate branch-age bias, both worktrees started from that exact master commit. Only `elastic_rule_table.rs`, `elastic_tiny.rs`, and the module declaration in `lib.rs` from the candidate were injected into the candidate worktree. One deterministic 1024-token canonical tokenizer was trained once and reused by both builds. The candidate read the exact baseline corpus. There were exactly 2 cases per length, 2 warmups and 11 measured runs.

Both sides reported zero semantic mismatches.

| Piece length | Baseline median | CSR-SoA median | Speedup |
| ---: | ---: | ---: | ---: |
| 8 | 610 ns | 250 ns | **2.440000x** |
| 16 | 1,713 ns | 546 ns | **3.137363x** |
| 32 | 6,301 ns | 1,973 ns | **3.193614x** |
| 64 | 25,757 ns | 7,699 ns | **3.345499x** |
| 96 | 60,537 ns | 17,151 ns | **3.529648x** |
| 128 | 116,982 ns | 32,781 ns | **3.568592x** |

Under this protocol the fitted six-class kernel sequence changed from baseline `[Heap, Indexed, Heap, Heap, Heap, Heap]` to candidate `[TinyScan, TinyScan, Heap, Heap, Heap, Heap]`, showing that the lookup redesign changes the best execution choice rather than merely improving an isolated microbenchmark.

## Post-Indexed integration gate

Indexed node/candidate compaction was merged independently as PR #1096 at master commit `928c0e5e2997a9281d08ab01812c33dbdd3353ca`. That phase changes `elastic_indexed.rs` and its evidence document only; the CSR-Tiny production files do not overlap it. After #1096 merged, this branch was updated deliberately so GitHub regenerates the pull-request merge ref and reruns the complete CI matrix against the combined master state before CSR-Tiny can merge.

These measurements are GitHub-hosted x86_64 evidence, not Jetson AGX Thor throughput claims. Target-hardware calibration remains a separate gate.
