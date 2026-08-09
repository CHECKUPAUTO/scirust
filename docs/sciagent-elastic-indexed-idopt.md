# SciAgent Indexed BPE ID-OPT evidence

This note records the isolated same-runner A/B used to validate compact node and candidate storage in the canonical `IndexedBpe` kernel.

## Scope

Only the Indexed working set changes in the candidate:

- compact node token/adjacency/generation fields use `u32`;
- compact candidate priority uses `PriorityKey(u64)` with compact right/generation/output fields;
- the packed `BTreeMap<PairKey, PackedRule>` rule lookup remains unchanged;
- inputs outside the checked compact domains execute the complete historical wide path;
- canonical merge ordering and output Token IDs are unchanged.

## Protocol

- baseline: `bccf562bbb067a07d6b84874e31a75a61c646e46`;
- candidate code: `cdb82f683914f320432834d20dcda1a6eb2243c1`;
- one deterministic 1024-token `canonical-rank-v1` tokenizer trained once and reused by both builds;
- candidate reads the exact baseline corpus;
- one case per length;
- 2 warmup runs and 7 measured runs;
- same GitHub-hosted Ubuntu runner;
- zero semantic mismatches required before ratios are accepted.

## Results

| Piece length | Baseline median (ns) | Compact median (ns) | Speedup |
| ---: | ---: | ---: | ---: |
| 32 | 1,562 | 1,382 | 1.130246x |
| 64 | 4,437 | 3,936 | 1.127287x |
| 128 | 13,520 | 11,157 | 1.211795x |
| 256 | 45,006 | 34,461 | 1.305998x |
| 512 | 170,292 | 117,023 | 1.455201x |
| 1024 | 769,239 | 491,157 | 1.566177x |
| 2048 | 3,425,030 | 2,281,135 | 1.501459x |

The candidate is faster at every measured length. The benefit increases materially once the Indexed working set reaches medium and large pieces, supporting the compact representation independently of later CSR rule-table work.

These numbers are hosted-runner evidence, not Jetson AGX Thor throughput claims. Target-hardware calibration remains separate.
