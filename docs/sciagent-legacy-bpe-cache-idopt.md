# SciAgent legacy BPE cache ID-OPT evidence

This note records the isolated measurements used to optimize the historical `legacy-parallel-v1` tokenizer without changing its merge semantics or JSON artifact format.

## Semantic contract

The production path keeps the historical repeated left-to-right parallel merge passes. Cache construction is internal and reconstructed at tokenizer load time. Existing artifacts are not reinterpreted.

The compact merge lookup uses `PairKey(u64) -> u32` only when every left, right, and output ID fits the checked `u32` domain. Otherwise the complete tokenizer uses the historical wide `(usize, usize) -> usize` lookup. Later duplicate rules keep the legacy behavior in which the last rule wins.

## Cache construction A/B

A deterministic 2048-token / 1,788-merge legacy tokenizer and 42,342 UTF-8 bytes from representative SciAgent sources were measured with 3 warmups and 11 rounds. Exact Token IDs were checked before timing.

| Path | Median | Relative |
| --- | ---: | ---: |
| Rebuild byte lookup + merge map per encode | 5,760,031 ns | 1.000000x |
| Cache byte lookup + merge map at load | 2,075,560 ns | **2.775170x** |

The production change therefore builds the 256-byte base-token LUT and merge lookup once.

## Packed merge-map A/B

With lookup construction already cached and `RandomState` unchanged:

| Representation | Raw key/value payload | Median | Relative |
| --- | ---: | ---: | ---: |
| `((usize, usize), usize)` | 24 B | 2,097,717 ns | 1.000000x |
| `(PairKey(u64), u32)` | 16 B | 1,854,423 ns | **1.131197x** |

The packed representation is therefore used for normal legacy tokenizers. This result is independent from the rejected identity-hasher experiment; `RandomState` remains unchanged.

## Rejected or deferred variants

Two narrower variants were measured and intentionally not integrated:

- `[u32; 256]` byte LUT: 2048 B to 1024 B on x86_64, but only `1.001039x` throughput; the production cache keeps the simpler `[usize; 256]` LUT until target-hardware evidence says otherwise.
- merge-pass double-buffer reuse: `0.998927x`; rejected because it added complexity without a throughput benefit.

Identity hashing for packed `u64` keys was also rejected after measuring `0.127212x` versus the default `RandomState` path.

## Validation

The production tests include an uncached historical reference encoder and exact-ID comparisons for both trained byte-level-v2 tokenizers and the embedded legacy tokenizer. They also cover cache/source consistency, duplicate-rule behavior, save/load reconstruction, and a checked wide fallback above the `u32` domain.

This phase was first revalidated after Indexed node/candidate compaction merged as PR #1096 at commit `928c0e5e2997a9281d08ab01812c33dbdd3353ca`. CSR-SoA TinyScan then merged independently as PR #1095 at commit `ed0e814073854fb9d74d92cff46372e2dfd8448b`. This branch is updated again after that merge so GitHub regenerates the pull-request merge ref and reruns the complete CI matrix against the current combined tokenizer state before the legacy cache can merge.

All timings above are GitHub-hosted x86_64 evidence, not Jetson AGX Thor throughput claims.
