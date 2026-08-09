# ElasticTokenizer autotune raw report v2

Raw autotune reports are performance evidence. A before/after ratio is meaningful only when both runs measured the same logical workload under the same timing protocol and hardware identity.

Schema v2 adds an exact `case_fingerprint` to the existing tokenizer and hardware identities.

## Exact-case fingerprint

The fingerprint is SHA-256 over a canonical byte stream with a versioned domain separator. The stream preserves:

1. total case count;
2. case order;
3. each `piece_len`;
4. each case's input-ID count;
5. every Token ID in order.

All integer fields are checked and encoded as big-endian `u64`, making the fingerprint independent of host pointer width and endianness.

Changing case order, a piece length, an input-ID count, or any Token ID changes the fingerprint. The fingerprint is computed only when a raw timing report is requested; profile fitting semantics are unchanged.

## Comparability contract

A strict comparator for v2 reports must reject comparison when tokenizer identity, exact-case fingerprint, hardware fingerprint, or timing protocol differ. It must also refuse to produce a speedup for a `(piece_len, kernel)` group containing a semantic mismatch, a missing side, or a different sample count.

Schema v1 reports intentionally remain non-comparable under that strict contract because they do not bind the exact calibration inputs.

## Validation

The report writer is gated by the repository's pinned nightly rustfmt/Clippy checks, exact MSRV 1.89 all-target check, and SciAgent tokenizer binary tests. The strict comparator remains a separate follow-up so report production and report comparison can be reviewed independently.

After Indexed working-set compaction merged as PR #1096 at master commit `928c0e5e2997a9281d08ab01812c33dbdd3353ca`, this branch was updated deliberately so GitHub regenerates its pull-request merge ref and reruns the complete CI matrix against the combined repository state. The report writer does not modify Indexed execution, but current-master validation remains mandatory before merge.
