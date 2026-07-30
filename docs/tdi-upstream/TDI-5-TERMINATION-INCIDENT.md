# TDI-5 Termination Incident

## Status

TDI-5 is frozen. This document records the termination incident without
repairing, rewriting, rehashing, or reinterpreting the TDI-5 protocol.

The TDI-5 partial output is non-final. It must not be analyzed as a completed
TDI-5 result, because the evaluator did not prove that progress remained
possible after the width-6 failure mode was reached.

## Root Cause

The frozen TDI-5 evaluator samples a non-empty successor set for each source
state. For width `w`, the full successor-set space has cardinality:

```text
2^(2^w)
```

At width 6 this is:

```text
2^(2^6) = 2^64 = 18_446_744_073_709_551_616
```

That exact value does not fit in `u64`, whose maximum is `2^64 - 1`.
The frozen implementation attempted the equivalent of:

```rust
1_u64.checked_shl(64)
```

That operation returns `None`. The width-6 non-empty mask count is
`2^64 - 1`, which fits in `u64`, but TDI-5 computed the full space in `u64`
before subtracting the empty set.

## Error-Masking Mechanism

The frozen TDI-5 evaluator then converted errors from exact candidate analysis
into absence of a record. This made structural and arithmetic failures
indistinguishable from valid preregistered sample rejections.

The generation loop continued searching for the requested accepted record count
without a deterministic maximum-attempt budget and without a deterministic
no-progress threshold. At width 6 this produced non-termination rather than an
explicit failure diagnostic.

## Scientific Disposition

TDI-5 remains frozen and unrepaired. TDI-5.1 is a new scientifically
independent experiment derived from TDI-5, with a new preregistration, new
evaluator, new reproduction script, new CI, and new SHA-256 manifests.

TDI-5.1 must not be treated as a continuation of the interrupted TDI-5 run.
