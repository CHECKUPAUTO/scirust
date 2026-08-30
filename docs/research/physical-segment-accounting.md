# Physical-segment accounting for tensor representations

Status: design gate for #1354; no implementation in this change.

## 1. Purpose

SciRust's existing `RepresentationPlan` remains the representation IR. This design adds an accounting vocabulary beneath representation declarations; it does not change `TensorType`, introduce allocation policy, or duplicate ElasticXxx transition machinery.

The problem is exact physical accounting when a logical representation is not a simple tree of privately owned typed tensors. Future formats can contain packed streams, metadata, alignment, elided values, and segments shared by several representations. Recursive per-consumer addition is then insufficient because a shared segment must be counted exactly once.

## 2. Required invariants

1. Logical tensor semantics and physical accounting remain separate.
2. Every physically stored bit belongs to exactly one physical segment in an accounting scope.
3. A segment has one owner in that scope and zero or more references.
4. References never add storage; the referenced segment is already present in the scope union.
5. Two references share storage only when they resolve to the same canonical segment identity and agree on content identity, representation/layout, size, alignment and lifetime.
6. Serialized and resident sizes are independent exact integer quantities.
7. Padding and allocation rounding are part of physical size, never inferred away.
8. Zero physical bits require a reconstruction contract; absence of storage is not absence of semantics.
9. `StorageBits` remains integer and authoritative. Ratios are derived only after exact totals exist.
10. Existing dense/factorized representations remain expressible without changing their logical contracts.
11. Declaration/accounting must remain deterministic and compatible with `no_std` + `alloc`.

## 3. Proposed additive vocabulary

The following is an API sketch, not committed Rust API.

```text
PhysicalSegmentId(u32)

PhysicalSegmentRole =
    Payload | Index | Scale | Codebook | Metadata | Residual | Auxiliary

PhysicalSegment {
    id,
    role,
    content_identity,
    layout_identity,
    raw_bits,
    serialized_bits,
    resident_materializations: [(MaterializationClass, resident_bits)],
    serialized_alignment_bits,
    lifetime,
    reconstruction_role,
}

SegmentUse = Own(PhysicalSegmentId) | Reference(PhysicalSegmentId)

RepresentationPhysicalLayout {
    uses: [SegmentUse],
}
```

`PhysicalSegmentId` is canonical only inside a declared accounting scope. It is not an address, device handle, graph node, or globally stable content hash.

`content_identity` answers whether two consumers require the same physical contents. `layout_identity` answers whether those contents have the same physical encoding. Equality of byte length alone is never evidence of sharing.

`MaterializationClass` is a named backend/materialization contract rather than a device pointer. A portable serialized representation may therefore be specified even when no resident backend layout is known yet.

`lifetime` makes amortization explicit: model-static, graph-static, request, sequence, tile/block, or another named scope. Cross-lifetime sharing is invalid unless the enclosing benchmark/accounting scope explicitly owns the longer-lived segment.

## 4. Accounting operation

For selected representations `r_1 ... r_n`, collect their segment uses and resolve every reference to exactly one owner. Let `U` be the set of unique owned segment identities reachable in the scope.

```text
serialized_bits(scope) = sum_{s in U} s.serialized_bits
resident_bits(scope, m) = sum_{s in U} s.resident_bits(m)
```

Both sums use checked integer arithmetic. Missing owners, conflicting definitions, unsupported materializations, or overflow are typed errors.

`raw_bits` is diagnostic only. It MUST NOT be substituted for serialized or resident size in effective-bits/value claims.

For a logical denominator `N > 0`:

```text
effective_serialized_rate = (serialized_bits(scope), N)
effective_resident_rate(m) = (resident_bits(scope, m), N)
```

The pair is an exact rational numerator/denominator. Decimal formatting is presentation only.

## 5. Reconstruction contract boundary

Physical accounting proves size, ownership and sharing. It does not by itself prove that a representation reconstructs a logical tensor.

Each bindable non-identity family still needs a reconstruction contract defining how its segments/components produce the logical value. Examples include affine dequantization, sparse scatter, factor contraction, base-plus-residual addition, dictionary lookup, or an implicit constant.

Therefore adding segment accounting MUST NOT make today's generic `Quantized` or `Sparse` skeletons bindable. Concrete layouts must separately prove their reconstruction geometry.

## 6. Worked examples

All examples count exact bits and intentionally use simple dimensions so arithmetic is auditable. They are examples of accounting semantics, not compression-performance claims.

### 6.1 Private packed 2-bit payload + scales

Logical block: 256 values.

- packed codes: `256 * 2 = 512` raw bits = 64 bytes;
- 8 FP16 scales: `8 * 16 = 128` raw bits = 16 bytes;
- serialized format aligns each segment to 8 bytes: both are already aligned;
- resident backend aligns allocations independently to 64 bytes.

Owned segments:

| role | raw | serialized | resident |
|---|---:|---:|---:|
| payload | 512 b | 512 b | 512 b |
| scales | 128 b | 128 b | 512 b |

Exact serialized numerator = `512 + 128 = 640 bits`, so rate = `640/256 = 2.5 bits/value`.

Exact resident numerator = `512 + 512 = 1024 bits`, so rate = `1024/256 = 4 bits/value`.

Calling this representation simply "2 bit" would describe payload width, not effective physical storage.

### 6.2 Two tensors sharing one codebook

Two logical tensors contain 1024 values each. Each owns a 4-bit index stream and both reference the same 256-entry FP16 codebook.

- tensor A indices: `1024 * 4 = 4096 b`;
- tensor B indices: `4096 b`;
- shared codebook: `256 * 16 = 4096 b`.

Assume no extra padding for this example. The union contains three owned segments: A indices, B indices, one codebook.

Exact numerator = `4096 + 4096 + 4096 = 12288 bits`.
Denominator = `2048 values`.
Effective rate = `12288/2048 = 6 bits/value`.

Incorrect recursive per-consumer accounting would count the codebook twice and produce `16384 bits = 8 bits/value`. Exact-once ownership prevents that error.

### 6.3 Sparse values + indices + metadata

Logical vector: 4096 values, 256 nonzeros.

- FP16 values: `256 * 16 = 4096 b`;
- U16 indices: `256 * 16 = 4096 b`;
- header: two U32 fields (`logical_len`, `nnz`) = `64 b`;
- serialized container pads the whole record to a 16-byte boundary.

Raw total = `4096 + 4096 + 64 = 8256 b = 1032 bytes`.
Next 16-byte multiple = 1040 bytes = `8320 b`.

Exact serialized numerator = `8320 bits`, not `8192` and not `8256`.
Effective serialized rate = `8320/4096 = 2.03125 bits/value`.

The sparse reconstruction contract must additionally state that indices select positions and unspecified positions reconstruct to zero.

### 6.4 Low-rank factors

Logical matrix: 128 x 128 = 16384 values. Rank 8, FP16 factors.

- left `[128,8]`: `1024 * 16 = 16384 b`;
- right `[8,128]`: `1024 * 16 = 16384 b`.

No additional metadata/padding in this example.
Exact numerator = `32768 bits`.
Rate = `32768/16384 = 2 bits/logical value`.

This maps directly to the existing factorized representation's two private component storages. Segment accounting generalizes the existing sum without changing factor contraction semantics.

### 6.5 Base + residual

Logical block: 256 values.

- base ternary packed stream: 2 physical bits/value = `512 b`;
- residual presence bitmap: `256 b`;
- 16 residual FP16 values: `16 * 16 = 256 b`;
- 16 U8 residual positions: `16 * 8 = 128 b`;
- metadata (`residual_count` U16): `16 b`;
- serialized record aligned to 8 bytes.

Raw total = `512 + 256 + 256 + 128 + 16 = 1168 b = 146 bytes`.
Next 8-byte multiple = 152 bytes = `1216 b`.

Exact serialized numerator = `1216 bits`.
Rate = `1216/256 = 4.75 bits/value`.

This deliberately demonstrates that a nominal low-bit base can have a substantially larger effective rate after residual machinery is counted.

### 6.6 Elided all-zero block

Logical block: 256 values known by the representation contract to reconstruct as the constant zero tensor.

If the enclosing format already identifies the block and the representation tag is owned outside this block's accounting scope, this block may own zero physical segments:

Exact local numerator = `0 bits`.
Local rate = `0/256 = 0 bits/value`.

This is legal only because reconstruction is `fill_zero(shape)`. It is not legal to use a generic empty payload as a zero-bit representation.

If a tag/bitmap is required to identify elision, that tag/bitmap must be owned and counted in the enclosing scope. A global effective-rate claim cannot exclude it.

### 6.7 Different serialized and resident alignment

One packed stream has 600 raw bits = 75 bytes.

Serialized format pads to 8-byte alignment:
`ceil(75/8)*8 = 80 bytes = 640 bits`.

Backend resident allocation rounds to 128 bytes:
`128 bytes = 1024 bits`.

For 300 logical values:
- serialized rate = `640/300 = 32/15 ~= 2.1333 bits/value`;
- resident rate = `1024/300 = 256/75 ~= 3.4133 bits/value`.

The two rates answer different questions and neither may replace the other.

## 7. Compatibility with current `RepresentationPlan`

A minimal implementation should avoid replacing recursive representation declarations. Instead, a concrete representation/layout should be able to expose or lower to a canonical physical-segment description for a requested accounting scope/materialization.

Dense compatibility mapping:

- one privately owned payload segment;
- raw/serialized bits equal today's checked dense `StorageBits` when no additional serialized alignment contract is requested;
- resident size may remain unavailable until a backend materialization is named.

Factorized compatibility mapping:

- union the physical segments of left and right components;
- existing private factors produce the same total as today's recursive addition;
- if a future factor is genuinely shared, segment identity naturally deduplicates it.

This permits the existing `storage_bits()` API to remain a compatibility surface for layouts whose serialized total is fully defined, while a richer accounting API supplies segment breakdown and resident totals.

## 8. Errors required by an implementation

At minimum:

- duplicate/conflicting owner for one segment ID;
- reference to an unowned segment;
- content/layout mismatch for a shared segment;
- lifetime/scope mismatch;
- invalid/non-power-of-two alignment if alignment is represented numerically;
- raw/serialized/resident size inconsistency;
- unavailable resident materialization;
- integer overflow;
- zero-bit representation without a reconstruction contract.

Exact Rust names are intentionally not frozen by this design note.

## 9. What is explicitly not selected yet

This design does not choose:

- an ElasticBitAllocation optimizer;
- a quantizer;
- an entropy coder;
- a sparse encoding;
- a codebook training algorithm;
- a hardware placement strategy;
- an LLM checkpoint or acceptance threshold.

Those decisions belong after exact accounting and the benchmark protocol are executable.

## 10. Implementation gate

Issue #1354's design gate is satisfied by this note only if review confirms that all seven worked examples are unambiguous and that dense/factorized compatibility is preserved.

If accepted, the first implementation PR should add only the generic segment/accounting primitives plus deterministic validation/tests. It should not add an allocator and should not simultaneously promote quantized/sparse skeletons into bindable representations.
