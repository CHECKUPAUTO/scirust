# FLAT M59 — Q1 vec4 MHA physical-Thor qualification

M59 qualifies FLAT M58's opt-in `Q1Vec4Mha` candidate against SciRust's existing resident multi-dispatch attention on the same physical NVIDIA Thor/Vulkan device. This is an evidence PR only: it changes no production routing and promotes no universal performance claim.

## Why this exists

The accepted M28 physical-Thor evidence showed the qualified Q4 vec4 MHA route slower than SciRust's existing resident multi-dispatch composition in all eight measured MHA rows. FLAT M58 introduced a different execution shape specifically for that bottleneck: one workgroup per query row with register-resident Q/output and vec4 K/V staging.

The M58 candidate is already correctness-qualified in FLAT. M59 performs the missing product-side paired measurement against the same SciRust baseline that established the negative M28 result.

## Canonical benchmark and route selection

M59 intentionally reuses `scirust-gpu/examples/flat_m28_naive_vs_fused.rs` rather than cloning the benchmark. The harness now accepts `SCIRUST_M28_FLAT_ROUTE`:

- unset or `q4_vec4_mha`: the accepted historical M28 behavior, using `with_vectorization(..., true)`;
- `q1_vec4_mha`: the M59 qualification path, using `with_q1_vec4_mha(..., true)`.

The default therefore remains the historical Q4 vec4 MHA baseline. M59 sets the route explicitly in its workflow, and the benchmark emits `flat_route=<route>` before the unchanged CSV schema so existing M28 result consumers remain compatible.

The two measured implementations execute on one `WgpuContext`:

1. SciRust resident multi-dispatch attention: `Q·K^T`, scale/causal mask, row softmax, probability·V;
2. the selected FLAT fused grouped-forward pipeline with a reused resident output buffer, plus a separately reported fresh-output scope.

Both paths receive independent resident buffers populated from identical deterministic bytes. H2D upload and D2H readback are outside timing. Both outputs are checked against `forward_reference_grouped` before any timing sample is accepted.

For `q1_vec4_mha`, the benchmark also inspects `kernel_variant_for_shape` and fails before timing unless the selected debug identity is exactly `Q1Vec4Mha`; an unexpected fallback is therefore not accepted as M59 evidence.

## Qualified geometry

The physical workflow covers the same MHA geometry family as the accepted M28 evidence:

- batch = 1;
- q_heads = kv_heads = 1;
- seq_len = 128 and 512;
- head_dim = 64 and 128;
- causal and non-causal;
- 3 warmups;
- 9 measured repeats per row.

The three measured scopes (SciRust naive, FLAT fresh-output, FLAT reused-output) rotate through first/middle/last execution position over complete three-iteration cycles.

## Physical evidence protocol

`.github/workflows/flat-m59-q1-thor-qualification.yml` requires:

- an exact PR-head checkout and SHA verification;
- a persistent `tarek-scirust-arm64-01..04` self-hosted Thor runner;
- an executable build directory under the GitHub workspace;
- `/dev/nvidia0` cross-workflow locking with independent lock-contention proof;
- NVIDIA Thor inventory and `NVIDIA Tegra NVIDIA Thor` Vulkan visibility;
- 300 seconds of continuous verified empty compute occupancy before timing;
- continuous rejection of `cuda_pretrain` contamination during timing;
- fail-closed handling when GPU occupancy cannot be queried;
- exact `flat_route=q1_vec4_mha` provenance and exactly two CSV timing rows per `(seq_len, head_dim)` geometry;
- empty post-run compute occupancy.

No software Vulkan timing is accepted as M59 physical evidence.

## Decision boundary

A green workflow proves exact-head correctness and protocol validity. It does **not** require the candidate to win and does not alter production routing.

After the exact-head measurements are available, the Phase O retention rule applies:

- retain/promote the candidate only for workload(s) where the measured evidence justifies it;
- otherwise record the negative result and keep or remove the candidate according to reproducibility needs;
- never generalize beyond the exact measured device, backend and geometry without new evidence.

`performance_claim=measurement_only_no_production_routing_change` remains in force for this qualification PR.
