# Automatic integration — `ccos-memory-runtime` → CCOS

> **For an automated integrator (Claude Opus 4.8) working inside the CCOS repo.**
> Follow this file top to bottom. It is self-sufficient: the crate to integrate
> **is this package** — you place it, wire two manifest edits, verify, and open a
> draft PR. Do **not** redesign the crate.

## What this crate is

A **zero-dependency** cognitive-memory middleware:

- `traits::MemoryProvider` — the neutral backend interface.
- `telemetry::{MemoryState, MemoryBackend, CompressionResult}` — the neutral
  vocabulary the runtime + monitors share.
- `AlignedMemoryPage` / `AlignedPayload` — a 128-byte, 128-aligned opaque page.
- `backend::slha_adapter::SlhaAdapter` — a **self-contained** SLHAv2 tile backend.

Lifecycle owned by the runtime: **HOT** (latent + residual) → **WARM** (residual
removed, latent kept, 96 B) → **COLD** (paged out; persistence external to CCOS).

## Core principle — no `scirust` dependency, anywhere

`scirust` (the SLHAv2 kernel) is a **development toolbox**, not a runtime
dependency. Where we need a piece of it — here, the 128-byte tile ABI and its
grouped-INT4 decode — we **vendor** (copy) that minimal logic into this crate.
Result: **CCOS works without scirust, and this SLHA option works without
scirust.** Verified: `cargo tree` prints only `ccos-memory-runtime`.

## Steps

1. **Place the crate.** Copy this entire package into the CCOS repo at
   `crates/ccos-memory-runtime/` (i.e. `Cargo.toml`, `src/`, `tests/`, and this
   file may travel with it). Keep the files byte-for-byte.

2. **Make CCOS a Cargo workspace.** In the **root** `Cargo.toml`, add:
   ```toml
   [workspace]
   members = ["crates/ccos-memory-runtime"]
   ```

3. **Wire SLHAv2 as an OPT-IN feature of the `ccos` crate.** In the **root**
   `Cargo.toml`:
   ```toml
   # under [dependencies]
   ccos-memory-runtime = { path = "crates/ccos-memory-runtime", optional = true }

   # under [features]
   slhav2 = ["dep:ccos-memory-runtime"]
   ```
   Do **not** add `slhav2` to `default` (keep `default = ["syn-parser"]`).

## Hard rules

- Do **not** modify any existing CCOS `.rs` file. Only add this crate and edit
  the root `Cargo.toml`.
- The default CCOS build stays behaviorally unchanged.
- No `scirust` in any configuration (including `--features slhav2`).

## Verify (all must pass)

```bash
cargo build                                   # default: unchanged
cargo build --features slhav2                 # pulls only the zero-dep runtime
cargo tree --features slhav2 | grep -i scirust && echo "LEAK — FAIL" || echo "no scirust anywhere: OK"
cargo test -p ccos-memory-runtime             # 5/5 pass
cargo clippy -p ccos-memory-runtime --all-targets -- -D warnings
cargo fmt && cargo fmt --check
git status                                    # only: crates/ccos-memory-runtime/** + root Cargo.toml
```

## Deliver

Commit on a feature branch and open a **draft** PR:

> **Title:** `feat: SLHAv2 memory backend as opt-in `slhav2` feature (zero-dep ccos-memory-runtime)`
>
> **Body:** Default build unchanged. `slhav2` is opt-in and pulls **no** scirust
> (verified via `cargo tree`). WARM footprint = 96 B. The adapter is
> self-contained and exposes no backend-specific type in its public API.

## Design notes (settled — keep as-is)

- **Zero dependencies.** The grouped-INT4 decode is vendored from scirust's tile
  ABI; scirust itself is never pulled in.
- **WARM = 96 B** (latent 64 + metadata 32; only the 32-byte residual is freed).
  Not 64 — 64 is the latent size, not the WARM footprint.
- **Restore is approximate** (latent-only; the residual is gone after WARM).
  `RestoreMode::Exact` is a reserved forward-compat variant.
- **Page lookup is O(n)** over a `Vec` (constraint: no HashMap). Fine at
  page-cache scale.
- **Zero `unsafe`** (`#![forbid(unsafe_code)]`); `cpu_cycles` is an elapsed-ns
  proxy — wire a real TSC/PMU counter later if true cycles are needed.
- **Feature name:** `slhav2`. If you prefer to avoid confusion with the existing
  runtime license flag `Feature::SlhAv2Embeddings`, rename the Cargo feature to
  `slha-backend` consistently (manifest only).

## Optional: exact-kernel decode (`scirust-kernel`) — STANDALONE SLHA build only

**Do NOT apply this when integrating into CCOS.** Declaring `scirust` — even as an
*optional* dependency — pulls it into CCOS's resolution graph and breaks the
"CCOS builds without scirust" guarantee. Use it only when building
`ccos-memory-runtime` on its own and you want bit-exact kernel parity + the NF4
codebook. This variant is verified: `cargo test --features scirust-kernel` → 5/5,
`cargo tree` = crate only by default, crate + scirust with the feature.

1. `Cargo.toml` — add the optional dep + feature:
   ```toml
   [dependencies]
   scirust = { git = "https://github.com/CHECKUPAUTO/SLHAv2", optional = true }

   [features]
   default = []
   scirust-kernel = ["dep:scirust"]
   ```

2. In `src/backend/slha_adapter.rs`, cfg-split `decode_latent` and add a tile
   reconstructor (move `const GROUP_DIM` into the non-feature `decode_latent`, or
   `#[cfg(not(feature = "scirust-kernel"))]`-gate it, so it isn't unused):
   ```rust
   #[cfg(not(feature = "scirust-kernel"))]
   fn decode_latent(b: &[u8; 128]) -> [f32; 128] { /* existing self-contained body */ }

   #[cfg(feature = "scirust-kernel")]
   fn decode_latent(b: &[u8; 128]) -> [f32; 128] { tile_from_payload(b).dequant_latent() }

   #[cfg(feature = "scirust-kernel")]
   fn tile_from_payload(b: &[u8; 128]) -> scirust::attention::slha_v2::SciRustSlhaTile {
       use scirust::attention::slha_v2::SciRustSlhaTile;
       const OFF_TOKEN: usize = OFF_SIGMA + 4;
       const OFF_POS: usize = OFF_TOKEN + 4;
       const OFF_HEAD: usize = OFF_POS + 4;
       let mut latent_kv = [0u8; LATENT_BYTES];
       latent_kv.copy_from_slice(&b[OFF_LATENT..OFF_LATENT + LATENT_BYTES]);
       let mut residual_bitmap = [0u64; RESIDUAL_WORDS];
       for (w, c) in residual_bitmap.iter_mut().zip(b[OFF_RESIDUAL..OFF_SCALE].chunks_exact(8)) {
           *w = u64::from_le_bytes(c.try_into().unwrap());
       }
       let mut group_scales = [0u8; 8];
       group_scales.copy_from_slice(&b[OFF_GROUPS..OFF_GROUPS + 8]);
       SciRustSlhaTile {
           latent_kv, residual_bitmap,
           scale: read_f32(b, OFF_SCALE), dynamic_lambda: read_f32(b, OFF_LAMBDA),
           residual_sigma: read_f32(b, OFF_SIGMA),
           token_id: read_u32(b, OFF_TOKEN), position: read_u32(b, OFF_POS),
           head_id: read_u16(b, OFF_HEAD), flags: read_u16(b, OFF_FLAGS), group_scales,
       }
   }
   #[cfg(feature = "scirust-kernel")]
   #[inline]
   fn read_u32(b: &[u8; 128], o: usize) -> u32 { u32::from_le_bytes(b[o..o + 4].try_into().unwrap()) }
   ```
