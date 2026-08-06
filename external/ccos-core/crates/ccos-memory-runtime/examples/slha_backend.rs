//! **The SLHAv2 memory backend in action** — the HOT → WARM → COLD page lifecycle the
//! `MemoryProvider` trait exposes, driven by `SlhaAdapter`. Zero dependencies: the 128-byte tile is
//! decoded by the crate's own vendored grouped-INT4 logic, so no `scirust` is pulled in any build.
//!
//! Run: `cargo run -p ccos-memory-runtime --example slha_backend`

use ccos_memory_runtime::backend::slha_adapter::SlhaAdapter;
use ccos_memory_runtime::telemetry::MemoryState;
use ccos_memory_runtime::traits::MemoryProvider;
use ccos_memory_runtime::AlignedMemoryPage;

fn main() {
    println!("# SLHAv2 memory backend — HOT → WARM → COLD lifecycle (zero-dependency)\n");
    let mut adapter = SlhaAdapter::new();

    // Ingest a page carrying a realistic 128-byte SLHAv2 tile (deterministic fill).
    let mut page = AlignedMemoryPage::new(42);
    for (i, b) in page.payload.bytes.iter_mut().enumerate() {
        *b = (i as u8).wrapping_mul(3).wrapping_add(1);
    }
    adapter.upsert(page);
    println!(
        "  ingest page 42                 → state {:?}",
        adapter.state(42).unwrap()
    );
    assert_eq!(adapter.state(42).unwrap(), MemoryState::Hot);

    // HOT → WARM: compress reclaims the 32-byte residual; the latent K/V is preserved.
    let r = adapter.compress(42).unwrap();
    let latent = r.representation.as_ref().map(|v| v.len()).unwrap_or(0);
    println!(
        "  compress(42)  {:?} → {:?}   → {} B → {} B  ({} B reclaimed),  latent = {} f32,  Δquality {:.4}",
        r.previous_state,
        r.new_state,
        r.bytes_before,
        r.bytes_after,
        r.bytes_before - r.bytes_after,
        latent,
        r.quality_delta,
    );
    assert_eq!((r.bytes_before, r.bytes_after), (128, 96));
    assert_eq!(adapter.state(42).unwrap(), MemoryState::Warm);

    // compress is a HOT→WARM transition ONLY — a WARM page is rejected (the transition guard).
    match adapter.compress(42) {
        Err(_) => println!("  compress(42) again             → rejected (already WARM, not HOT) ✓"),
        Ok(_) => panic!("a WARM page must not re-compress"),
    }

    // WARM → HOT: restore is approximate — the residual is gone, the latent drives reconstruction.
    adapter.restore(42).unwrap();
    println!(
        "  restore(42)   WARM → {:?}      → approximate (latent-only)",
        adapter.state(42).unwrap()
    );
    assert_eq!(adapter.state(42).unwrap(), MemoryState::Hot);

    // → COLD: evict. Persistence is external — CCOS owns the durable store, not the runtime.
    adapter.evict_page(42).unwrap();
    println!(
        "  evict_page(42)                 → state {:?}",
        adapter.state(42).unwrap()
    );
    assert_eq!(adapter.state(42).unwrap(), MemoryState::Cold);

    println!(
        "\n→ WARM footprint = 96 B (latent 64 + metadata 32; only the 32-byte residual is freed),\n\
         matching `scirust::ccos_core::WARM_BYTES`. No `scirust` is pulled — the decode is vendored, so\n\
         the default CCOS build stays byte-identical; the backend is reached only via `--features slhav2`."
    );
}
