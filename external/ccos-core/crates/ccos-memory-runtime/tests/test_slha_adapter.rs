//! Integration tests for the SLHAv2 backend adapter, exercising the lifecycle
//! contract through the neutral `MemoryProvider` interface only.

use ccos_memory_runtime::backend::slha_adapter::SlhaAdapter;
use ccos_memory_runtime::telemetry::{MemoryBackend, MemoryState};
use ccos_memory_runtime::traits::{MemoryProvider, MemoryRuntimeError};
use ccos_memory_runtime::AlignedMemoryPage;

#[test]
fn hot_compress_warm_restore_hot() {
    let mut adapter = SlhaAdapter::new();

    adapter.load_page(1).unwrap();
    assert_eq!(adapter.state(1).unwrap(), MemoryState::Hot);

    let result = adapter.compress(1).unwrap();
    assert_eq!(result.backend, MemoryBackend::SlhaV2);
    assert_eq!(result.previous_state, MemoryState::Hot);
    assert_eq!(result.new_state, MemoryState::Warm);
    assert!(result.representation.is_some());
    assert_eq!(result.bytes_before, 128);
    assert_eq!(result.bytes_after, 96); // residual's 32 B reclaimed
    assert_eq!(adapter.state(1).unwrap(), MemoryState::Warm);

    adapter.restore(1).unwrap();
    assert_eq!(adapter.state(1).unwrap(), MemoryState::Hot);
}

#[test]
fn compress_decodes_a_real_payload() {
    // A deterministic non-zero payload exercises the INT4 latent decode path.
    let mut page = AlignedMemoryPage::new(7);
    for (i, byte) in page.payload.bytes.iter_mut().enumerate() {
        *byte = (i as u8).wrapping_mul(3).wrapping_add(1);
    }

    let mut adapter = SlhaAdapter::new();
    adapter.upsert(page);

    let result = adapter.compress(7).unwrap();
    let representation = result.representation.expect("representation present");
    assert_eq!(representation.len(), 128);
    assert!(representation.iter().all(|v| v.is_finite()));
    assert_eq!(adapter.state(7).unwrap(), MemoryState::Warm);
}

#[test]
fn missing_page_is_reported() {
    let adapter = SlhaAdapter::new();
    match adapter.state(999) {
        Err(MemoryRuntimeError::PageNotFound(999)) => {}
        other => panic!("expected PageNotFound(999), got {other:?}"),
    }
}

#[test]
fn compress_requires_hot() {
    let mut adapter = SlhaAdapter::new();
    adapter.load_page(2).unwrap();
    adapter.compress(2).unwrap(); // HOT → WARM

    // A second compress on a WARM page is not a HOT→WARM transition and fails.
    assert!(matches!(
        adapter.compress(2),
        Err(MemoryRuntimeError::BackendFailure(_))
    ));
}

#[test]
fn evict_moves_to_cold() {
    let mut adapter = SlhaAdapter::new();
    adapter.load_page(3).unwrap();
    adapter.evict_page(3).unwrap();
    assert_eq!(adapter.state(3).unwrap(), MemoryState::Cold);
}
