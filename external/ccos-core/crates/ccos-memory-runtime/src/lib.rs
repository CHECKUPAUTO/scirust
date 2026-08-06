//! `ccos-memory-runtime` — the neutral cognitive-memory middleware.
//!
//! ```text
//! CCOS Core ─▶ MemoryPolicy ─▶ ccos-memory-runtime ─▶ MemoryProvider ─▶ SlhaAdapter ─▶ SLHAv2 kernel
//! ```
//!
//! Architectural isolation is the whole point of this crate:
//! - CCOS Core must **not** know about SLHAv2, and SLHAv2 must **not** know
//!   about CCOS.
//! - The runtime speaks only [`telemetry::MemoryState`],
//!   [`telemetry::CompressionResult`] and [`traits::MemoryResult`].
//! - [`backend::slha_adapter::SlhaAdapter`] is the **only** bridge to the
//!   SLHAv2 (`scirust`) backend; all SLHAv2 knowledge is confined to it and no
//!   `scirust` type crosses this crate's public API.
//!
//! Lifecycle owned by the runtime: **HOT** (latent + residual) → **WARM**
//! (residual removed, latent preserved) → **COLD** (page out of active memory;
//! persistence handled externally by CCOS).

#![forbid(unsafe_code)]

pub mod backend;
pub mod telemetry;
pub mod traits;

use telemetry::MemoryState;

/// A 128-byte, 128-aligned opaque payload (one SLHAv2 tile's worth of bytes).
/// The runtime never interprets the layout — only a backend does.
#[repr(align(128))]
#[derive(Clone)]
pub struct AlignedPayload {
    pub bytes: [u8; 128],
}

/// A memory page as the runtime sees it: an id, a lifecycle state, and an opaque
/// payload whose internal format only the owning backend understands.
pub struct AlignedMemoryPage {
    pub page_id: u64,
    pub state: MemoryState,
    /// Opaque payload. The runtime does not know the internal format.
    pub payload: AlignedPayload,
}

impl AlignedMemoryPage {
    /// A fresh HOT page with a zeroed payload.
    pub fn new(page_id: u64) -> Self {
        Self {
            page_id,
            state: MemoryState::Hot,
            payload: AlignedPayload { bytes: [0u8; 128] },
        }
    }
}
