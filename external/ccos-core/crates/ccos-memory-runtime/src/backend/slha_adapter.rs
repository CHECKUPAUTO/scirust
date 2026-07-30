//! `SlhaAdapter` — the single bridge between the neutral memory runtime and the
//! SLHAv2 tile format.
//!
//! # Zero-dependency by design
//! This adapter is **self-contained**: it knows the 128-byte SLHAv2 tile ABI and
//! decodes it with its own minimal grouped-INT4 logic, so it does **not** depend
//! on `scirust`. CCOS therefore works without scirust, and so does this SLHA
//! option. (For bit-exact kernel parity + NF4, apply the opt-in patch in
//! `INTEGRATION.md` → "Optional exact-kernel decode" — it pulls `scirust` and is
//! meant for the standalone SLHA build, not CCOS.)
//!
//! # Isolation
//! All SLHAv2 knowledge is confined to this module; the public surface exposes
//! only runtime-neutral types ([`MemoryState`], [`CompressionResult`],
//! [`MemoryResult`]). This module never calls a Monitor — it only *produces* a
//! [`CompressionResult`] the runtime forwards on.
//!
//! # Payload mapping (owned here, opaque to the runtime)
//! The 128-byte payload is an SLHAv2 tile, `#[repr(C)]`, zero padding:
//! ```text
//! [  0.. 64) latent_kv       64 B  grouped INT4 low-rank base (128 dims)
//! [ 64.. 96) residual_bitmap 32 B  1-bit sign-LSH residual (256 bits)
//! [ 96..100) scale            4 B  f32
//! [100..104) dynamic_lambda   4 B  f32  (λ)
//! [104..108) residual_sigma   4 B  f32  (σ_E)
//! [108..112) token_id         4 B  u32
//! [112..116) position         4 B  u32
//! [116..118) head_id          2 B  u16
//! [118..120) flags            2 B  u16  (HOT / WARM)
//! [120..128) group_scales     8 B  MX micro-scales
//! ```
//! `compress` (HOT→WARM) reclaims the 32-byte residual — logical footprint
//! **128 → 96 B** — while preserving the latent.

use crate::telemetry::{CompressionResult, MemoryBackend, MemoryState};
use crate::traits::{MemoryProvider, MemoryResult, MemoryRuntimeError};
use crate::AlignedMemoryPage;

// SLHAv2 tile ABI (inlined so this crate depends on nothing).
const LATENT_BYTES: usize = 64; // 128 dims × INT4
const RESIDUAL_WORDS: usize = 4; // 256-bit residual = 4×u64
const GROUP_DIM: usize = 16; // dims per MX group (128 / 8)
const FLAG_WARM: u16 = 1 << 0;
const HOT_BYTES: usize = 128;
const WARM_BYTES: usize = 96; // HOT − 32-byte residual

// Byte offsets within the 128-byte payload (see module docs).
const OFF_LATENT: usize = 0;
const OFF_RESIDUAL: usize = LATENT_BYTES; // 64
const OFF_SCALE: usize = OFF_RESIDUAL + RESIDUAL_WORDS * 8; // 96
const OFF_LAMBDA: usize = OFF_SCALE + 4; // 100
const OFF_SIGMA: usize = OFF_LAMBDA + 4; // 104
const OFF_FLAGS: usize = OFF_SIGMA + 4 + 4 + 4 + 2; // 118 (after sigma, token, pos, head)
const OFF_GROUPS: usize = OFF_FLAGS + 2; // 120

/// How a WARM page is brought back to HOT. Kept for forward-compatibility with
/// an exact restore; today only [`RestoreMode::Approximate`] is reachable.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RestoreMode {
    /// Latent-only: the residual reclaimed at `compress` time is not re-hydrated.
    Approximate,
    /// Re-hydrate the residual from CCOS's external COLD store. Reserved.
    #[allow(dead_code)] // forward-compat variant, not yet constructed
    Exact,
}

/// SLHAv2 backend. Owns fixed page storage and all tile knowledge.
pub struct SlhaAdapter {
    pages: Vec<AlignedMemoryPage>,
    backend: MemoryBackend,
}

impl SlhaAdapter {
    /// New adapter with a default preallocated page capacity.
    pub fn new() -> Self {
        Self::with_capacity(1024)
    }

    /// New adapter with `capacity` page slots preallocated, so the hot-path
    /// `compress` / `restore` never grow the backing store.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            pages: Vec::with_capacity(capacity),
            backend: MemoryBackend::SlhaV2,
        }
    }

    /// Producer ingestion path: insert (or replace) a page carrying real tile
    /// bytes. Not part of the hot inference path — may allocate.
    pub fn upsert(&mut self, page: AlignedMemoryPage) {
        match self.find_index(page.page_id) {
            Some(i) => self.pages[i] = page,
            None => self.pages.push(page),
        }
    }

    /// Number of resident pages.
    pub fn len(&self) -> usize {
        self.pages.len()
    }

    /// Whether the adapter holds no pages.
    pub fn is_empty(&self) -> bool {
        self.pages.is_empty()
    }

    fn find_index(&self, id: u64) -> Option<usize> {
        self.pages.iter().position(|p| p.page_id == id)
    }

    /// Decode the 64-byte grouped-INT4 latent of a page into dequantized `f32`.
    /// Self-contained (no `scirust`); returns a runtime-neutral `[f32; 128]`.
    fn extract_latent(&self, page: &AlignedMemoryPage) -> [f32; 128] {
        decode_latent(&page.payload.bytes)
    }

    /// HOT→WARM residual removal on a page: mask the 32-byte residual, drop λ,
    /// and set the WARM flag — the latent and metadata are preserved.
    fn remove_residual(&mut self, page: &mut AlignedMemoryPage) {
        strip_residual(page);
    }

    fn restore_with(&mut self, index: usize, mode: RestoreMode) {
        match mode {
            RestoreMode::Approximate => self.pages[index].state = MemoryState::Hot,
            RestoreMode::Exact => self.pages[index].state = MemoryState::Hot,
        }
    }
}

impl Default for SlhaAdapter {
    fn default() -> Self {
        Self::new()
    }
}

impl MemoryProvider for SlhaAdapter {
    type PageId = u64;

    fn state(&self, id: u64) -> MemoryResult<MemoryState> {
        self.find_index(id)
            .map(|i| self.pages[i].state)
            .ok_or(MemoryRuntimeError::PageNotFound(id))
    }

    fn load_page(&mut self, id: u64) -> MemoryResult<()> {
        match self.find_index(id) {
            Some(i) => self.pages[i].state = MemoryState::Hot,
            None => self.pages.push(AlignedMemoryPage::new(id)),
        }
        Ok(())
    }

    fn evict_page(&mut self, id: u64) -> MemoryResult<()> {
        let i = self
            .find_index(id)
            .ok_or(MemoryRuntimeError::PageNotFound(id))?;
        // → COLD. The page leaves active memory; CCOS owns persistence.
        self.pages[i].state = MemoryState::Cold;
        Ok(())
    }

    fn compress(&mut self, id: u64) -> MemoryResult<CompressionResult> {
        let i = self
            .find_index(id)
            .ok_or(MemoryRuntimeError::PageNotFound(id))?;
        if self.pages[i].state != MemoryState::Hot {
            return Err(MemoryRuntimeError::BackendFailure(format!(
                "page {id} is {:?}; compress is a HOT→WARM transition",
                self.pages[i].state
            )));
        }

        let started = std::time::Instant::now();

        // Take the page out (a stack move — no heap allocation) so the
        // `remove_residual(&mut self, &mut page)` helper can hold both borrows.
        let mut page = std::mem::replace(&mut self.pages[i], AlignedMemoryPage::new(id));

        let representation = self.extract_latent(&page);
        let quality_delta = read_f32(&page.payload.bytes, OFF_SIGMA); // σ_E dropped
        self.remove_residual(&mut page);
        page.state = MemoryState::Warm;

        self.pages[i] = page;

        Ok(CompressionResult {
            backend: self.backend,
            previous_state: MemoryState::Hot,
            new_state: MemoryState::Warm,
            bytes_before: HOT_BYTES, // 128
            bytes_after: WARM_BYTES, // 96 — the residual's 32 B are reclaimed
            quality_delta,
            cpu_cycles: started.elapsed().as_nanos() as u64, // portable ns proxy
            representation: Some(representation),
        })
    }

    fn restore(&mut self, id: u64) -> MemoryResult<()> {
        let i = self
            .find_index(id)
            .ok_or(MemoryRuntimeError::PageNotFound(id))?;
        // WARM→HOT, approximate: the residual cannot be recovered from the WARM
        // page alone (latent-only restore).
        self.restore_with(i, RestoreMode::Approximate);
        Ok(())
    }
}

// ── self-contained SLHAv2 tile logic — no dependencies, no `unsafe` ───────────

/// Grouped-INT4 latent decode: `value(d) = (nibble(d) − 8) · scale ·
/// group_scales[d/16] / 255`. Self-contained; for exact kernel parity / NF4 see
/// the `scirust-kernel` opt-in patch in `INTEGRATION.md`.
fn decode_latent(b: &[u8; 128]) -> [f32; 128] {
    let scale = read_f32(b, OFF_SCALE);
    let mut out = [0.0f32; 128];
    for (d, slot) in out.iter_mut().enumerate() {
        let byte = b[OFF_LATENT + d / 2];
        let nibble = if d % 2 == 0 { byte & 0x0F } else { byte >> 4 };
        let group_scale = b[OFF_GROUPS + d / GROUP_DIM] as f32;
        *slot = (nibble as i32 - 8) as f32 * (scale * group_scale / 255.0);
    }
    out
}

/// Zero the 32-byte residual + λ and set `FLAG_WARM` in the payload (HOT→WARM).
fn strip_residual(page: &mut AlignedMemoryPage) {
    let b = &mut page.payload.bytes;
    for byte in &mut b[OFF_RESIDUAL..OFF_SCALE] {
        *byte = 0;
    }
    b[OFF_LAMBDA..OFF_LAMBDA + 4].copy_from_slice(&0f32.to_le_bytes());
    let flags = read_u16(b, OFF_FLAGS) | FLAG_WARM;
    b[OFF_FLAGS..OFF_FLAGS + 2].copy_from_slice(&flags.to_le_bytes());
}

#[inline]
fn read_f32(b: &[u8; 128], off: usize) -> f32 {
    f32::from_le_bytes(b[off..off + 4].try_into().unwrap())
}
#[inline]
fn read_u16(b: &[u8; 128], off: usize) -> u16 {
    u16::from_le_bytes(b[off..off + 2].try_into().unwrap())
}
