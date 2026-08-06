//! `MemoryProvider` — the neutral backend interface. A backend (the
//! [`SlhaAdapter`](crate::backend::slha_adapter::SlhaAdapter), a mock, an
//! external store) is the only thing that knows its own representation; the
//! runtime interacts through [`MemoryState`], [`CompressionResult`] and
//! [`MemoryResult`] alone.

use crate::telemetry::*;

pub type MemoryResult<T> = Result<T, MemoryRuntimeError>;

#[derive(Debug)]
pub enum MemoryRuntimeError {
    PageNotFound(u64),
    BackendFailure(String),
    HardwareFault,
}

/// The contract every memory backend implements. Generic over `PageId` so a
/// backend can pick its own key type (the SLHAv2 adapter uses `u64`).
pub trait MemoryProvider {
    type PageId;

    fn state(&self, id: Self::PageId) -> MemoryResult<MemoryState>;

    fn load_page(&mut self, id: Self::PageId) -> MemoryResult<()>;

    fn evict_page(&mut self, id: Self::PageId) -> MemoryResult<()>;

    fn compress(&mut self, id: Self::PageId) -> MemoryResult<CompressionResult>;

    fn restore(&mut self, id: Self::PageId) -> MemoryResult<()>;
}
