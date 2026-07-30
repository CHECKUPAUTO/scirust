//! Telemetry contract — the backend-neutral vocabulary shared by the runtime,
//! its backends, and the monitors. Contains **no** SLHAv2 (`scirust`) types.

/// Which backend produced a result. The runtime stays backend-agnostic; this is
/// purely for telemetry / routing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryBackend {
    SlhaV2,
    Mock,
    External,
}

/// Lifecycle state of a page, owned by the runtime.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryState {
    /// Full representation available (latent + residual).
    Hot,
    /// Reduced representation (residual removed/ignored; latent preserved).
    Warm,
    /// Paged out of active memory (persistence handled externally by CCOS).
    Cold,
}

/// The outcome of a compression (state transition) as exposed to the runtime.
/// The backend generates the `representation`; the Monitor consumes it — the
/// backend never calls the Monitor itself.
#[derive(Debug, Clone)]
pub struct CompressionResult {
    pub backend: MemoryBackend,
    pub previous_state: MemoryState,
    pub new_state: MemoryState,
    pub bytes_before: usize,
    pub bytes_after: usize,
    /// Measured semantic degradation (backend-defined proxy).
    pub quality_delta: f32,
    pub cpu_cycles: u64,
    /// Representation exposed to the runtime. The backend generates it, the
    /// Monitor (via the runtime) consumes it.
    pub representation: Option<[f32; 128]>,
}
