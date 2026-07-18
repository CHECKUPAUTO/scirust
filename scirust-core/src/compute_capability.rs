//! **Unified compute-capability registry** — one queryable view over the
//! workspace's three, previously uncoordinated, hardware-backend abstractions.
//!
//! The ANEE investigation
//! (`docs/research/ANEE_ADAPTIVE_NUMERICAL_EXECUTION_ENGINE_2026-07-17.md` §3,
//! confirmed by direct source reading) found that SciRust ships **three
//! independent hardware-dispatch notions that never talk to each other**:
//!
//! 1. **CPU SIMD** — `scirust_simd::dispatch::BackendKind` + `detect_backend()`
//!    (runtime CPUID/feature detection: Scalar/SSE2/AVX2/AVX-512/NEON/SVE/…);
//! 2. **Portable GPU** — `scirust-gpu`'s `RawComputeBackend` trait and
//!    `WgpuEngine` (wgpu/Vulkan/Metal/DX12, feature `wgpu`);
//! 3. **CUDA** — `scirust-cuda`'s feature-gated `CudaChain` (Jetson-class
//!    bf16 Tensor-core path), surfaced through `scirust-gpu`'s `CudaBackend`.
//!
//! The program's closing synthesis (`ANEE_PROGRAM_SYNTHESIS_2026-07-18.md` §7)
//! recommends unifying them as ordinary engineering. This module is that
//! unification, scoped deliberately:
//!
//! * **It is a read-side registry, not a dispatch mega-trait.** Each domain
//!   keeps its own, well-fitting dispatch abstraction (a slice-kernel vtable
//!   is not a device handle); what was missing was a single place to ask
//!   *"what compute paths exist in this process, and are they usable?"* —
//!   for diagnostics, logs, and benchmark provenance.
//! * **Dependency direction is respected.** `scirust-core` already depends on
//!   `scirust-simd`, so the CPU entry is seeded automatically from the real
//!   detector. `scirust-gpu` depends on `scirust-core` only under its `wgpu`
//!   feature — so GPU/CUDA entries are *pushed* by `scirust-gpu` (its
//!   `register_compute_capabilities()`, and automatically by
//!   `WgpuEngine::new()`) rather than pulled from here, which would create a
//!   dependency cycle. Configurations where `scirust-gpu` is built without
//!   `wgpu` (hence without a `scirust-core` dependency) can still register
//!   manually via [`register_capability`].
//!
//! Availability is tri-state ([`Capability::available`]): `Some(true/false)`
//! after a real probe (an adapter request, a CUDA dynamic-load attempt),
//! `None` when the path is compiled in but has not been probed — the honesty
//! policy of `scirust-gpu` (never claim a capability that was not verified),
//! applied to reporting.

use scirust_simd::dispatch::detect_backend;
use std::sync::{Mutex, OnceLock};

/// Which of the workspace's compute domains a capability belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum ComputeDomain {
    /// CPU SIMD tier (`scirust_simd::dispatch::BackendKind`).
    CpuSimd,
    /// Portable GPU compute (`scirust-gpu`, feature `wgpu`).
    GpuPortable,
    /// CUDA / Tensor-core path (`scirust-cuda` via `scirust-gpu`'s `cuda`).
    Cuda,
}

impl ComputeDomain {
    /// Stable short label (used in [`capability_summary`]).
    pub fn label(self) -> &'static str {
        match self
        {
            ComputeDomain::CpuSimd => "cpu-simd",
            ComputeDomain::GpuPortable => "gpu-portable",
            ComputeDomain::Cuda => "cuda",
        }
    }
}

/// One compute path known to this process.
#[derive(Debug, Clone, PartialEq)]
pub struct Capability {
    /// The domain it belongs to.
    pub domain: ComputeDomain,
    /// Stable identifier within the domain (e.g. `"x86_64/AVX-512"`, `"wgpu"`).
    pub label: String,
    /// Whether the code path is compiled into this binary.
    pub compiled: bool,
    /// Probed usability: `Some(true/false)` after a real probe, `None` if
    /// compiled in but not yet probed.
    pub available: Option<bool>,
    /// Free-form provenance/detail (e.g. the wgpu adapter name).
    pub detail: String,
}

fn registry() -> &'static Mutex<Vec<Capability>> {
    static REGISTRY: OnceLock<Mutex<Vec<Capability>>> = OnceLock::new();
    REGISTRY.get_or_init(|| {
        // Seed with the CPU SIMD tier from the real runtime detector — always
        // present, always probed (detection *is* the probe).
        Mutex::new(vec![Capability {
            domain: ComputeDomain::CpuSimd,
            label: detect_backend().label().to_string(),
            compiled: true,
            available: Some(true),
            detail: "runtime CPU-feature detection (scirust_simd::dispatch::detect_backend)"
                .to_string(),
        }])
    })
}

/// Register (or update — upsert keyed on `(domain, label)`) a capability.
///
/// Idempotent: re-registering the same `(domain, label)` replaces the entry,
/// so probes can refine an earlier `available: None` announcement.
pub fn register_capability(cap: Capability) {
    let mut reg = registry().lock().expect("capability registry poisoned");
    match reg
        .iter_mut()
        .find(|c| c.domain == cap.domain && c.label == cap.label)
    {
        Some(existing) => *existing = cap,
        None => reg.push(cap),
    }
}

/// Snapshot of every known capability, sorted by `(domain, label)` for
/// deterministic output.
pub fn compute_capabilities() -> Vec<Capability> {
    let reg = registry().lock().expect("capability registry poisoned");
    let mut out = reg.clone();
    out.sort_by(|a, b| (a.domain, &a.label).cmp(&(b.domain, &b.label)));
    out
}

/// One-line summary for logs, e.g.
/// `cpu-simd:x86_64/AVX-512=yes | gpu-portable:wgpu=unprobed`.
pub fn capability_summary() -> String {
    compute_capabilities()
        .iter()
        .map(|c| {
            let avail = match c.available
            {
                Some(true) => "yes",
                Some(false) => "no",
                None => "unprobed",
            };
            format!("{}:{}={avail}", c.domain.label(), c.label)
        })
        .collect::<Vec<_>>()
        .join(" | ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_simd_tier_is_always_seeded_from_the_real_detector() {
        let caps = compute_capabilities();
        let cpu: Vec<_> = caps
            .iter()
            .filter(|c| c.domain == ComputeDomain::CpuSimd)
            .collect();
        assert!(!cpu.is_empty(), "CPU tier must always be present");
        assert!(
            cpu.iter()
                .any(|c| c.label == detect_backend().label() && c.available == Some(true)),
            "seeded CPU entry must match the live detector: {cpu:?}"
        );
    }

    #[test]
    fn register_is_an_upsert_keyed_on_domain_and_label() {
        // Unique label so this test is independent of execution order of the
        // other tests sharing the process-global registry.
        let label = "test-upsert-gpu";
        register_capability(Capability {
            domain: ComputeDomain::GpuPortable,
            label: label.to_string(),
            compiled: true,
            available: None,
            detail: "announced".into(),
        });
        register_capability(Capability {
            domain: ComputeDomain::GpuPortable,
            label: label.to_string(),
            compiled: true,
            available: Some(true),
            detail: "probed".into(),
        });
        let matches: Vec<_> = compute_capabilities()
            .into_iter()
            .filter(|c| c.label == label)
            .collect();
        assert_eq!(matches.len(), 1, "upsert must not duplicate: {matches:?}");
        assert_eq!(matches[0].available, Some(true));
        assert_eq!(matches[0].detail, "probed");
    }

    #[test]
    fn snapshot_is_deterministically_sorted_and_summary_reports_every_entry() {
        register_capability(Capability {
            domain: ComputeDomain::Cuda,
            label: "test-summary-cuda".into(),
            compiled: true,
            available: Some(false),
            detail: String::new(),
        });
        let caps = compute_capabilities();
        let mut sorted = caps.clone();
        sorted.sort_by(|a, b| (a.domain, &a.label).cmp(&(b.domain, &b.label)));
        assert_eq!(caps, sorted, "snapshot must come out sorted");
        let summary = capability_summary();
        assert!(summary.contains("cuda:test-summary-cuda=no"), "{summary}");
        assert!(summary.contains("cpu-simd:"), "{summary}");
    }
}
