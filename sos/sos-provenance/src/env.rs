//! Deterministic environment capture.
//!
//! Fills the machine-detectable parts of an [`EnvRecord`] — the hardware class
//! and OS this build targets — so the reproducibility key
//! ([`EnvRecord::digest`]) reflects where an object was actually produced. The
//! toolchain string and backend versions cannot be auto-detected reliably at
//! runtime, so they are supplied by the caller ([`EnvCapture::new`] /
//! [`EnvCapture::backend`]).

use sos_core::{BackendVersion, EnvRecord};

/// The hardware class of this build: target architecture and pointer width,
/// e.g. `"x86_64/64-bit"`. Detected from compile-time target constants, so it is
/// deterministic for a given build.
#[must_use]
pub fn detect_hardware() -> String {
    format!("{}/{}-bit", std::env::consts::ARCH, usize::BITS)
}

/// The operating system this build targets, e.g. `"linux"`.
#[must_use]
pub fn detect_os() -> String {
    std::env::consts::OS.to_string()
}

/// Builder that captures an [`EnvRecord`], auto-filling hardware and OS while
/// taking the toolchain and backends from the caller.
#[derive(Debug, Clone)]
pub struct EnvCapture {
    toolchain: String,
    backends: Vec<BackendVersion>,
}

impl EnvCapture {
    /// Start a capture for a given toolchain identifier (e.g. `"1.89.0-stable"`).
    #[must_use]
    pub fn new(toolchain: impl Into<String>) -> Self {
        Self {
            toolchain: toolchain.into(),
            backends: Vec::new(),
        }
    }

    /// Record a computational backend and its exact version.
    #[must_use]
    pub fn backend(mut self, backend: BackendVersion) -> Self {
        self.backends.push(backend);
        self
    }

    /// Finish: build the [`EnvRecord`] with auto-detected hardware and OS.
    #[must_use]
    pub fn build(self) -> EnvRecord {
        EnvRecord::new(
            self.toolchain,
            self.backends,
            detect_hardware(),
            detect_os(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sos_core::HashAlgo;

    #[test]
    fn hardware_reports_arch_and_width() {
        let hw = detect_hardware();
        assert!(hw.contains(std::env::consts::ARCH));
        assert!(hw.contains("-bit"));
    }

    #[test]
    fn os_matches_target() {
        assert_eq!(detect_os(), std::env::consts::OS);
    }

    #[test]
    fn capture_is_deterministic_within_a_run() {
        let a = EnvCapture::new("1.89.0-stable").build();
        let b = EnvCapture::new("1.89.0-stable").build();
        assert_eq!(a, b);
        assert_eq!(a.digest(HashAlgo::default()), b.digest(HashAlgo::default()));
        assert_eq!(a.os, detect_os());
        assert_eq!(a.hardware, detect_hardware());
    }

    #[test]
    fn different_toolchain_changes_the_key() {
        let a = EnvCapture::new("1.89.0-stable").build();
        let b = EnvCapture::new("nightly-2026-07-02").build();
        assert_ne!(a.digest(HashAlgo::default()), b.digest(HashAlgo::default()));
    }
}
