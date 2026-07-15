//! Optional product-licensing gate for the GPU acceleration module
//! (`Module::Gpu`), enabled by the `license-gate` feature.
//!
//! # What this is
//! A **graceful-refusal** entitlement gate a licensed deployment calls **once,
//! before arming GPU dispatch**. It reuses `scirust-license`'s offline
//! Lamport/Merkle verification and the shared [`activation`](scirust_license::activation)
//! flow, so it inherits those guardrails:
//!
//! * **Never destructive.** A missing/invalid license yields
//!   [`BackendError::Unlicensed`] — the caller declines to run the GPU path; no
//!   computation starts and nothing is corrupted. Enforcement lives only at
//!   arm-time, never inside a partially-run dispatch.
//! * **Offline.** Verification is a pure function of `(license, root, now)`; the
//!   caller supplies `now`. No network, no phone-home.
//! * **Coarse, opt-in node-lock.** Floating licenses run anywhere; only a
//!   deliberately node-locked license is machine-checked, and against a
//!   **stable** id the caller chooses — never a volatile GPU driver string.
//! * **Recoverable.** The license loads from `SCIRUST_LICENSE` /
//!   `SCIRUST_LICENSE_FILE`, so a moved deployment is fixed by dropping in a new
//!   file — no vendor round-trip.
//!
//! # What this is NOT
//! This is product licensing (protecting revenue from unlicensed *use*), not
//! anti-clone protection. A source-level cloner can delete the call; its value is
//! against honest users running an unlicensed build, exactly as the IP-protection
//! audit scoped it.
//!
//! # Integration
//! ```ignore
//! use scirust_gpu::license::{self, GpuAccess};
//! use scirust_license::NodePolicy;
//!
//! // At startup, before constructing/using the GPU context:
//! let _gpu: GpuAccess = license::activate(now_unix_seconds(), &NodePolicy::Floating)?;
//! // Hold the token for the process; its existence proves a valid GPU entitlement.
//! ```

use crate::{BackendError, BackendResult};
use scirust_license::activation::{self, Activation, NodePolicy};
use scirust_license::{Hash, Module, hashsig};

scirust_license::module_gate! {
    /// Capability token proving a verified GPU entitlement. Zero-sized and
    /// unconstructible without a license covering [`Module::Gpu`].
    pub GpuAccess => Gpu
}

/// Pinned public verification root (Merkle root) the gate trusts.
///
/// **Demo default.** Equals `scirust-license`'s public demo root so the tests run
/// out of the box — a demo-issued license proves nothing. Replace it with your
/// HSM-backed production root before shipping enforcement; `gpu_root_is_pinned`
/// is the drift-guard that keeps this constant honest.
pub const GPU_LICENSE_ROOT_HEX: &str =
    "82728023e3de7243e982d04ab09a7aa20a7fdb1fa10a0df2920060abc93a7f02";

/// The pinned root as bytes.
fn gpu_root() -> Hash {
    let bytes =
        hashsig::hex_decode(GPU_LICENSE_ROOT_HEX).expect("GPU_LICENSE_ROOT_HEX is valid hex");
    let mut root = [0u8; 32];
    root.copy_from_slice(&bytes);
    root
}

/// Attempt to arm the GPU module, loading the license from the environment
/// (`SCIRUST_LICENSE` / `SCIRUST_LICENSE_FILE`).
///
/// `now` is Unix seconds, supplied by the caller (the crate keeps no clock).
/// `node` selects the node-lock policy — [`NodePolicy::Floating`] for a portable
/// license, or [`NodePolicy::Bound`] with a stable machine id.
///
/// Returns a [`GpuAccess`] token on success, or [`BackendError::Unlicensed`] with
/// a human-readable reason on refusal. Never panics; never returns a degraded
/// result.
pub fn activate(now: u64, node: &NodePolicy) -> BackendResult<GpuAccess> {
    let license = activation::license_from_env();
    activate_with(license.as_ref(), now, node)
}

/// Like [`activate`] but with an explicitly supplied license (bypassing the
/// environment) — useful for tests and for callers that manage the license file
/// themselves.
pub fn activate_with(
    license: Option<&scirust_license::SignedLicense>,
    now: u64,
    node: &NodePolicy,
) -> BackendResult<GpuAccess> {
    match activation::gate(license, &gpu_root(), now, node, &[Module::Gpu])
    {
        Activation::Granted(entitlements) =>
        {
            // `gate` already required Module::Gpu, so this unlock cannot fail; map
            // defensively to a graceful refusal rather than unwrapping.
            GpuAccess::unlock(&entitlements).map_err(|e| BackendError::Unlicensed(e.to_string()))
        },
        Activation::Denied(reason) => Err(BackendError::Unlicensed(reason.to_string())),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirust_license::{License, SignedLicense, demo_vendor};

    const NOW: u64 = 1_000;

    fn gpu_license(leaf: u32) -> SignedLicense {
        demo_vendor().issue_with_leaf(License::new("Acme", "L-GPU", [Module::Gpu], 0, None), leaf)
    }

    #[test]
    fn gpu_root_is_pinned() {
        // Drift-guard: the embedded root must equal the demo root it is derived
        // from. Replace both together when moving to a production seed.
        assert_eq!(gpu_root(), scirust_license::demo_root());
    }

    #[test]
    fn a_valid_gpu_license_arms_the_token() {
        let signed = gpu_license(0);
        assert!(activate_with(Some(&signed), NOW, &NodePolicy::Floating).is_ok());
    }

    #[test]
    fn no_license_is_refused_gracefully() {
        let err = activate_with(None, NOW, &NodePolicy::Floating).unwrap_err();
        assert!(matches!(err, BackendError::Unlicensed(_)));
    }

    #[test]
    fn a_cpu_only_license_is_refused() {
        let signed = demo_vendor()
            .issue_with_leaf(License::new("Acme", "L-CPU", [Module::Core], 0, None), 1);
        let err = activate_with(Some(&signed), NOW, &NodePolicy::Floating).unwrap_err();
        assert!(matches!(err, BackendError::Unlicensed(_)));
    }

    #[test]
    fn node_locked_license_is_machine_checked() {
        let signed = demo_vendor().issue_with_leaf(
            License::new("Acme", "L-NODE", [Module::Gpu], 0, None).with_node_lock("node-1"),
            2,
        );
        assert!(
            activate_with(Some(&signed), NOW, &NodePolicy::Bound("node-1".to_string())).is_ok()
        );
        assert!(
            activate_with(Some(&signed), NOW, &NodePolicy::Bound("node-2".to_string())).is_err()
        );
    }
}
