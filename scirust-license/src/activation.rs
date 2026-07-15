//! Graceful-refusal activation flow layered over [`verify_license`] /
//! [`verify_license_on_node`].
//!
//! This is the product-licensing entry point a gated crate calls **once, at
//! initialization**, to decide whether a high-value capability may run. It bakes
//! in the guardrails a licensing gate needs to be *defensible* rather than an
//! anti-feature that punishes paying customers:
//!
//! * **Graceful, never destructive.** Every path returns an [`Activation`] value;
//!   nothing panics, corrupts data, or degrades a result. A closed gate means
//!   "do not arm this capability" — the caller returns a clean error, never wrong
//!   output. A gate must never sit inside or after a partially-run computation.
//! * **Offline.** Verification is a pure function of `(license, root, now)`: no
//!   network, and no clock of its own (the caller supplies `now`). No phone-home.
//! * **Node-lock is opt-in and coarse.** A floating license runs anywhere; only a
//!   license the vendor deliberately node-locked is machine-checked. Bind, if at
//!   all, to a **stable, coarse** machine id (`/etc/machine-id`, a provisioned
//!   UUID) — never a volatile string like a GPU driver version, which would lock
//!   a paying user out on an ordinary driver upgrade.
//! * **Recoverable offline.** Licenses load from a file/env the user controls
//!   ([`license_from_env`]), so a changed environment is fixed by dropping in a
//!   new license file — no vendor round-trip. Ship a demo/eval license so a
//!   transient misconfiguration never hard-locks a legitimate user.

use crate::{
    Entitlements, Hash, LicenseError, Module, SignedLicense, verify_license, verify_license_on_node,
};

/// How the machine is presented to the node-lock check.
#[derive(Debug, Clone)]
pub enum NodePolicy {
    /// Present no machine id. Floating licenses are accepted anywhere; a
    /// node-locked license is refused ([`LicenseError::NodeRequired`]) because
    /// there is nothing to check it against.
    Floating,
    /// Present this opaque machine id. Floating licenses still run anywhere; a
    /// node-locked license must match. Use a **stable, coarse** identifier.
    Bound(String),
}

/// Why an activation attempt did not grant entitlements.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DenyReason {
    /// No license was supplied to the gate.
    NoLicense,
    /// A license was supplied but failed verification — bad/malformed signature,
    /// outside its validity window, node mismatch, or missing a required module.
    Invalid(LicenseError),
}

impl std::fmt::Display for DenyReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self
        {
            DenyReason::NoLicense => write!(
                f,
                "no license supplied (set SCIRUST_LICENSE or SCIRUST_LICENSE_FILE, \
                 or install your license file)"
            ),
            DenyReason::Invalid(e) => write!(f, "license rejected: {e}"),
        }
    }
}

/// The outcome of an activation attempt. Always returned — never panics.
#[derive(Debug, Clone)]
pub enum Activation {
    /// The license verified and covers every required module. The caller may arm
    /// the gated capability.
    Granted(Entitlements),
    /// The gate is closed; `reason` explains why. The caller refuses the
    /// capability gracefully (a clean error), never producing degraded output.
    Denied(DenyReason),
}

impl Activation {
    /// Whether the gate is open.
    pub fn is_granted(&self) -> bool {
        matches!(self, Activation::Granted(_))
    }

    /// The verified entitlements, if granted.
    pub fn entitlements(&self) -> Option<&Entitlements> {
        match self
        {
            Activation::Granted(e) => Some(e),
            Activation::Denied(_) => None,
        }
    }

    /// The denial reason, if the gate is closed.
    pub fn denied(&self) -> Option<&DenyReason> {
        match self
        {
            Activation::Denied(r) => Some(r),
            Activation::Granted(_) => None,
        }
    }
}

/// Run the graceful licensing gate.
///
/// Verifies `license` (if present) against the trusted `root` at `now`, honoring
/// `node`, then requires every module in `required`. Returns [`Activation`] — it
/// never panics and never returns anything but a clean grant/deny decision.
///
/// A gated crate typically calls this exactly once during initialization and
/// stores the resulting token; it must not be placed on a per-operation hot path.
pub fn gate(
    license: Option<&SignedLicense>,
    root: &Hash,
    now: u64,
    node: &NodePolicy,
    required: &[Module],
) -> Activation {
    let Some(signed) = license
    else
    {
        return Activation::Denied(DenyReason::NoLicense);
    };
    let verified = match node
    {
        NodePolicy::Floating => verify_license(signed, root, now),
        NodePolicy::Bound(id) => verify_license_on_node(signed, root, now, id),
    };
    match verified
    {
        Ok(entitlements) =>
        {
            for &module in required
            {
                if let Err(e) = entitlements.require(module)
                {
                    return Activation::Denied(DenyReason::Invalid(e));
                }
            }
            Activation::Granted(entitlements)
        },
        Err(e) => Activation::Denied(DenyReason::Invalid(e)),
    }
}

/// Load a license from the environment, tried in order:
/// 1. `SCIRUST_LICENSE` — the license JSON, inline.
/// 2. `SCIRUST_LICENSE_FILE` — a path to a license file.
///
/// Returns `None` if neither is set or the content does not parse. A missing or
/// malformed license is a **closed gate**, never a panic. This env/file read is
/// the only I/O in the crate and lives here, deliberately outside the pure
/// verification core.
pub fn license_from_env() -> Option<SignedLicense> {
    if let Ok(json) = std::env::var("SCIRUST_LICENSE")
    {
        if let Ok(license) = SignedLicense::from_json(&json)
        {
            return Some(license);
        }
    }
    if let Ok(path) = std::env::var("SCIRUST_LICENSE_FILE")
    {
        if let Ok(text) = std::fs::read_to_string(&path)
        {
            if let Ok(license) = SignedLicense::from_json(&text)
            {
                return Some(license);
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{License, demo_root, demo_vendor};

    const NOW: u64 = 1_000;

    fn gpu_license(leaf: u32) -> SignedLicense {
        demo_vendor().issue_with_leaf(License::new("Acme", "L-GPU", [Module::Gpu], 0, None), leaf)
    }

    #[test]
    fn floating_license_covering_gpu_is_granted() {
        let signed = gpu_license(0);
        let act = gate(
            Some(&signed),
            &demo_root(),
            NOW,
            &NodePolicy::Floating,
            &[Module::Gpu],
        );
        assert!(act.is_granted());
        assert_eq!(act.entitlements().unwrap().licensee(), "Acme");
    }

    #[test]
    fn missing_license_is_denied_gracefully() {
        let act = gate(
            None,
            &demo_root(),
            NOW,
            &NodePolicy::Floating,
            &[Module::Gpu],
        );
        assert_eq!(act.denied(), Some(&DenyReason::NoLicense));
    }

    #[test]
    fn license_without_the_required_module_is_denied() {
        let signed = demo_vendor()
            .issue_with_leaf(License::new("Acme", "L-CPU", [Module::Core], 0, None), 1);
        let act = gate(
            Some(&signed),
            &demo_root(),
            NOW,
            &NodePolicy::Floating,
            &[Module::Gpu],
        );
        assert_eq!(
            act.denied(),
            Some(&DenyReason::Invalid(LicenseError::NotEntitled(Module::Gpu)))
        );
    }

    #[test]
    fn expired_license_is_denied() {
        let signed = demo_vendor().issue_with_leaf(
            License::new("Acme", "L-EXP", [Module::Gpu], 0, Some(NOW - 1)),
            2,
        );
        let act = gate(
            Some(&signed),
            &demo_root(),
            NOW,
            &NodePolicy::Floating,
            &[Module::Gpu],
        );
        assert!(matches!(
            act.denied(),
            Some(DenyReason::Invalid(LicenseError::Expired { .. }))
        ));
    }

    #[test]
    fn a_foreign_root_rejects_the_signature() {
        let signed = gpu_license(3);
        let foreign = crate::Vendor::from_seed(&[0x24; 32], 4).root();
        let act = gate(
            Some(&signed),
            &foreign,
            NOW,
            &NodePolicy::Floating,
            &[Module::Gpu],
        );
        assert_eq!(
            act.denied(),
            Some(&DenyReason::Invalid(LicenseError::BadSignature))
        );
    }

    #[test]
    fn node_locked_license_matches_only_the_bound_machine() {
        let signed = demo_vendor().issue_with_leaf(
            License::new("Acme", "L-NODE", [Module::Gpu], 0, None).with_node_lock("machine-A"),
            4,
        );
        // Correct machine → granted.
        assert!(
            gate(
                Some(&signed),
                &demo_root(),
                NOW,
                &NodePolicy::Bound("machine-A".to_string()),
                &[Module::Gpu],
            )
            .is_granted()
        );
        // Wrong machine → graceful NodeMismatch.
        assert_eq!(
            gate(
                Some(&signed),
                &demo_root(),
                NOW,
                &NodePolicy::Bound("machine-B".to_string()),
                &[Module::Gpu],
            )
            .denied(),
            Some(&DenyReason::Invalid(LicenseError::NodeMismatch))
        );
        // Floating policy against a node-locked license → NodeRequired, not a crash.
        assert_eq!(
            gate(
                Some(&signed),
                &demo_root(),
                NOW,
                &NodePolicy::Floating,
                &[Module::Gpu],
            )
            .denied(),
            Some(&DenyReason::Invalid(LicenseError::NodeRequired))
        );
    }

    #[test]
    fn floating_license_runs_on_any_bound_node() {
        // A non-node-locked license is accepted even when a machine id is present.
        let signed = gpu_license(5);
        assert!(
            gate(
                Some(&signed),
                &demo_root(),
                NOW,
                &NodePolicy::Bound("whatever".to_string()),
                &[Module::Gpu],
            )
            .is_granted()
        );
    }
}
