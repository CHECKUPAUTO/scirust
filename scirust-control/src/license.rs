//! License-gated access to the control module.
//!
//! `scirust-control` is dual-licensed: the raw constructors ([`crate::dlqr`],
//! [`Pid::new`], [`LinearMpc::new`]) stay public for noncommercial use, while
//! **commercial** use unlocks the module through a signed entitlement.
//!
//! The gate is a capability token, [`ControlModule`]. It has no public
//! constructor — the *only* way to obtain one is [`ControlModule::unlock`],
//! which requires verified [`Entitlements`] (from
//! [`scirust_license::verify_license`]) that cover [`Module::Control`]. The
//! gated controller constructors hang off that token, so the licensed entry path
//! is unreachable without the entitlement: the check is enforced by the type
//! system, not by convention.
//!
//! ```
//! use scirust_control::license::ControlModule;
//! use scirust_license::{Vendor, License, Module, verify_license};
//!
//! // Vendor side (offline, holds the secret seed): issue a license covering Control.
//! let vendor = Vendor::from_seed(&[7u8; 32], 6);
//! let signed = vendor.issue_with_leaf(
//!     License::new("Acme", "L-1", [Module::Control], 0, None),
//!     0,
//! );
//!
//! // Runtime side (holds only the public root): verify, then unlock the module.
//! let ent = verify_license(&signed, &vendor.root(), 1).expect("valid license");
//! let ctrl = ControlModule::unlock(&ent).expect("entitled to Control");
//! let _pid = ctrl.pid(2.0, 1.0, 0.05, 0.1); // gated entry point
//! ```

use crate::lqr::dlqr;
use crate::mpc::LinearMpc;
use crate::pid::Pid;
use scirust_estimation::Mat;
use scirust_license::{Entitlements, LicenseError, Module};

/// The licensable module this crate belongs to.
pub const MODULE: Module = Module::Control;

/// A capability token proving its holder is entitled to the control module.
///
/// Constructed only by [`ControlModule::unlock`] against verified
/// [`Entitlements`]; the gated controller constructors are its methods. Because
/// the inner field is private, no value of this type can exist without having
/// passed the entitlement check.
#[derive(Debug, Clone, Copy)]
pub struct ControlModule {
    _sealed: (),
}

impl ControlModule {
    /// Unlock the control module against a verified entitlement set.
    ///
    /// Returns [`LicenseError::NotEntitled`] carrying [`Module::Control`] if the
    /// license does not cover this module. Obtaining the [`Entitlements`] in the
    /// first place already proves the signature and validity window checked out
    /// (see [`scirust_license::verify_license`]).
    pub fn unlock(entitlements: &Entitlements) -> Result<Self, LicenseError> {
        entitlements.require(MODULE)?;
        Ok(Self { _sealed: () })
    }

    /// Discrete infinite-horizon LQR gain (gated [`crate::dlqr`]).
    pub fn dlqr(&self, a: &Mat, b: &Mat, q: &Mat, r: &Mat) -> Option<Mat> {
        dlqr(a, b, q, r)
    }

    /// PID controller with anti-windup (gated [`Pid::new`]).
    pub fn pid(&self, kp: f64, ki: f64, kd: f64, dt: f64) -> Pid {
        Pid::new(kp, ki, kd, dt)
    }

    /// Condensed linear MPC with certified input bounds (gated [`LinearMpc::new`]).
    // Mirrors `LinearMpc::new`'s parameter list (with the `&self` token added).
    #[allow(clippy::too_many_arguments)]
    pub fn linear_mpc(
        &self,
        a: Mat,
        b: Mat,
        q: Mat,
        r: Mat,
        horizon: usize,
        u_min: f64,
        u_max: f64,
    ) -> LinearMpc {
        LinearMpc::new(a, b, q, r, horizon, u_min, u_max)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirust_license::{License, Vendor, verify_license};

    // A self-contained test vendor (height 6 → 64 one-time leaves).
    fn vendor() -> Vendor {
        Vendor::from_seed(&[7u8; 32], 6)
    }

    #[test]
    fn unlock_succeeds_for_a_control_license_and_gated_calls_match_the_raw_ones() {
        let v = vendor();
        let signed = v.issue_with_leaf(
            License::new("Acme", "L-ctrl", [Module::Control], 0, None),
            0,
        );
        let ent = verify_license(&signed, &v.root(), 1).expect("valid license");
        let ctrl = ControlModule::unlock(&ent).expect("entitled to Control");

        // The gated LQR is exactly the raw LQR — gating adds a check, not a change
        // in the computation.
        let a = Mat::new(1, 1, vec![1.2]);
        let b = Mat::new(1, 1, vec![1.0]);
        let q = Mat::new(1, 1, vec![1.0]);
        let r = Mat::new(1, 1, vec![1.0]);
        let k_gated = ctrl.dlqr(&a, &b, &q, &r).expect("lqr");
        let k_raw = dlqr(&a, &b, &q, &r).expect("lqr");
        assert_eq!(k_gated.data, k_raw.data);

        // The other gated constructors are reachable through the token.
        let _pid = ctrl.pid(2.0, 1.0, 0.05, 0.1);
        let _mpc = ctrl.linear_mpc(
            Mat::new(2, 2, vec![1.0, 0.1, 0.0, 1.0]),
            Mat::new(2, 1, vec![0.005, 0.1]),
            Mat::new(2, 2, vec![1.0, 0.0, 0.0, 1.0]),
            Mat::new(1, 1, vec![0.1]),
            5,
            -1.0,
            1.0,
        );
    }

    #[test]
    fn unlock_is_denied_when_control_is_not_entitled() {
        let v = vendor();
        // A valid license that covers a *different* module only.
        let signed =
            v.issue_with_leaf(License::new("Acme", "L-water", [Module::Water], 0, None), 1);
        let ent = verify_license(&signed, &v.root(), 1).expect("valid license");
        assert_eq!(
            ControlModule::unlock(&ent).err(),
            Some(LicenseError::NotEntitled(Module::Control)),
        );
    }

    #[test]
    fn an_expired_control_license_never_yields_entitlements_to_unlock() {
        let v = vendor();
        let signed = v.issue_with_leaf(
            License::new("Acme", "L-exp", [Module::Control], 1_000, Some(2_000)),
            2,
        );
        // Verification fails on the validity window, so `unlock` is never reached.
        assert!(matches!(
            verify_license(&signed, &v.root(), 2_001),
            Err(LicenseError::Expired { .. })
        ));
    }

    #[test]
    fn self_granting_control_by_editing_the_payload_fails_verification() {
        let v = vendor();
        // Issued WITHOUT Control; the customer edits the JSON to add it.
        let mut signed = v.issue_with_leaf(
            License::new("Acme", "L-tamper", [Module::Water], 0, None),
            3,
        );
        signed.license.modules.push(Module::Control);
        // The signature was over the original digest → verification fails, so the
        // attacker never obtains an `Entitlements` to call `unlock` with.
        assert_eq!(
            verify_license(&signed, &v.root(), 1).err(),
            Some(LicenseError::BadSignature),
        );
    }
}
