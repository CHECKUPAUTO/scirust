//! License-gated access to the grid module.
//!
//! `scirust-grid` is dual-licensed: the raw analytics ([`crate::grid_frequency`],
//! [`crate::rocof`], [`crate::synchrophasor`], [`crate::thd`]) stay public for
//! noncommercial use, while **commercial** use unlocks the module through a
//! signed entitlement.
//!
//! The gate is a capability token, [`GridModule`]. It has no public constructor —
//! the *only* way to obtain one is [`GridModule::unlock`], which requires verified
//! [`Entitlements`] (from [`scirust_license::verify_license`]) that cover
//! [`Module::Grid`]. The gated analytics are its methods, so the licensed entry
//! path is unreachable without the entitlement: the check is enforced by the type
//! system, not by convention. (Same pattern as `scirust-control`'s
//! `ControlModule`.)
//!
//! ```
//! use scirust_grid::license::GridModule;
//! use scirust_license::{Vendor, License, Module, verify_license};
//!
//! // Vendor side (offline, holds the secret seed): issue a license covering Grid.
//! let vendor = Vendor::from_seed(&[7u8; 32], 6);
//! let signed = vendor.issue_with_leaf(
//!     License::new("Utility Co", "L-1", [Module::Grid], 0, None),
//!     0,
//! );
//!
//! // Runtime side (holds only the public root): verify, then unlock the module.
//! let ent = verify_license(&signed, &vendor.root(), 1).expect("valid license");
//! let grid = GridModule::unlock(&ent).expect("entitled to Grid");
//!
//! // RoCoF of a 0.5 Hz/s downward frequency ramp (gated entry point).
//! let freqs: Vec<f64> = (0..10).map(|k| 50.0 - 0.5 * k as f64 * 0.1).collect();
//! assert!((grid.rocof(&freqs, 0.1) + 0.5).abs() < 1e-9);
//! ```

use crate::{grid_frequency, rocof, synchrophasor, thd};
use scirust_license::{Entitlements, LicenseError, Module};

/// The licensable module this crate belongs to.
pub const MODULE: Module = Module::Grid;

/// A capability token proving its holder is entitled to the grid module.
///
/// Constructed only by [`GridModule::unlock`] against verified [`Entitlements`];
/// the gated analytics are its methods. Because the inner field is private, no
/// value of this type can exist without having passed the entitlement check.
#[derive(Debug, Clone, Copy)]
pub struct GridModule {
    _sealed: (),
}

impl GridModule {
    /// Unlock the grid module against a verified entitlement set.
    ///
    /// Returns [`LicenseError::NotEntitled`] carrying [`Module::Grid`] if the
    /// license does not cover this module. Obtaining the [`Entitlements`] in the
    /// first place already proves the signature and validity window checked out
    /// (see [`scirust_license::verify_license`]).
    pub fn unlock(entitlements: &Entitlements) -> Result<Self, LicenseError> {
        entitlements.require(MODULE)?;
        Ok(Self { _sealed: () })
    }

    /// Dominant grid frequency near `nominal_hz` (gated [`crate::grid_frequency`]).
    pub fn grid_frequency(
        &self,
        signal: &[f64],
        sample_rate: f64,
        nominal_hz: f64,
        search_hz: f64,
    ) -> f64 {
        grid_frequency(signal, sample_rate, nominal_hz, search_hz)
    }

    /// Rate of change of frequency, Hz/s (gated [`crate::rocof`]).
    pub fn rocof(&self, freqs: &[f64], dt: f64) -> f64 {
        rocof(freqs, dt)
    }

    /// Synchrophasor magnitude/phase at `freq_hz` (gated [`crate::synchrophasor`]).
    pub fn synchrophasor(&self, signal: &[f64], sample_rate: f64, freq_hz: f64) -> (f64, f64) {
        synchrophasor(signal, sample_rate, freq_hz)
    }

    /// Total harmonic distortion (gated [`crate::thd`]).
    pub fn thd(
        &self,
        signal: &[f64],
        sample_rate: f64,
        fundamental_hz: f64,
        n_harmonics: usize,
    ) -> f64 {
        thd(signal, sample_rate, fundamental_hz, n_harmonics)
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
    fn unlock_succeeds_for_a_grid_license_and_gated_calls_match_the_raw_ones() {
        let v = vendor();
        let signed = v.issue_with_leaf(
            License::new("Utility", "L-grid", [Module::Grid], 0, None),
            0,
        );
        let ent = verify_license(&signed, &v.root(), 1).expect("valid license");
        let grid = GridModule::unlock(&ent).expect("entitled to Grid");

        // The gated analytics are exactly the raw ones — gating adds a check, not
        // a different computation.
        let freqs: Vec<f64> = (0..10).map(|k| 50.0 - 0.5 * k as f64 * 0.1).collect();
        assert_eq!(grid.rocof(&freqs, 0.1), rocof(&freqs, 0.1));

        let n = 4096usize;
        let sr = 4096.0;
        let sig: Vec<f64> = (0..n)
            .map(|i| (2.0 * core::f64::consts::PI * 50.2 * i as f64 / sr).sin())
            .collect();
        assert_eq!(
            grid.grid_frequency(&sig, sr, 50.0, 2.0),
            grid_frequency(&sig, sr, 50.0, 2.0),
        );
        assert_eq!(
            grid.synchrophasor(&sig, sr, 50.0),
            synchrophasor(&sig, sr, 50.0),
        );
        assert_eq!(grid.thd(&sig, sr, 50.0, 7), thd(&sig, sr, 50.0, 7));
    }

    #[test]
    fn unlock_is_denied_when_grid_is_not_entitled() {
        let v = vendor();
        // A valid license that covers a *different* module only.
        let signed = v.issue_with_leaf(
            License::new("Utility", "L-ctrl", [Module::Control], 0, None),
            1,
        );
        let ent = verify_license(&signed, &v.root(), 1).expect("valid license");
        assert_eq!(
            GridModule::unlock(&ent).err(),
            Some(LicenseError::NotEntitled(Module::Grid)),
        );
    }

    #[test]
    fn an_expired_grid_license_never_yields_entitlements_to_unlock() {
        let v = vendor();
        let signed = v.issue_with_leaf(
            License::new("Utility", "L-exp", [Module::Grid], 1_000, Some(2_000)),
            2,
        );
        // Verification fails on the validity window, so `unlock` is never reached.
        assert!(matches!(
            verify_license(&signed, &v.root(), 2_001),
            Err(LicenseError::Expired { .. })
        ));
    }

    #[test]
    fn self_granting_grid_by_editing_the_payload_fails_verification() {
        let v = vendor();
        // Issued WITHOUT Grid; the customer edits the JSON to add it.
        let mut signed = v.issue_with_leaf(
            License::new("Utility", "L-tamper", [Module::Control], 0, None),
            3,
        );
        signed.license.modules.push(Module::Grid);
        // The signature was over the original digest → verification fails, so the
        // attacker never obtains an `Entitlements` to call `unlock` with.
        assert_eq!(
            verify_license(&signed, &v.root(), 1).err(),
            Some(LicenseError::BadSignature),
        );
    }
}
