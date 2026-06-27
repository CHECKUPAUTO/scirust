//! # SciRust licensing
//!
//! A pure-Rust, deterministic, zero-FFI module-entitlement system. A vendor
//! issues a **signed license** that lists the [`Module`]s a customer may use;
//! the runtime [`verify_license`]s it against an embedded public key (a 32-byte
//! Merkle root) and gates access with [`Entitlements::require`].
//!
//! The signatures are hash-based (Lamport one-time signatures over a Merkle
//! tree, SHA-256 only — see [`hashsig`]). This keeps the whole platform's pure-
//! `sha2`, no-elliptic-curve, deterministic posture while still being genuinely
//! forgery-resistant: the vendor holds a secret seed, the binary embeds only the
//! root, and a customer cannot mint entitlements they were not sold without
//! inverting SHA-256.
//!
//! ## Two sides
//! * **Vendor** (offline, holds the secret seed): [`Vendor::issue_with_leaf`].
//! * **Runtime** (ships to customers, holds only the root): [`verify_license`]
//!   → [`Entitlements`].
//!
//! ## Example
//! ```
//! use scirust_license::{Vendor, License, Module, verify_license};
//!
//! let vendor = Vendor::from_seed(&[42u8; 32], 8);
//! let root = vendor.root(); // the only thing a verifier needs
//!
//! let license = License::new("Acme", "L-1", [Module::Navigation], 1_000, Some(2_000));
//! let signed = vendor.issue_with_leaf(license, 0);
//!
//! let ent = verify_license(&signed, &root, 1_500).expect("valid");
//! assert!(ent.allows(Module::Navigation));
//! assert!(!ent.allows(Module::Water));
//! ```

pub mod cli;
pub mod gate;
pub mod hashsig;
pub mod license;
pub mod module;

pub use hashsig::{Hash, MerkleSig, MerkleSigner, verify};
pub use license::License;
pub use module::Module;

use serde::{Deserialize, Serialize};

/// Why a license was refused.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LicenseError {
    /// The signature bytes could not be parsed.
    MalformedSignature,
    /// The signature did not verify against the trusted root.
    BadSignature,
    /// The license is outside its validity window.
    Expired {
        /// The license's expiry (Unix seconds).
        expired_at: u64,
        /// The time it was checked against (Unix seconds).
        now: u64,
    },
    /// A required module is not covered by this (otherwise valid) license.
    NotEntitled(Module),
}

impl std::fmt::Display for LicenseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self
        {
            LicenseError::MalformedSignature => write!(f, "license signature is malformed"),
            LicenseError::BadSignature =>
            {
                write!(
                    f,
                    "license signature does not verify against the trusted key"
                )
            },
            LicenseError::Expired { expired_at, now } =>
            {
                write!(f, "license expired at {expired_at} (checked at {now})")
            },
            LicenseError::NotEntitled(m) =>
            {
                write!(f, "module '{m}' is not covered by this license")
            },
        }
    }
}

impl std::error::Error for LicenseError {}

/// A license plus its hash-based signature (hex). This is what is shipped to a
/// customer as a `.license.json` file.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SignedLicense {
    /// The license payload.
    pub license: License,
    /// Hex of the [`MerkleSig`] over `license.digest()`.
    pub signature: String,
}

impl SignedLicense {
    /// Pretty-printed JSON form (the on-disk license file).
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).expect("SignedLicense always serializes")
    }

    /// Parse a license file. Errors carry a human message.
    pub fn from_json(text: &str) -> Result<Self, String> {
        serde_json::from_str(text).map_err(|e| e.to_string())
    }
}

/// The vendor side: holds the secret master seed (as a [`MerkleSigner`]) and
/// mints signed licenses. **Never ships to customers.**
pub struct Vendor {
    signer: MerkleSigner,
}

impl Vendor {
    /// Build a vendor from a 32-byte secret seed. `height` sets the number of
    /// issuable one-time licenses to `2^height`.
    pub fn from_seed(seed: &Hash, height: u32) -> Self {
        Self {
            signer: MerkleSigner::from_seed(seed, height),
        }
    }

    /// The public key (Merkle root) to embed in verifiers.
    pub fn root(&self) -> Hash {
        self.signer.root()
    }

    /// How many one-time licenses this vendor key can sign.
    pub fn capacity(&self) -> usize {
        self.signer.capacity()
    }

    /// Sign `license` using one-time leaf `leaf`.
    ///
    /// Each leaf must be used for **at most one** distinct license; reusing a
    /// leaf for two different licenses is cryptographically unsafe (it leaks
    /// Lamport secrets). The caller owns leaf allocation — e.g. a monotonically
    /// increasing counter persisted alongside the seed.
    ///
    /// # Panics
    /// If `leaf >= capacity()`.
    pub fn issue_with_leaf(&self, license: License, leaf: u32) -> SignedLicense {
        let digest = license.digest();
        let sig = self.signer.sign(leaf, &digest);
        SignedLicense {
            license,
            signature: sig.to_hex(),
        }
    }
}

/// The verified entitlements extracted from a valid signed license. Only
/// produced by [`verify_license`], so holding one is proof the signature and
/// validity window checked out.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Entitlements {
    licensee: String,
    license_id: String,
    modules: Vec<Module>,
    expires_at: Option<u64>,
}

impl Entitlements {
    /// Who the license belongs to.
    pub fn licensee(&self) -> &str {
        &self.licensee
    }

    /// The vendor-assigned license id.
    pub fn license_id(&self) -> &str {
        &self.license_id
    }

    /// The unlocked modules.
    pub fn modules(&self) -> &[Module] {
        &self.modules
    }

    /// Expiry (Unix seconds), or `None` if perpetual.
    pub fn expires_at(&self) -> Option<u64> {
        self.expires_at
    }

    /// Whether `module` is unlocked.
    pub fn allows(&self, module: Module) -> bool {
        self.modules.contains(&module)
    }

    /// Gate an operation on a module: `Ok(())` if unlocked, otherwise
    /// [`LicenseError::NotEntitled`].
    pub fn require(&self, module: Module) -> Result<(), LicenseError> {
        if self.allows(module)
        {
            Ok(())
        }
        else
        {
            Err(LicenseError::NotEntitled(module))
        }
    }
}

/// Verify a signed license against a trusted public `root` at time `now` (Unix
/// seconds), returning the [`Entitlements`] it grants.
///
/// Checks, in order: the signature parses, the signature verifies against
/// `root` (binding licensee, modules and validity window), and the license is
/// inside its validity window. Module-level gating is then done with
/// [`Entitlements::require`].
pub fn verify_license(
    signed: &SignedLicense,
    root: &Hash,
    now: u64,
) -> Result<Entitlements, LicenseError> {
    let sig = MerkleSig::from_hex(&signed.signature).ok_or(LicenseError::MalformedSignature)?;
    let digest = signed.license.digest();
    if !verify(root, &digest, &sig)
    {
        return Err(LicenseError::BadSignature);
    }
    if !signed.license.is_valid_at(now)
    {
        return Err(LicenseError::Expired {
            expired_at: signed.license.expires_at.unwrap_or(0),
            now,
        });
    }
    Ok(Entitlements {
        licensee: signed.license.licensee.clone(),
        license_id: signed.license.license_id.clone(),
        modules: signed.license.modules.clone(),
        expires_at: signed.license.expires_at,
    })
}

/// The demo vendor's secret seed.
///
/// **Demo only.** A real vendor generates a random 32-byte seed offline and
/// never ships it. It lives here purely so the bundled example, the CLI and the
/// tests can mint demo licenses that verify against [`demo_root`].
pub const fn demo_seed() -> Hash {
    *b"scirust-demo-license-seed::v1!!!"
}

/// Merkle height of the demo vendor key (`2^10 = 1024` issuable licenses).
pub const DEMO_HEIGHT: u32 = 10;

/// The demo public key (Merkle root), embedded as a verifier would embed it.
///
/// Pinned as a constant — `demo_root_matches_the_demo_vendor` proves it equals
/// the root derived from [`demo_seed`], so it can never silently drift.
pub const DEMO_ROOT_HEX: &str = "82728023e3de7243e982d04ab09a7aa20a7fdb1fa10a0df2920060abc93a7f02";

/// The demo vendor (holds [`demo_seed`]); use it to mint demo licenses.
pub fn demo_vendor() -> Vendor {
    Vendor::from_seed(&demo_seed(), DEMO_HEIGHT)
}

/// The demo public root as bytes, for passing to [`verify_license`].
pub fn demo_root() -> Hash {
    let bytes = hashsig::hex_decode(DEMO_ROOT_HEX).expect("DEMO_ROOT_HEX is valid hex");
    let mut root = [0u8; 32];
    root.copy_from_slice(&bytes);
    root
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vendor() -> Vendor {
        Vendor::from_seed(&[42u8; 32], 6)
    }

    fn nav_license() -> License {
        License::new(
            "Acme Robotics",
            "L-2026-001",
            [Module::Navigation, Module::Control],
            1_000,
            Some(2_000),
        )
    }

    #[test]
    fn issue_then_verify_grants_exactly_the_listed_modules() {
        let v = vendor();
        let signed = v.issue_with_leaf(nav_license(), 0);
        let ent = verify_license(&signed, &v.root(), 1_500).expect("valid");
        assert!(ent.allows(Module::Navigation));
        assert!(ent.allows(Module::Control));
        assert!(!ent.allows(Module::Water));
        assert_eq!(ent.require(Module::Navigation), Ok(()));
        assert_eq!(
            ent.require(Module::Water),
            Err(LicenseError::NotEntitled(Module::Water))
        );
        assert_eq!(ent.licensee(), "Acme Robotics");
    }

    #[test]
    fn a_tampered_module_list_fails_verification() {
        let v = vendor();
        let mut signed = v.issue_with_leaf(nav_license(), 1);
        // Customer tries to self-grant Water by editing the JSON payload. The
        // signature was over the original digest, so verification must fail.
        signed.license.modules.push(Module::Water);
        assert_eq!(
            verify_license(&signed, &v.root(), 1_500),
            Err(LicenseError::BadSignature)
        );
    }

    #[test]
    fn a_signature_from_another_vendor_is_rejected() {
        let real = vendor();
        let attacker = Vendor::from_seed(&[99u8; 32], 6);
        let signed = attacker.issue_with_leaf(nav_license(), 0);
        // Attacker can sign, but not against the real embedded root.
        assert_eq!(
            verify_license(&signed, &real.root(), 1_500),
            Err(LicenseError::BadSignature)
        );
    }

    #[test]
    fn an_expired_license_is_rejected_but_not_a_perpetual_one() {
        let v = vendor();
        let signed = v.issue_with_leaf(nav_license(), 2);
        assert_eq!(
            verify_license(&signed, &v.root(), 2_001),
            Err(LicenseError::Expired {
                expired_at: 2_000,
                now: 2_001
            })
        );
        let perpetual = License::new("Acme", "L-perp", [Module::Water], 1, None);
        let signed_perp = v.issue_with_leaf(perpetual, 3);
        assert!(verify_license(&signed_perp, &v.root(), u64::MAX).is_ok());
    }

    #[test]
    fn a_malformed_signature_string_is_reported_distinctly() {
        let v = vendor();
        let mut signed = v.issue_with_leaf(nav_license(), 4);
        signed.signature = "not-hex!!".to_string();
        assert_eq!(
            verify_license(&signed, &v.root(), 1_500),
            Err(LicenseError::MalformedSignature)
        );
    }

    #[test]
    fn signed_license_survives_a_json_round_trip_and_still_verifies() {
        let v = vendor();
        let signed = v.issue_with_leaf(nav_license(), 5);
        let json = signed.to_json();
        let back = SignedLicense::from_json(&json).expect("parses");
        assert_eq!(signed, back);
        assert!(verify_license(&back, &v.root(), 1_500).is_ok());
    }

    #[test]
    fn demo_root_matches_the_demo_vendor() {
        // The embedded constant must equal the root derived from the demo seed,
        // so a verifier using DEMO_ROOT_HEX accepts demo-vendor licenses.
        assert_eq!(demo_vendor().root(), demo_root());
    }

    #[test]
    fn demo_vendor_round_trip_under_the_embedded_root() {
        let signed = demo_vendor().issue_with_leaf(nav_license(), 7);
        let ent = verify_license(&signed, &demo_root(), 1_500).expect("valid under demo root");
        assert!(ent.allows(Module::Navigation));
    }
}
