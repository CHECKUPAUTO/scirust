//! The license payload and its canonical, signable encoding.

use crate::hashsig::{Hash, hex_encode};
use crate::module::Module;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// A license payload: who it is for, what it unlocks, when it expires, and
/// (optionally) which machine it is bound to.
///
/// The struct fields are authoritative. [`License::digest`] commits to *all* of
/// them via a length-prefixed canonical encoding, so a signature over the digest
/// binds the exact licensee, module set, validity window and node lock — changing
/// any field changes the digest and breaks the signature.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct License {
    /// Who the license is issued to (organisation or product name).
    pub licensee: String,
    /// Unique identifier for this license (vendor-assigned).
    pub license_id: String,
    /// The modules this license unlocks.
    pub modules: Vec<Module>,
    /// Issue time, Unix seconds.
    pub issued_at: u64,
    /// Expiry time, Unix seconds; `None` means perpetual.
    pub expires_at: Option<u64>,
    /// Node lock: the [`node_fingerprint`] of the single machine this license is
    /// bound to, or `None` for a *floating* license usable on any machine. This
    /// stores the **hash**, never the raw machine identifier, so the license file
    /// never leaks the machine's identity. Enforced by
    /// [`verify_license_on_node`](crate::verify_license_on_node).
    #[serde(default)]
    pub node_lock: Option<Hash>,
}

impl License {
    /// Construct a license, normalising the module set (sorted, de-duplicated)
    /// so equal entitlements always produce an equal canonical encoding.
    pub fn new(
        licensee: impl Into<String>,
        license_id: impl Into<String>,
        modules: impl IntoIterator<Item = Module>,
        issued_at: u64,
        expires_at: Option<u64>,
    ) -> Self {
        let mut modules: Vec<Module> = modules.into_iter().collect();
        modules.sort_by_key(|m| m.code());
        modules.dedup();
        Self {
            licensee: licensee.into(),
            license_id: license_id.into(),
            modules,
            issued_at,
            expires_at,
            node_lock: None,
        }
    }

    /// Bind this license to a single machine, identified by the opaque,
    /// host-supplied `machine_id` (a provisioned UUID, `/etc/machine-id`, a TPM
    /// value — whatever the deployment treats as the machine's stable identity).
    /// Only the [`node_fingerprint`] (a SHA-256) is stored, so the license never
    /// reveals the raw identifier. Builder-style: `License::new(..).with_node_lock(id)`.
    ///
    /// The crate deliberately does **not** discover the machine itself — that
    /// would require platform-specific I/O and break the pure, deterministic,
    /// `no_std`-friendly posture. The host presents its own id at verification
    /// time via [`verify_license_on_node`](crate::verify_license_on_node).
    pub fn with_node_lock(mut self, machine_id: &str) -> Self {
        self.node_lock = Some(node_fingerprint(machine_id));
        self
    }

    /// Canonical, unambiguous byte encoding of the license. Every variable-
    /// length field is length-prefixed and the module set is sorted by code, so
    /// two licenses encode to the same bytes **iff** they are semantically equal
    /// — no field-separator injection can make two different licenses collide.
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut o = Vec::new();
        // Magic + format version. v2 added the trailing node-lock field; the
        // version byte makes a v1 and a v2 encoding of "the same" license hash
        // differently, so signatures never cross formats silently.
        o.extend_from_slice(b"SRL2");
        put_bytes(&mut o, self.licensee.as_bytes());
        put_bytes(&mut o, self.license_id.as_bytes());
        o.extend_from_slice(&self.issued_at.to_le_bytes());
        match self.expires_at
        {
            Some(t) =>
            {
                o.push(1);
                o.extend_from_slice(&t.to_le_bytes());
            },
            None => o.push(0),
        }
        // Modules: sorted by code, length-prefixed, two bytes each.
        let mut codes: Vec<u16> = self.modules.iter().map(|m| m.code()).collect();
        codes.sort_unstable();
        codes.dedup();
        o.extend_from_slice(&(codes.len() as u32).to_le_bytes());
        for c in codes
        {
            o.extend_from_slice(&c.to_le_bytes());
        }
        // Node lock: presence byte, then the 32-byte fingerprint if bound.
        match self.node_lock
        {
            Some(fp) =>
            {
                o.push(1);
                o.extend_from_slice(&fp);
            },
            None => o.push(0),
        }
        o
    }

    /// SHA-256 of the canonical encoding — the 256-bit value that gets signed.
    pub fn digest(&self) -> Hash {
        let mut h = Sha256::new();
        h.update(self.canonical_bytes());
        h.finalize().into()
    }

    /// Hex of [`License::digest`], handy for logs and the CLI.
    pub fn digest_hex(&self) -> String {
        hex_encode(&self.digest())
    }

    /// Whether the license is within its validity window at `now` (Unix
    /// seconds). A perpetual license (`expires_at == None`) is always valid.
    pub fn is_valid_at(&self, now: u64) -> bool {
        match self.expires_at
        {
            None => true,
            Some(t) => now <= t,
        }
    }

    /// Whether this license lists `module`.
    pub fn grants(&self, module: Module) -> bool {
        self.modules.contains(&module)
    }

    /// Whether this license is bound to a specific machine.
    pub fn is_node_locked(&self) -> bool {
        self.node_lock.is_some()
    }

    /// Whether this license may run on the machine identified by `machine_id`:
    /// always `true` for a floating license, otherwise `true` iff the id's
    /// [`node_fingerprint`] matches the bound lock.
    pub fn allows_node(&self, machine_id: &str) -> bool {
        match self.node_lock
        {
            None => true,
            Some(fp) => fp == node_fingerprint(machine_id),
        }
    }
}

/// Fingerprint an opaque machine identifier for node-locking: a domain-separated
/// SHA-256 of `machine_id`. Deterministic and pure — the same id always yields
/// the same 32 bytes, and the raw id is never recoverable from the hash. The
/// domain tag keeps these fingerprints from colliding with any other SHA-256 use
/// in the platform.
pub fn node_fingerprint(machine_id: &str) -> Hash {
    let mut h = Sha256::new();
    h.update(b"scirust-node-lock:v1\0");
    h.update(machine_id.as_bytes());
    h.finalize().into()
}

/// Append `len(u32 LE) ‖ bytes`.
fn put_bytes(out: &mut Vec<u8>, bytes: &[u8]) {
    out.extend_from_slice(&(bytes.len() as u32).to_le_bytes());
    out.extend_from_slice(bytes);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn module_order_does_not_change_the_digest() {
        let a = License::new(
            "Acme",
            "L-1",
            [Module::Navigation, Module::Water, Module::Battery],
            1000,
            None,
        );
        let b = License::new(
            "Acme",
            "L-1",
            [Module::Battery, Module::Navigation, Module::Water],
            1000,
            None,
        );
        // Same entitlements in a different order → identical canonical bytes.
        assert_eq!(a.canonical_bytes(), b.canonical_bytes());
        assert_eq!(a.digest(), b.digest());
    }

    #[test]
    fn duplicate_modules_are_collapsed() {
        let a = License::new("Acme", "L-1", [Module::Water], 1, None);
        let b = License::new("Acme", "L-1", [Module::Water, Module::Water], 1, None);
        assert_eq!(a.digest(), b.digest());
        assert_eq!(b.modules.len(), 1);
    }

    #[test]
    fn changing_any_field_changes_the_digest() {
        let base = License::new("Acme", "L-1", [Module::Water], 1000, Some(2000));
        let cases = [
            License::new("Acme2", "L-1", [Module::Water], 1000, Some(2000)),
            License::new("Acme", "L-2", [Module::Water], 1000, Some(2000)),
            License::new("Acme", "L-1", [Module::Grid], 1000, Some(2000)),
            License::new("Acme", "L-1", [Module::Water], 1001, Some(2000)),
            License::new("Acme", "L-1", [Module::Water], 1000, Some(2001)),
            License::new("Acme", "L-1", [Module::Water], 1000, None),
            License::new(
                "Acme",
                "L-1",
                [Module::Water, Module::Grid],
                1000,
                Some(2000),
            ),
        ];
        for other in cases
        {
            assert_ne!(
                base.digest(),
                other.digest(),
                "digest collided with {other:?}"
            );
        }
    }

    #[test]
    fn separator_injection_cannot_forge_a_collision() {
        // Length-prefixing means moving bytes across the licensee/id boundary
        // produces different bytes, not a collision.
        let a = License::new("ab", "c", [], 0, None);
        let b = License::new("a", "bc", [], 0, None);
        assert_ne!(a.canonical_bytes(), b.canonical_bytes());
    }

    #[test]
    fn validity_window_is_inclusive_and_perpetual_never_expires() {
        let expiring = License::new("Acme", "L", [Module::Water], 100, Some(200));
        assert!(expiring.is_valid_at(100));
        assert!(expiring.is_valid_at(200)); // inclusive upper bound
        assert!(!expiring.is_valid_at(201));
        let perpetual = License::new("Acme", "L", [Module::Water], 100, None);
        assert!(perpetual.is_valid_at(u64::MAX));
    }

    #[test]
    fn json_round_trips_and_preserves_the_digest() {
        let lic = License::new(
            "Acme Robotics",
            "L-2026-001",
            [Module::Robotics, Module::Control],
            1_700_000_000,
            Some(1_800_000_000),
        );
        let json = serde_json::to_string(&lic).unwrap();
        let back: License = serde_json::from_str(&json).unwrap();
        assert_eq!(lic, back);
        assert_eq!(lic.digest(), back.digest());
    }

    #[test]
    fn node_fingerprint_is_deterministic_and_distinguishing() {
        // Same id → same hash; different id → different hash; raw id absent.
        assert_eq!(node_fingerprint("machine-A"), node_fingerprint("machine-A"));
        assert_ne!(node_fingerprint("machine-A"), node_fingerprint("machine-B"));
        let fp = node_fingerprint("machine-A");
        assert_ne!(fp, [0u8; 32], "fingerprint is not trivially zero");
    }

    #[test]
    fn binding_to_a_node_changes_the_digest_and_stores_only_the_hash() {
        let floating = License::new("Acme", "L-1", [Module::Water], 1000, None);
        let locked = floating.clone().with_node_lock("machine-A");
        // The lock is part of the signed encoding, so it shifts the digest.
        assert_ne!(floating.digest(), locked.digest());
        // Only the fingerprint is stored — never the raw identifier.
        assert_eq!(locked.node_lock, Some(node_fingerprint("machine-A")));
        let json = serde_json::to_string(&locked).unwrap();
        assert!(
            !json.contains("machine-A"),
            "raw machine id must not leak: {json}"
        );
    }

    #[test]
    fn allows_node_is_permissive_when_floating_and_strict_when_locked() {
        let floating = License::new("Acme", "L-1", [Module::Water], 1000, None);
        assert!(!floating.is_node_locked());
        assert!(floating.allows_node("anything")); // floating runs anywhere

        let locked = floating.with_node_lock("machine-A");
        assert!(locked.is_node_locked());
        assert!(locked.allows_node("machine-A"));
        assert!(!locked.allows_node("machine-B"));
    }

    #[test]
    fn a_locked_license_json_round_trips_and_old_json_defaults_to_floating() {
        let locked = License::new("Acme", "L-1", [Module::Water], 1, None).with_node_lock("m1");
        let back: License = serde_json::from_str(&serde_json::to_string(&locked).unwrap()).unwrap();
        assert_eq!(locked, back);
        assert_eq!(locked.digest(), back.digest());

        // A v1-style license file (no `node_lock` key) still parses, as floating.
        let legacy = r#"{"licensee":"Old","license_id":"L-0","modules":["water"],"issued_at":1,"expires_at":null}"#;
        let parsed: License = serde_json::from_str(legacy).unwrap();
        assert!(!parsed.is_node_locked());
    }
}
