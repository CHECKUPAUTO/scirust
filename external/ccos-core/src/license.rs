//! **Zero-knowledge, offline license gating** for CCOS *Pro* features.
//!
//! Design constraints (by the project owner):
//! - **Nothing leaves the host.** No network calls, no telemetry, no phone-home. A license is a
//!   locally-verified, signed token — the engine holds a **public key**, the vendor signs with the
//!   matching **private key**, and verification is a pure offline signature check. A customer can
//!   run CCOS fully air-gapped.
//! - **The core is never gated, never degraded.** Ingestion, the causal graph, and the Q-Page
//!   belief / decay / propagation primitives are always available in the free **community** tier. An
//!   unlicensed engine is *not* made "vague" or silently wrong — it simply **gates the advanced
//!   features and logs, explicitly, how to obtain a key**. (This is the fail-closed / announced
//!   model — the deliberately-deceptive "degrade confidence under an invalid license" idea is *not*
//!   implemented here, by design.)
//! - **The dollar funds the user's own control surface**, not surveillance: the Pro features are
//!   per-source authority weighting, cognitive-tension visualization in the logs, and audit-report
//!   generation — tools the operator points *at their own system*.
//!
//! This module is the **gate**: tiers, the feature set, and the explicit-logging policy. The gate and
//! the verifier are **pure** — the single [`load_license_blob`] helper is the one explicit, opt-in I/O
//! entry point (an env var or a local file; never a network call). The public-key signature check
//! ([`LicenseVerifier`]) is pluggable; the bundled ed25519 verifier ([`Ed25519Verifier`]) is provided
//! behind the `license` cargo feature so the default build pulls in no cryptography.

use std::fmt;

include!(concat!(env!("OUT_DIR"), "/license_build_keys.rs"));

/// Current signed-license token format.
pub const LICENSE_TOKEN_VERSION: u32 = 1;
/// Hard input bound applied before UTF-8, base64, JSON, or signature parsing.
pub const MAX_LICENSE_TOKEN_BYTES: usize = 64 * 1024;
/// Hard limit for a signed offline revocation list.
pub const MAX_REVOCATION_LIST_BYTES: usize = 1024 * 1024;
/// Current signed revocation-list payload version.
pub const REVOCATION_LIST_VERSION: u32 = 1;
const ED25519_ALGORITHM: &str = "ed25519";
const SLH_DSA_ALGORITHM: &str = "slh-dsa-shake-128s";

/// Build-time public-key provenance. This is deliberately public metadata: a
/// verification key is not secret, but its origin must be unambiguous.
pub fn license_build_profile() -> &'static str {
    LICENSE_BUILD_PROFILE
}

/// Key identifiers embedded by `CCOS_LICENSE_PUBLIC_KEYS_FILE` at build time.
pub fn embedded_license_key_ids() -> Vec<&'static str> {
    EMBEDDED_ED25519_KEYS
        .iter()
        .chain(EMBEDDED_SLH_DSA_KEYS.iter())
        .map(|(kid, _)| *kid)
        .collect()
}

/// A licensed (*Pro*) capability. The **core** of CCOS is never one of these — only advanced,
/// operator-facing tooling is gated.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Feature {
    /// Per-source **custom authority weighting** (vs. the uniform default authority).
    CustomAuthorityWeights,
    /// **Cognitive-tension visualization** in the logs (rendering `qbelief` conflict per claim).
    TensionVisualization,
    /// **Audit-report generation** (belief / conflict / provenance of the knowledge base).
    AuditReports,
    /// **SLHAv2 grouped INT4 embeddings** — the adaptive per-group INT4 quantization (group
    /// size 16) that keeps cosine fidelity high when vector magnitudes vary across dims (the
    /// "SLHAv2 two-level INT4" distilled from SCIRUST's KV-cache). A community session falls
    /// back to **uniform** INT4 (a single per-vector absmax scale); Pro keeps the grouped
    /// scheme. The core recall path is unchanged — only the *precision* of the semantic
    /// embedding store reflects the tier, exactly like [`Feature::CustomAuthorityWeights`].
    SlhAv2Embeddings,
    /// **Adaptive retrieval** — the `ccos_core::retrieval` self-improving feedback loop
    /// (`ImprovementLoop`: learn a projection from confirmed (query, relevant-doc) pairs so Recall@k
    /// climbs). The *core* retrieval (dense / BM25 / hybrid + metrics) is free and fully functional,
    /// exactly like the rest of CCOS's core; only the continuous-improvement tier is gated.
    AdaptiveRetrieval,
    /// **OctaSoma semantic memory** — the region-sharded, embedding-based semantic-anchor
    /// backend (`ccos_core::octa_index`, compiled behind the `octasoma` cargo feature): true-embedding
    /// recall resolved *within* a causal region and expanded through the causal graph — the
    /// validated scope→rerank cascade. The free core recall strategies (working-set / around /
    /// task / the INT4 TF-IDF `Semantic`/`Hybrid` entries) are untouched; only the
    /// OctaSoma-backed index is Pro, exactly like [`Feature::AdaptiveRetrieval`].
    OctaSomaMemory,
}

impl Feature {
    /// Stable human-readable name (used in logs and errors).
    pub fn name(self) -> &'static str {
        match self {
            Feature::CustomAuthorityWeights => "custom-authority-weights",
            Feature::TensionVisualization => "tension-visualization",
            Feature::AuditReports => "audit-reports",
            Feature::SlhAv2Embeddings => "slhav2-embeddings",
            Feature::AdaptiveRetrieval => "adaptive-retrieval",
            Feature::OctaSomaMemory => "octasoma-memory",
        }
    }

    /// Every Pro feature — for enumerating the gate.
    pub const ALL: [Feature; 6] = [
        Feature::CustomAuthorityWeights,
        Feature::TensionVisualization,
        Feature::AuditReports,
        Feature::SlhAv2Embeddings,
        Feature::AdaptiveRetrieval,
        Feature::OctaSomaMemory,
    ];
}

impl fmt::Display for Feature {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// The active licensing tier.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tier {
    /// Free — the full core, no Pro features.
    Community,
    /// Licensed — Pro features unlocked.
    Pro,
}

/// A **verified** license. Only a [`LicenseVerifier`] produces one (from a signed token); it is never
/// fabricated from untrusted input.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct License {
    /// Who the license was issued to (for the audit trail / logs).
    pub licensee: String,
    /// Expiry in unix seconds; `None` = perpetual.
    pub expires_at: Option<u64>,
    /// Host machine fingerprint; `None` = unlocked (runs on any host).
    pub machine: Option<String>,
}

impl License {
    /// Whether the license is still in force at `now` (unix seconds).
    pub fn is_valid_at(&self, now: u64) -> bool {
        self.expires_at.is_none_or(|e| now <= e)
    }
    /// Whether the license is bound to the given host fingerprint.
    /// An unbound license (`machine` is `None`) matches any host.
    pub fn machine_ok(&self, host_fp: Option<&str>) -> bool {
        match (&self.machine, host_fp) {
            (None, _) => true,
            (Some(bound), Some(host)) => bound == host,
            (Some(_), None) => false,
        }
    }
}

/// Why a Pro action was refused (or how verification failed). A refusal **never** degrades the core.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LicenseError {
    /// No license present — running in the free community tier.
    NoLicense,
    /// The license is past its expiry.
    Expired,
    /// Malformed token or bad signature — never trusted.
    Invalid(String),
    /// A Pro `feature` was requested without an active license.
    FeatureLocked(Feature),
}

impl fmt::Display for LicenseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LicenseError::NoLicense => write!(f, "no license present (community tier)"),
            LicenseError::Expired => write!(f, "license expired"),
            LicenseError::Invalid(why) => write!(f, "invalid license: {why}"),
            LicenseError::FeatureLocked(feat) => write!(
                f,
                "the Pro feature '{feat}' requires an active license (the core is unaffected)"
            ),
        }
    }
}

impl std::error::Error for LicenseError {}

/// Verifies a license **entirely locally** — no network, no telemetry, no data leaves the host. An
/// implementation MUST be pure (an offline signature + format + expiry check only): this is the
/// zero-knowledge contract that lets a customer run CCOS air-gapped. `now` is unix seconds, supplied
/// by the caller so the verifier itself reads no clock.
pub trait LicenseVerifier {
    fn verify(&self, blob: &[u8], now: u64) -> Result<License, LicenseError>;
}

/// The default verifier: it holds no public key, so every input is unlicensed → community tier. It
/// pulls in no cryptography; the real public-key (`ed25519`) verifier lives behind the `license`
/// cargo feature and also implements [`LicenseVerifier`].
#[derive(Debug, Default, Clone, Copy)]
pub struct CommunityVerifier;

impl LicenseVerifier for CommunityVerifier {
    fn verify(&self, _blob: &[u8], _now: u64) -> Result<License, LicenseError> {
        Err(LicenseError::NoLicense)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Offline ed25519 verifier + signed-token format (behind the `license` feature)
// ─────────────────────────────────────────────────────────────────────────────

/// The vendor's **ed25519 public key**, baked into the binary. A license token is signed by the
/// matching private key — held only by the vendor, never in this tree — and verification is a pure
/// offline signature check against this constant. A deployment with its own key replaces these 32
/// bytes with its own public key (its private half then signs that deployment's licenses). An unset
/// value (the placeholder below) or any non-point makes [`Ed25519Verifier`] license **nothing** →
/// community tier, so a build that never set a real key fails **closed**, never open.
///
/// Regenerate with `cargo run --features license --example license_sign keygen`.
#[cfg(feature = "license")]
pub const LICENSE_PUBLIC_KEY: [u8; 32] = [0u8; 32];

/// The signed-token payload: who, and until when. Compact-JSON + base64url is the token's first
/// segment. Shared by every compiled-in verifier (ed25519 behind `license`, SLH-DSA behind
/// `license-pq`), so it lives behind the union of those features.
#[cfg(any(feature = "license", feature = "license-pq"))]
#[derive(serde::Serialize, serde::Deserialize)]
struct TokenPayload {
    /// Licensee (organisation / deployment name) — surfaced in the audit log.
    licensee: String,
    /// Expiry, unix seconds. Absent = perpetual.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    exp: Option<u64>,
    /// Host machine fingerprint; `None` = unlocked (runs on any host).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    machine: Option<String>,
}

/// URL-safe base64 **without padding** (RFC 4648 §5: `-`/`_`, no `=`). Hand-rolled so neither license
/// feature's only new dependency is its signature primitive — the same reason CCOS hand-rolls its hex.
/// Shared by the ed25519 and SLH-DSA verifiers, and by `signed-sync` bundle signatures.
#[cfg(any(feature = "license", feature = "license-pq", feature = "signed-sync"))]
pub(crate) fn b64url_encode(bytes: &[u8]) -> String {
    const A: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for chunk in bytes.chunks(3) {
        let b0 = chunk[0] as u32;
        let b1 = *chunk.get(1).unwrap_or(&0) as u32;
        let b2 = *chunk.get(2).unwrap_or(&0) as u32;
        let n = (b0 << 16) | (b1 << 8) | b2;
        out.push(A[(n >> 18) as usize & 63] as char);
        out.push(A[(n >> 12) as usize & 63] as char);
        if chunk.len() > 1 {
            out.push(A[(n >> 6) as usize & 63] as char);
        }
        if chunk.len() > 2 {
            out.push(A[n as usize & 63] as char);
        }
    }
    out
}

/// Inverse of [`b64url_encode`]. `None` on any non-alphabet byte or a truncated group.
#[cfg(any(feature = "license", feature = "license-pq", feature = "signed-sync"))]
pub(crate) fn b64url_decode(s: &str) -> Option<Vec<u8>> {
    let val = |c: u8| -> Option<u32> {
        Some(match c {
            b'A'..=b'Z' => (c - b'A') as u32,
            b'a'..=b'z' => (c - b'a' + 26) as u32,
            b'0'..=b'9' => (c - b'0' + 52) as u32,
            b'-' => 62,
            b'_' => 63,
            _ => return None,
        })
    };
    let s = s.as_bytes();
    let mut out = Vec::with_capacity(s.len() / 4 * 3 + 3);
    for chunk in s.chunks(4) {
        if chunk.len() < 2 {
            return None; // a lone trailing char encodes no full byte
        }
        let mut n = 0u32;
        for (i, &c) in chunk.iter().enumerate() {
            n |= val(c)? << (18 - 6 * i);
        }
        out.push((n >> 16) as u8);
        if chunk.len() > 2 {
            out.push((n >> 8) as u8);
        }
        if chunk.len() > 3 {
            out.push(n as u8);
        }
    }
    Some(out)
}

/// Sign a license token with the 32-byte ed25519 **signing seed** (private key material): emits
/// `base64url(payload).base64url(signature)`, the signature taken over the first segment's ASCII
/// bytes (JWT convention). Vendor-side tooling and the tests use this; the engine only ever *verifies*.
#[cfg(feature = "license")]
pub fn sign_token(signing_seed: &[u8; 32], licensee: &str, exp: Option<u64>) -> String {
    use ed25519_dalek::{Signer, SigningKey};
    let payload = TokenPayload {
        licensee: licensee.to_string(),
        exp,
        machine: None,
    };
    let json = serde_json::to_vec(&payload).expect("payload serialises");
    let signing_input = b64url_encode(&json);
    let sk = SigningKey::from_bytes(signing_seed);
    let sig = sk.sign(signing_input.as_bytes());
    format!("{signing_input}.{}", b64url_encode(&sig.to_bytes()))
}

/// The offline **ed25519 license verifier**: a pure signature + format check against a public key
/// (the baked-in [`LICENSE_PUBLIC_KEY`] by default). No I/O, no clock, no network — the zero-knowledge
/// contract that lets a customer run air-gapped. An unset / invalid embedded key licenses nothing.
#[cfg(feature = "license")]
#[derive(Clone)]
pub struct Ed25519Verifier {
    key: Option<ed25519_dalek::VerifyingKey>,
}

#[cfg(feature = "license")]
impl Default for Ed25519Verifier {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(feature = "license")]
impl Ed25519Verifier {
    /// Verifier bound to the baked-in vendor key ([`LICENSE_PUBLIC_KEY`]). The all-zero placeholder
    /// shipped in this open tree means *no key was set* → it licenses nothing, so the default build is
    /// **fail-closed**: a deployment must paste its own public key (via the `license_sign keygen` tool)
    /// before any token can unlock Pro.
    pub fn new() -> Self {
        if LICENSE_PUBLIC_KEY == [0u8; 32] {
            return Self { key: None };
        }
        Self::with_public_key(&LICENSE_PUBLIC_KEY)
    }

    /// Verifier bound to an explicit public key — the tests sign with a throwaway keypair and verify
    /// against its public half, never the embedded vendor key.
    pub fn with_public_key(public_key: &[u8; 32]) -> Self {
        Self {
            key: ed25519_dalek::VerifyingKey::from_bytes(public_key).ok(),
        }
    }
}

#[cfg(feature = "license")]
impl LicenseVerifier for Ed25519Verifier {
    /// Verify `blob` (a `payload.sig` token, tolerant of trailing whitespace from a file) and return
    /// the encoded [`License`] on a good signature. Temporal validity is **not** checked here — a
    /// signature-valid but expired token still parses, and [`Licensing::tier`] reports it as community
    /// (so the CLI can say *expired on X* while keeping the licensee for the audit log). `now` is thus
    /// unused; the check is pure signature + format.
    fn verify(&self, blob: &[u8], _now: u64) -> Result<License, LicenseError> {
        let key = self
            .key
            .as_ref()
            .ok_or_else(|| LicenseError::Invalid("no embedded public key".into()))?;
        let token = std::str::from_utf8(blob)
            .map_err(|_| LicenseError::Invalid("token is not UTF-8".into()))?
            .trim();
        let (signing_input, sig_b64) = token
            .split_once('.')
            .ok_or_else(|| LicenseError::Invalid("token is not payload.signature".into()))?;
        let sig_bytes = b64url_decode(sig_b64)
            .filter(|s| s.len() == 64)
            .ok_or_else(|| LicenseError::Invalid("signature is not 64 base64url bytes".into()))?;
        let sig_array: [u8; 64] = sig_bytes.try_into().expect("length checked to be 64");
        let sig = ed25519_dalek::Signature::from_bytes(&sig_array);
        use ed25519_dalek::Verifier;
        key.verify(signing_input.as_bytes(), &sig)
            .map_err(|_| LicenseError::Invalid("bad signature".into()))?;
        let json = b64url_decode(signing_input)
            .ok_or_else(|| LicenseError::Invalid("payload is not base64url".into()))?;
        let payload: TokenPayload = serde_json::from_slice(&json)
            .map_err(|e| LicenseError::Invalid(format!("payload JSON: {e}")))?;
        Ok(License {
            licensee: payload.licensee,
            expires_at: payload.exp,
            machine: payload.machine,
        })
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Offline SLH-DSA (FIPS 205, post-quantum) verifier + signed-token format
// (behind the `license-pq` feature)
// ─────────────────────────────────────────────────────────────────────────────
//
// A second, fully independent verifier alongside ed25519. SLH-DSA is NIST's
// stateless hash-based post-quantum signature scheme (FIPS 205, formerly
// SPHINCS+); it relies only on hashes, so it is conjectured secure against a
// large-scale quantum computer where ed25519 (Discrete-Log) is not. We use the
// **SLH-DSA-SHAKE-128s** parameter set: a 32-byte public key (the same shape as
// ed25519, so the fail-closed all-zero placeholder transfers verbatim) and a
// 7,856-byte signature (~10.5 KB base64url) — the smallest FIPS 205 signature,
// NIST PQ category 1 (~128-bit post-quantum), a like-for-like PQ upgrade of
// ed25519's classical 128-bit security.
//
// The token format is `slhdsa.<payload_b64>.<sig_b64>` — a `slhdsa.` **scheme
// tag** prefixes the token (so [`Licensing::detect`] can dispatch a token to the
// right verifier without trial-and-error) AND is bound into the signed message
// (the signing input is the ASCII `"slhdsa.<payload_b64>"`), so a signature
// made under one scheme can never be replayed as another. The crate
// (`lattice-slh-dsa`, pure Rust, `#![forbid(unsafe_code)]`) is **not
// independently audited** — see `docs/DEPLOYMENT.md` §4b.

/// The SLH-DSA parameter set used for license tokens: **SLH-DSA-SHAKE-128s**
/// (FIPS 205). 32-byte public key, 64-byte secret key, 7,856-byte signature.
#[cfg(feature = "license-pq")]
const SLH_DSA_MODE: slh_dsa::SlhDsaMode = slh_dsa::params::SLH_DSA_SHAKE_128S;

/// Signature length in bytes for [`SLH_DSA_MODE`] (7,856 for SLH-DSA-SHAKE-128s),
/// evaluated at compile time from the parameter set.
#[cfg(feature = "license-pq")]
const SLH_DSA_SIG_LEN: usize = slh_dsa::params::SLH_DSA_SHAKE_128S.sig_bytes();

/// The vendor's **SLH-DSA public key** (32 bytes), baked into the binary. A license
/// token is signed by the matching 64-byte secret key — held only by the vendor, never
/// in this tree — and verification is a pure offline signature check against this
/// constant. As with [`LICENSE_PUBLIC_KEY`], the all-zero placeholder shipped here
/// means *no key was set* → [`SlhDsaVerifier`] licenses **nothing**, so the build is
/// **fail-closed** until a deployment pastes its own public key.
///
/// Regenerate with `cargo run --features license-pq --example license_sign_pq keygen`.
#[cfg(feature = "license-pq")]
pub const LICENSE_SLH_DSA_PUBLIC_KEY: [u8; 32] = [0u8; 32];

/// Sign a license token with the 64-byte SLH-DSA **secret key** (the `sk` half of a
/// `keygen_seed` keypair): emits `slhdsa.<payload_b64>.<sig_b64>`, the signature taken
/// over the ASCII bytes `slhdsa.<payload_b64>` (the scheme tag is bound into the signed
/// message, so it cannot be replayed as an ed25519 token). SLH-DSA signing here is
/// **deterministic** (the crate uses a fixed all-zero `optrand`), so a given secret key
/// and payload always yield the same token — vendor tokens are reproducible and tests
/// are stable. Vendor-side tooling and the tests use this; the engine only ever *verifies*.
#[cfg(feature = "license-pq")]
pub fn sign_token_slhdsa(signing_sk: &[u8; 64], licensee: &str, exp: Option<u64>) -> String {
    let payload = TokenPayload {
        licensee: licensee.to_string(),
        exp,
        machine: None,
    };
    let json = serde_json::to_vec(&payload).expect("payload serialises");
    let payload_b64 = b64url_encode(&json);
    let signing_input = format!("slhdsa.{payload_b64}");
    let sig = slh_dsa::sign(signing_sk, signing_input.as_bytes(), SLH_DSA_MODE);
    format!("slhdsa.{payload_b64}.{}", b64url_encode(&sig))
}

/// The offline **SLH-DSA license verifier**: a pure signature + format check against a
/// public key (the baked-in [`LICENSE_SLH_DSA_PUBLIC_KEY`] by default). No I/O, no clock,
/// no network — the same zero-knowledge contract as [`Ed25519Verifier`], post-quantum. An
/// unset / all-zero embedded key licenses nothing (fail-closed). The 7,856-byte signature
/// is heap-allocated by the crate, so there is no large stack frame.
#[cfg(feature = "license-pq")]
#[derive(Clone, Default)]
pub struct SlhDsaVerifier {
    /// The baked-in public key, or `None` when the placeholder is set (fail-closed).
    key: Option<[u8; 32]>,
}

#[cfg(feature = "license-pq")]
impl SlhDsaVerifier {
    /// Verifier bound to the baked-in vendor key ([`LICENSE_SLH_DSA_PUBLIC_KEY`]). The
    /// all-zero placeholder shipped in this open tree means *no key was set* → it licenses
    /// nothing, so the default build is **fail-closed**: a deployment must paste its own
    /// public key (via the `license_sign_pq keygen` tool) before any token can unlock Pro.
    pub fn new() -> Self {
        if LICENSE_SLH_DSA_PUBLIC_KEY == [0u8; 32] {
            return Self { key: None };
        }
        Self::with_public_key(&LICENSE_SLH_DSA_PUBLIC_KEY)
    }

    /// Verifier bound to an explicit public key — the tests derive a throwaway keypair
    /// and verify against its public half, never the embedded vendor key.
    pub fn with_public_key(public_key: &[u8; 32]) -> Self {
        Self {
            key: Some(*public_key),
        }
    }
}

#[cfg(feature = "license-pq")]
impl LicenseVerifier for SlhDsaVerifier {
    /// Verify `blob` (a `slhdsa.payload.sig` token, tolerant of trailing whitespace from a
    /// file) and return the encoded [`License`] on a good signature. As with ed25519,
    /// temporal validity is **not** checked here — a signature-valid but expired token still
    /// parses, and [`Licensing::tier`] reports it as community (the licensee is retained for
    /// the audit log). `now` is thus unused; the check is pure signature + format.
    fn verify(&self, blob: &[u8], _now: u64) -> Result<License, LicenseError> {
        let pk = self
            .key
            .as_ref()
            .ok_or_else(|| LicenseError::Invalid("no embedded SLH-DSA public key".into()))?;
        let token = std::str::from_utf8(blob)
            .map_err(|_| LicenseError::Invalid("token is not UTF-8".into()))?
            .trim();
        let rest = token
            .strip_prefix("slhdsa.")
            .ok_or_else(|| LicenseError::Invalid("token is not slhdsa.payload.signature".into()))?;
        let (payload_b64, sig_b64) = rest
            .split_once('.')
            .ok_or_else(|| LicenseError::Invalid("token is not slhdsa.payload.signature".into()))?;
        let sig_bytes = b64url_decode(sig_b64)
            .filter(|s| s.len() == SLH_DSA_SIG_LEN)
            .ok_or_else(|| {
                LicenseError::Invalid(format!(
                    "signature is not {SLH_DSA_SIG_LEN} base64url bytes"
                ))
            })?;
        // The scheme tag is bound into the signed message: the signing input is the
        // ASCII `"slhdsa.<payload_b64>"`, so this signature cannot verify as an ed25519
        // token (and vice-versa) — no scheme confusion, no cross-scheme replay.
        let signing_input = format!("slhdsa.{payload_b64}");
        if !slh_dsa::verify(pk, &sig_bytes, signing_input.as_bytes(), SLH_DSA_MODE) {
            return Err(LicenseError::Invalid("bad signature".into()));
        }
        let json = b64url_decode(payload_b64)
            .ok_or_else(|| LicenseError::Invalid("payload is not base64url".into()))?;
        let payload: TokenPayload = serde_json::from_slice(&json)
            .map_err(|e| LicenseError::Invalid(format!("payload JSON: {e}")))?;
        Ok(License {
            licensee: payload.licensee,
            expires_at: payload.exp,
            machine: payload.machine,
        })
    }
}

/// Load a license token from the host — **the one explicit I/O entry point** (the gate and verifier
/// are pure). Order: the `$CCOS_LICENSE` env var (the token text inline — handy in containers / CI),
/// else the file at `$CCOS_LICENSE_FILE`, else the XDG default `$XDG_CONFIG_HOME/ccos/license` (or
/// `~/.config/ccos/license`). Returns `None` when nothing is present → the community tier. Never
/// fails: an unreadable or absent file is simply "no license".
pub fn load_license_blob() -> Option<Vec<u8>> {
    if let Ok(token) = std::env::var("CCOS_LICENSE") {
        let token = token.trim();
        if !token.is_empty() {
            return Some(token.as_bytes().to_vec());
        }
    }
    let path = std::env::var_os("CCOS_LICENSE_FILE")
        .map(std::path::PathBuf::from)
        .or_else(default_license_path)?;
    std::fs::read(path).ok()
}

/// `$XDG_CONFIG_HOME/ccos/license`, else `$HOME/.config/ccos/license`.
fn default_license_path() -> Option<std::path::PathBuf> {
    use std::path::PathBuf;
    if let Some(xdg) = std::env::var_os("XDG_CONFIG_HOME") {
        return Some(PathBuf::from(xdg).join("ccos").join("license"));
    }
    std::env::var_os("HOME").map(|home| {
        PathBuf::from(home)
            .join(".config")
            .join("ccos")
            .join("license")
    })
}

/// Current unix time in seconds — a convenience for callers that gate features (the verifier itself
/// never reads a clock; `now` is always passed in). Saturates to 0 before the epoch.
pub fn now_unix() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Whether a **real vendor public key** is baked into this build (vs the all-zero placeholder that
/// licenses nothing). Without the `license` feature there is no ed25519 verifier, so this is always
/// `false`. Diagnostic only (surfaced by `ccos doctor`) — never part of verification.
pub fn embedded_key_is_set() -> bool {
    #[cfg(feature = "license")]
    {
        LICENSE_PUBLIC_KEY != [0u8; 32]
    }
    #[cfg(not(feature = "license"))]
    {
        false
    }
}

/// Whether a **real SLH-DSA vendor public key** is baked into this build (vs the all-zero
/// placeholder). Without the `license-pq` feature there is no SLH-DSA verifier, so this is always
/// `false`. Diagnostic only (surfaced by `ccos doctor`) — never part of verification. The PQ
/// analogue of [`embedded_key_is_set`].
pub fn embedded_slh_dsa_key_is_set() -> bool {
    #[cfg(feature = "license-pq")]
    {
        LICENSE_SLH_DSA_PUBLIC_KEY != [0u8; 32]
    }
    #[cfg(not(feature = "license-pq"))]
    {
        false
    }
}

/// The compiled-in verifier scheme(s), for `ccos doctor`: `"slh-dsa+ed25519"` when both
/// `license-pq` and `license` are on, `"slh-dsa"` / `"ed25519"` for one, `"none"` when no
/// verifier is compiled in (community only). Diagnostic — never part of verification.
pub fn compiled_verifier_scheme() -> &'static str {
    match (cfg!(feature = "license-pq"), cfg!(feature = "license")) {
        (true, true) => "slh-dsa+ed25519",
        (true, false) => "slh-dsa",
        (false, true) => "ed25519",
        (false, false) => "none",
    }
}

/// Resolve the license file path: `$CCOS_LICENSE_FILE` when set, else
/// `$XDG_CONFIG_HOME/ccos/license` (or `$HOME/.config/ccos/license`).
pub fn license_install_path() -> Option<std::path::PathBuf> {
    std::env::var_os("CCOS_LICENSE_FILE")
        .map(std::path::PathBuf::from)
        .or_else(default_license_path)
}

/// Parse, dispatch by scheme tag, and verify a license blob: `slhdsa.`-prefixed
/// tokens go through the SLH-DSA verifier (`license-pq`); everything else goes
/// through the ed25519 verifier (`license`). Returns a verified [`License`] on
/// success, or a [`LicenseError`] explaining the refusal.
pub fn verify_token_blob(blob: &[u8], now: u64) -> Result<License, LicenseError> {
    if blob.len() > MAX_LICENSE_TOKEN_BYTES {
        return Err(LicenseError::Invalid("token exceeds 64 KiB limit".into()));
    }
    let token = std::str::from_utf8(blob)
        .map_err(|_| LicenseError::Invalid("token is not UTF-8".into()))?
        .trim();
    let algorithm = if let Some(rest) = token.strip_prefix("ccoslic1.") {
        rest.split('.').next().unwrap_or_default()
    } else if token.starts_with("slhdsa.") {
        SLH_DSA_ALGORITHM
    } else {
        ED25519_ALGORITHM
    };
    #[cfg(feature = "license-pq")]
    if algorithm == SLH_DSA_ALGORITHM {
        return SlhDsaVerifier::new().verify(blob, now);
    }
    #[cfg(feature = "license")]
    if algorithm == ED25519_ALGORITHM {
        return Ed25519Verifier::new().verify(blob, now);
    }
    let _ = (blob, now);
    Err(LicenseError::Invalid(format!(
        "unknown or unavailable license algorithm: {algorithm}"
    )))
}

impl Licensing {
    /// Determine the active licensing from the host: load any local token ([`load_license_blob`]) and
    /// verify it with the compiled-in verifier. The token's **scheme tag** selects the verifier:
    /// a `slhdsa.`-prefixed token is checked by the offline [`SlhDsaVerifier`] (when `license-pq`
    /// is compiled in); any other token by the offline [`Ed25519Verifier`] (when `license` is
    /// compiled in). A build may compile in one, the other, or both. With neither feature (or a
    /// token whose scheme has no compiled-in verifier) there is no matching verifier, so the result
    /// is the community tier (the core is never gated). Pure beyond the single [`load_license_blob`]
    /// read; the one place CLI commands and the session obtain their licensing.
    pub fn detect(now: u64) -> Self {
        let Some(blob) = load_license_blob() else {
            return Self::community();
        };
        // Dispatch on the scheme tag. A token is SLH-DSA iff its trimmed text starts with
        // `slhdsa.`; everything else is treated as the legacy ed25519 `payload.sig` form.
        #[cfg(feature = "license-pq")]
        {
            let is_pq = std::str::from_utf8(&blob)
                .map(|s| s.trim().starts_with("slhdsa."))
                .unwrap_or(false);
            if is_pq {
                return Self::from_blob(&SlhDsaVerifier::new(), &blob, now);
            }
        }
        #[cfg(feature = "license")]
        {
            let is_pq = std::str::from_utf8(&blob)
                .map(|s| s.trim().starts_with("slhdsa."))
                .unwrap_or(false);
            if !is_pq {
                return Self::from_blob(&Ed25519Verifier::new(), &blob, now);
            }
        }
        // No compiled-in verifier matches this token's scheme (or no verifier at all) →
        // community. The core is never gated.
        let _ = (blob, now);
        Self::community()
    }
}

/// Runtime license state and the **feature gate**. Holds an optional verified [`License`] and never
/// performs I/O itself. Cloneable and cheap; a single instance is threaded through the engine.
#[derive(Debug, Clone, Default)]
pub struct Licensing {
    license: Option<License>,
}

impl Licensing {
    /// The free community tier — the full core, no Pro features.
    pub fn community() -> Self {
        Self { license: None }
    }

    /// A licensed engine from an already-verified [`License`] (produced by a [`LicenseVerifier`]).
    pub fn licensed(license: License) -> Self {
        Self {
            license: Some(license),
        }
    }

    /// Verify `blob` with `verifier` and build the licensing state. On **any** failure it falls back
    /// to the community tier — a missing or invalid license must never break the core, only gate Pro.
    pub fn from_blob(verifier: &impl LicenseVerifier, blob: &[u8], now: u64) -> Self {
        match verifier.verify(blob, now) {
            Ok(license) => Self::licensed(license),
            Err(_) => Self::community(),
        }
    }

    /// The active tier at `now` (an expired license reads as community).
    pub fn tier(&self, now: u64) -> Tier {
        match &self.license {
            Some(l) if l.is_valid_at(now) => Tier::Pro,
            _ => Tier::Community,
        }
    }

    /// The licensee, if any (for the audit log).
    pub fn licensee(&self) -> Option<&str> {
        self.license.as_ref().map(|l| l.licensee.as_str())
    }

    /// Whether `feature` is unlocked at `now`. Every advanced feature is Pro in this design, so this
    /// is simply "is the tier Pro".
    pub fn allows(&self, _feature: Feature, now: u64) -> bool {
        matches!(self.tier(now), Tier::Pro)
    }

    /// **Gate a Pro `feature`.** `Ok(())` when unlocked; otherwise it emits one explicit system-log
    /// line — stating that the core is fully functional and that an annual, **locally-verified**
    /// license unlocks the feature — and returns [`LicenseError::FeatureLocked`]. There is **no**
    /// silent downgrade and no side effect beyond that log: the caller decides what to do with the
    /// refusal (typically: skip the Pro path, keep the core result).
    pub fn require(&self, feature: Feature, now: u64) -> Result<(), LicenseError> {
        if self.allows(feature, now) {
            Ok(())
        } else {
            eprintln!(
                "[ccos] license: Pro feature '{feature}' is locked — the core (ingestion, causal \
                 graph, Q-Page belief/decay/propagation) is fully functional. An annual license \
                 unlocks it and is verified entirely locally (no data leaves your infrastructure)."
            );
            Err(LicenseError::FeatureLocked(feature))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const NOW: u64 = 1_000;

    fn license(expires_at: Option<u64>) -> License {
        License {
            licensee: "acme-corp".to_string(),
            expires_at,
            machine: None,
        }
    }

    #[test]
    fn community_gates_every_pro_feature_without_degrading() {
        let l = Licensing::community();
        assert_eq!(l.tier(NOW), Tier::Community);
        assert_eq!(l.licensee(), None);
        for f in Feature::ALL {
            assert!(!l.allows(f, NOW));
            assert_eq!(l.require(f, NOW), Err(LicenseError::FeatureLocked(f)));
        }
    }

    #[test]
    fn valid_license_unlocks_every_pro_feature() {
        let l = Licensing::licensed(license(Some(NOW + 100)));
        assert_eq!(l.tier(NOW), Tier::Pro);
        assert_eq!(l.licensee(), Some("acme-corp"));
        for f in Feature::ALL {
            assert!(l.allows(f, NOW));
            assert!(l.require(f, NOW).is_ok());
        }
    }

    #[test]
    fn expired_license_falls_back_to_community() {
        let l = Licensing::licensed(license(Some(NOW - 1)));
        assert_eq!(l.tier(NOW), Tier::Community);
        assert!(!l.allows(Feature::AuditReports, NOW));
    }

    #[test]
    fn perpetual_license_never_expires() {
        let l = Licensing::licensed(license(None));
        assert_eq!(l.tier(u64::MAX), Tier::Pro);
    }

    #[test]
    fn community_verifier_is_zero_knowledge_and_never_licenses() {
        // The default verifier holds no key and reaches no network — any blob is community.
        let s = Licensing::from_blob(&CommunityVerifier, b"any-token-at-all", NOW);
        assert_eq!(s.tier(NOW), Tier::Community);
    }

    // ── ed25519 verifier + token format (behind the `license` feature) ────────
    // A throwaway TEST key: its public half is derived at runtime and passed to
    // `with_public_key`, never the embedded vendor key — so no production private
    // key lives in the tree.
    #[cfg(feature = "license")]
    const TEST_SEED: [u8; 32] = [7u8; 32];

    #[cfg(feature = "license")]
    fn test_verifier() -> Ed25519Verifier {
        let sk = ed25519_dalek::SigningKey::from_bytes(&TEST_SEED);
        Ed25519Verifier::with_public_key(&sk.verifying_key().to_bytes())
    }

    #[cfg(feature = "license")]
    #[test]
    fn b64url_round_trips_without_padding() {
        let cases: [&[u8]; 8] = [
            b"",
            b"f",
            b"fo",
            b"foo",
            b"foob",
            b"fooba",
            b"foobar",
            &[0, 255, 1, 254],
        ];
        for case in cases {
            assert_eq!(b64url_decode(&b64url_encode(case)).as_deref(), Some(case));
        }
        assert!(!b64url_encode(b"any payload here").contains('='));
    }

    #[cfg(feature = "license")]
    #[test]
    fn signed_token_verifies_to_pro_and_unlocks_features() {
        let token = sign_token(&TEST_SEED, "acme-corp", Some(NOW + 1000));
        let s = Licensing::from_blob(&test_verifier(), token.as_bytes(), NOW);
        assert_eq!(s.tier(NOW), Tier::Pro);
        assert_eq!(s.licensee(), Some("acme-corp"));
        for f in Feature::ALL {
            assert!(s.require(f, NOW).is_ok());
        }
    }

    #[cfg(feature = "license")]
    #[test]
    fn perpetual_signed_token_is_pro_forever() {
        let token = sign_token(&TEST_SEED, "forever-inc", None);
        let s = Licensing::from_blob(&test_verifier(), token.as_bytes(), NOW);
        assert_eq!(s.tier(u64::MAX), Tier::Pro);
    }

    #[cfg(feature = "license")]
    #[test]
    fn trailing_whitespace_from_a_file_is_tolerated() {
        let token = format!("{}\n", sign_token(&TEST_SEED, "acme", None));
        assert!(test_verifier().verify(token.as_bytes(), NOW).is_ok());
    }

    #[cfg(feature = "license")]
    #[test]
    fn tampered_payload_is_rejected_and_falls_back_to_community() {
        let token = sign_token(&TEST_SEED, "acme-corp", Some(NOW + 1000));
        let mut bytes = token.into_bytes();
        bytes[0] ^= 0b1; // flip a payload char → signature no longer matches
        let v = test_verifier();
        assert!(matches!(
            v.verify(&bytes, NOW),
            Err(LicenseError::Invalid(_))
        ));
        assert_eq!(
            Licensing::from_blob(&v, &bytes, NOW).tier(NOW),
            Tier::Community
        );
    }

    #[cfg(feature = "license")]
    #[test]
    fn a_token_signed_by_another_key_is_rejected() {
        let token = sign_token(&[9u8; 32], "impostor", None); // different seed
        let v = test_verifier(); // expects TEST_SEED's public half
        assert!(matches!(
            v.verify(token.as_bytes(), NOW),
            Err(LicenseError::Invalid(_))
        ));
    }

    #[cfg(feature = "license")]
    #[test]
    fn malformed_tokens_are_invalid_and_never_panic() {
        let v = test_verifier();
        for bad in ["", "no-dot", "not.base64url-!!", "only.", ".only"] {
            assert!(v.verify(bad.as_bytes(), NOW).is_err(), "rejects {bad:?}");
        }
    }

    #[cfg(feature = "license")]
    #[test]
    fn unset_embedded_key_fails_closed_to_community() {
        // The placeholder key shipped in this tree licenses nothing — even a well-formed token
        // signed by some key is refused, so the default build is fail-closed (a vendor must paste
        // their own public key). Holds while LICENSE_PUBLIC_KEY is the all-zero placeholder.
        let token = sign_token(&TEST_SEED, "acme", None);
        let s = Licensing::from_blob(&Ed25519Verifier::new(), token.as_bytes(), NOW);
        assert_eq!(s.tier(NOW), Tier::Community);
    }

    #[cfg(feature = "license")]
    #[test]
    fn expired_signed_token_reads_community_but_keeps_licensee() {
        let token = sign_token(&TEST_SEED, "lapsed-llc", Some(NOW - 1));
        let s = Licensing::from_blob(&test_verifier(), token.as_bytes(), NOW);
        // Valid signature (licensee retained for the audit log) but past expiry, so the
        // tier is community — gated, never silently degraded.
        assert_eq!(s.licensee(), Some("lapsed-llc"));
        assert_eq!(s.tier(NOW), Tier::Community);
        assert!(!s.allows(Feature::AuditReports, NOW));
    }

    // ── SLH-DSA (post-quantum) verifier + token format (behind `license-pq`) ────
    // A throwaway TEST keypair: derived at runtime from a fixed 48-byte seed via
    // `keygen_seed`, its public half passed to `with_public_key` — never the
    // embedded vendor key, so no production private key lives in the tree.
    #[cfg(feature = "license-pq")]
    const TEST_SLH_SEED: [u8; 48] = [7u8; 48];

    #[cfg(feature = "license-pq")]
    fn test_slh_keypair() -> ([u8; 32], [u8; 64]) {
        let (pk, sk) = slh_dsa::keygen_seed(slh_dsa::params::SLH_DSA_SHAKE_128S, &TEST_SLH_SEED);
        assert_eq!(pk.len(), 32);
        assert_eq!(sk.len(), 64);
        (
            pk.try_into().expect("pk is 32 bytes"),
            sk.try_into().expect("sk is 64 bytes"),
        )
    }

    #[cfg(feature = "license-pq")]
    fn test_slh_verifier() -> SlhDsaVerifier {
        let (pk, _sk) = test_slh_keypair();
        SlhDsaVerifier::with_public_key(&pk)
    }

    #[cfg(feature = "license-pq")]
    #[test]
    fn slh_dsa_signed_token_verifies_to_pro_and_unlocks_features() {
        let (_pk, sk) = test_slh_keypair();
        let token = sign_token_slhdsa(&sk, "acme-corp", Some(NOW + 1000));
        assert!(token.starts_with("slhdsa."));
        let s = Licensing::from_blob(&test_slh_verifier(), token.as_bytes(), NOW);
        assert_eq!(s.tier(NOW), Tier::Pro);
        assert_eq!(s.licensee(), Some("acme-corp"));
        for f in Feature::ALL {
            assert!(s.require(f, NOW).is_ok());
        }
    }

    #[cfg(feature = "license-pq")]
    #[test]
    fn slh_dsa_perpetual_signed_token_is_pro_forever() {
        let (_pk, sk) = test_slh_keypair();
        let token = sign_token_slhdsa(&sk, "forever-inc", None);
        let s = Licensing::from_blob(&test_slh_verifier(), token.as_bytes(), NOW);
        assert_eq!(s.tier(u64::MAX), Tier::Pro);
    }

    #[cfg(feature = "license-pq")]
    #[test]
    fn slh_dsa_trailing_whitespace_from_a_file_is_tolerated() {
        let (_pk, sk) = test_slh_keypair();
        let token = format!("{}\n", sign_token_slhdsa(&sk, "acme", None));
        assert!(test_slh_verifier().verify(token.as_bytes(), NOW).is_ok());
    }

    #[cfg(feature = "license-pq")]
    #[test]
    fn slh_dsa_tampered_payload_is_rejected_and_falls_back_to_community() {
        let (_pk, sk) = test_slh_keypair();
        let token = sign_token_slhdsa(&sk, "acme-corp", Some(NOW + 1000));
        let mut bytes = token.into_bytes();
        // Flip a char inside the payload segment (after "slhdsa.", before the first '.') →
        // the signature no longer matches the signed `slhdsa.<payload>` input.
        bytes["slhdsa.".len()] ^= 0b1;
        let v = test_slh_verifier();
        assert!(matches!(
            v.verify(&bytes, NOW),
            Err(LicenseError::Invalid(_))
        ));
        assert_eq!(
            Licensing::from_blob(&v, &bytes, NOW).tier(NOW),
            Tier::Community
        );
    }

    #[cfg(feature = "license-pq")]
    #[test]
    fn slh_dsa_tampered_signature_is_rejected() {
        let (_pk, sk) = test_slh_keypair();
        let token = sign_token_slhdsa(&sk, "acme-corp", None);
        let mut bytes = token.into_bytes();
        // Flip a byte near the end (inside the signature segment).
        let last = bytes.len().checked_sub(3).unwrap();
        bytes[last] ^= 0b1;
        let v = test_slh_verifier();
        assert!(matches!(
            v.verify(&bytes, NOW),
            Err(LicenseError::Invalid(_))
        ));
    }

    #[cfg(feature = "license-pq")]
    #[test]
    fn slh_dsa_token_signed_by_another_key_is_rejected() {
        // A different seed → a different keypair; the verifier expects TEST_SLH_SEED's pk.
        let (pk_other, sk_other) =
            slh_dsa::keygen_seed(slh_dsa::params::SLH_DSA_SHAKE_128S, &[9u8; 48]);
        let _ = pk_other;
        let sk_other: [u8; 64] = sk_other.try_into().unwrap();
        let token = sign_token_slhdsa(&sk_other, "impostor", None);
        let v = test_slh_verifier();
        assert!(matches!(
            v.verify(token.as_bytes(), NOW),
            Err(LicenseError::Invalid(_))
        ));
    }

    #[cfg(feature = "license-pq")]
    #[test]
    fn slh_dsa_malformed_tokens_are_invalid_and_never_panic() {
        let v = test_slh_verifier();
        for bad in [
            "",
            "no-dot",
            "slhdsa.",
            "slhdsa.only",
            "slhdsa..",
            "notslhdsa.payload.sig",
            "slhdsa.payload.!!",
        ] {
            assert!(v.verify(bad.as_bytes(), NOW).is_err(), "rejects {bad:?}");
        }
    }

    #[cfg(feature = "license-pq")]
    #[test]
    fn slh_dsa_unset_embedded_key_fails_closed_to_community() {
        // The placeholder key shipped in this tree licenses nothing — even a well-formed
        // token signed by some key is refused, so the default build is fail-closed (a
        // vendor must paste its own public key). Holds while LICENSE_SLH_DSA_PUBLIC_KEY
        // is the all-zero placeholder.
        let (_pk, sk) = test_slh_keypair();
        let token = sign_token_slhdsa(&sk, "acme", None);
        let s = Licensing::from_blob(&SlhDsaVerifier::new(), token.as_bytes(), NOW);
        assert_eq!(s.tier(NOW), Tier::Community);
    }

    #[cfg(feature = "license-pq")]
    #[test]
    fn slh_dsa_expired_signed_token_reads_community_but_keeps_licensee() {
        let (_pk, sk) = test_slh_keypair();
        let token = sign_token_slhdsa(&sk, "lapsed-llc", Some(NOW - 1));
        let s = Licensing::from_blob(&test_slh_verifier(), token.as_bytes(), NOW);
        assert_eq!(s.licensee(), Some("lapsed-llc"));
        assert_eq!(s.tier(NOW), Tier::Community);
        assert!(!s.allows(Feature::AuditReports, NOW));
    }

    // ── cross-scheme isolation (both verifiers compiled in) ────────────────────
    #[cfg(all(feature = "license", feature = "license-pq"))]
    #[test]
    fn ed25519_verifier_rejects_a_slh_dsa_tagged_token() {
        let (_pk, sk) = test_slh_keypair();
        let pq_token = sign_token_slhdsa(&sk, "acme", None);
        // The ed25519 verifier expects `payload.sig` (no `slhdsa.` tag) and a 64-byte sig;
        // a SLH-DSA token (7,856-byte sig, tagged) must not verify as ed25519.
        let ed = Ed25519Verifier::with_public_key(&[1u8; 32]);
        assert!(ed.verify(pq_token.as_bytes(), NOW).is_err());
    }

    #[cfg(all(feature = "license", feature = "license-pq"))]
    #[test]
    fn slh_dsa_verifier_rejects_a_legacy_ed25519_token() {
        let token = sign_token(&TEST_SEED, "acme", None); // untagged ed25519 token
        let v = test_slh_verifier();
        // No `slhdsa.` prefix → rejected at the format check, before any crypto.
        assert!(matches!(
            v.verify(token.as_bytes(), NOW),
            Err(LicenseError::Invalid(_))
        ));
    }

    #[cfg(all(feature = "license", feature = "license-pq"))]
    #[test]
    fn detect_dispatches_on_the_scheme_tag() {
        // A slhdsa. token → verified by the SLH-DSA path (Pro); an ed25519 token → the
        // ed25519 path (Pro). `Licensing::detect` reads the host blob, so exercise the
        // dispatch via `from_blob`-equivalent: directly through each verifier.
        let (pk_pq, sk_pq) = test_slh_keypair();
        let pq_token = sign_token_slhdsa(&sk_pq, "pq-corp", None);
        let s_pq = Licensing::from_blob(
            &SlhDsaVerifier::with_public_key(&pk_pq),
            pq_token.as_bytes(),
            NOW,
        );
        assert_eq!(s_pq.tier(NOW), Tier::Pro);
        assert_eq!(s_pq.licensee(), Some("pq-corp"));

        let ed_token = sign_token(&TEST_SEED, "ed-corp", None);
        let s_ed = Licensing::from_blob(&test_verifier(), ed_token.as_bytes(), NOW);
        assert_eq!(s_ed.tier(NOW), Tier::Pro);
        assert_eq!(s_ed.licensee(), Some("ed-corp"));
    }
}
