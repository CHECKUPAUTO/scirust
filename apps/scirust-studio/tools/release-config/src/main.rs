//! Decide, from the environment alone, how a release build should be signed
//! — and refuse to build at all when that decision is ambiguous.
//!
//! # The state this exists to make impossible
//!
//! Signing a desktop application needs material this repository does not and
//! must not hold: an Authenticode certificate for Windows, and a minisign
//! private key for the updater. So a build can legitimately be *unsigned*,
//! and it can legitimately be *signed*. What it must never be is **half**.
//!
//! A half-configured build is the dangerous one, and both halves fail
//! differently:
//!
//! - An updater public key with no private key produces an application that
//!   checks for updates it can never verify. It will refuse every update it
//!   is offered, which looks like a broken server rather than a broken build.
//! - A private key with no public key in the shipped config produces update
//!   artifacts that are signed and an application that does not check the
//!   signature. That is strictly worse than no updater at all.
//! - A certificate with no password, or a password with no certificate,
//!   makes `tauri build` fall back to producing an unsigned installer while
//!   the pipeline believes it asked for a signed one.
//!
//! None of those announce themselves. Each produces a build that completes,
//! an artifact that runs, and a security property that silently is not there.
//! So this tool treats a partial configuration as a **hard error** and says
//! which variable is missing.
//!
//! # What it emits
//!
//! On a fully-configured signing setup, a Tauri config *patch* on stdout, to
//! be passed to `cargo tauri build --config`. Tauri merges a patch over the
//! discovered config rather than replacing it, so the patch carries only the
//! updater block and the Windows signing block.
//!
//! On a cleanly unconfigured setup, the literal `{}` — an empty patch — and
//! a note on stderr saying the build will be unsigned. An empty patch is a
//! valid thing to hand `--config`, so the caller has one code path rather
//! than two.
//!
//! # What it will not do
//!
//! - **Never generate a key.** A keypair this tool invented would have its
//!   private half in a CI log and its public half in a shipped binary.
//! - **Never print secret material.** The private key and the certificate
//!   password are read to check they are *present* and are never echoed,
//!   including in error messages.
//! - **Never default an endpoint.** An updater pointed at a URL nobody chose
//!   is a supply-chain hole with a plausible-looking config.

use std::collections::BTreeMap;
use std::process::ExitCode;

/// Environment variable holding the minisign **private** key Tauri signs
/// update artifacts with. Read only to check it is present.
const SIGNING_PRIVATE_KEY: &str = "TAURI_SIGNING_PRIVATE_KEY";

/// The matching **public** key, which is compiled into the application so it
/// can verify what it is offered.
const UPDATER_PUBKEY: &str = "SCIRUST_UPDATER_PUBKEY";

/// Where the application asks for updates. Comma-separated; no default.
const UPDATER_ENDPOINTS: &str = "SCIRUST_UPDATER_ENDPOINTS";

/// SHA-1 thumbprint of the Authenticode certificate in the runner's store.
const WINDOWS_THUMBPRINT: &str = "SCIRUST_WINDOWS_CERT_THUMBPRINT";

/// Timestamp authority for Authenticode. Required alongside the thumbprint:
/// an un-timestamped signature stops verifying the day the certificate
/// expires, which turns a shipped installer into a warning some months later
/// for no visible reason.
const WINDOWS_TIMESTAMP_URL: &str = "SCIRUST_WINDOWS_TIMESTAMP_URL";

/// What a set of environment variables adds up to.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Posture {
    /// Every variable in the group is set.
    Configured,
    /// None of them are.
    Absent,
    /// Some are and some are not — the state this tool refuses.
    Partial {
        /// Which variables were present.
        present: Vec<String>,
        /// Which were missing.
        missing: Vec<String>,
    },
}

/// Classify a group of variables as all-present, all-absent, or partial.
///
/// A variable set to the empty string counts as *absent*. CI systems
/// routinely expose an unset secret as an empty value, and treating that as
/// present is how a build ends up passing an empty private key to a signer.
pub fn classify(env: &BTreeMap<String, String>, names: &[&str]) -> Posture {
    let mut present = Vec::new();
    let mut missing = Vec::new();
    for name in names
    {
        match env.get(*name)
        {
            Some(value) if !value.trim().is_empty() => present.push((*name).to_string()),
            _ => missing.push((*name).to_string()),
        }
    }
    if missing.is_empty()
    {
        Posture::Configured
    }
    else if present.is_empty()
    {
        Posture::Absent
    }
    else
    {
        Posture::Partial { present, missing }
    }
}

/// The complete decision for a build.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Decision {
    /// The Tauri config patch, as JSON. `{}` when nothing is configured.
    pub patch: String,
    /// Human-facing lines describing what was decided, for stderr.
    pub notes: Vec<String>,
    /// Whether the resulting build carries an updater.
    pub updater: bool,
    /// Whether the resulting installer is Authenticode-signed.
    pub windows_signed: bool,
}

/// Errors that stop a build rather than producing a patch.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConfigError {
    /// What is wrong, in full.
    pub message: String,
}

impl std::fmt::Display for ConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.message)
    }
}

/// Escape a string for embedding in JSON.
///
/// Hand-written rather than pulled in as a dependency: this tool runs inside
/// the release pipeline, so every crate it links is a crate that can affect a
/// signed artifact. The inputs are a base64 key, a URL list and a hex
/// thumbprint, and the escapes below cover every character JSON requires.
fn json_escape(value: &str) -> String {
    let mut out = String::with_capacity(value.len() + 2);
    for c in value.chars()
    {
        match c
        {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out
}

/// Work out what to build, from the environment.
pub fn decide(env: &BTreeMap<String, String>) -> Result<Decision, ConfigError> {
    let updater_vars = [SIGNING_PRIVATE_KEY, UPDATER_PUBKEY, UPDATER_ENDPOINTS];
    let windows_vars = [WINDOWS_THUMBPRINT, WINDOWS_TIMESTAMP_URL];

    let updater = classify(env, &updater_vars);
    let windows = classify(env, &windows_vars);

    for (what, posture, why) in [
        (
            "updater signing",
            &updater,
            "an application that checks for updates it cannot verify, or update artifacts \
             nothing verifies",
        ),
        (
            "Windows code signing",
            &windows,
            "an installer the pipeline believes is signed and that is not",
        ),
    ]
    {
        if let Posture::Partial { present, missing } = posture
        {
            return Err(ConfigError {
                message: format!(
                    "{what} is half-configured, which is the one state this build refuses.\n  \
                     set: {}\n  missing: {}\n\nSet all of them or none of them. A partial \
                     configuration produces {why} — a build that completes, an artifact that \
                     runs, and a security property that silently is not there.",
                    present.join(", "),
                    missing.join(", ")
                ),
            });
        }
    }

    let mut notes = Vec::new();
    let mut blocks: Vec<String> = Vec::new();

    if updater == Posture::Configured
    {
        let pubkey = json_escape(env.get(UPDATER_PUBKEY).expect("classified as present"));
        let endpoints: Vec<String> = env
            .get(UPDATER_ENDPOINTS)
            .expect("classified as present")
            .split(',')
            .map(|e| e.trim())
            .filter(|e| !e.is_empty())
            .map(|e| format!("\"{}\"", json_escape(e)))
            .collect();
        if endpoints.is_empty()
        {
            return Err(ConfigError {
                message: format!(
                    "{UPDATER_ENDPOINTS} is set but contains no usable URL. An updater with no \
                     endpoint is not a smaller feature, it is a broken one."
                ),
            });
        }
        notes.push(format!(
            "updater: enabled, {} endpoint(s), artifacts signed with the key in \
             {SIGNING_PRIVATE_KEY}",
            endpoints.len()
        ));
        blocks.push(format!(
            "\"plugins\":{{\"updater\":{{\"pubkey\":\"{pubkey}\",\"endpoints\":[{}]}}}}",
            endpoints.join(",")
        ));
        // Tauri only emits the `.sig` files an updater consumes when asked.
        blocks.push("\"bundle\":{\"createUpdaterArtifacts\":true}".to_string());
    }
    else
    {
        notes.push(
            "updater: DISABLED. None of the signing variables are set, so this build ships \
             without an update mechanism rather than with one it cannot verify."
                .to_string(),
        );
    }

    if windows == Posture::Configured
    {
        let thumbprint = json_escape(env.get(WINDOWS_THUMBPRINT).expect("classified as present"));
        let timestamp = json_escape(
            env.get(WINDOWS_TIMESTAMP_URL)
                .expect("classified as present"),
        );
        notes.push("windows: Authenticode signing enabled, with timestamping".to_string());
        blocks.push(format!(
            "\"bundle\":{{\"windows\":{{\"certificateThumbprint\":\"{thumbprint}\",\
             \"digestAlgorithm\":\"sha256\",\"timestampUrl\":\"{timestamp}\"}}}}"
        ));
    }
    else
    {
        notes.push(
            "windows: UNSIGNED. No certificate thumbprint is configured, so the installer will \
             raise SmartScreen — see docs/studio/WINDOWS_DESKTOP_ACCEPTANCE.md."
                .to_string(),
        );
    }

    Ok(Decision {
        patch: merge_blocks(&blocks),
        notes,
        updater: updater == Posture::Configured,
        windows_signed: windows == Posture::Configured,
    })
}

/// Combine the top-level JSON blocks into one object.
///
/// Two of them can both be `"bundle"`, so those are merged by hand rather
/// than concatenated — a duplicate key in a JSON object is exactly the kind
/// of thing a parser resolves silently and differently from how a reader
/// would.
fn merge_blocks(blocks: &[String]) -> String {
    let mut bundle_parts: Vec<&str> = Vec::new();
    let mut others: Vec<&str> = Vec::new();
    for block in blocks
    {
        if let Some(inner) = block
            .strip_prefix("\"bundle\":{")
            .and_then(|b| b.strip_suffix('}'))
        {
            bundle_parts.push(inner);
        }
        else
        {
            others.push(block);
        }
    }
    let mut parts: Vec<String> = others.iter().map(|s| (*s).to_string()).collect();
    if !bundle_parts.is_empty()
    {
        parts.push(format!("\"bundle\":{{{}}}", bundle_parts.join(",")));
    }
    format!("{{{}}}", parts.join(","))
}

fn main() -> ExitCode {
    let env: BTreeMap<String, String> = std::env::vars().collect();
    match decide(&env)
    {
        Ok(decision) =>
        {
            for note in &decision.notes
            {
                eprintln!("release-config: {note}");
            }
            println!("{}", decision.patch);
            ExitCode::SUCCESS
        },
        Err(e) =>
        {
            eprintln!("release-config: {e}");
            ExitCode::FAILURE
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn env(pairs: &[(&str, &str)]) -> BTreeMap<String, String> {
        pairs
            .iter()
            .map(|(k, v)| ((*k).to_string(), (*v).to_string()))
            .collect()
    }

    fn signed_updater() -> Vec<(&'static str, &'static str)> {
        vec![
            (SIGNING_PRIVATE_KEY, "dW50cnVzdGVkIGNvbW1lbnQ6IGZha2U="),
            (UPDATER_PUBKEY, "dW50cnVzdGVkIGNvbW1lbnQ6IHB1Yg=="),
            (UPDATER_ENDPOINTS, "https://example.invalid/updates.json"),
        ]
    }

    fn signed_windows() -> Vec<(&'static str, &'static str)> {
        vec![
            (
                WINDOWS_THUMBPRINT,
                "0123456789ABCDEF0123456789ABCDEF01234567",
            ),
            (WINDOWS_TIMESTAMP_URL, "http://timestamp.invalid"),
        ]
    }

    /// The ordinary case for this repository: nothing configured, build
    /// proceeds, and it says plainly that it is unsigned.
    #[test]
    fn nothing_configured_builds_unsigned_with_an_empty_patch() {
        let decision = decide(&env(&[])).expect("an unconfigured build is legitimate");
        assert_eq!(decision.patch, "{}");
        assert!(!decision.updater);
        assert!(!decision.windows_signed);
        assert!(decision.notes.iter().any(|n| n.contains("DISABLED")));
        assert!(decision.notes.iter().any(|n| n.contains("UNSIGNED")));
    }

    #[test]
    fn a_fully_configured_updater_emits_a_patch() {
        let decision = decide(&env(&signed_updater())).expect("fully configured");
        assert!(decision.updater);
        assert!(decision.patch.contains("\"updater\""));
        assert!(
            decision
                .patch
                .contains("https://example.invalid/updates.json")
        );
        // Without this Tauri produces no `.sig` files and the updater has
        // nothing to verify.
        assert!(decision.patch.contains("createUpdaterArtifacts"));
    }

    /// The whole reason this tool exists. Each of the three updater
    /// variables, alone, must stop the build.
    #[test]
    fn every_partial_updater_configuration_is_refused() {
        for (index, _) in signed_updater().iter().enumerate()
        {
            let only_one = vec![signed_updater()[index]];
            let err =
                decide(&env(&only_one)).expect_err("a half-configured updater must not build");
            assert!(err.message.contains("half-configured"), "{}", err.message);
            assert!(
                err.message.contains(signed_updater()[index].0),
                "the error must name what WAS set: {}",
                err.message
            );
        }
    }

    /// And each pair, missing the third.
    #[test]
    fn a_public_key_without_a_private_one_is_refused() {
        let err = decide(&env(&[
            (UPDATER_PUBKEY, "pub"),
            (UPDATER_ENDPOINTS, "https://example.invalid/u.json"),
        ]))
        .expect_err("no private key means nothing can be signed");
        assert!(err.message.contains(SIGNING_PRIVATE_KEY));
    }

    #[test]
    fn a_private_key_without_a_public_one_is_refused() {
        let err = decide(&env(&[
            (SIGNING_PRIVATE_KEY, "priv"),
            (UPDATER_ENDPOINTS, "https://example.invalid/u.json"),
        ]))
        .expect_err("signed artifacts nothing verifies are worse than none");
        assert!(err.message.contains(UPDATER_PUBKEY));
    }

    #[test]
    fn a_windows_certificate_without_a_timestamp_url_is_refused() {
        let err = decide(&env(&[(WINDOWS_THUMBPRINT, "AABB")]))
            .expect_err("an un-timestamped signature expires with the certificate");
        assert!(err.message.contains(WINDOWS_TIMESTAMP_URL));
    }

    /// A CI system exposing an unset secret as the empty string must read as
    /// absent, not as present-and-empty. Otherwise an empty private key is
    /// handed to a signer.
    #[test]
    fn an_empty_variable_counts_as_absent() {
        let decision = decide(&env(&[
            (SIGNING_PRIVATE_KEY, ""),
            (UPDATER_PUBKEY, "   "),
            (UPDATER_ENDPOINTS, ""),
        ]))
        .expect("all empty is the same as all unset");
        assert!(!decision.updater);
        assert_eq!(decision.patch, "{}");
    }

    /// ...but one real value beside two empty ones is still partial.
    #[test]
    fn one_real_value_among_empty_ones_is_still_partial() {
        let err = decide(&env(&[
            (SIGNING_PRIVATE_KEY, "actually-set"),
            (UPDATER_PUBKEY, ""),
            (UPDATER_ENDPOINTS, ""),
        ]))
        .expect_err("this is exactly the half-configured state");
        assert!(err.message.contains("half-configured"));
    }

    #[test]
    fn an_endpoint_list_with_no_usable_url_is_refused() {
        let err = decide(&env(&[
            (SIGNING_PRIVATE_KEY, "priv"),
            (UPDATER_PUBKEY, "pub"),
            (UPDATER_ENDPOINTS, " , , "),
        ]))
        .expect_err("an updater with no endpoint is broken, not smaller");
        assert!(err.message.contains("no usable URL"));
    }

    #[test]
    fn several_endpoints_are_all_carried() {
        let decision = decide(&env(&[
            (SIGNING_PRIVATE_KEY, "priv"),
            (UPDATER_PUBKEY, "pub"),
            (
                UPDATER_ENDPOINTS,
                "https://a.invalid/u.json, https://b.invalid/u.json",
            ),
        ]))
        .expect("configured");
        assert!(decision.patch.contains("a.invalid"));
        assert!(decision.patch.contains("b.invalid"));
    }

    /// Both halves configured must produce ONE `bundle` object, not two.
    /// A duplicate key is resolved silently by a parser and differently by a
    /// reader.
    #[test]
    fn both_signing_paths_merge_into_a_single_bundle_object() {
        let mut pairs = signed_updater();
        pairs.extend(signed_windows());
        let decision = decide(&env(&pairs)).expect("both configured");
        assert!(decision.updater && decision.windows_signed);
        assert_eq!(
            decision.patch.matches("\"bundle\"").count(),
            1,
            "patch was: {}",
            decision.patch
        );
        assert!(decision.patch.contains("createUpdaterArtifacts"));
        assert!(decision.patch.contains("certificateThumbprint"));
        assert!(decision.patch.contains("timestampUrl"));
    }

    /// The private key must never reach the patch, which is printed to
    /// stdout and ends up in a build log.
    #[test]
    fn the_private_key_never_appears_in_the_output() {
        let mut pairs = signed_updater();
        pairs.extend(signed_windows());
        let decision = decide(&env(&pairs)).expect("configured");
        let secret = signed_updater()[0].1;
        assert!(
            !decision.patch.contains(secret),
            "the private key leaked into the patch"
        );
        for note in &decision.notes
        {
            assert!(!note.contains(secret), "the private key leaked into a note");
        }
    }

    #[test]
    fn a_quote_in_a_value_cannot_break_out_of_the_json() {
        let decision = decide(&env(&[
            (SIGNING_PRIVATE_KEY, "priv"),
            (UPDATER_PUBKEY, "pu\"b"),
            (UPDATER_ENDPOINTS, "https://x.invalid/u.json"),
        ]))
        .expect("configured");
        assert!(decision.patch.contains("pu\\\"b"), "{}", decision.patch);
    }

    #[test]
    fn classify_reports_both_sides_of_a_partial_set() {
        let posture = classify(&env(&[("A", "1")]), &["A", "B", "C"]);
        match posture
        {
            Posture::Partial { present, missing } =>
            {
                assert_eq!(present, vec!["A".to_string()]);
                assert_eq!(missing, vec!["B".to_string(), "C".to_string()]);
            },
            other => panic!("expected Partial, got {other:?}"),
        }
    }
}
