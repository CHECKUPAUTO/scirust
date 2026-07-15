//! # SciRust codegen provenance
//!
//! Offline **Lamport/Merkle signing** and public **verification** of artifacts
//! emitted by the SciRust transpiler, reusing the hash-based signature scheme in
//! [`scirust_license::hashsig`] (SHA-256 only, no elliptic curve, deterministic).
//!
//! ## What this is — and what it is not
//!
//! This is a **provenance / leak-attribution** tool, not an anti-clone shield.
//! Read that distinction carefully, because it decides how the evidence holds up:
//!
//! * It **does** give an unforgeable, court-usable proof that a *specific emitted
//!   artifact* was produced by a build holding your secret seed. The verifier
//!   holds only the 32-byte public root and **cannot forge** a signature. This is
//!   strong against a party who redistributes your emitted `.rs` verbatim (a
//!   leaked customer artifact, a repackager), and the embedded one-time-signature
//!   `leaf` doubles as a per-artifact serial for tracing *which* build leaked.
//! * It **does not** stop, or even detect, a competitor who *reimplements the
//!   transpiler from source*: their engine never invokes your signer, so their
//!   output carries no mark. A provenance mark protects the *artifact*, not the
//!   *tool*. Anyone who runs the emitted code through an AST round-trip
//!   (`syn` + `prettyplease`) also strips the banner. Do not represent this as
//!   protection against functional cloning.
//!
//! ## Output neutrality
//!
//! The signature rides in a single trailing `//` line comment. Rust discards
//! comments at lex time, so a signed artifact compiles **byte-identically** to the
//! unsigned one — no numeric result, reduction order, or literal is touched.
//! [`verify_artifact`] re-[`canonicalize`]s the source (strips **all** comments,
//! collapses whitespace) before hashing, so reformatting / `rustfmt` does not break
//! the mark; only token-level edits do.
//!
//! ## Secret custody (read before production use)
//!
//! The secret master seed **never ships**. Signing is an offline step
//! ([`sign_artifact`], the `prov sign` CLI) run on a trusted host; the shipped
//! verifier embeds only [`EMIT_ROOT_HEX`]. The default [`EMIT_ROOT_HEX`] equals
//! [`scirust_license::demo_root`] so this crate's tests run end-to-end — **but the
//! demo seed is public, so a demo-signed banner proves nothing.** Before you rely
//! on this in court you MUST:
//! 1. generate a real master seed inside an HSM, with a logged, access-controlled
//!    procedure;
//! 2. replace [`EMIT_ROOT_HEX`] with the hex of that signer's [`MerkleSigner::root`];
//! 3. publish that root to an immutable, externally-timestamped venue (a signed &
//!    pushed git tag, a certificate-transparency log) **before** distributing any
//!    signed artifact, so the root provably predates any suspect copy;
//! 4. allocate a **fresh `leaf` per distinct signed digest** (reusing a leaf for two
//!    different digests leaks OTS secrets — see [`scirust_license::hashsig`]).

use scirust_license::hashsig::{self, Hash, MerkleSig, MerkleSigner};
use sha2::{Digest, Sha256};

/// Domain-separation tag mixed into every artifact digest, mirroring the
/// `b"scirust-node-lock:v2\0"` discipline in `scirust-license`.
const DIGEST_DOMAIN: &[u8] = b"scirust-emit:v1\0";

/// Marker that opens the provenance comment line appended to a signed artifact.
pub const BANNER_PREFIX: &str = "// srl-emit:v1 ";

/// The pinned public Merkle root the verifier trusts.
///
/// **Demo default.** This equals [`scirust_license::demo_root`], so the tests here
/// sign and verify out of the box. The demo *seed* is public, so a banner that
/// verifies under this root proves only that *some* build using the public demo
/// seed produced it — replace it with your HSM-backed root before production (see
/// the crate-level "Secret custody" note). The `emit_root_is_pinned` test is the
/// drift-guard that keeps this constant honest.
pub const EMIT_ROOT_HEX: &str = "82728023e3de7243e982d04ab09a7aa20a7fdb1fa10a0df2920060abc93a7f02";

/// The pinned public root as bytes.
///
/// # Panics
/// If [`EMIT_ROOT_HEX`] is not exactly 32 bytes of valid hex (a compile-time
/// authoring error the `emit_root_is_pinned` test also catches).
pub fn emit_root() -> Hash {
    let bytes = hashsig::hex_decode(EMIT_ROOT_HEX).expect("EMIT_ROOT_HEX must be valid hex");
    assert_eq!(bytes.len(), 32, "EMIT_ROOT_HEX must decode to 32 bytes");
    let mut root = [0u8; 32];
    root.copy_from_slice(&bytes);
    root
}

/// The outcome of verifying an artifact's provenance banner.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Verdict {
    /// The banner is present and its signature verifies against the trusted root
    /// over the artifact's own canonical digest. `leaf` is the one-time-signature
    /// index, usable as a per-artifact serial to trace which build produced it.
    Verified {
        /// The one-time leaf that signed this artifact.
        leaf: u32,
    },
    /// No `// srl-emit` banner was found.
    NoBanner,
    /// A banner line exists but its `sig=` field is missing or not decodable.
    MalformedBanner,
    /// A well-formed banner is present but the signature does **not** verify: the
    /// artifact was altered after signing, or the banner was transplanted from a
    /// different artifact, or it was signed under a different root.
    Forged,
}

/// A parsed provenance banner.
#[derive(Debug, Clone)]
pub struct Banner {
    /// The first 4 bytes of the signing root, hex-encoded — a human-readable hint
    /// only; verification always uses the caller-supplied trusted root.
    pub root_hint: String,
    /// The recovered signature.
    pub sig: MerkleSig,
}

/// Canonicalize an emitted Rust artifact for hashing.
///
/// Strips every comment (respecting string, byte-string, char and raw-string
/// literals so a `//` or `/* */` *inside a string* is preserved), then collapses
/// each run of ASCII whitespace to a single space and trims. Consequently
/// whitespace and comment reformatting — including `rustfmt` and stripping the
/// provenance banner itself — does **not** change the digest, while any
/// token-level edit does.
pub fn canonicalize(src: &str) -> String {
    let b = src.as_bytes();
    let n = b.len();
    let mut out: Vec<u8> = Vec::with_capacity(n);
    let mut i = 0usize;

    while i < n
    {
        let c = b[i];

        // Line comment `// … <eol>` -> a single separating space.
        if c == b'/' && at(b, i + 1) == b'/'
        {
            i += 2;
            while i < n && b[i] != b'\n'
            {
                i += 1;
            }
            push_sep(&mut out);
            continue;
        }

        // Block comment `/* … */`, which nests in Rust -> a separating space.
        if c == b'/' && at(b, i + 1) == b'*'
        {
            i += 2;
            let mut depth = 1usize;
            while i < n && depth > 0
            {
                if at(b, i) == b'/' && at(b, i + 1) == b'*'
                {
                    depth += 1;
                    i += 2;
                }
                else if at(b, i) == b'*' && at(b, i + 1) == b'/'
                {
                    depth -= 1;
                    i += 2;
                }
                else
                {
                    i += 1;
                }
            }
            push_sep(&mut out);
            continue;
        }

        // Raw string `r"…"`, `r#"…"#`, `br"…"`, `br#"…"#` — copied verbatim.
        if c == b'r' || (c == b'b' && at(b, i + 1) == b'r')
        {
            if let Some(end) = raw_string_end(b, i)
            {
                out.extend_from_slice(&b[i..end]);
                i = end;
                continue;
            }
        }

        // Byte string `b"…"` or normal string `"…"` — copied verbatim, honoring
        // backslash escapes so an embedded `"` does not end it early.
        if c == b'"' || (c == b'b' && at(b, i + 1) == b'"')
        {
            let start = i;
            if c == b'b'
            {
                i += 1;
            }
            i += 1; // opening quote
            while i < n
            {
                if b[i] == b'\\'
                {
                    i += 2;
                }
                else if b[i] == b'"'
                {
                    i += 1;
                    break;
                }
                else
                {
                    i += 1;
                }
            }
            out.extend_from_slice(&b[start..i.min(n)]);
            continue;
        }

        // Char literal `'x'` / `'\n'` vs a lifetime `'a`.
        if c == b'\''
        {
            if at(b, i + 1) == b'\\'
            {
                // Escaped char literal: consume through the closing quote.
                let start = i;
                i += 2;
                while i < n && b[i] != b'\''
                {
                    i += 1;
                }
                if i < n
                {
                    i += 1;
                }
                out.extend_from_slice(&b[start..i.min(n)]);
                continue;
            }
            if i + 2 < n && b[i + 2] == b'\''
            {
                // Simple char literal `'x'`.
                out.extend_from_slice(&b[i..i + 3]);
                i += 3;
                continue;
            }
            // Otherwise a lifetime: emit the quote and continue normally.
            out.push(c);
            i += 1;
            continue;
        }

        // Whitespace -> collapse to a single separating space.
        if c == b' ' || c == b'\t' || c == b'\n' || c == b'\r' || c == 0x0c
        {
            push_sep(&mut out);
            i += 1;
            continue;
        }

        // Ordinary code byte (ASCII or a UTF-8 continuation byte) -> verbatim.
        out.push(c);
        i += 1;
    }

    // `out` is valid UTF-8 (we only copy whole literals / code bytes and push
    // ASCII spaces), so this never loses data; trim the framing whitespace.
    String::from_utf8(out)
        .unwrap_or_default()
        .trim()
        .to_string()
}

/// The domain-separated SHA-256 digest of an already-canonicalized artifact.
pub fn digest_of(canonical: &str) -> Hash {
    let mut h = Sha256::new();
    h.update(DIGEST_DOMAIN);
    h.update(canonical.as_bytes());
    h.finalize().into()
}

/// Canonicalize `src` and return its provenance digest — what actually gets
/// signed and verified.
pub fn digest_of_source(src: &str) -> Hash {
    digest_of(&canonicalize(src))
}

/// Format the trailing provenance comment for `sig` produced under `root`.
pub fn format_banner(root: &Hash, sig: &MerkleSig) -> String {
    format!(
        "{}root={} leaf={} sig={}",
        BANNER_PREFIX,
        hashsig::hex_encode(&root[..4]),
        sig.leaf,
        sig.to_hex()
    )
}

/// Remove any existing provenance banner line(s) from `src`, returning the base
/// artifact. Idempotent, and used by [`sign_artifact`] so re-signing never stacks
/// banners.
pub fn strip_banner(src: &str) -> String {
    let kept: Vec<&str> = src
        .lines()
        .filter(|line| !line.trim_start().starts_with(BANNER_PREFIX.trim_end()))
        .collect();
    let mut out = kept.join("\n");
    if src.ends_with('\n') && !out.is_empty()
    {
        out.push('\n');
    }
    out
}

/// Locate and parse the (last) provenance banner in `src`.
pub fn parse_banner(src: &str) -> Option<Banner> {
    let line = src
        .lines()
        .rfind(|l| l.trim_start().starts_with(BANNER_PREFIX.trim_end()))?;
    let mut root_hint = String::new();
    let mut sig = None;
    for tok in line.split_whitespace()
    {
        if let Some(v) = tok.strip_prefix("root=")
        {
            root_hint = v.to_string();
        }
        else if let Some(v) = tok.strip_prefix("sig=")
        {
            sig = MerkleSig::from_hex(v);
        }
    }
    sig.map(|sig| Banner { root_hint, sig })
}

/// Sign `src` with `signer` under one-time `leaf`, returning the artifact with a
/// provenance banner appended. Any pre-existing banner is stripped first, so the
/// digest is taken over the base artifact and signing is idempotent.
///
/// The caller owns leaf allocation: never reuse a `leaf` for two artifacts whose
/// canonical digests differ (doing so leaks OTS secrets).
pub fn sign_artifact(src: &str, signer: &MerkleSigner, leaf: u32) -> String {
    let base = strip_banner(src);
    let digest = digest_of_source(&base);
    let sig = signer.sign(leaf, &digest);
    let banner = format_banner(&signer.root(), &sig);

    let mut out = base;
    if !out.ends_with('\n')
    {
        out.push('\n');
    }
    out.push_str(&banner);
    out.push('\n');
    out
}

/// Verify the provenance banner in `src` against the trusted `root`.
pub fn verify_artifact(src: &str, root: &Hash) -> Verdict {
    let has_line = src
        .lines()
        .any(|l| l.trim_start().starts_with(BANNER_PREFIX.trim_end()));
    if !has_line
    {
        return Verdict::NoBanner;
    }
    let banner = match parse_banner(src)
    {
        Some(b) => b,
        None => return Verdict::MalformedBanner,
    };
    // The banner is a comment, so canonicalize(src) excludes it: we recompute the
    // exact digest that was signed straight from the (possibly reformatted) source.
    let digest = digest_of_source(src);
    if hashsig::verify(root, &digest, &banner.sig)
    {
        Verdict::Verified {
            leaf: banner.sig.leaf,
        }
    }
    else
    {
        Verdict::Forged
    }
}

/// Byte at `i`, or `0` past the end — keeps the scanner branch-simple.
#[inline]
fn at(b: &[u8], i: usize) -> u8 {
    if i < b.len() { b[i] } else { 0 }
}

/// Push a single separating space unless the output already ends with one (or is
/// empty), so collapsed comments/whitespace never merge two adjacent tokens.
#[inline]
fn push_sep(out: &mut Vec<u8>) {
    if let Some(&last) = out.last()
    {
        if last != b' '
        {
            out.push(b' ');
        }
    }
}

/// If a raw-string literal starts at `i` (`r"`, `r#…"`, or a `b`-prefixed form),
/// return the index just past its close; otherwise `None`. An unterminated raw
/// string is treated as running to end of input.
fn raw_string_end(b: &[u8], i: usize) -> Option<usize> {
    let n = b.len();
    let mut j = i;
    if at(b, j) == b'b'
    {
        j += 1;
    }
    if at(b, j) != b'r'
    {
        return None;
    }
    j += 1;
    let mut hashes = 0usize;
    while j < n && b[j] == b'#'
    {
        hashes += 1;
        j += 1;
    }
    if at(b, j) != b'"'
    {
        return None;
    }
    j += 1; // opening quote
    while j < n
    {
        if b[j] == b'"'
        {
            let mut k = j + 1;
            let mut cnt = 0usize;
            while k < n && cnt < hashes && b[k] == b'#'
            {
                cnt += 1;
                k += 1;
            }
            if cnt == hashes
            {
                return Some(k);
            }
            j += 1;
        }
        else
        {
            j += 1;
        }
    }
    Some(n)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn demo_signer() -> MerkleSigner {
        MerkleSigner::from_seed(&scirust_license::demo_seed(), scirust_license::DEMO_HEIGHT)
    }

    #[test]
    fn emit_root_is_pinned() {
        // Drift-guard: the embedded constant must equal the demo signer's root, so
        // a demo-signed artifact verifies under EMIT_ROOT. Replace both together
        // when you move to a production seed.
        assert_eq!(emit_root(), scirust_license::demo_root());
        assert_eq!(emit_root(), demo_signer().root());
    }

    #[test]
    fn canonicalize_strips_comments_and_collapses_whitespace() {
        let a = "pub fn f(x: f64) -> f64 {\n    // c\n    x * 2.0\n}";
        let b = "pub fn f(x: f64) -> f64 {   x * 2.0 }";
        assert_eq!(canonicalize(a), canonicalize(b));
        assert!(!canonicalize(a).contains("c"));
    }

    #[test]
    fn canonicalize_preserves_slashes_and_stars_inside_strings() {
        let s = r#"let u = "http://x // y /* z */";"#;
        let c = canonicalize(s);
        assert!(c.contains("http://x // y /* z */"), "string mangled: {c}");
    }

    #[test]
    fn canonicalize_handles_raw_and_byte_strings_and_chars() {
        let s =
            "let a = r#\"raw \"q\" // /* \"#; let b = b\"by\\\"te\"; let c = '\\''; let d = 'x';";
        // Must not panic and must not treat the in-literal comment tokens as comments.
        let c = canonicalize(s);
        assert!(c.contains("raw \"q\" // /*"));
        assert!(c.contains("'x'"));
    }

    #[test]
    fn canonicalize_does_not_merge_tokens_across_a_block_comment() {
        assert_eq!(canonicalize("a/*x*/b"), "a b");
    }

    #[test]
    fn sign_then_verify_roundtrips() {
        let signer = demo_signer();
        let art = "pub fn sq(x: f64) -> f64 { x * x }\n";
        let signed = sign_artifact(art, &signer, 0);
        assert!(signed.contains(BANNER_PREFIX));
        assert_eq!(
            verify_artifact(&signed, &emit_root()),
            Verdict::Verified { leaf: 0 }
        );
    }

    #[test]
    fn banner_is_output_neutral_for_the_digest() {
        // Stripping the banner and re-canonicalizing yields the same code, and the
        // banner itself never enters the canonical form (it is a comment).
        let signer = demo_signer();
        let art = "pub fn sq(x: f64) -> f64 { x * x }\n";
        let signed = sign_artifact(art, &signer, 1);
        assert_eq!(canonicalize(&signed), canonicalize(art));
        assert_eq!(strip_banner(&signed).trim(), art.trim());
    }

    #[test]
    fn verification_survives_reformatting() {
        let signer = demo_signer();
        let art = "pub fn add(a: f64, b: f64) -> f64 { a + b }\n";
        let signed = sign_artifact(art, &signer, 2);
        // Simulate a formatter: re-indent, add blank lines and an unrelated comment.
        let reformatted = signed
            .replace(
                "{ a + b }",
                "{\n    // reflowed by a formatter\n    a  +  b\n}",
            )
            .replace("pub fn add", "\n\npub  fn   add");
        assert_eq!(
            verify_artifact(&reformatted, &emit_root()),
            Verdict::Verified { leaf: 2 }
        );
    }

    #[test]
    fn tampering_with_a_token_is_detected() {
        let signer = demo_signer();
        let art = "pub fn sq(x: f64) -> f64 { x * x }\n";
        let signed = sign_artifact(art, &signer, 3);
        // Change a real token (the operator): digest changes, signature fails.
        let tampered = signed.replace("x * x", "x + x");
        assert_eq!(verify_artifact(&tampered, &emit_root()), Verdict::Forged);
    }

    #[test]
    fn transplanting_a_banner_onto_other_code_is_detected() {
        let signer = demo_signer();
        let a = sign_artifact("pub fn f() -> f64 { 1.0 }\n", &signer, 4);
        let banner = a
            .lines()
            .find(|l| l.trim_start().starts_with(BANNER_PREFIX.trim_end()))
            .unwrap();
        let forged = format!("pub fn g() -> f64 {{ 2.0 }}\n{banner}\n");
        assert_eq!(verify_artifact(&forged, &emit_root()), Verdict::Forged);
    }

    #[test]
    fn a_different_root_rejects_a_genuine_banner() {
        let signer = demo_signer();
        let signed = sign_artifact("pub fn f() -> f64 { 1.0 }\n", &signer, 5);
        let other = MerkleSigner::from_seed(&[0x11; 32], 4).root();
        assert_eq!(verify_artifact(&signed, &other), Verdict::Forged);
    }

    #[test]
    fn missing_and_malformed_banners_are_classified() {
        assert_eq!(
            verify_artifact("pub fn f() {}\n", &emit_root()),
            Verdict::NoBanner
        );
        let bad = format!("pub fn f() {{}}\n{}root=aa leaf=0 sig=zz\n", BANNER_PREFIX);
        assert_eq!(
            verify_artifact(&bad, &emit_root()),
            Verdict::MalformedBanner
        );
    }

    #[test]
    fn re_signing_does_not_stack_banners() {
        let signer = demo_signer();
        let once = sign_artifact("pub fn f() -> f64 { 1.0 }\n", &signer, 6);
        let twice = sign_artifact(&once, &signer, 6);
        let count = twice
            .lines()
            .filter(|l| l.trim_start().starts_with(BANNER_PREFIX.trim_end()))
            .count();
        assert_eq!(count, 1);
        assert_eq!(
            verify_artifact(&twice, &emit_root()),
            Verdict::Verified { leaf: 6 }
        );
    }
}
