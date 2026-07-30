//! # Unicode de-obfuscation sanitizer — surfacing hidden-character attacks
//!
//! A coding agent's context is assembled from text it did not write: files,
//! tool output, search results, pasted snippets. That text can carry characters
//! a human reviewer **cannot see** but a model still tokenises — the substrate of
//! a whole family of injection / obfuscation attacks:
//!
//! - **Bidirectional overrides** (`U+202A`–`U+202E`, `U+2066`–`U+2069`, the
//!   directional marks `U+200E/200F/061C`) — the *Trojan Source* attack
//!   (CVE-2021-42574): source that **reads** one way and **compiles/executes**
//!   another, because the visual order is reversed by an `RLO`/`PDI` dance.
//! - **Zero-width characters** (`U+200B` ZWSP, `U+200C/200D`, `U+2060` WJ,
//!   `U+FEFF` BOM, `U+00AD` soft hyphen) — invisible bytes spliced into
//!   identifiers, strings or instructions to defeat exact-match filters.
//! - **Unicode "Tags" block** (`U+E0000`–`U+E007F`) — *ASCII smuggling*: an
//!   entire instruction (`"ignore all previous rules"`) encoded in codepoints
//!   that render as **nothing**, decoded here back to the ASCII it shadows.
//! - **C0/C1 controls** and assorted default-ignorable / variation-selector
//!   codepoints used as covert channels.
//!
//! `guard.rs` already strips `char::is_control()` from model *output*, but that
//! only covers Unicode category **Cc** (the C0/C1 block) — every vector above
//! except the raw controls lives in category **Cf** (Format) and sails straight
//! through. This module is the **input** counterpart and it does the opposite of
//! a silent strip: it **surfaces** each hidden character as an explicit, visible
//! literal (`[U+202E RLO]`, `[U+200B ZWSP]`, `[U+E0041 TAG:A]`) and emits a
//! structured [`ScanReport`] of findings, so the de-obfuscation is **auditable**
//! — it can be recorded in CCOS's hash-chained event log like any other state
//! transition.
//!
//! ## What this is — and what it is *not*
//!
//! This is a **deterministic normalisation pass**, not an anti-prompt-injection
//! oracle. It closes the *hidden-character* class completely and verifiably, and
//! surfaces the **mixed-script homoglyph** class (see [`scan_confusables`]). It
//! deliberately does **not** attempt:
//! - **exhaustive** homoglyph coverage — [`scan_confusables`] flags only tokens
//!   that *mix* a real Latin letter with a Cyrillic/Greek look-alike (the
//!   identifier/URL spoofing signature, `раypal`), over a **curated high-value
//!   subset** of UTS-39. Pure single-script spoofs and the full confusables table
//!   are out of scope — the mixed-script rule is what keeps false positives off
//!   legitimate multilingual prose;
//! - **semantic** injection ("ignore your instructions" in plain visible text) —
//!   no character-level pass can catch a paraphrase. That is the job of the
//!   downstream [`crate::injection_classifier`] *signal*, and ultimately of
//!   privilege separation in the host.
//!
//! ## Determinism & cost
//!
//! Pure function of the input: one `O(n)` pass over `char_indices`, no RNG, no
//! `HashMap` iteration order in any output (the per-kind tally is a `BTreeMap`).
//! [`scan`] allocates **nothing** on the clean path (an empty `Vec`/`BTreeMap`),
//! and [`defang`] returns `Cow::Borrowed` — zero copy — when the input has no
//! anomalies, which is the overwhelmingly common case for real source.

use serde::{Deserialize, Serialize};
use std::borrow::Cow;
use std::collections::BTreeMap;
use std::fmt::Write as _;

/// A class of hidden / deceptive character, ordered by how actively it is abused.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum AnomalyKind {
    /// Bidirectional control or mark — the *Trojan Source* family. **High** risk.
    BidiControl,
    /// A codepoint from the Unicode Tags block (`U+E0000`–`U+E007F`): invisible
    /// ASCII smuggling. **High** risk.
    TagChar,
    /// Invisible zero-width / word-joining / BOM / soft-hyphen formatting.
    ZeroWidth,
    /// A C0 control (`U+0000`–`U+001F`) other than the permitted `\t`/`\n`/`\r`,
    /// or `DEL`/C1 (`U+007F`–`U+009F`).
    Control,
    /// Other default-ignorable / deprecated format codepoint or variation
    /// selector used as a covert channel. **Low** risk, surfaced for completeness.
    OtherFormat,
    /// A **confusable** (homoglyph): a Cyrillic/Greek character visually identical to a Latin
    /// letter, appearing in a token that *mixes* it with real Latin letters — the identifier /
    /// URL spoofing signature (`раypal`). **High** risk. Appended last so old reports stay
    /// deserializable. *(Detected per-token; see [`scan_confusables`].)*
    Confusable,
}

/// Coarse triage severity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Severity {
    Low,
    Medium,
    High,
}

impl AnomalyKind {
    /// Stable machine-readable tag (used as the [`ScanReport::counts`] key).
    pub fn as_str(self) -> &'static str {
        match self {
            AnomalyKind::BidiControl => "bidi-control",
            AnomalyKind::TagChar => "tag-char",
            AnomalyKind::ZeroWidth => "zero-width",
            AnomalyKind::Control => "control",
            AnomalyKind::OtherFormat => "other-format",
            AnomalyKind::Confusable => "confusable",
        }
    }

    /// Triage severity for the kind.
    pub fn severity(self) -> Severity {
        match self {
            AnomalyKind::BidiControl | AnomalyKind::TagChar | AnomalyKind::Confusable => {
                Severity::High
            }
            AnomalyKind::ZeroWidth | AnomalyKind::Control => Severity::Medium,
            AnomalyKind::OtherFormat => Severity::Low,
        }
    }
}

/// One hidden character located in the input, with everything needed to audit it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Finding {
    /// Byte offset of the character in the *original* input.
    pub byte_offset: usize,
    /// Char (scalar) index of the character in the original input.
    pub char_index: usize,
    /// The Unicode scalar value.
    pub codepoint: u32,
    /// Which class of anomaly this is.
    pub kind: AnomalyKind,
    /// Short mnemonic (`"RLO"`, `"ZWSP"`, `"TAG:A"`, `"ESC"`).
    pub label: String,
}

impl Finding {
    /// The explicit, visible literal this character is replaced with when defanged,
    /// e.g. `"[U+202E RLO]"`. Fully auditable: the codepoint is always shown.
    pub fn literal(&self) -> String {
        format!("[U+{:04X} {}]", self.codepoint, self.label)
    }

    /// Severity of this finding.
    pub fn severity(&self) -> Severity {
        self.kind.severity()
    }
}

/// The outcome of a scan: every finding, in input order, plus a per-kind tally.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ScanReport {
    /// Findings in ascending byte-offset order.
    pub findings: Vec<Finding>,
    /// `kind.as_str()` → count, in a deterministic (`BTreeMap`) order.
    pub counts: BTreeMap<String, usize>,
}

impl ScanReport {
    /// No hidden characters were found.
    pub fn is_clean(&self) -> bool {
        self.findings.is_empty()
    }

    /// Number of findings.
    pub fn len(&self) -> usize {
        self.findings.len()
    }

    /// True when there are no findings (mirrors [`ScanReport::is_clean`]).
    pub fn is_empty(&self) -> bool {
        self.findings.is_empty()
    }

    /// The highest severity present, or `None` when clean.
    pub fn highest_severity(&self) -> Option<Severity> {
        self.findings.iter().map(Finding::severity).max()
    }

    /// A one-line human summary, e.g. `"3 hidden chars: 1 bidi-control, 2 zero-width"`.
    pub fn summary(&self) -> String {
        if self.is_clean() {
            return "clean".to_string();
        }
        let mut s = format!("{} hidden char(s):", self.findings.len());
        for (kind, n) in &self.counts {
            let _ = write!(s, " {n} {kind},");
        }
        s.pop(); // trailing comma
        s
    }
}

/// What to do with a detected anomaly.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Action {
    /// Replace the character with its explicit visible literal (default — safe + auditable).
    Surface,
    /// Remove the character entirely.
    Strip,
    /// Leave it in place (detect-only — `defang` becomes a pure scan).
    Keep,
}

/// Tunables for the pass. Defaults are the safe choice for ingesting source / tool output.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SanitizerConfig {
    /// What to do with each detected anomaly.
    pub action: Action,
    /// Keep a literal tab (`U+0009`) rather than flagging it as a control.
    pub allow_tab: bool,
    /// Keep newlines (`U+000A`, `U+000D`) rather than flagging them.
    pub allow_newline: bool,
}

impl Default for SanitizerConfig {
    fn default() -> Self {
        Self {
            action: Action::Surface,
            allow_tab: true,
            allow_newline: true,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Classification — the heart of the pass.
// ─────────────────────────────────────────────────────────────────────────────

/// Classify a single scalar. Returns `None` for an ordinary, visible character
/// (or a permitted whitespace control); `Some((kind, label))` for an anomaly.
fn classify(c: char, cfg: &SanitizerConfig) -> Option<(AnomalyKind, String)> {
    let cp = c as u32;
    match cp {
        // ── Permitted whitespace controls (configurable) ───────────────────
        0x09 if cfg.allow_tab => None,
        0x0A | 0x0D if cfg.allow_newline => None,

        // ── C0 controls / DEL / C1 ─────────────────────────────────────────
        0x00..=0x1F | 0x7F..=0x9F => Some((AnomalyKind::Control, control_name(cp).to_string())),

        // ── Bidirectional controls & marks (Trojan Source, CVE-2021-42574) ──
        0x202A => bidi("LRE"),
        0x202B => bidi("RLE"),
        0x202C => bidi("PDF"),
        0x202D => bidi("LRO"),
        0x202E => bidi("RLO"),
        0x2066 => bidi("LRI"),
        0x2067 => bidi("RLI"),
        0x2068 => bidi("FSI"),
        0x2069 => bidi("PDI"),
        0x200E => bidi("LRM"),
        0x200F => bidi("RLM"),
        0x061C => bidi("ALM"),

        // ── Zero-width / invisible formatting ──────────────────────────────
        0x200B => zw("ZWSP"),
        0x200C => zw("ZWNJ"),
        0x200D => zw("ZWJ"),
        0x2060 => zw("WJ"),
        0xFEFF => zw("BOM"),
        0x00AD => zw("SHY"),
        0x180E => zw("MVS"),

        // ── Unicode Tags block — ASCII smuggling ───────────────────────────
        0xE0001 => Some((AnomalyKind::TagChar, "TAG:LANG".to_string())),
        0xE007F => Some((AnomalyKind::TagChar, "TAG:END".to_string())),
        0xE0020..=0xE007E => {
            // Shadows printable ASCII: U+E00xx mirrors the byte 0xxx.
            let ascii = (cp - 0xE0000) as u8 as char;
            Some((AnomalyKind::TagChar, format!("TAG:{ascii}")))
        }
        0xE0000 | 0xE0080..=0xE00FF => Some((AnomalyKind::TagChar, "TAG".to_string())),

        // ── Other default-ignorable / covert-channel codepoints ────────────
        0x2061..=0x2064 => other("INVIS-MATH"), // invisible times / separator / plus / fn
        0xFFF9..=0xFFFB => other("ANNOT"),      // interlinear annotation anchors
        0x115F | 0x1160 | 0x3164 | 0xFFA0 => other("HANGUL-FILLER"),
        0xFE00..=0xFE0F => other("VS"), // variation selectors 1–16
        0xE0100..=0xE01EF => other("VS-SUPP"), // variation selectors supplement
        0x2028 | 0x2029 => other("LINE-SEP"), // line / paragraph separator

        _ => None,
    }
}

#[inline]
fn bidi(name: &str) -> Option<(AnomalyKind, String)> {
    Some((AnomalyKind::BidiControl, name.to_string()))
}
#[inline]
fn zw(name: &str) -> Option<(AnomalyKind, String)> {
    Some((AnomalyKind::ZeroWidth, name.to_string()))
}
#[inline]
fn other(name: &str) -> Option<(AnomalyKind, String)> {
    Some((AnomalyKind::OtherFormat, name.to_string()))
}

/// The Latin letter a Cyrillic/Greek character is a **confusable** (homoglyph) of, or `None` if it
/// is not in the curated set. This is a **high-value curated subset** of Unicode UTS-39 confusables
/// (the letters actually used in identifier / domain spoofing), not the exhaustive table — honest
/// scope, matching the rest of the sanitizer. Deterministic pure `const`-style lookup.
fn confusable_skeleton(c: char) -> Option<char> {
    Some(match c {
        // ── Cyrillic → Latin (lowercase) ──
        '\u{0430}' => 'a', // а
        '\u{0435}' => 'e', // е
        '\u{043E}' => 'o', // о
        '\u{0440}' => 'p', // р
        '\u{0441}' => 'c', // с
        '\u{0443}' => 'y', // у
        '\u{0445}' => 'x', // х
        '\u{0456}' => 'i', // і (Ukrainian)
        '\u{0455}' => 's', // ѕ
        '\u{0458}' => 'j', // ј
        '\u{04BB}' => 'h', // һ
        // ── Cyrillic → Latin (uppercase) ──
        '\u{0410}' => 'A', // А
        '\u{0412}' => 'B', // В
        '\u{0415}' => 'E', // Е
        '\u{041A}' => 'K', // К
        '\u{041C}' => 'M', // М
        '\u{041D}' => 'H', // Н
        '\u{041E}' => 'O', // О
        '\u{0420}' => 'P', // Р
        '\u{0421}' => 'C', // С
        '\u{0422}' => 'T', // Т
        '\u{0423}' => 'Y', // У
        '\u{0425}' => 'X', // Х
        // ── Greek → Latin ──
        '\u{03BF}' => 'o', // ο
        '\u{03B1}' => 'a', // α
        '\u{03C1}' => 'p', // ρ
        '\u{03BD}' => 'v', // ν
        '\u{0391}' => 'A', // Α
        '\u{039F}' => 'O', // Ο
        '\u{03A1}' => 'P', // Ρ
        '\u{0392}' => 'B', // Β
        '\u{0397}' => 'H', // Η
        '\u{039A}' => 'K', // Κ
        '\u{03A4}' => 'T', // Τ
        '\u{03A7}' => 'X', // Χ
        _ => return None,
    })
}

/// Whether a character can be part of an identifier / URL / path token (letters, digits, and the
/// `. - _` connectors), for the confusable tokenizer.
fn is_token_char(c: char) -> bool {
    c.is_alphanumeric() || c == '.' || c == '-' || c == '_'
}

/// Per-**token** confusable detection: find identifier-like tokens that **mix** a real ASCII-Latin
/// letter with a Cyrillic/Greek look-alike — the `раypal` spoofing signature — and emit a
/// [`Finding`] for each confusable character in them. Pure single-script tokens (all-Latin, or
/// legitimate all-Cyrillic prose) are **not** flagged, so this keeps false positives to the actual
/// mixed-script attack. Findings come out in ascending byte order.
pub fn scan_confusables(input: &str) -> Vec<Finding> {
    let mut out = Vec::new();
    // Walk tokens: a token is a maximal run of `is_token_char`. Within one, collect its confusable
    // characters and note whether it also contains a plain ASCII-Latin letter (the mixed-script
    // signal). On a token boundary, emit the confusables only if the token was mixed.
    let mut has_ascii_latin = false;
    let mut confusables: Vec<(usize, usize, char, char)> = Vec::new(); // (byte, char_idx, char, skeleton)
    let flush = |confusables: &mut Vec<(usize, usize, char, char)>,
                 has_ascii_latin: &mut bool,
                 out: &mut Vec<Finding>| {
        if *has_ascii_latin {
            for &(byte_offset, char_index, c, skel) in confusables.iter() {
                out.push(Finding {
                    byte_offset,
                    char_index,
                    codepoint: c as u32,
                    kind: AnomalyKind::Confusable,
                    label: format!("→{skel}"),
                });
            }
        }
        confusables.clear();
        *has_ascii_latin = false;
    };
    for (char_index, (byte_offset, c)) in input.char_indices().enumerate() {
        if is_token_char(c) {
            if c.is_ascii_alphabetic() {
                has_ascii_latin = true;
            }
            if let Some(skel) = confusable_skeleton(c) {
                confusables.push((byte_offset, char_index, c, skel));
            }
        } else {
            flush(&mut confusables, &mut has_ascii_latin, &mut out);
        }
    }
    flush(&mut confusables, &mut has_ascii_latin, &mut out);
    out
}

/// Short mnemonic for a C0/DEL/C1 control codepoint.
fn control_name(cp: u32) -> &'static str {
    const C0: [&str; 32] = [
        "NUL", "SOH", "STX", "ETX", "EOT", "ENQ", "ACK", "BEL", "BS", "HT", "LF", "VT", "FF", "CR",
        "SO", "SI", "DLE", "DC1", "DC2", "DC3", "DC4", "NAK", "SYN", "ETB", "CAN", "EM", "SUB",
        "ESC", "FS", "GS", "RS", "US",
    ];
    match cp {
        0x00..=0x1F => C0[cp as usize],
        0x7F => "DEL",
        _ => "C1", // U+0080..=U+009F
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Passes.
// ─────────────────────────────────────────────────────────────────────────────

/// Detect hidden characters with the default config. Zero allocation on a clean input.
pub fn scan(input: &str) -> ScanReport {
    scan_with(input, &SanitizerConfig::default())
}

/// Detect hidden characters with an explicit config.
pub fn scan_with(input: &str, cfg: &SanitizerConfig) -> ScanReport {
    let mut report = ScanReport::default();
    for (char_index, (byte_offset, c)) in input.char_indices().enumerate() {
        if let Some((kind, label)) = classify(c, cfg) {
            *report.counts.entry(kind.as_str().to_string()).or_insert(0) += 1;
            report.findings.push(Finding {
                byte_offset,
                char_index,
                codepoint: c as u32,
                kind,
                label,
            });
        }
    }
    // Merge the per-token confusable findings and restore ascending byte order (defang relies on
    // it). Confusable characters never coincide with a hidden-char finding (disjoint codepoints),
    // so no offset collides.
    for f in scan_confusables(input) {
        *report
            .counts
            .entry(f.kind.as_str().to_string())
            .or_insert(0) += 1;
        report.findings.push(f);
    }
    report.findings.sort_by_key(|f| f.byte_offset);
    report
}

/// Produce a de-obfuscated copy and the scan report. Returns `Cow::Borrowed`
/// (no allocation, no copy) when the input is clean.
pub fn defang(input: &str) -> (Cow<'_, str>, ScanReport) {
    defang_with(input, &SanitizerConfig::default())
}

/// [`defang`] with an explicit config.
pub fn defang_with<'a>(input: &'a str, cfg: &SanitizerConfig) -> (Cow<'a, str>, ScanReport) {
    let report = scan_with(input, cfg);
    if report.is_clean() || cfg.action == Action::Keep {
        return (Cow::Borrowed(input), report);
    }
    // Rebuild once, substituting each finding. Findings are in byte order, so a
    // single forward walk over the original bytes suffices.
    let mut out = String::with_capacity(input.len() + report.findings.len() * 8);
    let mut next = 0usize; // index into report.findings
    for (byte_offset, c) in input.char_indices() {
        let is_finding = report
            .findings
            .get(next)
            .is_some_and(|f| f.byte_offset == byte_offset);
        if is_finding {
            let f = &report.findings[next];
            next += 1;
            match cfg.action {
                Action::Surface => out.push_str(&f.literal()),
                Action::Strip => {}
                Action::Keep => out.push(c), // unreachable (handled above), kept for totality
            }
        } else {
            out.push(c);
        }
    }
    (Cow::Owned(out), report)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clean_ascii_is_clean_and_zero_copy() {
        let src = "fn main() {\n    let x = 1; // ok\n}\n";
        let report = scan(src);
        assert!(report.is_clean());
        assert_eq!(report.summary(), "clean");
        let (out, r2) = defang(src);
        assert!(r2.is_clean());
        // Cow::Borrowed → same backing pointer, no copy.
        assert!(matches!(out, Cow::Borrowed(_)));
        assert_eq!(out.as_ptr(), src.as_ptr());
    }

    #[test]
    fn confusable_mixed_script_token_is_flagged_and_surfaced() {
        // Cyrillic р (U+0440) + а (U+0430) mixed with Latin y,p,a,l — the "paypal" homoglyph spoof.
        let spoof = "login at \u{0440}\u{0430}ypal.com now";
        let report = scan(spoof);
        let confs: Vec<&Finding> = report
            .findings
            .iter()
            .filter(|f| f.kind == AnomalyKind::Confusable)
            .collect();
        assert_eq!(
            confs.len(),
            2,
            "both look-alikes flagged: {:?}",
            report.findings
        );
        assert_eq!(report.highest_severity(), Some(Severity::High));
        // defang surfaces each explicitly (auditable), never silently normalising.
        let (out, _) = defang(spoof);
        assert!(
            out.contains("→p") && out.contains("→a"),
            "surfaced skeletons: {out}"
        );
    }

    #[test]
    fn legitimate_single_script_text_is_not_a_confusable() {
        // Pure ASCII: no confusables (and no false positive on a real domain).
        assert!(scan("please login at paypal.com now")
            .findings
            .iter()
            .all(|f| f.kind != AnomalyKind::Confusable));
        // Pure Cyrillic prose (привет) — no ASCII-Latin in the token, so not a spoof signature.
        let russian = "\u{043F}\u{0440}\u{0438}\u{0432}\u{0435}\u{0442}";
        assert!(
            scan(russian)
                .findings
                .iter()
                .all(|f| f.kind != AnomalyKind::Confusable),
            "legitimate Cyrillic text is not flagged"
        );
    }

    #[test]
    fn confusable_scan_is_deterministic() {
        let s = "\u{0430}dmin panel \u{0440}oot access";
        assert_eq!(scan(s).findings, scan(s).findings);
    }

    #[test]
    fn detects_trojan_source_bidi_override() {
        // An RLO ... PDI pair: the classic Trojan-Source reordering trick.
        let src = "let access = \u{202E}// \u{2069}admin;";
        let report = scan(src);
        assert_eq!(report.findings.len(), 2);
        assert_eq!(report.findings[0].kind, AnomalyKind::BidiControl);
        assert_eq!(report.findings[0].label, "RLO");
        assert_eq!(report.findings[0].codepoint, 0x202E);
        assert_eq!(report.findings[1].label, "PDI");
        assert_eq!(report.highest_severity(), Some(Severity::High));

        let (out, _) = defang(src);
        assert!(out.contains("[U+202E RLO]"));
        assert!(out.contains("[U+2069 PDI]"));
        // No invisible codepoints survive.
        assert!(!out.contains('\u{202E}'));
        assert!(!out.contains('\u{2069}'));
    }

    #[test]
    fn detects_zero_width_inside_identifier() {
        let src = "del\u{200B}ete_all()";
        let report = scan(src);
        assert_eq!(report.findings.len(), 1);
        assert_eq!(report.findings[0].kind, AnomalyKind::ZeroWidth);
        assert_eq!(report.findings[0].label, "ZWSP");
        let (out, _) = defang(src);
        assert_eq!(out, "del[U+200B ZWSP]ete_all()");
    }

    #[test]
    fn decodes_unicode_tag_ascii_smuggling() {
        // The Tags block spells "Hi" invisibly: U+E0048 U+E0069.
        let src = "ok\u{E0048}\u{E0069}";
        let report = scan(src);
        assert_eq!(report.findings.len(), 2);
        assert_eq!(report.findings[0].kind, AnomalyKind::TagChar);
        assert_eq!(report.findings[0].label, "TAG:H");
        assert_eq!(report.findings[1].label, "TAG:i");
        let (out, _) = defang(src);
        assert_eq!(out, "ok[U+E0048 TAG:H][U+E0069 TAG:i]");
    }

    #[test]
    fn flags_c0_controls_but_keeps_tab_and_newline() {
        let src = "a\tb\nc\u{7}d"; // BEL (U+0007) is the only anomaly
        let report = scan(src);
        assert_eq!(report.findings.len(), 1);
        assert_eq!(report.findings[0].kind, AnomalyKind::Control);
        assert_eq!(report.findings[0].label, "BEL");
        let (out, _) = defang(src);
        assert_eq!(out, "a\tb\nc[U+0007 BEL]d");
    }

    #[test]
    fn byte_and_char_offsets_account_for_multibyte() {
        // "é" is 2 bytes; the ZWSP follows it.
        let src = "é\u{200B}x";
        let report = scan(src);
        assert_eq!(report.findings.len(), 1);
        assert_eq!(report.findings[0].char_index, 1);
        assert_eq!(report.findings[0].byte_offset, 2);
    }

    #[test]
    fn strip_action_removes_without_literal() {
        let cfg = SanitizerConfig {
            action: Action::Strip,
            ..Default::default()
        };
        let (out, report) = defang_with("a\u{200B}b", &cfg);
        assert_eq!(out, "ab");
        assert_eq!(report.findings.len(), 1); // still reported, just not surfaced
    }

    #[test]
    fn keep_action_is_detect_only() {
        let cfg = SanitizerConfig {
            action: Action::Keep,
            ..Default::default()
        };
        let (out, report) = defang_with("a\u{200B}b", &cfg);
        assert_eq!(out, "a\u{200B}b"); // unchanged
        assert!(matches!(out, Cow::Borrowed(_)));
        assert_eq!(report.findings.len(), 1);
    }

    #[test]
    fn deterministic_across_runs() {
        let src = "x\u{202E}y\u{200B}z\u{E0041}\u{7}";
        let a = scan(src);
        let b = scan(src);
        assert_eq!(a, b);
        assert_eq!(a.counts.get("bidi-control"), Some(&1));
        assert_eq!(a.counts.get("zero-width"), Some(&1));
        assert_eq!(a.counts.get("tag-char"), Some(&1));
        assert_eq!(a.counts.get("control"), Some(&1));
    }

    #[test]
    fn summary_is_human_readable() {
        let src = "\u{202E}\u{200B}\u{200B}";
        let report = scan(src);
        // BTreeMap order: "bidi-control" < "zero-width"
        assert_eq!(
            report.summary(),
            "3 hidden char(s): 1 bidi-control, 2 zero-width"
        );
    }
}
