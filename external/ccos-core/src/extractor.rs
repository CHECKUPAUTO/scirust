//! **Cognitive distillation** — turn raw text into Q-Page [`Assertion`]s (`Supports`/`Contradicts`
//! edges carrying a per-source **authority**).
//!
//! Provider-agnostic by construction: the contract is the [`Extractor`] trait; the LLM-backed
//! implementation lives behind the `llm` feature (any backend — a local model, a hosted API — fits),
//! and a deterministic [`MockExtractor`] lets the conflict-resolution logic be **measured and tested
//! with no model running**.
//!
//! **Determinism / `replay == live`.** Extraction (the model call) is the *only* non-deterministic
//! step and it runs **once, at ingest**. Its output is applied as the same explicit, replayable
//! assertions slice 1 already records (`Op::Assert`), so a replay re-applies the recorded edges and
//! **never re-calls the model** — the reconstructed graph is identical. The model is a sub-processor
//! that produces events; the events, not the model, are the source of truth.

use serde::{Deserialize, Serialize};

/// Which evidence surface an assertion lands on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Stance {
    /// Evidence *for* the claim (the affirmative surface `S_A`).
    Supports,
    /// Evidence *against* the claim (the negative surface `S_¬A`).
    Contradicts,
}

/// One distilled cognitive edge: `source` asserts, with a given **authority**, that the evidence it
/// carries `supports`/`contradicts` `claim`. This is exactly the input to
/// `CcosMemory::assert_support` / `assert_contradiction` — the extractor's job is to produce these
/// from text; recording them is the unchanged, replayable assertion path.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Assertion {
    /// The claim node this assertion bears on.
    pub claim: String,
    /// The asserting source's node id — it carries the authority weight onto the claim.
    pub source: String,
    /// `Supports` or `Contradicts`.
    pub stance: Stance,
    /// Source **authority** in `[0, 1]`: how strongly this source's word weighs on the claim's
    /// belief. Distinct sources with different authorities are how the "conflict of origins" resolves.
    pub authority: f64,
}

impl Assertion {
    /// Authority clamped to the valid `[0, 1]` range (defensive — an extractor or model may emit
    /// out-of-range values).
    pub fn clamped_authority(&self) -> f64 {
        self.authority.clamp(0.0, 1.0)
    }
}

/// What went wrong distilling text into [`Assertion`]s.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExtractError {
    /// The underlying model/backend failed (network, timeout, refusal, …).
    Backend(String),
    /// The model's output was not the expected JSON shape.
    Parse(String),
}

impl std::fmt::Display for ExtractError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ExtractError::Backend(e) => write!(f, "extraction backend error: {e}"),
            ExtractError::Parse(e) => write!(f, "could not parse extracted assertions: {e}"),
        }
    }
}

impl std::error::Error for ExtractError {}

/// Distills text into [`Assertion`]s. The single non-deterministic step in ingestion — its output is
/// recorded as replayable assertions, so the graph stays deterministic (see the module docs).
pub trait Extractor {
    fn extract(&self, text: &str) -> Result<Vec<Assertion>, ExtractError>;
}

/// The JSON envelope an LLM extractor asks the model to emit — `{ "assertions": [ … ] }`. Public so
/// the `llm`-backed extractor and any external tool agree on the schema, and so the prompt can be
/// validated against it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExtractionOutput {
    pub assertions: Vec<Assertion>,
}

impl ExtractionOutput {
    /// Parse a model response that should contain the `{ "assertions": [...] }` JSON. Tolerates
    /// surrounding prose by extracting the first balanced `{...}` block, since chat models often wrap
    /// JSON in commentary.
    pub fn parse(response: &str) -> Result<Self, ExtractError> {
        let json = first_json_object(response).unwrap_or(response);
        serde_json::from_str(json).map_err(|e| ExtractError::Parse(e.to_string()))
    }
}

/// Return the first balanced `{ … }` substring (naive brace matching, ignoring braces inside double
/// quotes), or `None`. Lets us pull the JSON object out of a chatty model reply.
fn first_json_object(s: &str) -> Option<&str> {
    let bytes = s.as_bytes();
    let start = s.find('{')?;
    let mut depth = 0usize;
    let mut in_str = false;
    let mut escaped = false;
    for (i, &b) in bytes.iter().enumerate().skip(start) {
        if in_str {
            match b {
                b'\\' if !escaped => escaped = true,
                b'"' if !escaped => in_str = false,
                _ => escaped = false,
            }
            continue;
        }
        match b {
            b'"' => in_str = true,
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    return Some(&s[start..=i]);
                }
            }
            _ => {}
        }
    }
    None
}

/// A deterministic extractor that returns a fixed assertion set regardless of input — the reference
/// the LLM extractor must reproduce, and the engine behind the conflict-of-origins bench and the
/// tests (no model required).
#[derive(Debug, Clone, Default)]
pub struct MockExtractor {
    pub assertions: Vec<Assertion>,
}

impl MockExtractor {
    pub fn new(assertions: Vec<Assertion>) -> Self {
        Self { assertions }
    }
}

impl Extractor for MockExtractor {
    fn extract(&self, _text: &str) -> Result<Vec<Assertion>, ExtractError> {
        Ok(self.assertions.clone())
    }
}

/// The **LLM-backed** extractor — distills raw text into [`Assertion`]s by asking a model (behind the
/// `llm` feature; the endpoint and model are configurable, so any compatible backend works — a local
/// model or a hosted one). Its [`extract`](LlmExtractor::extract) is **async** (a model call is
/// inherently async), so it deliberately does *not* implement the synchronous [`Extractor`] trait:
/// deterministic extractors use that trait, the model is a separate sub-processor. The output is
/// meant to be recorded via `assert_support` / `assert_contradiction` (`Op::Assert`), so a **replay
/// never re-calls the model** — `replay == live` holds.
#[cfg(feature = "llm")]
pub struct LlmExtractor {
    client: crate::llm::LlmClient,
}

#[cfg(feature = "llm")]
impl LlmExtractor {
    /// Build an extractor over the given LLM configuration (endpoint, model, timeouts, guard).
    pub fn new(config: crate::llm::LlmConfig) -> Self {
        Self {
            client: crate::llm::LlmClient::new(config),
        }
    }

    /// The system prompt: it pins the model to emit *only* the [`ExtractionOutput`] JSON schema.
    pub fn system_prompt() -> &'static str {
        "You are a cognitive distiller for a causal-memory engine. From the user's text, extract \
         each factual CLAIM and, for every source that bears on it, whether that source SUPPORTS or \
         CONTRADICTS the claim, plus an authority in [0,1] reflecting how reliable the source is. \
         Respond with ONLY this JSON and no other prose: \
         {\"assertions\":[{\"claim\":\"<short claim id>\",\"source\":\"<source id>\",\
         \"stance\":\"supports\"|\"contradicts\",\"authority\":<number 0..1>}]}"
    }

    /// Distill `text` into assertions via the model. The caller records the result as `assert_*`
    /// events, keeping the graph replayable.
    pub async fn extract(&self, text: &str) -> Result<Vec<Assertion>, ExtractError> {
        let resp = self.client.query(text, Some(Self::system_prompt())).await;
        if resp.is_fallback {
            return Err(ExtractError::Backend(
                "model unavailable (fallback response)".to_string(),
            ));
        }
        Ok(ExtractionOutput::parse(&resp.sanitized_output)?.assertions)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn a(claim: &str, source: &str, stance: Stance, authority: f64) -> Assertion {
        Assertion {
            claim: claim.into(),
            source: source.into(),
            stance,
            authority,
        }
    }

    #[test]
    fn mock_extractor_is_deterministic() {
        let m = MockExtractor::new(vec![a("x", "A", Stance::Supports, 0.9)]);
        assert_eq!(m.extract("any").unwrap(), m.extract("other").unwrap());
    }

    #[test]
    fn assertion_authority_is_clamped() {
        assert_eq!(a("x", "A", Stance::Supports, 1.5).clamped_authority(), 1.0);
        assert_eq!(
            a("x", "A", Stance::Contradicts, -0.2).clamped_authority(),
            0.0
        );
    }

    #[test]
    fn parses_assertions_json() {
        let out = ExtractionOutput::parse(
            r#"{"assertions":[{"claim":"x","source":"A","stance":"supports","authority":0.9}]}"#,
        )
        .unwrap();
        assert_eq!(out.assertions.len(), 1);
        assert_eq!(out.assertions[0].stance, Stance::Supports);
        assert_eq!(out.assertions[0].source, "A");
    }

    #[test]
    fn parses_json_wrapped_in_prose() {
        // Chat models often wrap JSON in commentary; we pull the first balanced object out.
        let reply = "Sure! Here are the assertions:\n{\"assertions\": [\
            {\"claim\":\"api is safe\",\"source\":\"docs\",\"stance\":\"contradicts\",\"authority\":0.3}\
            ]}\nLet me know if you need more.";
        let out = ExtractionOutput::parse(reply).unwrap();
        assert_eq!(out.assertions[0].stance, Stance::Contradicts);
        assert!((out.assertions[0].authority - 0.3).abs() < 1e-9);
    }

    #[test]
    fn malformed_output_is_a_parse_error() {
        assert!(matches!(
            ExtractionOutput::parse("no json here"),
            Err(ExtractError::Parse(_))
        ));
    }

    #[cfg(feature = "llm")]
    #[test]
    fn llm_extractor_system_prompt_pins_the_schema() {
        let p = super::LlmExtractor::system_prompt();
        assert!(
            p.contains("\"assertions\"") && p.contains("\"stance\"") && p.contains("authority"),
            "the prompt must specify the extraction JSON schema"
        );
        assert!(
            p.to_lowercase().contains("only"),
            "the prompt must instruct JSON-only output"
        );
    }
}
