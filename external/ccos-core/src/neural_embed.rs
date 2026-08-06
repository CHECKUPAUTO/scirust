//! **Quarantined neural embedder** — an [`Encoder`] backed by a *local*
//! Ollama-style `/api/embeddings` endpoint, behind the off-by-default `neural-embed` feature.
//!
//! This is the paper's future-work item 1, landed *as a quarantine*: the default build must stay
//! deterministic, dependency-free and bit-for-bit replayable, and a neural embedder structurally
//! cannot promise that (its vectors depend on model weights, server version, and hardware). So the
//! entire module sits behind a feature flag, and the contract is explicit:
//!
//! - **Off by default.** `cargo build` compiles none of this and pulls nothing new; the feature
//!   enables only `reqwest`'s *blocking* client — a crate already in the tree for the `llm` feature.
//! - **Local only.** The endpoint is meant to be a locally-running embedding server (e.g. Ollama at
//!   `http://127.0.0.1:11434` with `nomic-embed-text`). CCOS sends it document/query **text** and
//!   nothing else; with a local server, nothing leaves the host — the air-gap posture is preserved,
//!   but replay-exactness is **not**: re-running against a different model build may produce
//!   different vectors. That is the quarantine boundary, stated plainly.
//! - **Fail fast, degrade visibly.** [`NeuralEncoder::try_new`] probes the endpoint once and errors
//!   immediately if it is unreachable (no silent fallback that would fake semantics). A *transient*
//!   failure mid-run yields a zero vector — which ranks last under cosine — and increments
//!   [`NeuralEncoder::errors`], so a degraded run is measurable, never silent.
//!
//! Use it anywhere an [`Encoder`] fits — e.g.
//! `SemanticRetriever::new(NeuralEncoder::try_new(endpoint, model)?)` — and compare it against the
//! deterministic encoders with `examples/neural_vs_lsa.rs`.

use crate::retrieval::Encoder;
use std::time::Duration;

/// Why the neural endpoint could not be used.
#[derive(Debug)]
pub enum NeuralEmbedError {
    /// The endpoint did not answer the probe (connection refused, timeout, HTTP error).
    Unreachable(String),
    /// The endpoint answered, but not with an embedding this module understands.
    BadResponse(String),
    /// The endpoint host is not on the egress allowlist (air-gap: localhost only).
    EgressDenied(String),
}

impl std::fmt::Display for NeuralEmbedError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unreachable(e) => write!(f, "embedding endpoint unreachable: {e}"),
            Self::BadResponse(e) => write!(f, "embedding endpoint returned no usable vector: {e}"),
            Self::EgressDenied(e) => write!(f, "embedding endpoint not on egress allowlist: {e}"),
        }
    }
}

impl std::error::Error for NeuralEmbedError {}

/// Parse an embeddings-API response body into a vector. Accepts the two shapes in the wild:
/// classic Ollama `{"embedding": [f32, …]}` and the batched `{"embeddings": [[f32, …], …]}`
/// (first row). Pure and offline-testable — the only part of this module that can be.
fn parse_embedding(body: &str) -> Result<Vec<f32>, String> {
    let v: serde_json::Value = serde_json::from_str(body).map_err(|e| e.to_string())?;
    let arr = if let Some(a) = v.get("embedding").and_then(|a| a.as_array()) {
        a
    } else if let Some(a) = v
        .get("embeddings")
        .and_then(|a| a.as_array())
        .and_then(|rows| rows.first())
        .and_then(|r| r.as_array())
    {
        a
    } else {
        return Err(format!(
            "no `embedding`/`embeddings` array in response: {}",
            &body[..body.len().min(120)]
        ));
    };
    let out: Vec<f32> = arr
        .iter()
        .map(|x| x.as_f64().unwrap_or(0.0) as f32)
        .collect();
    if out.is_empty() {
        return Err("empty embedding".into());
    }
    Ok(out)
}

/// The quarantined neural [`Encoder`]: text → dense vector via a local `/api/embeddings` endpoint.
/// See the module docs for the contract (off by default, local only, fail-fast, degrade visibly).
pub struct NeuralEncoder {
    client: reqwest::blocking::Client,
    url: String,
    model: String,
    dim: usize,
    errors: usize,
}

impl NeuralEncoder {
    /// Connect to an Ollama-style server (`endpoint` like `http://127.0.0.1:11434`) and probe it
    /// once to learn the model's embedding dimension. Errors immediately — rather than degrading
    /// silently — if the endpoint is unreachable or answers with no usable vector.
    pub fn try_new(endpoint: &str, model: &str) -> Result<Self, NeuralEmbedError> {
        // Air-gap control (CCOS_EXTENDED plan P4): refuse a non-loopback endpoint
        // before any connection attempt. The neural embedder is quarantined *and*
        // localhost-only by default; an operator may widen this via
        // `CCOS_EGRESS_ALLOW`, but until they do a remote endpoint is denied here
        // — fail fast, never silently egress text off-host.
        let url = format!("{}/api/embeddings", endpoint.trim_end_matches('/'));
        let validated = crate::egress::EgressAllowlist::from_env()
            .validate(&url)
            .map_err(|e| NeuralEmbedError::EgressDenied(e.to_string()))?;

        let client = crate::egress::secure_blocking_client(
            &validated,
            Duration::from_secs(3),
            Duration::from_secs(30),
        )
        .map_err(|e| NeuralEmbedError::Unreachable(e.to_string()))?;

        let mut enc = Self {
            client,
            url,
            model: model.to_string(),
            dim: 0,
            errors: 0,
        };
        let probe = enc
            .request("dimension probe")
            .map_err(NeuralEmbedError::Unreachable)?;
        let v = parse_embedding(&probe).map_err(NeuralEmbedError::BadResponse)?;
        enc.dim = v.len();
        Ok(enc)
    }

    /// Transient failures observed since construction (each yielded a zero vector). A run that ends
    /// with `errors() > 0` is degraded and should say so — never report its ranking as clean.
    pub fn errors(&self) -> usize {
        self.errors
    }

    fn request(&self, text: &str) -> Result<String, String> {
        let body = serde_json::json!({ "model": self.model, "prompt": text });
        let resp = self
            .client
            .post(&self.url)
            .json(&body)
            .send()
            .map_err(|e| e.to_string())?;
        if !resp.status().is_success() {
            return Err(format!("HTTP {}", resp.status()));
        }
        let body = crate::egress::blocking_response_bytes_limited(resp, 8 * 1024 * 1024)?;
        String::from_utf8(body).map_err(|_| "embedding response is not UTF-8".to_string())
    }
}

impl Encoder for NeuralEncoder {
    fn embedding_dim(&self) -> usize {
        self.dim
    }

    /// One text → one vector. A transient endpoint failure yields a zero vector (ranks last under
    /// cosine) and bumps [`Self::errors`] — degraded visibly, never silently wrong.
    fn encode(&mut self, text: &str) -> Vec<f32> {
        match self.request(text).and_then(|b| parse_embedding(&b)) {
            Ok(mut v) => {
                v.resize(self.dim, 0.0); // guard: a model swap mid-run must not corrupt the index
                v
            }
            Err(e) => {
                self.errors += 1;
                eprintln!("neural-embed: transient failure ({e}) — zero vector substituted");
                vec![0.0; self.dim]
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_embedding_accepts_classic_ollama_shape() {
        let v = parse_embedding(r#"{"embedding": [0.5, -1.25, 3.0]}"#).unwrap();
        assert_eq!(v, vec![0.5, -1.25, 3.0]);
    }

    #[test]
    fn parse_embedding_accepts_batched_shape_first_row() {
        let v = parse_embedding(r#"{"embeddings": [[1.0, 2.0], [9.0, 9.0]]}"#).unwrap();
        assert_eq!(v, vec![1.0, 2.0]);
    }

    #[test]
    fn parse_embedding_rejects_missing_or_empty_vectors() {
        assert!(parse_embedding(r#"{"ok": true}"#).is_err());
        assert!(parse_embedding(r#"{"embedding": []}"#).is_err());
        assert!(parse_embedding("not json").is_err());
    }

    #[test]
    fn try_new_fails_fast_on_unreachable_endpoint() {
        // Port 1 refuses immediately — the constructor must error, not hang or fake a dimension.
        // The egress allowlist check runs first; either Denied or Unreachable is acceptable.
        let err = NeuralEncoder::try_new("http://127.0.0.1:1", "any-model");
        assert!(matches!(
            err,
            Err(NeuralEmbedError::Unreachable(_) | NeuralEmbedError::EgressDenied(_))
        ),);
    }
}
