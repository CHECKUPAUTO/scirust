//! Trading agent — the orchestrator that ties market → indicators → model →
//! certify → LLM narration → proof.
//!
//! **Golden rule**: the LLM never decides alone. The `TradingAgent` emits a
//! `CertifiedPrediction` first (pure SciRust), then asks the LLM to narrate
//! and sanity-check it. If the LLM's narration falls outside the certified
//! bounds, the decision is **blocked** and an alert is raised.

use serde::{Deserialize, Serialize};

use crate::certify::{CertifiedBounds, certify, feature_attribution};
use crate::indicators::IndicatorSet;
use crate::market::{MarketFeed, MarketSnapshot};
use crate::model::{PricePredictor, build_features};
use crate::proof::{DecisionProof, DecisionRecord};

/// The action the agent can take.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum Action {
    Long,
    Short,
    Flat,
}

impl Action {
    pub fn from_prediction(pred: f32, threshold: f32) -> Self {
        if pred > threshold
        {
            Action::Long
        }
        else if pred < -threshold
        {
            Action::Short
        }
        else
        {
            Action::Flat
        }
    }

    pub fn label(&self) -> &'static str {
        match self
        {
            Action::Long => "LONG",
            Action::Short => "SHORT",
            Action::Flat => "FLAT",
        }
    }
}

/// A certified prediction — produced by SciRust, before any LLM involvement.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CertifiedPrediction {
    pub symbol: String,
    pub action: Action,
    pub raw_prediction: f32,
    pub bounds: CertifiedBounds,
    pub feature_attribution: std::collections::BTreeMap<String, f32>,
    pub snapshot_fingerprint: String,
    pub weights_fingerprint: String,
    pub last_close: f32,
}

/// The LLM client trait — swappable backend.
pub trait LlmClient {
    /// Narrate a certified prediction in plain language.
    /// Returns the explanation text.
    fn narrate(&self, pred: &CertifiedPrediction) -> String;

    /// Sanity-check the prediction against external context.
    /// Returns `true` if the narration is consistent with the bounds.
    fn sanity_check(&self, pred: &CertifiedPrediction, narration: &str) -> bool;
}

/// A stub LLM client — deterministic, no network.
/// Used when Ollama is not available.
pub struct StubLlm;

impl LlmClient for StubLlm {
    fn narrate(&self, pred: &CertifiedPrediction) -> String {
        format!(
            "Model predicts {:?} for {} with raw return {:.4} and certified interval [{:.4}, {:.4}]. \
             Uncertainty: {:.4}. Last close: {:.2}.",
            pred.action,
            pred.symbol,
            pred.raw_prediction,
            pred.bounds.output.lo,
            pred.bounds.output.hi,
            pred.bounds.uncertainty,
            pred.last_close,
        )
    }

    fn sanity_check(&self, pred: &CertifiedPrediction, narration: &str) -> bool {
        // The stub always passes — it just reports the bounds.
        // A real LLM would flag inconsistencies (e.g. claiming a huge move
        // outside the certified interval).
        let _ = pred;
        let _ = narration;
        true
    }
}

/// An Ollama LLM client — connects to a local Ollama instance via HTTP.
///
/// Calls the `/api/generate` endpoint with the certified prediction as
/// context. The LLM is instructed (via the system prompt) that it MUST NOT
/// announce any number outside the certified interval.
pub struct OllamaClient {
    pub model: String,
    pub base_url: String,
}

/// Request body for Ollama `/api/generate`.
#[cfg(feature = "live")]
#[derive(serde::Serialize)]
struct OllamaRequest {
    model: String,
    prompt: String,
    stream: bool,
}

/// Response body from Ollama `/api/generate` (non-streaming).
#[cfg(feature = "live")]
#[derive(serde::Deserialize)]
struct OllamaResponse {
    response: String,
}

impl OllamaClient {
    pub fn new(model: &str, base_url: &str) -> Self {
        Self {
            model: model.to_string(),
            base_url: base_url.to_string(),
        }
    }

    /// Build the prompt for the LLM.
    fn build_prompt(pred: &CertifiedPrediction) -> String {
        let mut top_attrs: Vec<String> = pred
            .feature_attribution
            .iter()
            .map(|(k, v)| format!("{}={:.4}", k, v))
            .collect();
        top_attrs.sort_by(|a, b| b.cmp(a));
        let attrs_str = top_attrs
            .iter()
            .take(5)
            .cloned()
            .collect::<Vec<_>>()
            .join(", ");
        format!(
            "You are a crypto trading analyst. The SciRust model has produced a \
             CERTIFIED prediction (mathematically proven bounds via Interval Bound Propagation):\n\n\
             Symbol: {}\n\
             Action: {}\n\
             Raw predicted return: {:.6}\n\
             Certified output interval: [{:.6}, {:.6}]\n\
             Uncertainty (half-width): {:.6}\n\
             Last close price: {:.2}\n\
             Snapshot fingerprint: {}\n\
             Top feature attributions: {}\n\n\
             RULES:\n\
             1. You CANNOT announce any predicted return outside the certified interval [{:.6}, {:.6}].\n\
             2. Narrate the decision in 2-3 sentences.\n\
             3. Explain the key drivers from the feature attribution.\n\
             4. State the confidence level based on the uncertainty.\n\
             5. Be concise and factual.",
            pred.symbol,
            pred.action.label(),
            pred.raw_prediction,
            pred.bounds.output.lo,
            pred.bounds.output.hi,
            pred.bounds.uncertainty,
            pred.last_close,
            pred.snapshot_fingerprint,
            attrs_str,
            pred.bounds.output.lo,
            pred.bounds.output.hi,
        )
    }

    /// Call Ollama `/api/generate`. Returns the raw text response, or an
    /// error string prefixed with `[error]` if the call fails.
    ///
    /// Networking is opt-in: build with `--features live`. The default build
    /// keeps SciRust pure-Rust (no TLS/C dependency) and returns an error
    /// string instead of making a request.
    #[cfg(not(feature = "live"))]
    fn call_ollama(&self, _prompt: &str) -> String {
        format!(
            "[error: live feature disabled for {} at {} — build scirust-trader with --features live]",
            self.model, self.base_url
        )
    }

    /// Call Ollama `/api/generate`. Returns the raw text response, or an
    /// error string prefixed with `[error]` if the call fails.
    #[cfg(feature = "live")]
    fn call_ollama(&self, prompt: &str) -> String {
        let url = format!("{}/api/generate", self.base_url);
        let body = OllamaRequest {
            model: self.model.clone(),
            prompt: prompt.to_string(),
            stream: false,
        };
        let client = match reqwest::blocking::Client::builder()
            .timeout(std::time::Duration::from_secs(30))
            .build()
        {
            Ok(c) => c,
            Err(e) => return format!("[error: cannot build HTTP client: {}]", e),
        };
        match client.post(&url).json(&body).send()
        {
            Ok(resp) => match resp.json::<OllamaResponse>()
            {
                Ok(parsed) => parsed.response,
                Err(e) => format!("[error: cannot parse Ollama response: {}]", e),
            },
            Err(e) => format!("[error: cannot reach Ollama at {}: {}]", url, e),
        }
    }
}

impl LlmClient for OllamaClient {
    fn narrate(&self, pred: &CertifiedPrediction) -> String {
        let prompt = Self::build_prompt(pred);
        self.call_ollama(&prompt)
    }

    fn sanity_check(&self, pred: &CertifiedPrediction, narration: &str) -> bool {
        // A narration is consistent if:
        // 1. It does not start with "[error" (HTTP call succeeded)
        // 2. It does not contain any number that looks like a *return prediction*
        //    outside the certified bounds. We only flag numbers in [-1, 1] that
        //    appear near keywords like "predict", "return", "expected".
        if narration.starts_with("[error") || narration.is_empty()
        {
            return false;
        }
        let lo = pred.bounds.output.lo;
        let hi = pred.bounds.output.hi;
        let tolerance = (hi - lo).abs().max(0.01) * 3.0;
        let keywords = ["predict", "return", "expected", "forecast"];
        let words: Vec<&str> = narration.split_whitespace().collect();
        for (i, token) in words.iter().enumerate()
        {
            let cleaned = token
                .trim_matches(|c: char| !c.is_ascii_digit() && c != '.' && c != '-' && c != '+');
            if let Ok(v) = cleaned.parse::<f32>()
            {
                // Only flag small numbers (potential returns, not prices).
                if v.abs() < 1.0
                {
                    // Check if a keyword is nearby (within 5 words).
                    let nearby = words[i.saturating_sub(5)..=i.min(words.len() - 1)]
                        .join(" ")
                        .to_lowercase();
                    if keywords.iter().any(|kw| nearby.contains(kw))
                        && (v < lo - tolerance || v > hi + tolerance)
                    {
                        return false;
                    }
                }
            }
        }
        true
    }
}

/// The trading agent — orchestrates the full pipeline.
pub struct TradingAgent {
    pub model: PricePredictor,
    pub llm: Box<dyn LlmClient>,
    pub action_threshold: f32,
    pub certify_eps: f32,
    pub lookback: usize,
}

impl TradingAgent {
    pub fn new(model: PricePredictor, llm: Box<dyn LlmClient>) -> Self {
        Self {
            model,
            llm,
            action_threshold: 0.001,
            certify_eps: 0.01,
            lookback: 10,
        }
    }

    /// Process one market snapshot → certified prediction → LLM narration.
    pub fn process(&mut self, snapshot: &MarketSnapshot) -> DecisionRecord {
        let closes = snapshot.closes();
        let _n = closes.len();
        let highs: Vec<f32> = snapshot.candles.iter().map(|c| c.high).collect();
        let lows: Vec<f32> = snapshot.candles.iter().map(|c| c.low).collect();

        let indicators =
            IndicatorSet::from_ohlcv(&highs, &lows, &closes, 14, 12, 26, 9, 14, 20, 2.0);

        let features = build_features(
            &closes,
            &indicators.rsi,
            &indicators.macd_hist,
            &indicators.atr,
            self.lookback,
        );

        let raw_pred = self.model.predict(&features);
        let action = Action::from_prediction(raw_pred, self.action_threshold);

        let weights = self.model.export_weights();
        let bounds = certify(&weights, &features, self.certify_eps);

        let feature_names: Vec<String> = (0..self.lookback)
            .map(|i| format!("close_{}", i))
            .chain(["rsi", "macd_hist", "atr"].iter().map(|s| s.to_string()))
            .collect();
        let attribution = feature_attribution(&features, &feature_names);

        let pred = CertifiedPrediction {
            symbol: snapshot.symbol.clone(),
            action,
            raw_prediction: raw_pred,
            bounds: bounds.clone(),
            feature_attribution: attribution,
            snapshot_fingerprint: snapshot.fingerprint(),
            weights_fingerprint: weights.fingerprint.clone(),
            last_close: snapshot.last_close().unwrap_or(0.0),
        };

        let narration = self.llm.narrate(&pred);
        let consistent = self.llm.sanity_check(&pred, &narration);

        DecisionRecord {
            prediction: pred,
            narration,
            llm_consistent: consistent,
            timestamp_ms: chrono::Utc::now().timestamp_millis(),
        }
    }

    /// Run a backtest on a mock feed and produce proofs.
    pub fn backtest(
        &mut self,
        feed: &mut impl MarketFeed,
        num_steps: usize,
        window: usize,
    ) -> Vec<DecisionRecord> {
        let mut records = Vec::with_capacity(num_steps);
        for _ in 0..num_steps
        {
            if let Some(snapshot) = feed.next_snapshot(window)
            {
                let record = self.process(&snapshot);
                records.push(record);
            }
            else
            {
                break;
            }
        }
        records
    }

    /// Seal all decisions into a proof file.
    pub fn seal_proof(&self, records: &[DecisionRecord]) -> DecisionProof {
        DecisionProof::from_records(records)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::market::MockExchange;

    #[test]
    fn agent_processes_snapshot() {
        let model = PricePredictor::new(13, &[16, 8], 42);
        let mut agent = TradingAgent::new(model, Box::new(StubLlm));
        agent.lookback = 10;

        let mut feed = MockExchange::new(42, 50_000.0);
        let snapshot = feed.next_snapshot(50).unwrap();
        let record = agent.process(&snapshot);

        assert!(!record.narration.is_empty());
        assert!(record.llm_consistent);
        assert_eq!(record.prediction.symbol, "BTC/USDT");
    }

    #[test]
    fn backtest_produces_multiple_records() {
        let model = PricePredictor::new(13, &[16, 8], 42);
        let mut agent = TradingAgent::new(model, Box::new(StubLlm));
        agent.lookback = 10;

        let mut feed = MockExchange::new(42, 50_000.0);
        let records = agent.backtest(&mut feed, 5, 50);
        assert_eq!(records.len(), 5);
    }

    #[test]
    fn ollama_client_builds_prompt() {
        let model = PricePredictor::new(13, &[16, 8], 42);
        let mut agent = TradingAgent::new(
            model,
            // Use a port that's almost certainly closed so the test is fast.
            Box::new(OllamaClient::new("qwen3:8b", "http://127.0.0.1:1")),
        );
        agent.lookback = 10;

        let mut feed = MockExchange::new(42, 50_000.0);
        let snapshot = feed.next_snapshot(50).unwrap();
        let record = agent.process(&snapshot);
        // The narration will be an error string (no Ollama at port 1).
        assert!(!record.narration.is_empty());
        assert!(
            !record.llm_consistent,
            "error narration should be inconsistent"
        );
    }

    #[test]
    fn action_threshold_works() {
        assert_eq!(Action::from_prediction(0.01, 0.001), Action::Long);
        assert_eq!(Action::from_prediction(-0.01, 0.001), Action::Short);
        assert_eq!(Action::from_prediction(0.0, 0.001), Action::Flat);
    }

    #[test]
    fn proof_seals_records() {
        let model = PricePredictor::new(13, &[16, 8], 42);
        let mut agent = TradingAgent::new(model, Box::new(StubLlm));
        agent.lookback = 10;

        let mut feed = MockExchange::new(42, 50_000.0);
        let records = agent.backtest(&mut feed, 3, 50);
        let proof = agent.seal_proof(&records);
        assert_eq!(proof.num_decisions, 3);
        assert_eq!(proof.manifest_hash.len(), 64);
    }
}
