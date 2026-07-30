use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GuardConfig {
    pub max_output_length: usize,
    pub require_valid_json: bool,
    pub reliability_threshold: f64,
    pub sanitize_control_chars: bool,
    pub max_nesting_depth: usize,
}

impl Default for GuardConfig {
    fn default() -> Self {
        Self {
            max_output_length: 8192,
            require_valid_json: true,
            reliability_threshold: 0.7,
            sanitize_control_chars: true,
            max_nesting_depth: 32,
        }
    }
}

impl GuardConfig {
    /// Read overrides from the environment, falling back to [`Default`] for any unset/unparsable
    /// variable — the same convention as `ScoringWeights::from_env` and `MemoryGraph::new_from_env`.
    /// Recognised: `CCOS_GUARD_MAX_OUTPUT`, `CCOS_GUARD_REQUIRE_JSON`, `CCOS_GUARD_RELIABILITY`,
    /// `CCOS_GUARD_SANITIZE`, `CCOS_GUARD_MAX_DEPTH`. Booleans accept `1/true/yes/on` (else false).
    pub fn from_env() -> Self {
        let d = Self::default();
        let usize_var = |k: &str, f: usize| {
            std::env::var(k)
                .ok()
                .and_then(|v| v.trim().parse::<usize>().ok())
                .unwrap_or(f)
        };
        let f64_var = |k: &str, f: f64| {
            std::env::var(k)
                .ok()
                .and_then(|v| v.trim().parse::<f64>().ok())
                .filter(|x| x.is_finite())
                .unwrap_or(f)
        };
        let bool_var = |k: &str, f: bool| {
            std::env::var(k)
                .ok()
                .map(|v| {
                    matches!(
                        v.trim().to_ascii_lowercase().as_str(),
                        "1" | "true" | "yes" | "on"
                    )
                })
                .unwrap_or(f)
        };
        Self {
            max_output_length: usize_var("CCOS_GUARD_MAX_OUTPUT", d.max_output_length),
            require_valid_json: bool_var("CCOS_GUARD_REQUIRE_JSON", d.require_valid_json),
            reliability_threshold: f64_var("CCOS_GUARD_RELIABILITY", d.reliability_threshold),
            sanitize_control_chars: bool_var("CCOS_GUARD_SANITIZE", d.sanitize_control_chars),
            max_nesting_depth: usize_var("CCOS_GUARD_MAX_DEPTH", d.max_nesting_depth),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GuardResult {
    pub passed: bool,
    pub sanitized_output: String,
    pub reliability_score: f64,
    pub warnings: Vec<String>,
    pub blocked_reason: Option<String>,
}

#[derive(Debug, Clone)]
pub struct GuardLayer {
    config: GuardConfig,
}

impl GuardLayer {
    pub fn new(config: GuardConfig) -> Self {
        Self { config }
    }

    pub fn validate_and_sanitize(&self, raw_output: &str) -> GuardResult {
        let mut warnings = Vec::new();

        if raw_output.is_empty() {
            return GuardResult {
                passed: false,
                sanitized_output: Self::fallback_response(),
                reliability_score: 0.0,
                warnings: vec!["empty output".into()],
                blocked_reason: Some("empty output".into()),
            };
        }

        if raw_output.len() > self.config.max_output_length {
            warnings.push(format!(
                "output truncated from {} to {} chars",
                raw_output.len(),
                self.config.max_output_length
            ));
        }
        let truncated: String = raw_output
            .chars()
            .take(self.config.max_output_length)
            .collect();

        let sanitized = if self.config.sanitize_control_chars {
            Self::sanitize_control_characters(&truncated)
        } else {
            truncated
        };

        let json_valid = if self.config.require_valid_json {
            Self::is_valid_json(&sanitized)
        } else {
            true
        };

        if !json_valid {
            warnings.push("output is not valid JSON".into());
        }

        // Enforce the configured nesting bound: deeply nested JSON is a common
        // denial-of-service / stack-exhaustion vector for downstream consumers.
        let depth = Self::json_nesting_depth(&sanitized);
        let depth_ok = depth <= self.config.max_nesting_depth;
        if !depth_ok {
            warnings.push(format!(
                "nesting depth {} exceeds limit {}",
                depth, self.config.max_nesting_depth
            ));
        }

        let reliability = self.compute_reliability_score(&sanitized, json_valid);

        let passed = reliability >= self.config.reliability_threshold && json_valid && depth_ok;

        let blocked_reason = if !passed {
            Some(if !depth_ok {
                format!(
                    "nesting depth {} exceeds limit {}",
                    depth, self.config.max_nesting_depth
                )
            } else {
                format!(
                    "reliability {:.2} below threshold {:.2}",
                    reliability, self.config.reliability_threshold
                )
            })
        } else {
            None
        };

        let final_output = if passed {
            sanitized
        } else {
            Self::fallback_response()
        };

        GuardResult {
            passed,
            sanitized_output: final_output,
            reliability_score: reliability,
            warnings,
            blocked_reason,
        }
    }

    fn sanitize_control_characters(input: &str) -> String {
        input
            .chars()
            .filter(|c| !c.is_control() || c.is_whitespace() || *c == '\n' || *c == '\t')
            .collect()
    }

    fn is_valid_json(input: &str) -> bool {
        let trimmed = input.trim();
        if trimmed.is_empty() {
            return false;
        }
        // The entire payload must parse as a single JSON value. A previous
        // version accepted any valid *prefix* (e.g. `{"ok":1} <injected text>`),
        // which let trailing hallucinated or prompt-injected content slip past
        // the guard. Requiring whole-string validity closes that hole and is
        // also O(n) instead of the old O(n^2) prefix scan.
        serde_json::from_str::<Value>(trimmed).is_ok()
    }

    /// Maximum nesting depth of `{`/`[` brackets, ignoring brackets that appear
    /// inside JSON string literals.
    fn json_nesting_depth(input: &str) -> usize {
        let mut depth = 0usize;
        let mut max = 0usize;
        let mut in_string = false;
        let mut escaped = false;
        for ch in input.chars() {
            if escaped {
                escaped = false;
                continue;
            }
            match ch {
                '\\' if in_string => escaped = true,
                '"' => in_string = !in_string,
                '{' | '[' if !in_string => {
                    depth += 1;
                    max = max.max(depth);
                }
                '}' | ']' if !in_string => depth = depth.saturating_sub(1),
                _ => {}
            }
        }
        max
    }

    fn compute_reliability_score(&self, output: &str, is_json: bool) -> f64 {
        let mut score: f64 = 0.5;

        if is_json {
            score += 0.3;
        }
        if !output.is_empty() && output.len() > 10 {
            score += 0.1;
        }
        if output.len() < self.config.max_output_length {
            score += 0.05;
        }
        // Penalize very short outputs
        if output.len() < 20 {
            score -= 0.15;
        }
        // Check for balanced braces
        if output.matches('{').count() == output.matches('}').count()
            && output.matches('[').count() == output.matches(']').count()
        {
            score += 0.05;
        }

        score.clamp(0.0, 1.0)
    }

    pub fn fallback_response() -> String {
        serde_json::json!({
            "status": "fallback",
            "message": "Guard layer blocked the output. Deterministic fallback response provided.",
            "analysis": {
                "summary": "No analysis available due to guard rejection.",
                "confidence": 0.0,
                "dependencies": []
            }
        })
        .to_string()
    }

    pub fn reliability_threshold(&self) -> f64 {
        self.config.reliability_threshold
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn guard_config_from_env_defaults_when_unset() {
        // No CCOS_GUARD_* set → identical to Default (env-override convention, default-identical).
        let c = GuardConfig::from_env();
        let d = GuardConfig::default();
        assert_eq!(c.max_output_length, d.max_output_length);
        assert_eq!(c.require_valid_json, d.require_valid_json);
        assert_eq!(c.reliability_threshold, d.reliability_threshold);
        assert_eq!(c.sanitize_control_chars, d.sanitize_control_chars);
        assert_eq!(c.max_nesting_depth, d.max_nesting_depth);
    }

    fn make_guard() -> GuardLayer {
        GuardLayer::new(GuardConfig::default())
    }

    #[test]
    fn test_empty_output_blocked() {
        let guard = make_guard();
        let result = guard.validate_and_sanitize("");
        assert!(!result.passed);
        assert_eq!(result.reliability_score, 0.0);
    }

    #[test]
    fn test_valid_json_passes() {
        let guard = make_guard();
        let input = r#"{"key": "value", "list": [1,2,3]}"#;
        let result = guard.validate_and_sanitize(input);
        assert!(result.passed);
        assert!(result.reliability_score > 0.7);
    }

    #[test]
    fn test_invalid_json_blocked() {
        let guard = make_guard();
        let input = "not json at all {{{";
        let result = guard.validate_and_sanitize(input);
        assert!(!result.passed);
    }

    #[test]
    fn test_control_char_sanitization() {
        let guard = make_guard();
        let input = "hello\x00world\x01";
        let result = guard.validate_and_sanitize(input);
        assert!(result.sanitized_output.contains("helloworld") || !result.passed);
    }

    #[test]
    fn test_fallback_response_is_valid_json() {
        let fallback = GuardLayer::fallback_response();
        assert!(serde_json::from_str::<Value>(&fallback).is_ok());
    }

    #[test]
    fn test_nesting_depth_counted() {
        assert_eq!(GuardLayer::json_nesting_depth("{}"), 1);
        assert_eq!(GuardLayer::json_nesting_depth(r#"{"a":{"b":[1]}}"#), 3);
        // Brackets inside strings must not count.
        assert_eq!(GuardLayer::json_nesting_depth(r#"{"a":"{{{["}"#), 1);
    }

    #[test]
    fn test_deep_nesting_blocked() {
        let config = GuardConfig {
            max_nesting_depth: 4,
            ..GuardConfig::default()
        };
        let guard = GuardLayer::new(config);
        let deep = format!("{}true{}", "[".repeat(10), "]".repeat(10));
        let result = guard.validate_and_sanitize(&deep);
        assert!(!result.passed, "nesting beyond the limit must be blocked");
        assert!(serde_json::from_str::<Value>(&result.sanitized_output).is_ok());
    }
}
