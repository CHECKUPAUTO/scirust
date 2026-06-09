use crate::core::{Reasoner, Result};
use std::collections::HashMap;

/// A lightweight probabilistic-logic engine. Atoms carry a prior probability and
/// weighted rules `body => head (w)` contribute evidence. The probability of an
/// event is combined with a **noisy-OR**: independent sources each have a chance
/// to make the event true.
pub struct ProbabilisticLogic {
    pub confidence_threshold: f64,
    /// Prior probability of an atom being true.
    priors: HashMap<String, f64>,
    /// Weighted rules: (body atoms, head, weight).
    rules: Vec<(Vec<String>, String, f64)>,
}

impl ProbabilisticLogic {
    pub fn new(threshold: f64) -> Self {
        Self {
            confidence_threshold: threshold,
            priors: HashMap::new(),
            rules: Vec::new(),
        }
    }

    /// Set the prior probability of an atom.
    pub fn add_fact(&mut self, atom: &str, prob: f64) {
        self.priors.insert(atom.to_string(), prob.clamp(0.0, 1.0));
    }

    /// Add a weighted rule `body => head (weight)`.
    pub fn add_rule(&mut self, body: Vec<&str>, head: &str, weight: f64) {
        self.rules.push((
            body.into_iter().map(|s| s.to_string()).collect(),
            head.to_string(),
            weight.clamp(0.0, 1.0),
        ));
    }

    /// Probability of `event`, combining its prior with each supporting rule via
    /// noisy-OR. Rule bodies are scored from atom priors (single level).
    pub fn infer_probability(&self, event: &str) -> Result<f64> {
        let base = self.priors.get(event).copied().unwrap_or(0.0);
        let mut complement = 1.0 - base;
        for (body, head, weight) in &self.rules {
            if head != event {
                continue;
            }
            let p_body: f64 = body
                .iter()
                .map(|b| self.priors.get(b).copied().unwrap_or(0.0))
                .product();
            let contribution = (weight * p_body).clamp(0.0, 1.0);
            complement *= 1.0 - contribution;
        }
        Ok(1.0 - complement)
    }

    /// True iff `infer_probability(event)` exceeds the configured threshold.
    pub fn holds(&self, event: &str) -> Result<bool> {
        Ok(self.infer_probability(event)? > self.confidence_threshold)
    }
}

impl Reasoner for ProbabilisticLogic {
    fn name(&self) -> &str {
        "ProbabilisticLogic"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn noisy_or_combines_prior_and_rule() {
        let mut pl = ProbabilisticLogic::new(0.5);
        pl.add_fact("rain", 0.3);
        pl.add_fact("cloudy", 0.8);
        pl.add_rule(vec!["cloudy"], "rain", 0.5);
        // 1 - (1-0.3)(1 - 0.5*0.8) = 1 - 0.7*0.6 = 0.58
        let p = pl.infer_probability("rain").unwrap();
        assert!((p - 0.58).abs() < 1e-9, "got {p}");
        assert!(pl.holds("rain").unwrap());
    }

    #[test]
    fn unknown_event_has_zero_probability() {
        let pl = ProbabilisticLogic::new(0.5);
        assert_eq!(pl.infer_probability("unknown").unwrap(), 0.0);
    }
}
