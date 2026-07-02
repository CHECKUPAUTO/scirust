//! Cause-and-effect matrix: the standard process-safety artifact mapping
//! detected conditions ("causes", e.g. "High Pressure PT-101") to safety
//! actions ("effects", e.g. "Close XV-201"). A named, structured matrix
//! rather than a spreadsheet — so it can be evaluated deterministically
//! against a set of active causes and, unlike a spreadsheet, its evolution
//! can be hash-chain audited (a C&E link changing is itself a safety-
//! relevant event).

use crate::error::{SisError, SisResult};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CauseEffectMatrix {
    causes: Vec<String>,
    effects: Vec<String>,
    /// `links[cause_index][effect_index]`.
    links: Vec<Vec<bool>>,
}

impl CauseEffectMatrix {
    pub fn new(causes: Vec<String>, effects: Vec<String>) -> Self {
        let links = vec![vec![false; effects.len()]; causes.len()];
        Self {
            causes,
            effects,
            links,
        }
    }

    pub fn causes(&self) -> &[String] {
        &self.causes
    }

    pub fn effects(&self) -> &[String] {
        &self.effects
    }

    fn cause_index(&self, cause: &str) -> SisResult<usize> {
        self.causes
            .iter()
            .position(|c| c == cause)
            .ok_or_else(|| SisError::UnknownCause(cause.to_string()))
    }

    fn effect_index(&self, effect: &str) -> SisResult<usize> {
        self.effects
            .iter()
            .position(|e| e == effect)
            .ok_or_else(|| SisError::UnknownEffect(effect.to_string()))
    }

    /// Links `cause` to `effect` (idempotent — linking twice is a no-op).
    pub fn link(&mut self, cause: &str, effect: &str) -> SisResult<()> {
        let ci = self.cause_index(cause)?;
        let ei = self.effect_index(effect)?;
        self.links[ci][ei] = true;
        Ok(())
    }

    pub fn unlink(&mut self, cause: &str, effect: &str) -> SisResult<()> {
        let ci = self.cause_index(cause)?;
        let ei = self.effect_index(effect)?;
        self.links[ci][ei] = false;
        Ok(())
    }

    pub fn is_linked(&self, cause: &str, effect: &str) -> SisResult<bool> {
        let ci = self.cause_index(cause)?;
        let ei = self.effect_index(effect)?;
        Ok(self.links[ci][ei])
    }

    /// Given the causes currently active, returns the effects that must
    /// execute — the union of every active cause's linked effects, in
    /// matrix column order (deterministic, independent of `active_causes`'
    /// input order).
    pub fn apply(&self, active_causes: &[&str]) -> SisResult<Vec<String>> {
        let mut triggered = vec![false; self.effects.len()];
        for &cause in active_causes
        {
            let ci = self.cause_index(cause)?;
            for (ei, &linked) in self.links[ci].iter().enumerate()
            {
                if linked
                {
                    triggered[ei] = true;
                }
            }
        }
        Ok(self
            .effects
            .iter()
            .zip(triggered)
            .filter(|(_, on)| *on)
            .map(|(effect, _)| effect.clone())
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> CauseEffectMatrix {
        let mut m = CauseEffectMatrix::new(
            vec![
                "High Pressure PT-101".to_string(),
                "Low Level LT-201".to_string(),
            ],
            vec![
                "Close XV-201".to_string(),
                "Trip Compressor".to_string(),
                "Open PSV-301".to_string(),
            ],
        );
        m.link("High Pressure PT-101", "Close XV-201").unwrap();
        m.link("High Pressure PT-101", "Trip Compressor").unwrap();
        m.link("Low Level LT-201", "Trip Compressor").unwrap();
        m
    }

    #[test]
    fn apply_returns_union_of_linked_effects() {
        let m = sample();
        let effects = m.apply(&["High Pressure PT-101"]).unwrap();
        assert_eq!(
            effects,
            vec!["Close XV-201".to_string(), "Trip Compressor".to_string()]
        );
    }

    #[test]
    fn apply_deduplicates_shared_effect_across_causes() {
        let m = sample();
        let effects = m
            .apply(&["High Pressure PT-101", "Low Level LT-201"])
            .unwrap();
        assert_eq!(
            effects,
            vec!["Close XV-201".to_string(), "Trip Compressor".to_string()]
        );
    }

    #[test]
    fn apply_with_no_active_causes_triggers_nothing() {
        let m = sample();
        assert!(m.apply(&[]).unwrap().is_empty());
    }

    #[test]
    fn unlink_removes_a_link() {
        let mut m = sample();
        m.unlink("High Pressure PT-101", "Close XV-201").unwrap();
        assert!(!m.is_linked("High Pressure PT-101", "Close XV-201").unwrap());
        let effects = m.apply(&["High Pressure PT-101"]).unwrap();
        assert_eq!(effects, vec!["Trip Compressor".to_string()]);
    }

    #[test]
    fn rejects_unknown_cause_or_effect() {
        let m = sample();
        assert!(m.apply(&["Nonexistent Cause"]).is_err());
        let mut m2 = sample();
        assert!(
            m2.link("High Pressure PT-101", "Nonexistent Effect")
                .is_err()
        );
    }
}
