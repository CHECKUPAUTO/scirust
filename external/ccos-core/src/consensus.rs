use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct LlmVote {
    pub model: String,
    pub output: String,
    pub confidence: f64,
}

#[derive(Debug, Clone)]
pub struct ConsensusResult {
    pub output: String,
    pub vote_count: u32,
    pub total_votes: u32,
    pub agreement_ratio: f64,
    pub models_in_agreement: Vec<String>,
    pub consensus_reached: bool,
}

pub struct ConsensusEngine {
    pub min_agreement_ratio: f64,
}

impl ConsensusEngine {
    pub fn new() -> Self {
        Self {
            min_agreement_ratio: 0.5,
        }
    }

    pub fn with_threshold(threshold: f64) -> Self {
        Self {
            min_agreement_ratio: threshold.clamp(0.0, 1.0),
        }
    }

    pub fn resolve(&self, votes: &[LlmVote]) -> ConsensusResult {
        if votes.is_empty() {
            return ConsensusResult {
                output: "NO_CONSENSUS".to_string(),
                vote_count: 0,
                total_votes: 0,
                agreement_ratio: 0.0,
                models_in_agreement: vec![],
                consensus_reached: false,
            };
        }

        let mut score_map: HashMap<String, (u32, Vec<String>)> = HashMap::new();

        for vote in votes {
            let entry = score_map
                .entry(vote.output.clone())
                .or_insert((0, Vec::new()));
            entry.0 += 1;
            entry.1.push(vote.model.clone());
        }

        let (best_output, (count, models)) = score_map
            .into_iter()
            .max_by_key(|(_, (score, _))| *score)
            .unwrap_or_else(|| (String::from("NO_CONSENSUS"), (0, vec![])));

        let total = votes.len() as u32;
        let ratio = count as f64 / total.max(1) as f64;
        let reached = ratio >= self.min_agreement_ratio;

        ConsensusResult {
            output: best_output,
            vote_count: count,
            total_votes: total,
            agreement_ratio: ratio,
            models_in_agreement: models,
            consensus_reached: reached,
        }
    }

    pub fn resolve_weighted(&self, votes: &[LlmVote]) -> ConsensusResult {
        if votes.is_empty() {
            return ConsensusResult {
                output: "NO_CONSENSUS".to_string(),
                vote_count: 0,
                total_votes: 0,
                agreement_ratio: 0.0,
                models_in_agreement: vec![],
                consensus_reached: false,
            };
        }

        let mut score_map: HashMap<String, (f64, Vec<String>)> = HashMap::new();

        for vote in votes {
            let entry = score_map
                .entry(vote.output.clone())
                .or_insert((0.0, Vec::new()));
            entry.0 += vote.confidence.max(0.0);
            entry.1.push(vote.model.clone());
        }

        let (best_output, (weighted_score, models)) = score_map
            .into_iter()
            .max_by(|a, b| {
                a.1 .0
                    .partial_cmp(&b.1 .0)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap_or_else(|| (String::from("NO_CONSENSUS"), (0.0, vec![])));

        let total_confidence: f64 = votes.iter().map(|v| v.confidence.max(0.0)).sum();
        let ratio = if total_confidence > 0.0 {
            weighted_score / total_confidence
        } else {
            0.0
        };
        let reached = ratio >= self.min_agreement_ratio;

        ConsensusResult {
            output: best_output,
            vote_count: models.len() as u32,
            total_votes: votes.len() as u32,
            agreement_ratio: ratio,
            models_in_agreement: models,
            consensus_reached: reached,
        }
    }
}

impl Default for ConsensusEngine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_consensus_selects_majority() {
        let votes = vec![
            LlmVote {
                model: "a".into(),
                output: "X".into(),
                confidence: 0.9,
            },
            LlmVote {
                model: "b".into(),
                output: "X".into(),
                confidence: 0.8,
            },
            LlmVote {
                model: "c".into(),
                output: "Y".into(),
                confidence: 0.7,
            },
        ];

        let engine = ConsensusEngine::new();
        let result = engine.resolve(&votes);
        assert_eq!(result.output, "X");
        assert_eq!(result.vote_count, 2);
        assert!(result.consensus_reached);
    }

    #[test]
    fn test_consensus_no_majority() {
        let votes = vec![
            LlmVote {
                model: "a".into(),
                output: "A".into(),
                confidence: 0.9,
            },
            LlmVote {
                model: "b".into(),
                output: "B".into(),
                confidence: 0.8,
            },
            LlmVote {
                model: "c".into(),
                output: "C".into(),
                confidence: 0.7,
            },
        ];

        let engine = ConsensusEngine::new();
        let result = engine.resolve(&votes);
        assert!(!result.consensus_reached);
        assert_eq!(result.agreement_ratio, 1.0 / 3.0);
    }

    #[test]
    fn test_consensus_empty_votes() {
        let engine = ConsensusEngine::new();
        let result = engine.resolve(&[]);
        assert_eq!(result.output, "NO_CONSENSUS");
        assert!(!result.consensus_reached);
    }

    #[test]
    fn test_weighted_consensus_favors_high_confidence() {
        let votes = vec![
            LlmVote {
                model: "a".into(),
                output: "X".into(),
                confidence: 0.95,
            },
            LlmVote {
                model: "b".into(),
                output: "Y".into(),
                confidence: 0.2,
            },
            LlmVote {
                model: "c".into(),
                output: "Y".into(),
                confidence: 0.3,
            },
        ];

        let engine = ConsensusEngine::new();
        let result = engine.resolve_weighted(&votes);
        // X has 0.95 weighted, Y has 0.5 weighted — X wins despite fewer votes
        assert_eq!(result.output, "X");
        assert!(result.consensus_reached);
    }

    #[test]
    fn test_high_threshold_requires_strong_agreement() {
        let votes = vec![
            LlmVote {
                model: "a".into(),
                output: "X".into(),
                confidence: 0.9,
            },
            LlmVote {
                model: "b".into(),
                output: "X".into(),
                confidence: 0.8,
            },
            LlmVote {
                model: "c".into(),
                output: "Y".into(),
                confidence: 0.7,
            },
        ];

        // 2/3 = 0.66 — with threshold 0.8 this should NOT reach consensus
        let engine = ConsensusEngine::with_threshold(0.8);
        let result = engine.resolve(&votes);
        assert!(
            !result.consensus_reached,
            "2/3 majority must not reach 0.8 threshold"
        );
    }
}
