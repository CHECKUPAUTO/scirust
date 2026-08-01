//! Canonical first-counterexample search over immutable local corpora.

use crate::{Corpus, ToyPoint};

/// First point which refutes a named relation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Counterexample {
    relation_id: String,
    prime: u64,
    a: u64,
    b: u64,
    point_index: u64,
    point: ToyPoint,
}

impl Counterexample {
    pub fn relation_id(&self) -> &str {
        &self.relation_id
    }

    pub const fn curve_key(&self) -> (u64, u64, u64) {
        (self.prime, self.a, self.b)
    }

    pub const fn point_index(&self) -> u64 {
        self.point_index
    }

    pub const fn point(&self) -> ToyPoint {
        self.point
    }
}

/// Exact outcome of a bounded first-counterexample search.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FalsificationResult {
    counterexample: Option<Counterexample>,
    evaluated_tuples: u64,
}

impl FalsificationResult {
    /// The first canonical counterexample within the declared budget, if one exists.
    pub const fn counterexample(&self) -> Option<&Counterexample> {
        self.counterexample.as_ref()
    }

    /// Number of relation calls actually performed.
    pub const fn evaluated_tuples(&self) -> u64 {
        self.evaluated_tuples
    }

    /// Consumes the result and returns its counterexample, if any.
    pub fn into_counterexample(self) -> Option<Counterexample> {
        self.counterexample
    }
}

/// Searches curves and points in canonical order and returns the first failure.
pub fn first_point_counterexample<F>(
    corpus: &Corpus,
    relation_id: impl Into<String>,
    relation: F,
) -> Option<Counterexample>
where
    F: FnMut(crate::ToyCurve, ToyPoint) -> bool,
{
    first_point_counterexample_bounded(corpus, relation_id, u64::MAX, relation)
        .into_counterexample()
}

/// Searches at most `tuple_budget` points in canonical order.
///
/// The relation is never called after the budget is exhausted. A result without
/// a counterexample is therefore not evidence of complete coverage unless its
/// `evaluated_tuples` equals the corpus point count.
pub fn first_point_counterexample_bounded<F>(
    corpus: &Corpus,
    relation_id: impl Into<String>,
    tuple_budget: u64,
    mut relation: F,
) -> FalsificationResult
where
    F: FnMut(crate::ToyCurve, ToyPoint) -> bool,
{
    let relation_id = relation_id.into();
    let mut evaluated_tuples = 0u64;
    for entry in corpus.curves()
    {
        if evaluated_tuples == tuple_budget
        {
            break;
        }
        let curve = entry.curve();
        for (point_index, point) in curve.enumerate_points().into_iter().enumerate()
        {
            if evaluated_tuples == tuple_budget
            {
                return FalsificationResult {
                    counterexample: None,
                    evaluated_tuples,
                };
            }
            evaluated_tuples = evaluated_tuples
                .checked_add(1)
                .expect("bounded toy corpus tuple count fits in u64");
            if !relation(curve, point)
            {
                return FalsificationResult {
                    counterexample: Some(Counterexample {
                        relation_id,
                        prime: curve.prime().value(),
                        a: curve.a(),
                        b: curve.b(),
                        point_index: u64::try_from(point_index).expect("point index fits in u64"),
                        point,
                    }),
                    evaluated_tuples,
                };
            }
        }
    }
    FalsificationResult {
        counterexample: None,
        evaluated_tuples,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CorpusKind, ExperimentManifest, LocalResearchCase};

    fn corpus() -> Corpus {
        Corpus::generate(ExperimentManifest::new(
            LocalResearchCase::new(23, CorpusKind::IndependentHoldout, 1, 100)
                .expect("valid local case"),
        ))
    }

    #[test]
    fn bounded_search_stops_before_calling_the_relation_past_its_budget() {
        let corpus = corpus();
        let mut calls = 0u64;
        let result = first_point_counterexample_bounded(&corpus, "always-true", 1, |_, _| {
            calls += 1;
            true
        });
        assert_eq!(calls, 1);
        assert_eq!(result.evaluated_tuples(), 1);
        assert!(result.counterexample().is_none());
    }

    #[test]
    fn exhaustive_wrapper_retains_complete_search_semantics() {
        let corpus = corpus();
        let result = first_point_counterexample_bounded(&corpus, "always-true", u64::MAX, |_, _| {
            true
        });
        assert_eq!(result.evaluated_tuples(), corpus.total_points());
        assert!(result.counterexample().is_none());
    }
}
