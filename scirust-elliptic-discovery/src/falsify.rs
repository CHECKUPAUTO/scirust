//! Canonical first-counterexample search over immutable local corpora.

use crate::{Corpus, ToyPoint};

/// First point which refutes a named relation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Counterexample
{
    relation_id: String,
    prime: u64,
    a: u64,
    b: u64,
    point_index: u64,
    point: ToyPoint,
}

impl Counterexample
{
    pub fn relation_id(&self) -> &str
    {
        &self.relation_id
    }

    pub const fn curve_key(&self) -> (u64, u64, u64)
    {
        (self.prime, self.a, self.b)
    }

    pub const fn point_index(&self) -> u64
    {
        self.point_index
    }

    pub const fn point(&self) -> ToyPoint
    {
        self.point
    }
}

/// Searches curves and points in canonical order and returns the first failure.
pub fn first_point_counterexample<F>(
    corpus: &Corpus,
    relation_id: impl Into<String>,
    mut relation: F,
) -> Option<Counterexample>
where
    F: FnMut(crate::ToyCurve, ToyPoint) -> bool,
{
    let relation_id = relation_id.into();
    for entry in corpus.curves()
    {
        let curve = entry.curve();
        for (point_index, point) in curve.enumerate_points().into_iter().enumerate()
        {
            if !relation(curve, point)
            {
                return Some(Counterexample {
                    relation_id,
                    prime: curve.prime().value(),
                    a: curve.a(),
                    b: curve.b(),
                    point_index: u64::try_from(point_index).expect("point index fits in u64"),
                    point,
                });
            }
        }
    }
    None
}
