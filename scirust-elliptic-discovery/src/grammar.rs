//! Finite typed grammar for exact unary point relations.

use std::collections::BTreeSet;

use crate::{CurveError, RelationSignature, ToyCurve, ToyPoint};

/// A point-valued expression. Construction is finite and contains no external data.
#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum PointExpression {
    Input,
    Identity,
    Negate(Box<Self>),
    Double(Box<Self>),
    ScalarMultiply { scalar: u64, point: Box<Self> },
    Add(Box<Self>, Box<Self>),
}

impl PointExpression {
    /// Evaluates with exact group operations.
    pub fn evaluate(&self, curve: ToyCurve, input: ToyPoint) -> Result<ToyPoint, CurveError> {
        match self
        {
            Self::Input => Ok(input),
            Self::Identity => Ok(curve.identity()),
            Self::Negate(point) => curve.negate(point.evaluate(curve, input)?),
            Self::Double(point) =>
            {
                let value = point.evaluate(curve, input)?;
                curve.add(value, value)
            },
            Self::ScalarMultiply { scalar, point } =>
            {
                curve.scalar_mul(point.evaluate(curve, input)?, *scalar)
            },
            Self::Add(left, right) =>
            {
                curve.add(left.evaluate(curve, input)?, right.evaluate(curve, input)?)
            },
        }
    }

    /// Maximum syntax-tree depth.
    pub fn depth(&self) -> u8 {
        match self
        {
            Self::Input | Self::Identity => 0,
            Self::Negate(point) | Self::Double(point) | Self::ScalarMultiply { point, .. } =>
            {
                point.depth().saturating_add(1)
            },
            Self::Add(left, right) => left.depth().max(right.depth()).saturating_add(1),
        }
    }
}

/// A boolean relation evaluated on one curve and one point.
#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum Relation {
    PointEqual(PointExpression, PointExpression),
    IsInfinity(PointExpression),
    CurveAEquals(u64),
    CurveJEquals(u64),
}

impl Relation {
    /// Exact relation evaluation. Partial point operations propagate an error.
    pub fn evaluate(&self, curve: ToyCurve, input: ToyPoint) -> Result<bool, CurveError> {
        match self
        {
            Self::PointEqual(left, right) =>
            {
                Ok(left.evaluate(curve, input)? == right.evaluate(curve, input)?)
            },
            Self::IsInfinity(point) => Ok(point.evaluate(curve, input)?.is_infinity()),
            Self::CurveAEquals(value) => Ok(curve.a() == value % curve.prime().value()),
            Self::CurveJEquals(value) => Ok(crate::CurveInvariants::compute(curve).j_invariant()
                == value % curve.prime().value()),
        }
    }

    /// Recognizes structural forms which belong to the known catalog.
    pub fn signature(&self) -> RelationSignature {
        match self
        {
            Self::PointEqual(left, right)
                if is_double_negation_of_input(left) && matches!(right, PointExpression::Input) =>
            {
                RelationSignature::NegationInvolution
            },
            Self::PointEqual(left, right)
                if is_additive_inverse(left, right) || is_additive_inverse(right, left) =>
            {
                RelationSignature::AdditiveInverse
            },
            Self::PointEqual(left, right)
                if is_scalar_composition(left, right) || is_scalar_composition(right, left) =>
            {
                RelationSignature::ScalarComposition
            },
            _ => RelationSignature::Unrecognized,
        }
    }
}

fn is_double_negation_of_input(expression: &PointExpression) -> bool {
    matches!(
        expression,
        PointExpression::Negate(outer)
            if matches!(outer.as_ref(), PointExpression::Negate(inner)
                if matches!(inner.as_ref(), PointExpression::Input))
    )
}

fn is_additive_inverse(left: &PointExpression, right: &PointExpression) -> bool {
    matches!(
        (left, right),
        (PointExpression::Add(first, second), PointExpression::Identity)
            if matches!(first.as_ref(), PointExpression::Input)
                && matches!(second.as_ref(), PointExpression::Negate(point)
                    if matches!(point.as_ref(), PointExpression::Input))
    )
}

fn is_scalar_composition(left: &PointExpression, right: &PointExpression) -> bool {
    let PointExpression::ScalarMultiply {
        scalar: outer,
        point: outer_point,
    } = left
    else
    {
        return false;
    };
    let PointExpression::ScalarMultiply {
        scalar: inner,
        point: inner_point,
    } = outer_point.as_ref()
    else
    {
        return false;
    };
    let PointExpression::ScalarMultiply {
        scalar: product,
        point: right_point,
    } = right
    else
    {
        return false;
    };
    matches!(inner_point.as_ref(), PointExpression::Input)
        && matches!(right_point.as_ref(), PointExpression::Input)
        && outer.checked_mul(inner) == Some(product)
}

/// Generates a deterministic prefix of the finite relation grammar.
pub fn generate_relations(max_depth: u8, max_scalar: u64, budget: usize) -> Vec<Relation> {
    if budget == 0
    {
        return Vec::new();
    }
    let mut all = BTreeSet::from([PointExpression::Identity, PointExpression::Input]);
    let mut frontier = all.clone();
    for _ in 0..max_depth
    {
        let previous: Vec<_> = frontier.into_iter().collect();
        let universe: Vec<_> = all.iter().cloned().collect();
        let mut next = BTreeSet::new();
        for point in &previous
        {
            next.insert(PointExpression::Negate(Box::new(point.clone())));
            next.insert(PointExpression::Double(Box::new(point.clone())));
            for scalar in 0..=max_scalar
            {
                next.insert(PointExpression::ScalarMultiply {
                    scalar,
                    point: Box::new(point.clone()),
                });
            }
            for other in &universe
            {
                next.insert(PointExpression::Add(
                    Box::new(point.clone()),
                    Box::new(other.clone()),
                ));
            }
        }
        next.retain(|expression| expression.depth() <= max_depth);
        frontier = next.difference(&all).cloned().collect();
        all.extend(frontier.iter().cloned());
    }

    let expressions: Vec<_> = all.into_iter().collect();
    let mut relations = Vec::with_capacity(budget);
    'outer: for (left_index, left) in expressions.iter().enumerate()
    {
        for right in &expressions[left_index..]
        {
            relations.push(Relation::PointEqual(left.clone(), right.clone()));
            if relations.len() == budget
            {
                break 'outer;
            }
        }
    }
    relations
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generation_is_bounded_sorted_and_reproducible() {
        let left = generate_relations(2, 3, 50);
        let right = generate_relations(2, 3, 50);
        assert_eq!(left, right);
        assert_eq!(left.len(), 50);
        assert!(left.windows(2).all(|pair| pair[0] <= pair[1]));
    }

    #[test]
    fn known_syntax_is_recognized_before_candidate_classification() {
        let input = PointExpression::Input;
        let relation = Relation::PointEqual(
            PointExpression::Negate(Box::new(PointExpression::Negate(Box::new(input.clone())))),
            input,
        );
        assert_eq!(relation.signature(), RelationSignature::NegationInvolution);
    }
}
