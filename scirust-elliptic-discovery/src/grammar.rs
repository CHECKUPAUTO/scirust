//! Finite typed grammar for exact unary point relations.

use std::collections::BTreeSet;

use crate::{CurveError, RelationSignature, ToyCurve, ToyPoint};

/// Maximum expression depth accepted by the local grammar.
pub const MAX_GRAMMAR_DEPTH: u8 = 4;
/// Maximum scalar accepted by the local grammar.
pub const MAX_GRAMMAR_SCALAR: u64 = 32;
/// Maximum number of relations produced by one local grammar request.
pub const MAX_GENERATED_RELATIONS: usize = 10_000;

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
                if (is_double_negation_of_input(left) && is_input(right))
                    || (is_input(left) && is_double_negation_of_input(right)) =>
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

fn is_input(expression: &PointExpression) -> bool {
    matches!(expression, PointExpression::Input)
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
    let (PointExpression::Add(first, second), PointExpression::Identity) = (left, right)
    else
    {
        return false;
    };
    (is_input(first) && is_negation_of_input(second))
        || (is_negation_of_input(first) && is_input(second))
}

fn is_negation_of_input(expression: &PointExpression) -> bool {
    matches!(expression, PointExpression::Negate(point) if is_input(point))
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
        && outer.checked_mul(*inner) == Some(*product)
}

/// Generates a deterministic, resource-bounded sequence of local relations.
///
/// The request is clamped to the public toy-domain limits so direct callers
/// cannot bypass `SearchPlan`. The sequence round-robins the public relation
/// variants instead of materializing the full expression grammar.
pub fn generate_relations(max_depth: u8, max_scalar: u64, budget: usize) -> Vec<Relation> {
    let budget = budget.min(MAX_GENERATED_RELATIONS);
    if budget == 0
    {
        return Vec::new();
    }

    let max_depth = max_depth.min(MAX_GRAMMAR_DEPTH);
    let max_scalar = max_scalar.min(MAX_GRAMMAR_SCALAR);
    let expression_budget = expression_count_for_pair_budget(budget);
    let expressions = generate_expression_prefix(max_depth, max_scalar, expression_budget);

    let mut point_equalities = point_equalities(&expressions, budget).into_iter();
    let mut infinities = expressions.into_iter().map(Relation::IsInfinity);
    let mut curve_a_values = (0..=max_scalar).map(Relation::CurveAEquals);
    let mut curve_j_values = j_invariant_values(max_scalar)
        .into_iter()
        .map(Relation::CurveJEquals);
    let mut relations = Vec::with_capacity(budget);

    while relations.len() < budget
    {
        let mut progressed = false;
        if let Some(relation) = point_equalities.next()
        {
            relations.push(relation);
            progressed = true;
        }
        if relations.len() == budget
        {
            break;
        }
        if let Some(relation) = infinities.next()
        {
            relations.push(relation);
            progressed = true;
        }
        if relations.len() == budget
        {
            break;
        }
        if let Some(relation) = curve_a_values.next()
        {
            relations.push(relation);
            progressed = true;
        }
        if relations.len() == budget
        {
            break;
        }
        if let Some(relation) = curve_j_values.next()
        {
            relations.push(relation);
            progressed = true;
        }
        if !progressed
        {
            break;
        }
    }

    relations
}

fn expression_count_for_pair_budget(budget: usize) -> usize {
    let mut expressions = 0usize;
    let mut pairs = 0usize;
    while pairs < budget
    {
        expressions += 1;
        pairs += expressions;
    }
    expressions
}

fn generate_expression_prefix(
    max_depth: u8,
    max_scalar: u64,
    expression_budget: usize,
) -> Vec<PointExpression> {
    let mut all = BTreeSet::new();
    all.insert(PointExpression::Input);
    if all.len() < expression_budget
    {
        all.insert(PointExpression::Identity);
    }
    let mut frontier = all.clone();

    for _ in 0..max_depth
    {
        if all.len() == expression_budget || frontier.is_empty()
        {
            break;
        }
        let universe: Vec<_> = all.iter().cloned().collect();
        let mut next = BTreeSet::new();
        'expand: for point in &frontier
        {
            if insert_expression(
                &mut next,
                &all,
                expression_budget,
                PointExpression::Negate(Box::new(point.clone())),
            )
            {
                break 'expand;
            }
            if insert_expression(
                &mut next,
                &all,
                expression_budget,
                PointExpression::Double(Box::new(point.clone())),
            )
            {
                break 'expand;
            }
            for scalar in 0..=max_scalar
            {
                if insert_expression(
                    &mut next,
                    &all,
                    expression_budget,
                    PointExpression::ScalarMultiply {
                        scalar,
                        point: Box::new(point.clone()),
                    },
                )
                {
                    break 'expand;
                }
            }
            for other in &universe
            {
                if insert_expression(
                    &mut next,
                    &all,
                    expression_budget,
                    PointExpression::Add(Box::new(point.clone()), Box::new(other.clone())),
                )
                {
                    break 'expand;
                }
            }
        }
        if next.is_empty()
        {
            break;
        }
        all.extend(next.iter().cloned());
        frontier = next;
    }

    all.into_iter().collect()
}

fn insert_expression(
    next: &mut BTreeSet<PointExpression>,
    all: &BTreeSet<PointExpression>,
    expression_budget: usize,
    expression: PointExpression,
) -> bool {
    if all.len() + next.len() == expression_budget
    {
        return true;
    }
    if !all.contains(&expression)
    {
        next.insert(expression);
    }
    all.len() + next.len() == expression_budget
}

fn point_equalities(expressions: &[PointExpression], budget: usize) -> Vec<Relation> {
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

fn j_invariant_values(max_scalar: u64) -> Vec<u64> {
    let mut values: Vec<_> = (0..=max_scalar).collect();
    if !values.contains(&1728)
    {
        values.push(1728);
    }
    values
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generation_is_bounded_and_reproducible() {
        let left = generate_relations(2, 3, 50);
        let right = generate_relations(2, 3, 50);
        assert_eq!(left, right);
        assert_eq!(left.len(), 50);
        assert_eq!(left.iter().collect::<BTreeSet<_>>().len(), left.len());
    }

    #[test]
    fn every_relation_variant_is_reachable_in_a_bounded_sequence() {
        let relations = generate_relations(0, 0, 4);
        assert_eq!(relations.len(), 4);
        assert!(matches!(relations[0], Relation::PointEqual(_, _)));
        assert!(matches!(relations[1], Relation::IsInfinity(_)));
        assert!(matches!(relations[2], Relation::CurveAEquals(0)));
        assert!(matches!(relations[3], Relation::CurveJEquals(0)));
        assert!(generate_relations(0, 0, 20).contains(&Relation::CurveJEquals(1728)));
    }

    #[test]
    fn direct_generation_cannot_exceed_the_local_resource_bound() {
        let relations = generate_relations(u8::MAX, u64::MAX, MAX_GENERATED_RELATIONS + 1);
        assert_eq!(relations.len(), MAX_GENERATED_RELATIONS);
    }

    #[test]
    fn known_syntax_is_recognized_independently_of_tree_order() {
        let input = PointExpression::Input;
        let double_negation =
            PointExpression::Negate(Box::new(PointExpression::Negate(Box::new(input.clone()))));
        for relation in [
            Relation::PointEqual(double_negation.clone(), input.clone()),
            Relation::PointEqual(input.clone(), double_negation),
        ]
        {
            assert_eq!(relation.signature(), RelationSignature::NegationInvolution);
        }

        let negated_input = PointExpression::Negate(Box::new(input.clone()));
        for relation in [
            Relation::PointEqual(
                PointExpression::Add(Box::new(input.clone()), Box::new(negated_input.clone())),
                PointExpression::Identity,
            ),
            Relation::PointEqual(
                PointExpression::Add(Box::new(negated_input), Box::new(input)),
                PointExpression::Identity,
            ),
        ]
        {
            assert_eq!(relation.signature(), RelationSignature::AdditiveInverse);
        }
    }
}
