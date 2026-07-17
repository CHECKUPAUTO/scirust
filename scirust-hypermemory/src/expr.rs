//! Explicitly parenthesized relations over concepts.
//!
//! Non-associativity is represented as **data**, never as an emergent property
//! of a scalar code. A relation is a binary tree of [`S16Expr`]; `(a·b)·c` and
//! `a·(b·c)` are independently representable, produce different digests, and (in
//! general) different sedenion codes. Reconstruction always reads the stored
//! tree — we never attempt to invert a sedenion product (ill-posed under zero
//! divisors and non-associativity).
//!
//! Evaluation is **iterative** (an explicit heap stack, two-stack post-order),
//! so a deep tree cannot exhaust the native call stack. Two guards —
//! [`ExprLimits`] `max_depth` and `max_size` — surface pathological input as
//! typed errors rather than crashes.

use scirust_simd::hypercomplex::SedenionSimd;

use crate::diagnostics::ProductDiagnostics;
use crate::digest::{DOMAIN_EXPRESSION, Digest32, DomainHasher};
use crate::error::{HypermemoryError, Result};
use crate::id::ConceptId;
use crate::store::S16Store;

/// Hard crate ceiling on expression depth. `ExprLimits` cannot be constructed
/// with a larger `max_depth`; combined with iterative evaluation this bounds
/// worst-case work regardless of caller input.
pub const MAX_SUPPORTED_DEPTH: usize = 4096;
/// Hard crate ceiling on expression node count.
pub const MAX_SUPPORTED_SIZE: usize = 1 << 20;

/// Depth and size limits for expression construction and evaluation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ExprLimits {
    max_depth: usize,
    max_size: usize,
}

impl ExprLimits {
    /// Default maximum depth.
    pub const DEFAULT_DEPTH: usize = 64;
    /// Default maximum node count.
    pub const DEFAULT_SIZE: usize = 4096;

    /// Construct limits, validating them against the hard crate ceilings.
    ///
    /// Returns [`HypermemoryError::InvalidRelation`] if either bound is zero or
    /// exceeds its ceiling.
    pub fn new(max_depth: usize, max_size: usize) -> Result<Self> {
        if max_depth == 0 || max_size == 0
        {
            return Err(HypermemoryError::InvalidRelation {
                reason: "expression limits must be non-zero",
            });
        }
        if max_depth > MAX_SUPPORTED_DEPTH
        {
            return Err(HypermemoryError::InvalidRelation {
                reason: "max_depth exceeds the crate ceiling",
            });
        }
        if max_size > MAX_SUPPORTED_SIZE
        {
            return Err(HypermemoryError::InvalidRelation {
                reason: "max_size exceeds the crate ceiling",
            });
        }
        Ok(Self {
            max_depth,
            max_size,
        })
    }

    /// The maximum depth.
    #[inline]
    #[must_use]
    pub const fn max_depth(&self) -> usize {
        self.max_depth
    }

    /// The maximum node count.
    #[inline]
    #[must_use]
    pub const fn max_size(&self) -> usize {
        self.max_size
    }
}

impl Default for ExprLimits {
    fn default() -> Self {
        Self {
            max_depth: Self::DEFAULT_DEPTH,
            max_size: Self::DEFAULT_SIZE,
        }
    }
}

/// A relation expression: an atom (a concept) or a parenthesized product.
///
/// The variants are public so a stored relation can be *read back* (its whole
/// point). Prefer [`S16Expr::try_product`] to build, so limits are enforced up
/// front; evaluation enforces them regardless.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum S16Expr {
    /// A leaf referencing a stored concept.
    Atom(ConceptId),
    /// An ordered, parenthesized product `left · right`.
    Product {
        /// Left operand.
        left: Box<S16Expr>,
        /// Right operand.
        right: Box<S16Expr>,
    },
}

impl S16Expr {
    /// An atom leaf.
    #[must_use]
    pub fn atom(id: ConceptId) -> Self {
        Self::Atom(id)
    }

    /// A product node without limit checking. Evaluation still enforces limits,
    /// so this cannot cause a crash — but [`Self::try_product`] rejects an
    /// over-limit tree at construction time, which is usually what you want.
    #[must_use]
    pub fn product(left: S16Expr, right: S16Expr) -> Self {
        Self::Product {
            left: Box::new(left),
            right: Box::new(right),
        }
    }

    /// A product node, rejected with a typed error if the resulting tree would
    /// exceed `limits`.
    pub fn try_product(left: S16Expr, right: S16Expr, limits: &ExprLimits) -> Result<Self> {
        let (dl, sl) = left.metrics();
        let (dr, sr) = right.metrics();
        let depth = 1 + dl.max(dr);
        let size = 1 + sl + sr;
        if depth > limits.max_depth()
        {
            return Err(HypermemoryError::ExpressionDepthLimit {
                limit: limits.max_depth(),
            });
        }
        if size > limits.max_size()
        {
            return Err(HypermemoryError::ExpressionSizeLimit {
                limit: limits.max_size(),
            });
        }
        Ok(Self::product(left, right))
    }

    /// The tree's depth (a single atom has depth 1). Iterative — never recurses.
    #[must_use]
    pub fn depth(&self) -> usize {
        self.metrics().0
    }

    /// The tree's node count. Iterative — never recurses.
    #[must_use]
    pub fn size(&self) -> usize {
        self.metrics().1
    }

    /// Compute `(depth, size)` in one iterative traversal.
    fn metrics(&self) -> (usize, usize) {
        let mut stack: Vec<(&S16Expr, usize)> = vec![(self, 1)];
        let mut max_depth = 0usize;
        let mut size = 0usize;
        while let Some((node, depth)) = stack.pop()
        {
            size += 1;
            if depth > max_depth
            {
                max_depth = depth;
            }
            if let S16Expr::Product { left, right } = node
            {
                stack.push((left, depth + 1));
                stack.push((right, depth + 1));
            }
        }
        (max_depth, size)
    }

    /// Evaluate the expression to its sedenion code, resolving each atom to its
    /// concept's **anchor** (raw immutable code, not the normalized effective
    /// vector) so norm dynamics — and thus zero divisors — remain observable.
    ///
    /// Failure modes:
    ///
    /// * exceeding `limits.max_depth()` → [`HypermemoryError::ExpressionDepthLimit`];
    /// * exceeding `limits.max_size()` → [`HypermemoryError::ExpressionSizeLimit`];
    /// * an atom on a stale concept → [`HypermemoryError::StaleId`];
    /// * an atom on a vacant / out-of-range slot → [`HypermemoryError::MissingAtom`].
    ///
    /// Evaluation is iterative, so an arbitrarily deep (caller-constructed) tree
    /// fails with a typed error rather than overflowing the stack.
    pub fn evaluate(&self, store: &S16Store, limits: &ExprLimits) -> Result<SedenionSimd> {
        // Pass 1: post-order collection with depth/size guards.
        let post = self.post_order(limits)?;

        // Pass 2: fold with an explicit value stack.
        let mut values: Vec<SedenionSimd> = Vec::new();
        for node in post.iter().rev()
        {
            match node
            {
                S16Expr::Atom(id) =>
                {
                    let anchor = store.get(*id).map_err(map_atom_error)?.anchor();
                    values.push(anchor);
                },
                S16Expr::Product { .. } =>
                {
                    let right = values.pop().ok_or(HypermemoryError::InvariantViolation {
                        detail: "value stack underflow (right)",
                    })?;
                    let left = values.pop().ok_or(HypermemoryError::InvariantViolation {
                        detail: "value stack underflow (left)",
                    })?;
                    values.push(left * right);
                },
            }
        }
        match values.len()
        {
            1 => Ok(values[0]),
            _ => Err(HypermemoryError::InvariantViolation {
                detail: "evaluation did not reduce to a single value",
            }),
        }
    }

    /// Evaluate, additionally returning [`ProductDiagnostics`] for the **root**
    /// product's two evaluated operands (`None` if the root is an atom).
    ///
    /// This is what surfaces the zero-divisor condition for a top-level relation
    /// such as `(e₁+e₁₀)·(e₄−e₁₅)`.
    pub fn evaluate_with_diagnostics(
        &self,
        store: &S16Store,
        limits: &ExprLimits,
        threshold: f32,
    ) -> Result<(SedenionSimd, Option<ProductDiagnostics>)> {
        match self
        {
            S16Expr::Atom(id) =>
            {
                let anchor = store.get(*id).map_err(map_atom_error)?.anchor();
                Ok((anchor, None))
            },
            S16Expr::Product { left, right } =>
            {
                let l = left.evaluate(store, limits)?;
                let r = right.evaluate(store, limits)?;
                let diagnostics = ProductDiagnostics::measure(&l, &r, threshold);
                Ok((l * r, Some(diagnostics)))
            },
        }
    }

    /// Two-stack post-order traversal producing nodes children-before-parent,
    /// with depth and size guards applied during the walk.
    fn post_order<'a>(&'a self, limits: &ExprLimits) -> Result<Vec<&'a S16Expr>> {
        let mut s1: Vec<(&S16Expr, usize)> = vec![(self, 1)];
        let mut collected: Vec<&S16Expr> = Vec::new();
        let mut count = 0usize;
        while let Some((node, depth)) = s1.pop()
        {
            if depth > limits.max_depth()
            {
                return Err(HypermemoryError::ExpressionDepthLimit {
                    limit: limits.max_depth(),
                });
            }
            count += 1;
            if count > limits.max_size()
            {
                return Err(HypermemoryError::ExpressionSizeLimit {
                    limit: limits.max_size(),
                });
            }
            collected.push(node);
            if let S16Expr::Product { left, right } = node
            {
                s1.push((left, depth + 1));
                s1.push((right, depth + 1));
            }
        }
        // `collected` is a reverse post-order; the caller iterates it in reverse
        // to obtain left-subtree-before-right-subtree-before-node order.
        Ok(collected)
    }

    /// A stable, order-sensitive digest of the *structure* (independent of the
    /// floating-point code it evaluates to).
    ///
    /// Domain-separated SHA-256 over a prefix-free pre-order serialization:
    /// each `Product` emits a `0x02` tag then its left then right subtrees; each
    /// `Atom` emits `0x01` then its slot and generation (little-endian). This
    /// uniquely encodes the parenthesization, so `(a·b)·c` and `a·(b·c)` differ.
    /// Iterative — safe on deep trees.
    #[must_use]
    pub fn digest(&self) -> Digest32 {
        let mut hasher = DomainHasher::new(DOMAIN_EXPRESSION);
        let mut stack: Vec<&S16Expr> = vec![self];
        while let Some(node) = stack.pop()
        {
            match node
            {
                S16Expr::Atom(id) =>
                {
                    hasher.update(&[0x01]);
                    hasher.update(&id.slot().to_le_bytes());
                    hasher.update(&id.generation().to_le_bytes());
                },
                S16Expr::Product { left, right } =>
                {
                    hasher.update(&[0x02]);
                    // Push right first so left is serialized first (pre-order,
                    // left-before-right).
                    stack.push(right);
                    stack.push(left);
                },
            }
        }
        hasher.finalize()
    }
}

/// Map a store lookup error into the relation-atom error vocabulary: a
/// generation mismatch stays [`HypermemoryError::StaleId`]; a vacant or
/// out-of-range slot becomes [`HypermemoryError::MissingAtom`].
fn map_atom_error(err: HypermemoryError) -> HypermemoryError {
    match err
    {
        HypermemoryError::VacantSlot { slot } | HypermemoryError::SlotOutOfRange { slot, .. } =>
        {
            HypermemoryError::MissingAtom { slot }
        },
        other => other,
    }
}

/// A relation identifier, minted by the caller.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct RelationId(u64);

impl RelationId {
    /// Construct a relation id from a raw value.
    #[must_use]
    pub const fn new(value: u64) -> Self {
        Self(value)
    }

    /// The raw identifier value.
    #[inline]
    #[must_use]
    pub const fn value(self) -> u64 {
        self.0
    }
}

/// A stored relation: its explicit expression, evaluated code, structural
/// digest, and an optional exact label/payload.
#[derive(Clone, Debug, PartialEq)]
pub struct S16Relation {
    id: RelationId,
    expr: S16Expr,
    code: SedenionSimd,
    digest: Digest32,
    label: Option<Vec<u8>>,
}

impl S16Relation {
    /// Build a relation by evaluating `expr` against `store` under `limits`.
    ///
    /// The evaluated sedenion code and the structural digest are computed once
    /// and stored. Fails with the same errors as [`S16Expr::evaluate`].
    pub fn build(
        id: RelationId,
        expr: S16Expr,
        store: &S16Store,
        limits: &ExprLimits,
    ) -> Result<Self> {
        let code = expr.evaluate(store, limits)?;
        let digest = expr.digest();
        Ok(Self {
            id,
            expr,
            code,
            digest,
            label: None,
        })
    }

    /// Attach an exact label/payload.
    #[must_use]
    pub fn with_label(mut self, label: Vec<u8>) -> Self {
        self.label = Some(label);
        self
    }

    /// The relation identifier.
    #[inline]
    #[must_use]
    pub const fn id(&self) -> RelationId {
        self.id
    }

    /// The explicit expression (the authoritative structure; reconstruction
    /// reads this, never the code).
    #[inline]
    #[must_use]
    pub const fn expr(&self) -> &S16Expr {
        &self.expr
    }

    /// The evaluated sedenion code.
    #[inline]
    #[must_use]
    pub const fn code(&self) -> SedenionSimd {
        self.code
    }

    /// The structural digest.
    #[inline]
    #[must_use]
    pub const fn digest(&self) -> &Digest32 {
        &self.digest
    }

    /// The optional exact label/payload.
    #[inline]
    #[must_use]
    pub fn label(&self) -> Option<&[u8]> {
        self.label.as_deref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::{ConceptSpec, S16Store};

    fn store_with(anchors: &[(u8, SedenionSimd)]) -> (S16Store, Vec<ConceptId>) {
        let mut store = S16Store::new();
        let mut ids = Vec::new();
        for &(tag, anchor) in anchors
        {
            ids.push(
                store
                    .insert(ConceptSpec::new(vec![tag], anchor, 1.0, 0))
                    .unwrap(),
            );
        }
        (store, ids)
    }

    #[test]
    fn atom_evaluates_to_its_anchor() {
        let anchor = SedenionSimd::unit(1) + SedenionSimd::unit(10);
        let (store, ids) = store_with(&[(0, anchor)]);
        let expr = S16Expr::atom(ids[0]);
        let limits = ExprLimits::default();
        assert_eq!(
            expr.evaluate(&store, &limits).unwrap().to_array(),
            anchor.to_array()
        );
        assert_eq!(expr.depth(), 1);
        assert_eq!(expr.size(), 1);
    }

    #[test]
    fn binary_product_matches_direct_multiplication() {
        let a = SedenionSimd::unit(1);
        let b = SedenionSimd::unit(2);
        let (store, ids) = store_with(&[(0, a), (1, b)]);
        let limits = ExprLimits::default();
        let expr =
            S16Expr::try_product(S16Expr::atom(ids[0]), S16Expr::atom(ids[1]), &limits).unwrap();
        assert_eq!(
            expr.evaluate(&store, &limits).unwrap().to_array(),
            (a * b).to_array()
        );
        assert_eq!(expr.depth(), 2);
        assert_eq!(expr.size(), 3);
    }

    #[test]
    fn parenthesization_is_significant() {
        // Non-associativity is preserved by the tree: (a·b)·c ≠ a·(b·c) for a
        // known octonion-embedded case (e1, e2, e4).
        let a = SedenionSimd::unit(1);
        let b = SedenionSimd::unit(2);
        let c = SedenionSimd::unit(4);
        let (store, ids) = store_with(&[(0, a), (1, b), (2, c)]);
        let limits = ExprLimits::default();
        let atom = |i: usize| S16Expr::atom(ids[i]);

        let left = S16Expr::product(S16Expr::product(atom(0), atom(1)), atom(2)); // (a·b)·c
        let right = S16Expr::product(atom(0), S16Expr::product(atom(1), atom(2))); // a·(b·c)

        let lv = left.evaluate(&store, &limits).unwrap();
        let rv = right.evaluate(&store, &limits).unwrap();
        // Match the hand oracle (direct multiplication) exactly...
        assert_eq!(lv.to_array(), ((a * b) * c).to_array());
        assert_eq!(rv.to_array(), (a * (b * c)).to_array());
        // ...and the two parenthesizations genuinely differ.
        assert_ne!(lv.to_array(), rv.to_array());
        // Distinct structures → distinct digests.
        assert_ne!(left.digest(), right.digest());
    }

    #[test]
    fn expression_digest_is_stable() {
        let (store, ids) = store_with(&[(0, SedenionSimd::unit(1)), (1, SedenionSimd::unit(2))]);
        let _ = &store;
        let e1 = S16Expr::product(S16Expr::atom(ids[0]), S16Expr::atom(ids[1]));
        let e2 = S16Expr::product(S16Expr::atom(ids[0]), S16Expr::atom(ids[1]));
        assert_eq!(e1.digest(), e2.digest(), "same structure → same digest");
    }

    #[test]
    fn missing_atom_is_reported() {
        let (mut store, ids) = store_with(&[(0, SedenionSimd::unit(1))]);
        let removed = ids[0];
        store.remove(removed).unwrap();
        let expr = S16Expr::atom(removed);
        let limits = ExprLimits::default();
        // A removed slot is vacant → MissingAtom.
        assert_eq!(
            expr.evaluate(&store, &limits),
            Err(HypermemoryError::MissingAtom {
                slot: removed.slot()
            })
        );
    }

    #[test]
    fn stale_atom_is_reported() {
        let (mut store, ids) = store_with(&[(0, SedenionSimd::unit(1))]);
        let stale = ids[0];
        store.remove(stale).unwrap();
        // Reuse the slot: the old id is now stale (generation mismatch).
        let _new = store
            .insert(ConceptSpec::new(vec![9], SedenionSimd::unit(2), 1.0, 0))
            .unwrap();
        let expr = S16Expr::atom(stale);
        let limits = ExprLimits::default();
        assert_eq!(
            expr.evaluate(&store, &limits),
            Err(HypermemoryError::StaleId {
                slot: stale.slot(),
                id_generation: stale.generation(),
                current_generation: stale.generation() + 1,
            })
        );
    }

    #[test]
    fn depth_limit_is_enforced() {
        let (store, ids) = store_with(&[(0, SedenionSimd::unit(1))]);
        // Build a deep left-leaning chain by hand (unchecked), then evaluate
        // under a tight depth limit.
        let mut expr = S16Expr::atom(ids[0]);
        for _ in 0..10
        {
            expr = S16Expr::product(expr, S16Expr::atom(ids[0]));
        }
        let limits = ExprLimits::new(4, 4096).unwrap();
        assert_eq!(
            expr.evaluate(&store, &limits),
            Err(HypermemoryError::ExpressionDepthLimit { limit: 4 })
        );
        // try_product refuses to build past the limit. Depth exactly at the
        // limit (4) is allowed; one deeper (5) is rejected.
        let base = S16Expr::atom(ids[0]);
        let three = S16Expr::try_product(
            S16Expr::try_product(base.clone(), base.clone(), &limits).unwrap(),
            base.clone(),
            &limits,
        )
        .unwrap();
        assert_eq!(three.depth(), 3);
        let four = S16Expr::try_product(three, base.clone(), &limits).unwrap();
        assert_eq!(four.depth(), 4, "depth exactly at the limit is allowed");
        assert_eq!(
            S16Expr::try_product(four, base, &limits),
            Err(HypermemoryError::ExpressionDepthLimit { limit: 4 })
        );
    }

    #[test]
    fn size_limit_is_enforced() {
        let (store, ids) = store_with(&[(0, SedenionSimd::unit(1))]);
        let mut expr = S16Expr::atom(ids[0]);
        for _ in 0..8
        {
            expr = S16Expr::product(expr, S16Expr::atom(ids[0]));
        }
        // 8 products + 9 atoms = 17 nodes; cap at 5.
        let limits = ExprLimits::new(64, 5).unwrap();
        assert_eq!(
            expr.evaluate(&store, &limits),
            Err(HypermemoryError::ExpressionSizeLimit { limit: 5 })
        );
    }

    #[test]
    fn limits_reject_out_of_ceiling_values() {
        assert!(ExprLimits::new(0, 10).is_err());
        assert!(ExprLimits::new(10, 0).is_err());
        assert!(ExprLimits::new(MAX_SUPPORTED_DEPTH + 1, 10).is_err());
        assert!(ExprLimits::new(10, MAX_SUPPORTED_SIZE + 1).is_err());
        assert!(ExprLimits::new(10, 10).is_ok());
    }

    #[test]
    fn relation_record_carries_structure_code_and_digest() {
        let a = SedenionSimd::unit(1) + SedenionSimd::unit(10);
        let b = SedenionSimd::unit(4) - SedenionSimd::unit(15);
        let (store, ids) = store_with(&[(0, a), (1, b)]);
        let limits = ExprLimits::default();
        let expr = S16Expr::product(S16Expr::atom(ids[0]), S16Expr::atom(ids[1]));
        let rel = S16Relation::build(RelationId::new(7), expr.clone(), &store, &limits)
            .unwrap()
            .with_label(b"a-times-b".to_vec());
        assert_eq!(rel.id().value(), 7);
        assert_eq!(rel.expr(), &expr);
        assert_eq!(rel.digest(), &expr.digest());
        assert_eq!(rel.label(), Some(&b"a-times-b"[..]));
        // The zero-divisor product evaluates to exactly zero.
        assert_eq!(rel.code().to_array(), [0.0f32; 16]);
    }

    #[test]
    fn root_diagnostics_flag_zero_divisor_and_structure_is_recoverable() {
        let a = SedenionSimd::unit(1) + SedenionSimd::unit(10);
        let b = SedenionSimd::unit(4) - SedenionSimd::unit(15);
        let (store, ids) = store_with(&[(0, a), (1, b)]);
        let limits = ExprLimits::default();
        let expr = S16Expr::product(S16Expr::atom(ids[0]), S16Expr::atom(ids[1]));
        let (code, diag) = expr
            .evaluate_with_diagnostics(
                &store,
                &limits,
                crate::diagnostics::DEFAULT_NEAR_ZERO_THRESHOLD,
            )
            .unwrap();
        assert_eq!(code.to_array(), [0.0f32; 16], "product is exactly zero");
        let diag = diag.unwrap();
        assert_eq!(diag.lhs_norm_sqr(), 2.0);
        assert_eq!(diag.rhs_norm_sqr(), 2.0);
        assert!(diag.near_zero_divisor());
        assert!(diag.finite());
        // The structure is still fully recoverable from the stored expression —
        // we never tried to invert the (zero) product.
        match &expr
        {
            S16Expr::Product { left, right } =>
            {
                assert_eq!(**left, S16Expr::Atom(ids[0]));
                assert_eq!(**right, S16Expr::Atom(ids[1]));
            },
            _ => panic!("expected a product"),
        }
    }
}
