//! [`Scope`] — a theory's domain of validity.

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};
use sos_core::canonical::{Canonical, CanonicalEncoder};

/// The **domain of validity** of a theory: the region where it claims to hold.
///
/// A `Scope` is a conjunction of qualitative **predicates** — a theory holds
/// exactly where *all* of its predicates are satisfied. Fewer predicates ⇒ a
/// broader claim; the empty set is the **universal** scope ("everywhere, no
/// restriction"). This deterministic, set-based model captures what the Theory
/// Engine needs — narrowing a successor's domain and comparing rivals over their
/// overlapping domain (RFC-0002 §07.3) — without floating-point bounds. (A
/// quantitative fixed-point `Scope` with numeric intervals can extend this later,
/// per the same determinism discipline.)
///
/// # Region semantics
///
/// Read a scope as the *set of worlds* where its predicates all hold. Then:
/// * [`Scope::contains`] — `a.contains(b)` iff region(a) ⊇ region(b), i.e. `a`'s
///   predicates are a subset of `b`'s (`a` is the broader domain).
/// * [`Scope::overlap`] — the shared sub-domain (predicate union / conjunction).
/// * [`Scope::narrows`] — a strict narrowing (more predicates, smaller region).
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct Scope {
    predicates: BTreeSet<String>,
}

impl Scope {
    /// The universal scope: no predicates, holds everywhere.
    #[must_use]
    pub fn universal() -> Self {
        Self {
            predicates: BTreeSet::new(),
        }
    }

    /// A scope from a set of predicate labels (deduplicated and sorted).
    pub fn from_predicates<I, S>(predicates: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        Self {
            predicates: predicates.into_iter().map(Into::into).collect(),
        }
    }

    /// The predicates, in sorted order.
    pub fn predicates(&self) -> impl Iterator<Item = &str> {
        self.predicates.iter().map(String::as_str)
    }

    /// The number of predicates constraining this scope.
    #[must_use]
    pub fn predicate_count(&self) -> usize {
        self.predicates.len()
    }

    /// Whether this is the universal scope (holds everywhere).
    #[must_use]
    pub fn is_universal(&self) -> bool {
        self.predicates.is_empty()
    }

    /// Whether this scope's region **contains** `other`'s — every predicate of
    /// `self` also constrains `other`, so `self` is at least as broad. The
    /// universal scope contains every scope.
    #[must_use]
    pub fn contains(&self, other: &Scope) -> bool {
        self.predicates.is_subset(&other.predicates)
    }

    /// The shared sub-domain where **both** scopes hold: the conjunction of all
    /// their predicates (the union of the two predicate sets).
    #[must_use]
    pub fn overlap(&self, other: &Scope) -> Scope {
        Self {
            predicates: self.predicates.union(&other.predicates).cloned().collect(),
        }
    }

    /// Whether `self` is a **strict narrowing** of `other`: same-or-more
    /// constrained and not equal (a smaller region). This is the relationship a
    /// revised theory's domain bears to its parent's when it is confined to a
    /// limiting case (e.g. the low-velocity limit).
    #[must_use]
    pub fn narrows(&self, other: &Scope) -> bool {
        self != other && other.predicates.is_subset(&self.predicates)
    }
}

impl Canonical for Scope {
    fn encode(&self, enc: &mut CanonicalEncoder) {
        // `BTreeSet` iterates in sorted order, so this is canonical.
        let items: Vec<&String> = self.predicates.iter().collect();
        enc.seq(&items);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn universal_contains_everything() {
        let u = Scope::universal();
        let low_v = Scope::from_predicates(["low-velocity"]);
        assert!(u.is_universal());
        assert!(u.contains(&low_v));
        assert!(!low_v.contains(&u));
    }

    #[test]
    fn narrowing_and_overlap() {
        let low_v = Scope::from_predicates(["low-velocity"]);
        let low_v_weak = Scope::from_predicates(["low-velocity", "weak-field"]);
        assert!(low_v_weak.narrows(&low_v)); // more constrained
        assert!(!low_v.narrows(&low_v_weak));
        // Overlap is the conjunction of both.
        let ov = low_v.overlap(&Scope::from_predicates(["weak-field"]));
        assert_eq!(ov, low_v_weak);
    }

    #[test]
    fn canonical_is_order_independent() {
        let a = Scope::from_predicates(["b", "a"]);
        let b = Scope::from_predicates(["a", "b"]);
        assert_eq!(a.canonical_bytes(), b.canonical_bytes());
        assert_ne!(a.canonical_bytes(), Scope::universal().canonical_bytes());
    }
}
