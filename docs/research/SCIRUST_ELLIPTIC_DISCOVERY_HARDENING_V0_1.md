# SciRust Elliptic Discovery — bounded hardening v0.1

## Status

This document defines phase 7 before any modification of the production code.
It complements phases 0 to 6 without widening their scope: only toy curves,
generated locally in bounded prime fields, are representable.

## Verifiable motivation

The reviews of the previous phases noted four behaviors that must be treated as
experimental safety defects, and not as optional optimizations:

1. the falsification traversal continued past the declared budget;
2. the grammar could materialize a very large set before applying the candidate
   budget;
3. certain known identities depended on the arbitrary order of the syntax
   trees;
4. the \(j=0\) automorphism check used a scale that was not a cube root of
   unity in the first field encountered.

The validation of this phase also revealed an integration defect already
present on `master`: the canonical report used an unescaped opening brace in a
Rust format string. The phase restores `{{` and verifies that the JSON report
indeed begins with the expected literal brace.

The observations are traceable in the review comments of
[#918](https://github.com/Memorithm/scirust/pull/918#discussion_r3696552709),
[#919](https://github.com/Memorithm/scirust/pull/919#discussion_r3696555831)
and [#920](https://github.com/Memorithm/scirust/pull/920#discussion_r3696556768).

## Objective

Phase 7 makes the execution bounded in the operational sense: a declared limit
effectively bounds the calls to a relation and the working memory of the
generation. It neither claims to increase the mathematical scope of the engine,
nor to produce a discovery.

## Design decisions

### Bounded and observable falsification

A new bounded falsification primitive returns:

- the first possible canonical counterexample;
- the exact number of tuples evaluated;
- a result without counterexample when the limit is reached.

The traversal stops before calling the relation on a tuple located beyond the
limit. A G2, G3 or G6 gate only passes when all the required tuples have been
evaluated without counterexample. A limit reached before that coverage produces
`InsufficientCoverage`.

The historical exhaustive search API remains available and delegates to this
primitive with a maximal limit; it therefore does not change the meaning of the
existing reports.

### Budget-driven generation

`generate_relations` remains a local and deterministic generation, but its
working space is now determined before expansion: the number of expressions
needed for the point equalities is the smallest value \(n\) such that
\(n(n+1)/2\) covers the requested budget. The public bounds of the crate are
also applied to this function so that a direct call cannot bypass
`SearchPlan`.

The selection is a fixed canonical sequence and not a promise to materialize
the complete universe of the grammar. It interleaves, in this order:

1. point equalities;
2. infinity predicates;
3. equalities of the coefficient \(a\);
4. j-invariant equalities.

Thus, the four public variants of `Relation` are reachable in a bounded
search. The \(j\) constants explicitly include `0` and `1728`. The order and
the caps are part of the reproducible behavior and are encoded indirectly in
the phase 6 receipts.

Since the candidate sequence changes, phase 7 moves the canonical plan,
evaluation and receipt domains to `V2`, and the crate version to `0.2.0`.
There is no external receipt decoder: this domain separation is enough to
prevent fingerprint confusion between the two behaviors.

### Structural normalization of known laws

The negation signature recognizes both sides of an equality. The additive
inverse signature recognizes both orders of the summands. These rules do not
change the exact evaluation: they only prevent an already catalogued law from
being presented as an unrecognized relation because of the order in which the
tree is built.

### Conditional \(j=0\) check

The universal negative check is evaluated in \(\mathbb{F}_7\) with
\(\zeta=2\), because \(2^3 = 1 \pmod 7\) and \(2 \ne 1\). It ignores the other
fields and first checks the `a = 0` curves; its first counterexample must
therefore come from a curve of the same field with `a != 0`. The test
demonstrates that the check distinguishes the \(j=0\) condition from the mere
invalidity of a scale constant.

### Unchanged status boundary

`CandidateUnclassified` remains a final status of `review_candidate`, after an
attempted justification and a complete human review. Passing G6 can therefore
not attribute it alone: the automated state remains
`NeedsLiteratureReview`. This decision preserves the boundary established in
phase 5 between automated result and absence of conflict after independent
review.

## Invariants

| Invariant | Phase 7 guarantee |
|---|---|
| Inputs | No new public input of key, address, SEC 1 point or network. |
| Exactness | Only integers, prime fields and exact group operations. |
| Reproducibility | Fixed traversal and generation orders, without clock or hidden randomness. |
| Budget | No relation call beyond the declared limit. |
| Memory | The generation never builds a universe proportional to an unbounded depth. |
| Non-novelty | Recognized identities remain `Known`; unknown ones remain subject to review. |
| Pure Rust | No `unsafe`, FFI or new dependency. |

## Exit tests

The phase is finished when:

- a true predicate with a budget of one is called exactly once;
- an insufficiently covered gate actually stops at the budget;
- a request for maximal depth and scalar remains capped by the candidate
  budget;
- each variant of `Relation` appears in a sufficiently budgeted sequence;
- the inverted forms of double negation and of additive inverse are
  catalogued;
- the \(j=0\) check uses a non-trivial cube root and its counterexample has
  `p = 7` and `a != 0`;
- the tests, Clippy, formatting and MSRV verification applicable to the
  workspace pass in CI.

## Out of scope

- serialization or deserialization of external relations;
- arbitrary curves, secp256k1, Bitcoin addresses, public or private keys;
- blockchain, network, RPC, key files or secret recovery targets;
- novelty declaration, general proof or extension beyond the established local
  bounds.
