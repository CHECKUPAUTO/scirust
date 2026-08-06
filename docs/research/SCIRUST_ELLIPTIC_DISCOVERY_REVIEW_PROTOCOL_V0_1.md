# Elliptic discovery review protocol v0.1

## Purpose

This procedure governs human review of relations emitted by
`scirust-elliptic-discovery`. It applies only to locally generated toy curves.
An automated result is never a novelty claim.

## Required evidence

The reviewer must receive one deterministic report containing:

1. the manifest and corpus fingerprints;
2. complete `ExhaustiveSmall` coverage;
3. the separately seeded `IndependentHoldout` result;
4. the first canonical counterexample, when one exists;
5. the known-property catalog comparison;
6. the `ScaleLadder` result for any unknown surviving relation;
7. the exact proof certificate or the recorded reason no certificate was produced.

Missing or partial coverage forces `Inconclusive`. A counterexample forces `Refuted`.
Catalog matches force `Known` or `RepresentationArtifact`.

## Independent literature review

The human reviewer must search by mathematical structure rather than by generated
identifier. At minimum, compare the relation with:

- negation, identity and scalar-composition laws;
- exceptional automorphism families at `j = 0` and `j = 1728`;
- roots of unity available in the base field;
- twists and coordinate isomorphisms;
- GLV-type endomorphisms and their minimal polynomials;
- point-representation and sign conventions.

The report must name the reviewer and record an ordered, nonempty source list.
The code rejects a completed review without both fields.

Baseline references:

- J. H. Silverman, *The Arithmetic of Elliptic Curves*;
- R. Gallant, R. Lambert and S. Vanstone, *Faster Point Multiplication on
  Elliptic Curves with Efficient Endomorphisms*, CRYPTO 2001;
- Standards for Efficient Cryptography, SEC 1 v2.

## Decisions

- `Known`: the literature or executable catalog identifies the property.
- `RepresentationArtifact`: normalization removes the apparent relation.
- `Refuted`: an exact counterexample exists.
- `Inconclusive`: coverage or justification is insufficient.
- `NeedsLiteratureReview`: automated gates passed but human review is pending.
- `CandidateUnclassified`: gates passed, a proof attempt was recorded and an
  independent reviewer found no catalog conflict in the recorded sources.

`CandidateUnclassified` means only that the relation remains a hypothesis. It
does not mean new, novel, proved beyond the stated certificate, or valid outside
the declared toy domain.

## Prohibited scope expansion

The review must reject any report derived from a Bitcoin address, third-party
public key, SEC 1 input, blockchain endpoint, real wallet target or externally
supplied curve instance. Such inputs are outside the representable API of the
crate and outside this protocol.
