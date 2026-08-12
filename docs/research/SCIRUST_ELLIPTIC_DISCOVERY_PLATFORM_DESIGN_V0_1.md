# Elliptic discovery platform — design v0.1

## Status

- Type: design and audit document; no production code is added by this deliverable.
- Repository: Memorithm/scirust.
- Working branch: `research/elliptic-discovery-platform`.
- Audited remote base: `origin/master` at `3d6615c8e4784149d0b8b97e9f58631edf5e6f90` (2026-08-01).
- Initial state of this branch: clean after creation from `origin/master`.

This platform has as its sole purpose reproducible mathematical research on toy
elliptic curves, locally generated instances, and explicitly authorized research
sets. It must never accept, derive, search for, or compare a Bitcoin address,
a third-party public key, a real blockchain target, or any data standing in for them.

A tested relation is a hypothesis. It cannot be described as a new discovery
without passing the validation gates defined in "Falsification Protocol".

## Audit method and workspace conventions

The audit covered in particular:

- `scirust-hypercrypto` and its research documentation;
- `scirust-symbolic`;
- `scirust-neuro-symbolic`;
- `scirust-solvers`;
- `scirust-evo`;
- `scirust-core`;
- the neighbors directly needed for the decision: `scirust-modalg`, `scirust-sim`, and
  `scirust-algogen`.

The conventions observed at the root are: Rust edition 2021 for the majority of existing
crates, MSRV 1.89, Cargo resolver v2, CI toolchain `nightly-2026-07-02`, `rustfmt`
(maximum 100 columns), and Clippy with `-D warnings`. The reference commands are
documented in [CONTRIBUTING.md](../../CONTRIBUTING.md) and in
[.github/workflows/ci.yml](../../.github/workflows/ci.yml).

The future implementation must be pure Rust, without FFI, and its own crate must carry
`#![forbid(unsafe_code)]`. A dependency will only be added after verifying that no
existing abstraction covers the need.

## Initial Git state observed

| Check | Command executed | Result observed |
|---|---|---|
| Fetch of the remote branch | `git fetch origin --prune` | `origin/master` advanced from `25f272a` to `3d6615c`. |
| Isolated creation | `git worktree add -b research/elliptic-discovery-platform … origin/master` | Branch created at commit `3d6615c`, tracking `origin/master`. |
| State of the created worktree | `git status --short --branch` | `## research/elliptic-discovery-platform...origin/master`, with no modified file. |
| Initial diff consistency | `git diff --check` | Success, no output. |

Preexisting worktrees containing unrelated modifications were preserved. All
modifications of this work are limited to the isolated worktree above.

## Reading of scirust-hypercrypto

The crate [scirust-hypercrypto](../../scirust-hypercrypto/) is an experimental falsification
harness for a hypercomplex keyed permutation construction. Its README and
[its v0.1 specification](../../scirust-hypercrypto/docs/research/SCIRUST_HYPERCRYPTO_SPEC_V0_1.md)
explicitly delimit a research target, not a production cryptographic primitive.

Its reusable qualities are methodological, not structural:

- deterministic analysis, exact algebra, and negative controls;
- coverage explicitly annotated `Exhaustive` or `Sampled { count, seed }` in
  [analysis/util.rs](../../scirust-hypercrypto/src/analysis/util.rs);
- verdicts and control indicators in
  [analysis/battery.rs](../../scirust-hypercrypto/src/analysis/battery.rs);
- canonical reports and SHA-256 fingerprints in
  [analysis/report.rs](../../scirust-hypercrypto/src/analysis/report.rs);
- direct dependency on `scirust-modalg` for exact algebra.

Its `algebra.rs` facade re-exports modular word integers, quaternions, octonions, and
modular matrices specific to its domain. Adding elliptic curves to this facade
would mix two research objects without a shared invariant, would suggest an operational
cryptographic component, and would force an artificial inverted dependency.

## Architecture decision

**Decision: later create a new generic crate
`scirust-elliptic-discovery`; do not extend `scirust-hypercrypto`.**

| Option | Advantage | Decisive problem | Decision |
|---|---|---|---|
| Extend `scirust-hypercrypto` | Apparent reuse of falsification reports | Incompatible domain, API, and experimental cryptographic purpose; misleading coupling | Rejected |
| Extend `scirust-modalg` | Maximal reuse of the arithmetic | `modalg` is a generic algebraic library; placing experiment orchestration, the catalog, and the guardrails there would create a mixing of layers | Rejected |
| New `scirust-elliptic-discovery` | Explicit scope, safety, and reproducibility; minimal dependencies toward the existing foundations | New manifest and tests to maintain | Retained |

The new crate will be a thin consumer of `scirust-modalg`, not a replacement. It may
reuse the reporting ideas of `scirust-hypercrypto`, without importing its business API.

The proposed manifest, to be added only in phase 1, is:

    [package]
    name = "scirust-elliptic-discovery"
    version = "0.1.0"
    edition = "2021"
    publish = false
    rust-version = "1.89"

    [dependencies]
    scirust-modalg = { path = "../scirust-modalg" }
    scirust-sim = { path = "../scirust-sim", default-features = false }

No new external dependency is planned. `scirust-sim` would only be used for its
deterministic generator with explicit state; no floating-point draw would be part of an
algebraic computation or a validity decision.

## Inventory of reusable components

| Need | Existing component | Assessment and planned use |
|---|---|---|
| Exact multi-precision integers | `scirust-modalg::bigint::BigInt` | Arbitrary signed integer, exact operations, GCD, and decimal conversion. To be reserved for bounds or certificates that exceed `u64`. |
| Number theory and prime modulus | `scirust-modalg::numtheory` | Deterministic `is_prime` on `u64`, `pow_mod`, `mulmod`, `inv_mod`, factorization, and divisors. Foundation to reuse for \(\mathbb F_p\). |
| Finite fields and polynomials | `scirust-modalg::poly::Poly`, `extfield::ExtField` | Canonical polynomials over \(\mathrm{GF}(p)\), irreducibility test, extensions, and exact Frobenius. Reusable for an attempted symbolic justification; v0.1 of the curves stays on \(\mathbb F_p\). |
| Symbolic computation | `scirust-symbolic` | Expressions, differentiation, and simplification are present, but constants and evaluation are in `f64`. Suitable neither for the algebraic base nor for a proof. No direct link planned. |
| Neuro-symbolic reasoning | `scirust-neuro-symbolic` | CSP/SAT/Datalog/e-graph are available, but over integer or floating-point domains, hashed containers, and no finite-field certificates. Design inspiration only. |
| Solvers | `scirust-solvers` | Polynomial roots and unified interface mostly in `f64`. Unsuitable for exact equalities in \(\mathbb F_p\). No direct link planned. |
| Evolutionary search | `scirust-evo` | Deterministic routines under a seed, but genotypes and fitness in `f64`, with `rand`/Rayon. May inspire a separate exploratory phase, never validate a relation. |
| Reproducibility | `scirust-sim::SplitMix64` | Public pure-Rust generator, explicit seed, and reference vectors. Direct candidate, limited to non-algebraic sampling. |
| Fingerprints and replayability | `scirust-algogen` and `scirust-hypercrypto` | Canonical identity, campaign archive, and ordered reports are design precedents. Do not depend on `algogen` because its programs are floating-point. |
| Reproducible reductions | `scirust-core::reproducible` | Targets reproducible floating-point reductions; outside the need for the exact base. `scirust-core` also contains `unsafe` areas and FFI/BLAS backends; no direct link. |

The important observations are the following:

1. `scirust-modalg` already covers the useful exact primitives. Rewriting a modular
   inverse, a factorization, or a primality test would be risky.
2. No audited component today provides complete toy elliptic curve arithmetic over
   \(\mathbb F_p\), with enumeration, order, symmetry catalog, and certified falsification.
3. The audited symbolic, solver, and evolutionary components use floating point.
   They must not participate in establishing an equality, a counterexample, or a
   discovery status.
4. `scirust-core` is too broad a dependency for this subsystem: its inventory includes
   `unsafe` paths, system calls, and optional backends. That would violate the
   pure-Rust/no-FFI scope of this crate.

## Duplication risks and countermeasures

| Risk | Mandatory countermeasure |
|---|---|
| Reimplementing \(\mathbb F_p\) over naive operations and diverging from `modalg` | A thin type facade can delegate operations to `numtheory`; cross-tests over the toy domain. |
| Creating a second homegrown PRNG | Reuse `scirust-sim::SplitMix64` or document why an existing API is insufficient. Seed, algorithm, and version are part of the report. |
| Duplicating HyperCrypto's reports without canonical order | Adopt the principle: ordered structures, explicit-length encoding, fingerprint of the corpus and of the research program. |
| Confusing pattern search with proof | Strictly separate candidate generator, falsifier, classifier, and proof attempt. |
| Implicitly extending the domain to real keys | Dedicated types `ToyPrime`, `ToyCurve`, and `LocalResearchCase`; no SEC 1 decoding, address, public key, or RPC API. |
| Depending on `scirust-core`, `symbolic`, `solvers`, or `evo` to speed up v0.1 | Forbidden until a safety, correctness, and MSRV audit justifies a discrete and exact interface. |

## Minimal algebraic model

The initial phase is limited to short Weierstrass curves over a toy prime field:

\[
E_{a,b}/\mathbb F_p : y^2 = x^3 + ax + b,
\qquad p \text{ premier impair},
\qquad 4a^3 + 27b^2 \not\equiv 0 \pmod p.
\]

The v0.1 domain is bounded to \(5 \le p \le 4093\). This limit makes exact enumeration
practicable, does not claim to represent a production curve, and must not be raised
without new cost and protocol analysis.

The future public types must be intentionally narrow:

- `ToyPrime`: verified, odd prime within the research bound;
- `Fp`: canonical residue \([0,p-1]\), without floating point;
- `ToyCurve`: parameters \((p,a,b)\) with nonzero discriminant;
- `ToyPoint`: point at infinity or coordinates validated on **its own** `ToyCurve`;
- `LocalResearchCase`: seed, domain, local source, and explicit authorization;
- `ExperimentId`: canonical fingerprint of the manifest, seed, and corpus.

None of these types must implement a public key parser, a SEC 1 deserialization,
an address import, a chain URL, a network client, or an implicit conversion from external
bytes.

### Exact arithmetic and enumeration

1. Verify `p` with the deterministic test of `scirust-modalg`.
2. Reduce `a` and `b` canonically, then reject a zero discriminant.
3. Build an ordered table \(r \mapsto [y]\) of all \(y^2 \bmod p\).
4. Walk over increasing \(x\), compute \(x^3+ax+b\), and emit the solutions in
   \((x,y)\) order, preceded by the point at infinity.
5. Set \(\#E(\mathbb F_p)\) equal to the number of points thus enumerated. The Hasse bound is a
   consistency check, never a substitute for enumeration.
6. Compute the order of a point \(P\) starting from \(\#E\), factoring that number with
   `scirust-modalg`, then dividing by each factor only if
   \((\#E/q)P=\mathcal O\).

Addition, doubling, the inverse, and the special cases (\(\mathcal O\), opposites,
vertical tangent) will be tested against this exhaustive enumeration. Denominator
inverses are provided by the existing modular arithmetic; no real computation, no
tolerance, and no discrete logarithm are needed.

## Proposed architecture of the future crate

    scirust-elliptic-discovery/
      src/
        lib.rs              # surface publique, forbid unsafe
        scope.rs            # garde-fous LocalResearchCase
        field.rs            # ToyPrime et Fp, façade exacte sur modalg
        curve.rs            # ToyCurve, ToyPoint, lois de groupe
        enumerate.rs        # liste exacte et ordre canonique
        orders.rs           # ordres et certificats de division
        invariant.rs        # invariants exacts observables
        grammar.rs          # langage fini de relations candidates
        catalog.rs          # propriétés connues et signatures
        classify.rs         # Known / Artifact / Refuted / Candidate
        falsify.rs          # recherche ordonnée du premier contre-exemple
        proof.rs            # tentative symbolique exacte et certificat
        experiment.rs       # graines, corpus, partitions, manifeste
        canonical.rs        # sérialisation canonique et empreintes
        report.rs           # rapport stable et relecture
      tests/
        field_and_curve.rs
        exhaustive_small.rs
        known_catalog.rs
        counterexamples.rs
        reproducibility.rs

The modules are oriented in a single direction: `scope` and `field` ground `curve`;
`curve` grounds `enumerate` and `orders`; candidates only access immutable
invariants; the falsifier and the classifier produce a report but modify neither the
corpus nor the catalog.

## Research language and classification

A candidate is a typed, bounded, and entirely exact expression over points and scalars.
The initial kernel includes: identity, negation, addition, doubling, scalar multiplication
by a bounded integer, valid coordinates, \(j\), discriminant, point order, and group
cardinality. Partial operators return `Undefined` rather than inventing a value.

Each result carries one of the following statuses:

| Status | Meaning |
|---|---|
| `Refuted` | A canonical counterexample is archived. |
| `Known` | Matches a catalog property, with reference. |
| `RepresentationArtifact` | Disappears after representation normalization or coordinate/encoding change. |
| `NeedsLiteratureReview` | Persistent pattern but outside a sufficiently verified catalog. |
| `Inconclusive` | Insufficient coverage or non-determinable result. |
| `CandidateUnclassified` | Passed all automatic gates, but is **not** a new discovery. |

There is deliberately no `New` or `Discovered` status. Any eventual human
conclusion requires a literature review and an independent proof or justification.

## Initial catalog of known properties

The engine must recognize and exclude at least the following families before classifying a
pattern as an unclassified candidate.

| Known family | Exact or testable signature | Classification |
|---|---|---|
| Negation and identity | \(-(x,y)=(x,-y)\), \(P+(-P)=\mathcal O\), \(-\mathcal O=\mathcal O\) | `Known` |
| Group linearity | \(m(nP)=(mn)P\), associativity, order divisibility | `Known` |
| \(j=0\) automorphisms | For \(E:y^2=x^3+b\), \((x,y)\mapsto(\zeta x,y)\) if \(\zeta^3=1\) | `Known` |
| Cube roots of unity | \(\zeta^3=1\), \(\zeta\ne1\), only when the field contains the nontrivial root | `Known` or conditional |
| Nearby \(j=1728\) case | Additional automorphisms of \(y^2=x^3+ax\) when the field contains the required constants | `Known` |
| GLV-type endomorphisms | Relation \(\phi(P)=[\lambda]P\) arising from a known endomorphism and its minimal polynomial | `Known`; never presented as new |
| Coordinate changes | Isomorphism \(x=u^2x', y=u^3y'\) and the corresponding transformation of \(a,b\) | `RepresentationArtifact` or `Known` |
| Twists and \(j\) classes | Same invariant \(j\) without automatic point-group identity | `Known`; avoid abusive cross-cutting conclusions |
| Encoding symmetries | Choice of sign of \(y\), point at infinity, order, or coordinate form | `RepresentationArtifact` |
| Sub-corpus artifacts | Pattern true only for \(j=0\), a congruence of \(p\), or a fixed order | `RepresentationArtifact` or `Refuted` after independent partition |

The \(j=0\) automorphism is verified directly:
\((\zeta x)^3+b=\zeta^3x^3+b=x^3+b\).
The existence of nontrivial cube roots, the exceptional classes \(j=0\) and
\(j=1728\), and the twists must be marked as corpus conditions, not as
discovered exceptions. The base references are [Silverman, Arithmetic of Elliptic Curves](https://www.math.brown.edu/johsilve/AECHome.html)
and the documentation of [SageMath elliptic curves over finite fields](https://doc.sagemath.org/html/en/reference/arithmetic_curves/sage/schemes/elliptic_curves/ell_finite_field.html).

GLV endomorphisms are a known scalar multiplication acceleration technique, not
a signal of novel structure; see [Gallant, Lambert, and Vanstone, CRYPTO 2001](https://link.springer.com/chapter/10.1007/3-540-44647-8_11).
Compressed representations must be treated as encodings: SEC 1 describes
point representation formats, not a new algebraic symmetry
([SEC 1 v2](https://www.secg.org/sec1-v2.pdf)). v0.1 will nevertheless not implement any of these
formats.

## Falsification protocol

A relation crosses the following gates only in this order:

1. **G0 — Authorized domain.** Every case is `LocalResearchCase`, labeled toy and generated
   locally; any other input type is impossible to represent in the API.
2. **G1 — Exact base.** Evaluation uses only integers, \(\mathbb F_p\), and the verified group
   laws; no floating-point value or heuristic decides the verdict.
3. **G2 — Small exhaustiveness.** Test all nonsingular curves and all necessary
   points on the exhaustive corpus defined below.
4. **G3 — Independent set.** Test a separate corpus, determined by a distinct seed and manifest;
   never choose the sample after seeing the candidate.
5. **G4 — Counterexample.** Enumerate the inputs in a canonical order and archive the first
   counterexample \((p,a,b,\text{tuple de points},\text{expression})\), if one exists.
6. **G5 — Catalog.** Compare the relation and its normalizations against the catalog above,
   notably under negation, isomorphisms, and encodings.
7. **G6 — Scale-up and justification.** Pass the scale defined below, then attempt
   an exact polynomial identity, a finite computation certificate, or a symbolic
   justification. A failure keeps the `CandidateUnclassified` or `Inconclusive` status.

The mandatory negative controls include: a deliberately false negation formula,
a doubling with the wrong sign, a property valid only at \(j=0\) wrongly presented as
universal, an encoding sign symmetry, and an expression overfitted to the training
corpus. They must be refuted by the falsifier.

The results of G2 and G3 are distinct. A property that passes G2 but fails G3 is
`Refuted`; a property that does not receive sufficient coverage is `Inconclusive`.
No experimental success, even exhaustive over the toy bound, amounts to a generalization
outside its domain.

## Deterministic corpora

| Set | Construction | Purpose |
|---|---|---|
| `ExhaustiveSmall` | All nonsingular \((a,b)\) for \(p\in\{5,7,11,13\}\), all points and all tuples within the declared budget | Complete falsification at small size |
| `IndependentHoldout` | Primes \(17\) to \(97\), deterministic seed-based partitions, local curves and exactly enumerated points | Independent validation |
| `ScaleLadder` | \(p\in\{127,251,509,1021,2039,4093\}\), all curves selected by manifest, exact point enumeration; high-arity tuples under explicitly counted coverage | Scale-up study |

The report must contain, for each set: selection algorithm, seed, crate version,
bounds, number of curves examined, number of points/tuples evaluated, traversal order,
canonical fingerprint, and verdict. Any future parallelization must produce the same first
counterexample and the same report as a sequential execution.

## Safety, correctness, and reproducibility invariants

| Invariant | Verifiable requirement |
|---|---|
| Authorized research only | No API accepts addresses, public keys, SEC 1 encodings, network endpoints, or blockchain data. |
| Local instances | All curves come from validated toy parameters or a local seed-based generator. |
| Exactness | No `f32`, `f64`, epsilon, numerical root, implicitly ordered hash map, or hidden draw on the verdict path. |
| Reproducibility | Seed, version, manifest, bounds, canonical order, and fingerprint are present in every report. |
| Determinism | Output containers are ordered; the first counterexample is defined by lexicographic order. |
| Rust purity | `#![forbid(unsafe_code)]`, no FFI, no BLAS backend, no network I/O. |
| Transparency | Assumptions, negative controls, coverage limits, and proof failures are reported, never suppressed. |
| Non-attribution | Statuses do not include "new"; the known catalog is consulted before any candidate. |

## Roadmap

### Phase 0 — Audit and research contract

Finish this document, keep the repository without production changes, have the scope
and the catalog reviewed. Exit criterion: agreement on the forbidden input types and the G0–G6 gates.

### Phase 1 — Minimal exact kernel

Create the crate, add it to the workspace, implement `ToyPrime`, `Fp`, `ToyCurve`,
`ToyPoint`, enumeration, and orders, with cross-references against `scirust-modalg`.
Exit criterion: group laws and the `ExhaustiveSmall` corpus pass without floating point.

### Phase 2 — Experimental harness

Add canonical manifest, local corpus, `SplitMix64` with explicit seed, stable report,
fingerprint, and first counterexample. Exit criterion: two identical executions produce
identical bytes.

### Phase 3 — Catalog and controls

Implement the recognition rules for negation, \(j=0\), cube roots, coordinate
changes, encoding artifacts, and known GLV. Add the negative controls. Exit criterion:
each control is classified correctly.

### Phase 4 — Candidate generation and falsification

Add a typed finite grammar, exhaustive search with a fixed budget, the training/validation
partitions, and the scale-up. Exit criterion: no candidate can
avoid G2–G6.

### Phase 5 — Justification and review

Add the exact certificates/identities feasible with `modalg::poly`, a readable report
export, and a human literature review procedure. Exit criterion:
reports formally separate proof, counterexample, known property, and hypothesis.

### Phase 6 — Local execution and replay

Add an execution boundary over `SearchPlan`, a complete canonical receipt, and a strict replay
that detects any divergence. No external data decoder is added. Exit
criterion: the same plan and version produce the same bytes, and any tampering with the receipt
is detected. The detailed specification is
[`SCIRUST_ELLIPTIC_DISCOVERY_EXECUTION_REPLAY_V0_1.md`](SCIRUST_ELLIPTIC_DISCOVERY_EXECUTION_REPLAY_V0_1.md).

### Phase 7 — Hardening of bounds and controls

Operationally enforce the falsification budget, bound the generation memory
by the candidate budget, normalize the known syntactic symmetries, and exercise the conditional
\(j=0\) control with a valid cube root. Exit criterion: no relation is
evaluated beyond its budget, the four relation variants are reachable, and the known
laws do not depend on the construction order of their tree. The detailed specification
is [`SCIRUST_ELLIPTIC_DISCOVERY_HARDENING_V0_1.md`](SCIRUST_ELLIPTIC_DISCOVERY_HARDENING_V0_1.md).

### Phase 8 — Local campaign orchestration

Connect the mandatory controls, the bounded search, the receipt, the justification
attempts, and the pending human reviews in a single canonical artifact.
Add a stable Markdown report and a strict replay of the complete campaign.
Exit criterion: a campaign executes the six controls in
order, keeps all evaluations, and detects any divergence without
ever fabricating a review decision or a novelty status. The
detailed specification is
[`SCIRUST_ELLIPTIC_DISCOVERY_CAMPAIGN_ORCHESTRATION_V0_1.md`](SCIRUST_ELLIPTIC_DISCOVERY_CAMPAIGN_ORCHESTRATION_V0_1.md).

## Validations to apply

After adding the crate, run at minimum:

    cargo +nightly-2026-07-02 fmt --all -- --check
    cargo +nightly-2026-07-02 clippy -p scirust-elliptic-discovery --all-targets --locked -- -D warnings
    cargo +nightly-2026-07-02 test -p scirust-elliptic-discovery --locked
    cargo +1.89.0 check -p scirust-elliptic-discovery --locked
    git diff --check

For the present documentary deliverable, the applicable validation is `git diff --check` and an
inspection of the diff. In the current audit environment, `cargo` and `rustup` are not
installed in `PATH`; therefore, no Cargo validation may be presented as executed or
successful.

## References

- [Workspace contribution documentation](../../CONTRIBUTING.md)
- [Workspace CI](../../.github/workflows/ci.yml)
- [HyperCrypto v0.1 specification](../../scirust-hypercrypto/docs/research/SCIRUST_HYPERCRYPTO_SPEC_V0_1.md)
- [HyperCrypto phase 1 falsification report](../../scirust-hypercrypto/docs/research/SCIRUST_HYPERCRYPTO_FALSIFICATION_PHASE1.md)
- [J. H. Silverman, The Arithmetic of Elliptic Curves](https://www.math.brown.edu/johsilve/AECHome.html)
- [R. Gallant, R. Lambert and S. Vanstone, Faster Point Multiplication on Elliptic Curves with Efficient Endomorphisms](https://link.springer.com/chapter/10.1007/3-540-44647-8_11)
- [Standards for Efficient Cryptography, SEC 1 v2](https://www.secg.org/sec1-v2.pdf)
- [SageMath — Elliptic curves over finite fields](https://doc.sagemath.org/html/en/reference/arithmetic_curves/sage/schemes/elliptic_curves/ell_finite_field.html)
