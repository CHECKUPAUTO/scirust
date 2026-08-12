# SciRust Elliptic Discovery — campaign orchestration v0.1

## Status

This document defines phase 8 before any modification of the production code.
It complements phases 0 to 7 without extending their domain: only the
integrated toy curves, generated locally on bounded prime fields, are
representable.

## Verifiable observation

The components needed for a campaign exist, but remain separate:

- `run_control` runs a mandatory check on a corpus;
- `run_search` returns the detailed candidate evaluations;
- `execute_local` produces a search receipt, but does not run the checks and
  only keeps the fingerprints and a summary;
- `attempt_justification` attempts an exact proof;
- `review_candidate` applies the human review boundary;
- `ReviewReport` produces a readable report for a candidate.

Thus, no single public entry point currently guarantees that a search, its
checks, its justifications and its pending reviews describe the same
execution. Phase 8 closes this orchestration gap. It adds neither new grammar,
nor new corpus, nor novelty declaration.

## Objective

Add a local boundary `execute_campaign(SearchPlan)` that produces a complete,
deterministic and replayable artifact containing:

1. the phase 6 search receipt;
2. the six mandatory checks in a canonical order;
3. all the detailed evaluations in generation order;
4. an exact justification attempt for each candidate;
5. an explicitly pending human review;
6. a stable Markdown report and a campaign SHA-256 fingerprint.

## Design decisions

### Single construction of corpora and results

A campaign constructs `ResearchCorpora` only once. The checks use
`ExhaustiveSmall`; the search uses the three existing partitions. The receipt
is built from the already computed corpora and evaluations, without silently
re-running the search.

The execution order is fixed:

1. generation of the corpora;
2. mandatory checks;
3. generation and falsification of the candidates;
4. construction of the receipt;
5. justification attempt;
6. creation of the `Pending` reviews;
7. canonical encoding and readable report.

### Mandatory checks and expectations

The canonical sequence and its expected results are:

| Order | Check | Expected status | Expected counterexample |
|---:|---|---|---|
| 0 | `TrueNegation` | `Known` | no |
| 1 | `FalseNegationKeepsY` | `Refuted` | yes |
| 2 | `FalseDoublingSign` | `Refuted` | yes |
| 3 | `JZeroClaimedUniversal` | `Refuted` | yes |
| 4 | `EncodingSignClaimedNovel` | `RepresentationArtifact` | no |
| 5 | `OverfitAZero` | `Refuted` | yes |

`CampaignRun::controls_valid` requires exactly this sequence. A campaign whose
check diverges remains inspectable, but cannot be presented as a valid
reference execution.

### Human review boundary

The campaign calls `attempt_justification` for each evaluation, then
`review_candidate` with `LiteratureReview::pending()`. It never fabricates a
reviewer, a source or a human decision.

Consequently, no automated campaign execution can attribute
`CandidateUnclassified`. A relation that requires a review remains
`NeedsLiteratureReview` until a separate and auditable human intervention.

### Canonical artifact

`CampaignRun::canonical_bytes` uses the domain:

    SCIRUST-ELLIPTIC-DISCOVERY/CAMPAIGN/V1

The encoding contains, in this order:

1. the schema version;
2. the canonical bytes of the execution receipt;
3. each check, its classification and its possible counterexample;
4. for each candidate, its evaluation fingerprint and the corresponding pending
   review report.

The lengths are explicit and all integers are in big-endian order via
`CanonicalEncoder`. The campaign fingerprint is the SHA-256 of these bytes.

Adding this public surface brings the crate to `0.3.0`. The V2 domains of the
plan, evaluations and receipt remain unchanged: they already encode the crate
version. The new artifact receives its own `CAMPAIGN/V1` domain.

The phase adds no external data decoder. The replay receives a `CampaignRun`
object already built by the local API and recomputes the campaign from the
validated `SearchPlan` it contains.

### Readable report

`CampaignReport` produces a deterministic Markdown including:

- fingerprints of the plan, the receipt and the campaign;
- validity and results of the checks;
- exact summary of the automated statuses;
- a section ordered by candidate with coverage, counterexample, catalogue,
  justification and review state;
- an explicit warning that no status constitutes a discovery.

The report is a view of the artifact, never a separate source of authority.

### Strict replay

`replay_campaign` recomputes `execute_campaign(expected.plan())`, compares the
complete canonical bytes and keeps the recomputed observation even in case of
divergence. A deletion, permutation or alteration of a result must therefore
be detected.

## Reuse and absence of duplication

- `ResearchCorpora` and `run_search` remain the only sources of the corpora and
  evaluations;
- the internal construction of the receipt is extracted from `execute_local`
  and reused;
- the canonical encoding of classifications and counterexamples is shared with
  `execution`;
- `ReviewReport` remains the detailed readable representation of a candidate;
- no dependency is added.

## Safety and reproducibility invariants

| Invariant | Phase 8 guarantee |
|---|---|
| Inputs | Only a validated `SearchPlan` is accepted. |
| Corpora | Only the three integrated toy corpora. |
| Checks | Six checks, fixed order and expectations. |
| Exactness | No float, epsilon or heuristic in the verdict. |
| Reproducibility | Same plan and same version give the same bytes. |
| Review | Always `Pending` during automated execution. |
| Non-novelty | No `New` or `Discovered` status; no invented review. |
| Rust purity | `unsafe` forbidden, no FFI, no network. |
| Resources | The bounds of `SearchPlan` and of phase 7 remain applicable. |

## Exit tests

The phase is finished when:

- two identical campaigns produce the same bytes and the same fingerprint;
- two distinct seeds produce distinct fingerprints;
- the six checks are present in order and valid;
- the campaign receipt is identical to the local receipt built on the same
  plan;
- each candidate has a justification and a `Pending` review;
- no automated campaign produces `CandidateUnclassified`;
- an alteration of the campaign is detected at replay;
- the report contains the checks, the coverage, the counterexamples, the
  justifications and the non-novelty warning;
- formatting, Clippy, tests, MSRV and workspace verifications pass.

## Out of scope

- CLI, file reading or writing and deserialization of artifacts;
- user-provided curves or external corpora;
- addresses, keys, SEC 1, network, RPC or blockchain;
- new mathematical grammar or new certificates;
- automatic literature review;
- discovery, novelty or out-of-toy-domain scope declaration.
