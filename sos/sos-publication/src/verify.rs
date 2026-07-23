//! The verification pipeline: structural validation, dependency resolution and
//! closure, per-claim assessment under a [`SupportPolicy`], the reproducibility
//! summary, the executable-exhibit drift check, and the release-consistency
//! check.
//!
//! Verification answers, mechanically, the questions a reader of a published
//! result should be able to ask: *which object supports this claim? is anything
//! contradicted? does every dependency resolve? is the declared scope complete?
//! is the reproducibility bar met? can the figures still be regenerated? has the
//! document changed since it was released?* Every answer is **report data** — a
//! [`VerificationReport`], a [`ExhibitReport`], a [`ReleaseConsistency`] — never
//! a thrown error, so a contradiction is surfaced, never hidden.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use sos_core::{DeterminismLevel, Object, ObjectId};

use crate::claim::Claim;
use crate::error::Result;
use crate::exhibit::RegenPolicy;
use crate::key::{ClaimKey, FigureKey, RefKey, SectionId, TableKey};
use crate::policy::{BindingContext, ClaimContext, PolicyId, SupportPolicy};
use crate::publication::Publication;
use crate::release::ReleaseManifest;
use crate::section::Block;
use crate::source::{PublicationObjectSource, dependency_closure};

/// The standing of a claim after assessment. Categorical, never a hidden score
/// (Invariant VI); the exact conditions that produce each are documented on
/// [`StandardPolicy`](crate::policy::StandardPolicy).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum ClaimStatus {
    /// Backed by resolved, in-scope **direct** evidence, meeting the repro bar.
    Supported,
    /// Backed only by **indirect** in-scope evidence (corroboration / supplies-*).
    PartiallySupported,
    /// A resolved object **contradicts** it. Never hidden; dominates support.
    Contradicted,
    /// Bindings exist but none is effective (support is out of the declared
    /// scope, or the edges are only context).
    Unresolved,
    /// No bindings at all — nothing to verify against the graph.
    Unverifiable,
    /// The claim's key is duplicated (ambiguous handle).
    StructurallyInvalid,
    /// A bound object is absent from the graph.
    DependencyMissing,
    /// In-scope support exists but falls below the required determinism level.
    ReproducibilityFailed,
    /// The policy rejected the claim's wiring (e.g. an object bound as both
    /// support and contradiction).
    PolicyRejected,
}

impl ClaimStatus {
    /// A short, stable code.
    #[must_use]
    pub const fn code(self) -> &'static str {
        match self
        {
            Self::Supported => "supported",
            Self::PartiallySupported => "partially-supported",
            Self::Contradicted => "contradicted",
            Self::Unresolved => "unresolved",
            Self::Unverifiable => "unverifiable",
            Self::StructurallyInvalid => "structurally-invalid",
            Self::DependencyMissing => "dependency-missing",
            Self::ReproducibilityFailed => "reproducibility-failed",
            Self::PolicyRejected => "policy-rejected",
        }
    }

    /// Whether the claim stands well enough to publish (fully or partially
    /// supported). Every other status is a problem to resolve before release.
    #[must_use]
    pub const fn is_publishable(self) -> bool {
        matches!(self, Self::Supported | Self::PartiallySupported)
    }
}

impl core::fmt::Display for ClaimStatus {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(self.code())
    }
}

/// A structural defect in a publication's *form* (as opposed to a claim's
/// standing). Reported so "is it structurally complete?" has a concrete answer.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum StructuralIssue {
    /// The publication has no claims.
    NoClaims,
    /// The publication declares no root objects, so its scope is undefined.
    EmptyDeclaredRoots,
    /// Two claims share a key.
    DuplicateClaimKey(ClaimKey),
    /// Two figures share a key.
    DuplicateFigureKey(FigureKey),
    /// Two tables share a key.
    DuplicateTableKey(TableKey),
    /// Two bibliography entries share a key.
    DuplicateReferenceKey(RefKey),
    /// A claim has no bindings — it asserts something with no edges into the graph.
    ClaimWithoutBindings(ClaimKey),
    /// A regenerable figure declares no source objects.
    FigureWithoutSources(FigureKey),
    /// A regenerable table declares no source objects.
    TableWithoutSources(TableKey),
    /// A section references a claim key that is not in the registry.
    UnknownClaimReference {
        /// The referencing section.
        section: SectionId,
        /// The dangling key.
        key: ClaimKey,
    },
    /// A section references a figure key that is not in the registry.
    UnknownFigureReference {
        /// The referencing section.
        section: SectionId,
        /// The dangling key.
        key: FigureKey,
    },
    /// A section references a table key that is not in the registry.
    UnknownTableReference {
        /// The referencing section.
        section: SectionId,
        /// The dangling key.
        key: TableKey,
    },
    /// A section references a bibliography key that is not in the registry.
    UnknownCitationReference {
        /// The referencing section.
        section: SectionId,
        /// The dangling key.
        key: RefKey,
    },
}

/// The per-claim result: its status plus the objects behind the decision.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ClaimAssessment {
    /// The claim's handle.
    pub key: ClaimKey,
    /// The claim's content address (independent of this publication).
    pub content_id: ObjectId,
    /// The decided status.
    pub status: ClaimStatus,
    /// Resolved, in-scope supporting objects.
    pub supporting: Vec<ObjectId>,
    /// Resolved contradicting objects.
    pub contradicting: Vec<ObjectId>,
    /// Bound objects absent from the source.
    pub missing: Vec<ObjectId>,
    /// Resolved supporting objects that lie **outside** the declared scope.
    pub out_of_scope: Vec<ObjectId>,
}

/// The reproducibility summary over all in-scope supporting evidence.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum ReproVerdict {
    /// The publication declared no reproducibility bar.
    NotRequired,
    /// The weakest in-scope supporting level meets the required minimum.
    Satisfied {
        /// The required minimum level.
        required: DeterminismLevel,
        /// The weakest realized supporting level (`L3` if there is no evidence).
        realized: DeterminismLevel,
    },
    /// Some in-scope supporting evidence falls below the required minimum.
    Insufficient {
        /// The required minimum level.
        required: DeterminismLevel,
        /// The weakest realized supporting level.
        realized: DeterminismLevel,
    },
}

impl ReproVerdict {
    /// Whether the bar is met (or none was set).
    #[must_use]
    pub const fn is_met(self) -> bool {
        matches!(self, Self::NotRequired | Self::Satisfied { .. })
    }
}

/// The full report of verifying a publication against an object source.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VerificationReport {
    /// The policy the claims were judged under.
    pub policy: PolicyId,
    /// Structural defects in the document's form (empty ⇒ structurally complete).
    pub structural: Vec<StructuralIssue>,
    /// Per-claim assessments, in registry order.
    pub claims: Vec<ClaimAssessment>,
    /// Every bound object that did not resolve, sorted and deduplicated.
    pub missing_objects: Vec<ObjectId>,
    /// The size of the declared dependency closure.
    pub closure_size: usize,
    /// The reproducibility summary.
    pub reproducibility: ReproVerdict,
}

impl VerificationReport {
    /// Whether the document has no structural defects.
    #[must_use]
    pub fn is_structurally_complete(&self) -> bool {
        self.structural.is_empty()
    }

    /// The claims that are contradicted.
    #[must_use]
    pub fn contradicted(&self) -> Vec<&ClaimAssessment> {
        self.claims
            .iter()
            .filter(|c| c.status == ClaimStatus::Contradicted)
            .collect()
    }

    /// The claims that are neither fully nor partially supported — the ones a
    /// referee would flag.
    #[must_use]
    pub fn problems(&self) -> Vec<&ClaimAssessment> {
        self.claims
            .iter()
            .filter(|c| !c.status.is_publishable())
            .collect()
    }

    /// Whether the publication is fit to release: structurally complete, no
    /// missing dependencies, every claim at least partially supported, and the
    /// reproducibility bar met. This is a *necessary* bar, not a claim of truth.
    #[must_use]
    pub fn is_publishable(&self) -> bool {
        self.structural.is_empty()
            && self.missing_objects.is_empty()
            && self.reproducibility.is_met()
            && self.claims.iter().all(|c| c.status.is_publishable())
    }
}

/// First-duplicate-encounter list of items that appear more than once.
fn duplicates<T: Ord + Clone>(items: impl Iterator<Item = T>) -> Vec<T> {
    let mut seen: BTreeSet<T> = BTreeSet::new();
    let mut reported: BTreeSet<T> = BTreeSet::new();
    let mut out = Vec::new();
    for item in items
    {
        if !seen.insert(item.clone()) && reported.insert(item.clone())
        {
            out.push(item);
        }
    }
    out
}

/// Collect the structural defects of a publication's form.
#[must_use]
pub fn structural_issues(publication: &Publication) -> Vec<StructuralIssue> {
    let mut issues = Vec::new();

    if publication.claims.is_empty()
    {
        issues.push(StructuralIssue::NoClaims);
    }
    if publication.declared_roots.is_empty()
    {
        issues.push(StructuralIssue::EmptyDeclaredRoots);
    }

    for key in duplicates(publication.claims.iter().map(|c| c.key.clone()))
    {
        issues.push(StructuralIssue::DuplicateClaimKey(key));
    }
    for claim in &publication.claims
    {
        if claim.bindings.is_empty()
        {
            issues.push(StructuralIssue::ClaimWithoutBindings(claim.key.clone()));
        }
    }

    for key in duplicates(publication.figures.iter().map(|f| f.key.clone()))
    {
        issues.push(StructuralIssue::DuplicateFigureKey(key));
    }
    for figure in &publication.figures
    {
        if figure.sources.is_empty() && figure.regeneration != RegenPolicy::StaticAsset
        {
            issues.push(StructuralIssue::FigureWithoutSources(figure.key.clone()));
        }
    }

    for key in duplicates(publication.tables.iter().map(|t| t.key.clone()))
    {
        issues.push(StructuralIssue::DuplicateTableKey(key));
    }
    for table in &publication.tables
    {
        if table.sources.is_empty() && table.regeneration != RegenPolicy::StaticAsset
        {
            issues.push(StructuralIssue::TableWithoutSources(table.key.clone()));
        }
    }

    for key in duplicates(publication.bibliography.iter().map(|r| r.key().clone()))
    {
        issues.push(StructuralIssue::DuplicateReferenceKey(key));
    }

    // Dangling references from section blocks into the registries.
    let claim_keys: BTreeSet<&ClaimKey> = publication.claims.iter().map(|c| &c.key).collect();
    let figure_keys: BTreeSet<&FigureKey> = publication.figures.iter().map(|f| &f.key).collect();
    let table_keys: BTreeSet<&TableKey> = publication.tables.iter().map(|t| &t.key).collect();
    let ref_keys: BTreeSet<&RefKey> = publication.bibliography.iter().map(|r| r.key()).collect();
    for section in &publication.sections
    {
        for block in &section.blocks
        {
            match block
            {
                Block::Claim(key) if !claim_keys.contains(key) =>
                {
                    issues.push(StructuralIssue::UnknownClaimReference {
                        section: section.id.clone(),
                        key: key.clone(),
                    });
                },
                Block::Figure(key) if !figure_keys.contains(key) =>
                {
                    issues.push(StructuralIssue::UnknownFigureReference {
                        section: section.id.clone(),
                        key: key.clone(),
                    });
                },
                Block::Table(key) if !table_keys.contains(key) =>
                {
                    issues.push(StructuralIssue::UnknownTableReference {
                        section: section.id.clone(),
                        key: key.clone(),
                    });
                },
                Block::Cite(key) if !ref_keys.contains(key) =>
                {
                    issues.push(StructuralIssue::UnknownCitationReference {
                        section: section.id.clone(),
                        key: key.clone(),
                    });
                },
                _ =>
                {},
            }
        }
    }

    issues
}

/// Verify `publication` against `source` under `policy`.
///
/// Resolves every bound object, takes the dependency closure of the declared
/// roots, and assesses each claim. The returned [`VerificationReport`] is total:
/// scientific problems are recorded, not raised. The only `Err` is an
/// operational fault reading the source.
///
/// # Errors
/// [`PublicationError::Source`](crate::error::PublicationError::Source) if the
/// object source faults or a stored header cannot be decoded.
pub fn verify<S, P>(publication: &Publication, source: &S, policy: &P) -> Result<VerificationReport>
where
    S: PublicationObjectSource + ?Sized,
    P: SupportPolicy,
{
    let structural = structural_issues(publication);
    let closure = dependency_closure(source, &publication.declared_roots)?;
    let duplicate_claim_keys: BTreeSet<ClaimKey> =
        duplicates(publication.claims.iter().map(|c| c.key.clone()))
            .into_iter()
            .collect();

    let mut claims = Vec::with_capacity(publication.claims.len());
    let mut missing_global: BTreeSet<ObjectId> = BTreeSet::new();
    let mut effective_levels: Vec<DeterminismLevel> = Vec::new();

    for claim in &publication.claims
    {
        let bindings = resolve_bindings(claim, source, &closure)?;
        let ctx = ClaimContext {
            claim,
            bindings,
            requirement: publication.reproducibility,
            duplicate_key: duplicate_claim_keys.contains(&claim.key),
        };
        let status = policy.assess(&ctx);

        let bindings = &ctx.bindings;
        let supporting = collect(bindings, |b| b.present && b.in_scope && b.role.is_support());
        let out_of_scope = collect(bindings, |b| {
            b.present && !b.in_scope && b.role.is_support()
        });
        let contradicting = collect(bindings, |b| b.present && b.role.is_contradiction());
        let missing = collect(bindings, |b| !b.present);

        for object in &missing
        {
            missing_global.insert(*object);
        }
        for binding in bindings
        {
            if binding.is_effective_support()
            {
                if let Some(level) = binding.level
                {
                    effective_levels.push(level);
                }
            }
        }

        claims.push(ClaimAssessment {
            key: claim.key.clone(),
            content_id: claim.content_id(),
            status,
            supporting,
            contradicting,
            missing,
            out_of_scope,
        });
    }

    let realized = DeterminismLevel::min_over(effective_levels.iter().copied());
    let reproducibility = match publication.reproducibility.minimum()
    {
        None => ReproVerdict::NotRequired,
        Some(required) if realized >= required => ReproVerdict::Satisfied { required, realized },
        Some(required) => ReproVerdict::Insufficient { required, realized },
    };

    Ok(VerificationReport {
        policy: policy.id(),
        structural,
        claims,
        missing_objects: missing_global.into_iter().collect(),
        closure_size: closure.len(),
        reproducibility,
    })
}

/// Resolve a claim's bindings against the source and the declared closure.
fn resolve_bindings<S: PublicationObjectSource + ?Sized>(
    claim: &Claim,
    source: &S,
    closure: &BTreeSet<ObjectId>,
) -> Result<Vec<BindingContext>> {
    let mut out = Vec::with_capacity(claim.bindings.len());
    for binding in &claim.bindings
    {
        let facts = source.facts(binding.object)?;
        let present = facts.is_some();
        let in_scope = present && closure.contains(&binding.object);
        out.push(BindingContext {
            role: binding.role,
            object: binding.object,
            present,
            in_scope,
            level: facts.as_ref().map(|f| f.level),
            kind: facts.map(|f| f.kind),
        });
    }
    Ok(out)
}

/// The objects of the bindings matching `pred`, in binding order.
fn collect(bindings: &[BindingContext], pred: impl Fn(&BindingContext) -> bool) -> Vec<ObjectId> {
    bindings
        .iter()
        .filter(|b| pred(b))
        .map(|b| b.object)
        .collect()
}

/// Which registry an exhibit came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExhibitClass {
    /// A figure.
    Figure,
    /// A table.
    Table,
}

impl ExhibitClass {
    /// A short, stable code.
    #[must_use]
    pub const fn code(self) -> &'static str {
        match self
        {
            Self::Figure => "figure",
            Self::Table => "table",
        }
    }
}

/// The drift verdict for one exhibit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum ExhibitVerdict {
    /// The re-rendered artifact hashed to the pinned expected address.
    Reproduced,
    /// The re-rendered artifact hashed to a different address than pinned.
    Drifted,
    /// The exhibit is checked and pinned, but no re-derivation was supplied.
    Missing,
    /// The exhibit is not subject to the drift check (not `MustRegenerate`, or no
    /// expected address pinned).
    NotChecked,
}

/// One exhibit's line in an [`ExhibitReport`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExhibitAssessment {
    /// Whether it is a figure or a table.
    pub class: ExhibitClass,
    /// The exhibit's handle.
    pub key: String,
    /// The pinned expected artifact address, if any.
    pub expected: Option<ObjectId>,
    /// The supplied re-derived artifact address, if any.
    pub rederived: Option<ObjectId>,
    /// The verdict.
    pub verdict: ExhibitVerdict,
}

/// The result of the executable-exhibit drift check.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExhibitReport {
    /// Per-exhibit verdicts (figures then tables, in registry order).
    pub exhibits: Vec<ExhibitAssessment>,
}

impl ExhibitReport {
    /// Whether every *checked* exhibit reproduced (unchecked ones do not count
    /// against it, but a checked-and-missing one does).
    #[must_use]
    pub fn reproduced(&self) -> bool {
        self.exhibits.iter().all(|e| {
            matches!(
                e.verdict,
                ExhibitVerdict::Reproduced | ExhibitVerdict::NotChecked
            )
        })
    }

    /// The first drifted exhibit, if any.
    #[must_use]
    pub fn first_drift(&self) -> Option<&ExhibitAssessment> {
        self.exhibits
            .iter()
            .find(|e| e.verdict == ExhibitVerdict::Drifted)
    }
}

/// Check whether each pinned, must-regenerate exhibit re-derives to its expected
/// address, given a fresh re-rendering supplied by the Workflow Engine.
///
/// `figures` and `tables` map an exhibit key to the content address a fresh
/// re-render produced. This crate does **not** render anything — producing those
/// addresses is the executor's job; here we only compare against what was pinned
/// and localize drift. A figure with policy other than
/// [`RegenPolicy::MustRegenerate`], or with no pinned `expected`, is
/// [`ExhibitVerdict::NotChecked`].
#[must_use]
pub fn verify_exhibits(
    publication: &Publication,
    figures: &BTreeMap<FigureKey, ObjectId>,
    tables: &BTreeMap<TableKey, ObjectId>,
) -> ExhibitReport {
    let mut exhibits = Vec::new();

    for figure in &publication.figures
    {
        let rederived = figures.get(&figure.key).copied();
        let verdict = exhibit_verdict(figure.regeneration, figure.expected, rederived);
        exhibits.push(ExhibitAssessment {
            class: ExhibitClass::Figure,
            key: figure.key.0.clone(),
            expected: figure.expected,
            rederived,
            verdict,
        });
    }
    for table in &publication.tables
    {
        let rederived = tables.get(&table.key).copied();
        let verdict = exhibit_verdict(table.regeneration, table.expected, rederived);
        exhibits.push(ExhibitAssessment {
            class: ExhibitClass::Table,
            key: table.key.0.clone(),
            expected: table.expected,
            rederived,
            verdict,
        });
    }

    ExhibitReport { exhibits }
}

/// The verdict for one exhibit given its policy, pinned address, and re-derivation.
fn exhibit_verdict(
    policy: RegenPolicy,
    expected: Option<ObjectId>,
    rederived: Option<ObjectId>,
) -> ExhibitVerdict {
    let Some(expected) = expected
    else
    {
        return ExhibitVerdict::NotChecked;
    };
    if !policy.is_checked()
    {
        return ExhibitVerdict::NotChecked;
    }
    match rederived
    {
        None => ExhibitVerdict::Missing,
        Some(actual) if actual == expected => ExhibitVerdict::Reproduced,
        Some(_) => ExhibitVerdict::Drifted,
    }
}

/// Whether a released publication still matches its release manifest.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReleaseConsistency {
    /// Whether the sealed publication's id equals the id the manifest attests —
    /// `false` means the publication changed since release.
    pub matches_publication: bool,
    /// Whether the sealed publication's content address verifies.
    pub publication_verified: bool,
}

impl ReleaseConsistency {
    /// Whether the release is consistent: the publication verifies and is the one
    /// the manifest attests.
    #[must_use]
    pub fn is_consistent(self) -> bool {
        self.matches_publication && self.publication_verified
    }
}

/// Check a sealed publication against a release manifest.
///
/// Answers "has it changed since release?": if the caller re-seals the current
/// publication and its id no longer equals `manifest.publication`, the document
/// was edited after release. The engine reports the divergence; it never
/// silently reinterprets a released publication.
#[must_use]
pub fn check_release(
    published: &Object<Publication>,
    manifest: &ReleaseManifest,
) -> ReleaseConsistency {
    ReleaseConsistency {
        matches_publication: published.id == manifest.publication,
        publication_verified: published.verify_id(),
    }
}
