//! [`SupportPolicy`] — the versioned engine that decides a claim's standing —
//! and the [`StandardPolicy`] (v1) that implements the default, fully explicit
//! decision procedure.
//!
//! Invariant VI forbids opaque scoring: a claim's [`ClaimStatus`] is a
//! *categorical function of stated conditions*, never a hidden numeric verdict.
//! The whole procedure is written out in [`StandardPolicy::assess`] and every
//! branch is reachable and tested. The policy is **versioned** ([`PolicyId`]) and
//! recorded in the publication, so a document is always judged by a named,
//! reproducible rule set — and a future, stricter policy is a new version, not a
//! silent change to this one.

use serde::{Deserialize, Serialize};
use sos_core::canonical::{Canonical, CanonicalEncoder};
use sos_core::{DeterminismLevel, Kind, ObjectId};

use crate::claim::{BindingRole, Claim};
use crate::publication::ReproRequirement;
use crate::verify::ClaimStatus;

/// The name and version of a support policy. Recorded in a
/// [`Publication`](crate::publication::Publication) and in a
/// [`ReleaseManifest`](crate::release::ReleaseManifest) so a document names the
/// exact rule set it is judged by.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct PolicyId {
    /// The policy name (e.g. `"standard"`).
    pub name: String,
    /// The policy version.
    pub version: u32,
}

impl PolicyId {
    /// Construct a policy id.
    #[must_use]
    pub fn new(name: impl Into<String>, version: u32) -> Self {
        Self {
            name: name.into(),
            version,
        }
    }

    /// The id of the built-in [`StandardPolicy`] (`standard@v1`).
    #[must_use]
    pub fn standard() -> Self {
        Self::new("standard", StandardPolicy::VERSION)
    }
}

impl Canonical for PolicyId {
    fn encode(&self, enc: &mut CanonicalEncoder) {
        enc.str(&self.name);
        enc.u64(u64::from(self.version));
    }
}

impl core::fmt::Display for PolicyId {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{}@v{}", self.name, self.version)
    }
}

/// One of a claim's bindings, resolved against the object source and the declared
/// scope — the input a [`SupportPolicy`] reasons over.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BindingContext {
    /// How the object bears on the claim.
    pub role: BindingRole,
    /// The bound object's content address.
    pub object: ObjectId,
    /// Whether the object was found in the source.
    pub present: bool,
    /// Whether the object lies within the publication's declared dependency
    /// closure (its scope). Always `false` when `present` is `false`.
    pub in_scope: bool,
    /// The object's realized determinism level (`Some` iff `present`).
    pub level: Option<DeterminismLevel>,
    /// The object's kind (`Some` iff `present`).
    pub kind: Option<Kind>,
}

impl BindingContext {
    /// Whether this is a **resolved, in-scope** supporting edge (the kind that
    /// can actually raise a claim's standing).
    #[must_use]
    pub fn is_effective_support(&self) -> bool {
        self.present && self.in_scope && self.role.is_support()
    }
}

/// Everything a [`SupportPolicy`] needs to judge one claim.
#[derive(Debug, Clone)]
pub struct ClaimContext<'a> {
    /// The claim under assessment.
    pub claim: &'a Claim,
    /// Its bindings, each resolved against source and scope.
    pub bindings: Vec<BindingContext>,
    /// The publication's reproducibility bar.
    pub requirement: ReproRequirement,
    /// Whether this claim's key is duplicated elsewhere in the publication (an
    /// ambiguity the policy treats as structurally invalid).
    pub duplicate_key: bool,
}

/// A versioned rule set that decides a claim's [`ClaimStatus`] from its resolved
/// bindings. Implementations must be **pure and total** — same context, same
/// status — and must never infer standing from claim wording.
pub trait SupportPolicy {
    /// The policy's id (recorded in reports).
    fn id(&self) -> PolicyId;

    /// Decide the claim's status from its resolved [`ClaimContext`].
    fn assess(&self, ctx: &ClaimContext) -> ClaimStatus;
}

/// The default support policy (version 1).
///
/// The decision procedure, in strict order (first match wins):
///
/// 1. **Duplicate key** → [`ClaimStatus::StructurallyInvalid`]. An ambiguous
///    handle makes the claim unaddressable.
/// 2. **No bindings** → [`ClaimStatus::Unverifiable`]. A claim with no edges into
///    the graph cannot be checked; a well-written sentence is not evidence.
/// 3. **Incoherent wiring** (some object bound *both* as support and as
///    contradiction) → [`ClaimStatus::PolicyRejected`]. The author's own edges
///    disagree; the engine refuses to guess.
/// 4. **A resolved contradiction** → [`ClaimStatus::Contradicted`]. Never hidden,
///    and it dominates any support.
/// 5. **A missing dependency** (any bound object absent from the source) →
///    [`ClaimStatus::DependencyMissing`]. The claim leans on something not in the
///    graph.
/// 6. **No effective support** (no resolved, in-scope supporting edge — the
///    support is out of the declared scope, or the edges are only context) →
///    [`ClaimStatus::Unresolved`].
/// 7. **Reproducibility bar unmet** (weakest in-scope supporting level below the
///    required minimum) → [`ClaimStatus::ReproducibilityFailed`].
/// 8. Otherwise **direct** in-scope support → [`ClaimStatus::Supported`]; only
///    **indirect** in-scope support → [`ClaimStatus::PartiallySupported`].
#[derive(Debug, Clone, Copy, Default)]
pub struct StandardPolicy;

impl StandardPolicy {
    /// This policy's version.
    pub const VERSION: u32 = 1;

    /// Construct the standard policy.
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

impl SupportPolicy for StandardPolicy {
    fn id(&self) -> PolicyId {
        PolicyId::standard()
    }

    fn assess(&self, ctx: &ClaimContext) -> ClaimStatus {
        // 1. Ambiguous handle.
        if ctx.duplicate_key
        {
            return ClaimStatus::StructurallyInvalid;
        }
        // 2. No edges into the graph.
        if ctx.bindings.is_empty()
        {
            return ClaimStatus::Unverifiable;
        }
        // 3. Incoherent wiring: an object bound as both support and contradiction.
        if self.has_incoherent_binding(ctx)
        {
            return ClaimStatus::PolicyRejected;
        }
        // 4. A resolved contradiction dominates and is never hidden.
        if ctx
            .bindings
            .iter()
            .any(|b| b.role.is_contradiction() && b.present)
        {
            return ClaimStatus::Contradicted;
        }
        // 5. Any bound object absent from the graph.
        if ctx.bindings.iter().any(|b| !b.present)
        {
            return ClaimStatus::DependencyMissing;
        }
        // 6. Effective (resolved, in-scope) support.
        let has_direct = ctx
            .bindings
            .iter()
            .any(|b| b.role.is_direct_support() && b.in_scope);
        let has_indirect = ctx
            .bindings
            .iter()
            .any(|b| b.role.is_indirect_support() && b.in_scope);
        if !has_direct && !has_indirect
        {
            return ClaimStatus::Unresolved;
        }
        // 7. Reproducibility bar over the in-scope supporting evidence.
        if let Some(required) = ctx.requirement.minimum()
        {
            let weakest = DeterminismLevel::min_over(
                ctx.bindings
                    .iter()
                    .filter(|b| b.is_effective_support())
                    .filter_map(|b| b.level),
            );
            if weakest < required
            {
                return ClaimStatus::ReproducibilityFailed;
            }
        }
        // 8. Supported (direct) or partially supported (indirect only).
        if has_direct
        {
            ClaimStatus::Supported
        }
        else
        {
            ClaimStatus::PartiallySupported
        }
    }
}

impl StandardPolicy {
    /// Whether any single object is bound with both a supporting and a
    /// contradicting role — an incoherence in the claim's own wiring.
    fn has_incoherent_binding(&self, ctx: &ClaimContext) -> bool {
        ctx.bindings.iter().any(|b| {
            b.role.is_support()
                && ctx
                    .bindings
                    .iter()
                    .any(|o| o.object == b.object && o.role.is_contradiction())
        })
    }
}
