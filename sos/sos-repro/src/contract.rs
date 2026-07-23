//! The reproduction contract: level-aware verification of a re-run against its
//! declarations (RFC-0002 §09.4, §09.8).

use serde::{Deserialize, Serialize};
use sos_core::{DeterminismLevel, ObjectId};

use crate::error::{ReproError, Result};

/// What "reproduced" *means* at a given [`DeterminismLevel`] — the spine of every
/// reproducibility claim (RFC-0002 §09.8).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MatchRule {
    /// **L3 — bit-exact.** The re-run must yield the byte-identical object, i.e.
    /// the same [`ObjectId`]. Decidable from ids alone.
    BitExact,
    /// **L2 — within certificate.** The re-run must agree up to the declared
    /// tolerance; the numeric backend supplies the verdict.
    WithinCertificate,
    /// **L1 — in distribution.** The re-run must match in distribution given the
    /// recorded seed; the statistics backend supplies the verdict.
    InDistribution,
    /// **L0 — replay-identical.** A recorded observation reproduces by replay,
    /// i.e. the same recorded [`ObjectId`]. Decidable from ids alone.
    ReplayIdentical,
}

impl MatchRule {
    /// The rule that governs reproduction at `level`.
    #[must_use]
    pub const fn of(level: DeterminismLevel) -> Self {
        match level
        {
            DeterminismLevel::L3 => Self::BitExact,
            DeterminismLevel::L2 => Self::WithinCertificate,
            DeterminismLevel::L1 => Self::InDistribution,
            DeterminismLevel::L0 => Self::ReplayIdentical,
        }
    }

    /// Whether this rule is decidable from object-id equality alone (`L3`/`L0`),
    /// as opposed to needing a backend's numeric / statistical verdict
    /// (`L2`/`L1`).
    #[must_use]
    pub const fn is_id_decidable(self) -> bool {
        matches!(self, Self::BitExact | Self::ReplayIdentical)
    }
}

/// A node's declared reproducibility claim: its content id and the level it
/// realized.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct NodeClaim {
    /// The declared object id.
    pub id: ObjectId,
    /// The determinism level the node declared.
    pub level: DeterminismLevel,
}

impl NodeClaim {
    /// Construct a node claim.
    #[must_use]
    pub fn new(id: ObjectId, level: DeterminismLevel) -> Self {
        Self { id, level }
    }
}

/// The outcome a re-execution supplied for a node, in the form appropriate to its
/// level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Reproduced {
    /// The object id the re-execution produced — for `L3` (bit-exact) and `L0`
    /// (replay) nodes, which this engine checks by id equality.
    Id(ObjectId),
    /// A backend-supplied verdict — for `L2` (within certificate) and `L1` (in
    /// distribution) nodes, which this engine cannot evaluate alone. `true` means
    /// the backend certified the reproduction agrees at that node's level.
    Certified(bool),
}

/// The verdict for a single node.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NodeVerdict {
    /// The node reproduced at its declared level.
    Reproduced,
    /// The node did **not** reproduce — a localized, declared deviation.
    Diverged,
    /// The supplied evidence did not fit the node's level (an [`Reproduced::Id`]
    /// for an `L2`/`L1` node, or a [`Reproduced::Certified`] for an `L3`/`L0`
    /// node) — a caller error, not a reproduction result.
    Mismatched,
}

/// Decide a single node's verdict under the reproduction contract.
#[must_use]
pub fn verify_node(claim: &NodeClaim, reproduced: &Reproduced) -> NodeVerdict {
    match (MatchRule::of(claim.level), reproduced)
    {
        (MatchRule::BitExact | MatchRule::ReplayIdentical, Reproduced::Id(rid)) =>
        {
            if *rid == claim.id
            {
                NodeVerdict::Reproduced
            }
            else
            {
                NodeVerdict::Diverged
            }
        },
        (MatchRule::WithinCertificate | MatchRule::InDistribution, Reproduced::Certified(ok)) =>
        {
            if *ok
            {
                NodeVerdict::Reproduced
            }
            else
            {
                NodeVerdict::Diverged
            }
        },
        _ => NodeVerdict::Mismatched,
    }
}

/// One node's line in a [`VerifyReport`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct NodeReport {
    /// The declared node id.
    pub node: ObjectId,
    /// Its declared level.
    pub level: DeterminismLevel,
    /// The rule applied.
    pub rule: MatchRule,
    /// The verdict reached.
    pub verdict: NodeVerdict,
}

/// The result of verifying a re-run against a sub-DAG's declarations — the
/// machine-checkable form of "reproduced" (RFC-0002 §09.7–8). Any deviation is
/// **localized** to a specific node and its declared level; nothing is a mystery.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VerifyReport {
    /// Per-node verdicts, in the order supplied.
    pub nodes: Vec<NodeReport>,
    /// The propagated determinism level of the sub-DAG (the weakest / `meet` over
    /// all node levels).
    pub level: DeterminismLevel,
}

impl VerifyReport {
    /// Whether **every** node reproduced at its declared level.
    #[must_use]
    pub fn reproduced(&self) -> bool {
        self.nodes
            .iter()
            .all(|n| n.verdict == NodeVerdict::Reproduced)
    }

    /// The first node that did not reproduce (diverged or mismatched), localizing
    /// the deviation.
    #[must_use]
    pub fn first_deviation(&self) -> Option<&NodeReport> {
        self.nodes
            .iter()
            .find(|n| n.verdict != NodeVerdict::Reproduced)
    }
}

/// Verify a whole sub-DAG's reproduction: check each `claims[i]` against
/// `reproduced[i]` under the contract, and propagate the sub-DAG's determinism
/// level as the weakest over all nodes.
///
/// # Errors
/// [`ReproError::LengthMismatch`] if `claims` and `reproduced` differ in length.
pub fn verify_reproduction(
    claims: &[NodeClaim],
    reproduced: &[Reproduced],
) -> Result<VerifyReport> {
    if claims.len() != reproduced.len()
    {
        return Err(ReproError::LengthMismatch {
            claims: claims.len(),
            reproduced: reproduced.len(),
        });
    }
    let nodes = claims
        .iter()
        .zip(reproduced)
        .map(|(claim, outcome)| NodeReport {
            node: claim.id,
            level: claim.level,
            rule: MatchRule::of(claim.level),
            verdict: verify_node(claim, outcome),
        })
        .collect();
    let level = DeterminismLevel::min_over(claims.iter().map(|c| c.level));
    Ok(VerifyReport { nodes, level })
}

#[cfg(test)]
mod tests {
    use super::*;
    use sos_core::HashAlgo;

    fn oid(tag: &[u8]) -> ObjectId {
        ObjectId::compute(HashAlgo::default(), b"sos-obj:N:v1", tag)
    }

    #[test]
    fn l3_matches_by_id_and_flags_divergence() {
        let claim = NodeClaim::new(oid(b"x"), DeterminismLevel::L3);
        assert_eq!(
            verify_node(&claim, &Reproduced::Id(oid(b"x"))),
            NodeVerdict::Reproduced
        );
        assert_eq!(
            verify_node(&claim, &Reproduced::Id(oid(b"y"))),
            NodeVerdict::Diverged
        );
        // Wrong evidence kind for an L3 node.
        assert_eq!(
            verify_node(&claim, &Reproduced::Certified(true)),
            NodeVerdict::Mismatched
        );
    }

    #[test]
    fn l2_and_l1_take_the_backend_verdict() {
        let l2 = NodeClaim::new(oid(b"a"), DeterminismLevel::L2);
        assert_eq!(
            verify_node(&l2, &Reproduced::Certified(true)),
            NodeVerdict::Reproduced
        );
        assert_eq!(
            verify_node(&l2, &Reproduced::Certified(false)),
            NodeVerdict::Diverged
        );
        // An id cannot decide an L2 node.
        assert_eq!(
            verify_node(&l2, &Reproduced::Id(oid(b"a"))),
            NodeVerdict::Mismatched
        );
        assert!(!MatchRule::of(DeterminismLevel::L1).is_id_decidable());
    }

    #[test]
    fn report_localizes_the_deviation_and_propagates_level() {
        let claims = [
            NodeClaim::new(oid(b"1"), DeterminismLevel::L3),
            NodeClaim::new(oid(b"2"), DeterminismLevel::L1),
            NodeClaim::new(oid(b"3"), DeterminismLevel::L3),
        ];
        let reproduced = [
            Reproduced::Id(oid(b"1")),    // ok
            Reproduced::Certified(false), // diverges
            Reproduced::Id(oid(b"3")),    // ok
        ];
        let report = verify_reproduction(&claims, &reproduced).unwrap();
        assert!(!report.reproduced());
        // The weakest level over the sub-DAG is L1.
        assert_eq!(report.level, DeterminismLevel::L1);
        // The deviation is localized to node 2.
        assert_eq!(report.first_deviation().unwrap().node, oid(b"2"));
    }

    #[test]
    fn length_mismatch_is_an_error() {
        let claims = [NodeClaim::new(oid(b"1"), DeterminismLevel::L3)];
        assert!(matches!(
            verify_reproduction(&claims, &[]),
            Err(ReproError::LengthMismatch { .. })
        ));
    }
}
