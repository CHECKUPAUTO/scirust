//! The [`Planner`] trait and the myopic [`GreedyPlanner`].

use crate::error::{PlannerError, Result};
use crate::estimate::Estimate;
use crate::plan::{Candidate, Plan, RankedDesign, StopVerdict};
use crate::policy::{EXCLUDED, UtilityPolicy};

/// The planner syscall (SDE §05.6): given candidate designs, a utility policy,
/// and the information-exhaustion floor, recommend the next experiment (or signal
/// that information is exhausted).
pub trait Planner {
    /// Rank `candidates` by utility and produce a [`Plan`]. `eig_floor_milli` is
    /// the `ε` below which a design teaches too little to be worth running; if no
    /// candidate's EIG clears it, the verdict is
    /// [`StopVerdict::InformationExhausted`].
    ///
    /// # Errors
    /// [`PlannerError::NoCandidates`] if `candidates` is empty.
    fn recommend(
        &self,
        candidates: &[Candidate],
        policy: UtilityPolicy,
        eig_floor_milli: i64,
    ) -> Result<Plan>;
}

/// The default **myopic**, one-step-greedy planner: it maximizes immediate
/// utility (`argmax_ξ U(ξ)`), which is cheap and often optimal enough (SDE
/// §05.6). Non-myopic finite-horizon planning is a documented research direction,
/// not implemented here — no stub.
#[derive(Debug, Clone, Copy, Default)]
pub struct GreedyPlanner;

impl GreedyPlanner {
    /// Construct the greedy planner.
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

impl Planner for GreedyPlanner {
    fn recommend(
        &self,
        candidates: &[Candidate],
        policy: UtilityPolicy,
        eig_floor_milli: i64,
    ) -> Result<Plan> {
        if candidates.is_empty()
        {
            return Err(PlannerError::NoCandidates);
        }

        let mut ranked: Vec<RankedDesign> = candidates
            .iter()
            .map(|c| RankedDesign {
                experiment: c.experiment,
                eig: c.eig,
                cost: c.cost,
                utility: policy.utility(&c.eig, &c.cost),
            })
            .collect();

        // Deterministic ranking: highest utility first, ties broken by object id.
        ranked.sort_by(|a, b| {
            b.utility
                .cmp(&a.utility)
                .then(a.experiment.cmp(&b.experiment))
        });

        // Information exhaustion: recommend the top-utility design that both
        // clears the EIG floor and is not excluded by a hard constraint (e.g.
        // over budget); if none qualifies, stop. Because `ranked` is already
        // sorted by utility, the first qualifying design is the recommendation.
        let verdict = match ranked
            .iter()
            .find(|d| d.utility != EXCLUDED && clears_floor(&d.eig, eig_floor_milli))
        {
            Some(top) => StopVerdict::Recommend(top.experiment),
            None => StopVerdict::InformationExhausted,
        };

        Ok(Plan {
            policy,
            eig_floor_milli,
            ranked,
            verdict,
        })
    }
}

/// Whether a design's EIG clears the exhaustion floor: its point estimate is at
/// least `ε` **and** it is significantly informative (its value exceeds its own
/// noise), so a `0.02 ± 0.03` bit estimate never counts.
fn clears_floor(eig: &Estimate, floor_milli: i64) -> bool {
    eig.bits_milli >= floor_milli && eig.is_significant()
}
