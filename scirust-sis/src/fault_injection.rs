//! Fault injection against a voting architecture: given a real process
//! demand (or its absence) and a set of channels stuck dangerous-undetected
//! (never vote trip, regardless of the real condition), determine whether
//! the architecture still does its job — and classify the outcome.
//!
//! This is the SIS analogue of `scirust-func-safety::fault_injection`
//! (there: bit-flips/stuck-at faults on neural network weights; here:
//! stuck-failed sensor/logic/final-element channels), used to empirically
//! demonstrate the safety property a `PFDavg` number only states abstractly
//! — e.g. that 2oo3 tolerates one failed channel and still trips correctly,
//! while 2oo2 does not.

use crate::error::SisResult;
use crate::voting::Architecture;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TripOutcome {
    /// A real demand was present and the group correctly tripped.
    SafeTrip,
    /// No demand was present and the group correctly stayed put.
    SafeIdle,
    /// A real demand was present but the group failed to trip — the
    /// dangerous failure mode `PFDavg` quantifies.
    DangerousFailure,
    /// No demand was present but the group tripped anyway (spurious trip —
    /// costly but not dangerous).
    SpuriousTrip,
}

impl TripOutcome {
    pub fn is_dangerous(&self) -> bool {
        matches!(self, TripOutcome::DangerousFailure)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TripSimulationResult {
    pub architecture: Architecture,
    pub demand_present: bool,
    pub failed_channels: Vec<usize>,
    pub votes: Vec<bool>,
    pub tripped: bool,
    pub outcome: TripOutcome,
}

/// Simulates one demand scenario. `failed_channels` lists the (0-indexed)
/// channels that have suffered a dangerous-undetected failure and therefore
/// never vote trip, independent of `demand_present`. Every healthy channel
/// votes trip iff `demand_present`.
pub fn simulate_demand(
    architecture: Architecture,
    demand_present: bool,
    failed_channels: &[usize],
) -> SisResult<TripSimulationResult> {
    let n = architecture.n as usize;
    let votes: Vec<bool> = (0..n)
        .map(|i| demand_present && !failed_channels.contains(&i))
        .collect();
    let tripped = architecture.evaluate_votes(&votes)?;

    let outcome = match (demand_present, tripped)
    {
        (true, true) => TripOutcome::SafeTrip,
        (true, false) => TripOutcome::DangerousFailure,
        (false, false) => TripOutcome::SafeIdle,
        (false, true) => TripOutcome::SpuriousTrip,
    };

    Ok(TripSimulationResult {
        architecture,
        demand_present,
        failed_channels: failed_channels.to_vec(),
        votes,
        tripped,
        outcome,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn healthy_group_trips_on_demand() {
        let r = simulate_demand(Architecture::TWO_OO3, true, &[]).unwrap();
        assert!(r.tripped);
        assert_eq!(r.outcome, TripOutcome::SafeTrip);
    }

    #[test]
    fn healthy_group_stays_idle_without_demand() {
        let r = simulate_demand(Architecture::TWO_OO3, false, &[]).unwrap();
        assert!(!r.tripped);
        assert_eq!(r.outcome, TripOutcome::SafeIdle);
    }

    #[test]
    fn two_oo3_tolerates_one_failed_channel() {
        // 2 of the 3 channels still vote correctly ⇒ still trips.
        let r = simulate_demand(Architecture::TWO_OO3, true, &[0]).unwrap();
        assert!(r.tripped, "2oo3 should tolerate a single failed channel");
        assert_eq!(r.outcome, TripOutcome::SafeTrip);
    }

    #[test]
    fn two_oo3_fails_dangerous_with_two_failed_channels() {
        let r = simulate_demand(Architecture::TWO_OO3, true, &[0, 1]).unwrap();
        assert!(!r.tripped);
        assert_eq!(r.outcome, TripOutcome::DangerousFailure);
    }

    #[test]
    fn two_oo2_fails_dangerous_with_a_single_failed_channel() {
        // 2oo2 has zero tolerance for a dangerous failure — this is the
        // trade-off for its lower spurious-trip rate.
        let r = simulate_demand(Architecture::TWO_OO2, true, &[0]).unwrap();
        assert!(
            !r.tripped,
            "2oo2 has no redundancy against dangerous failure"
        );
        assert_eq!(r.outcome, TripOutcome::DangerousFailure);
    }

    #[test]
    fn oo2_tolerates_one_failed_channel() {
        // 1oo2: either channel tripping is enough.
        let r = simulate_demand(Architecture::OO2, true, &[0]).unwrap();
        assert!(r.tripped);
        assert_eq!(r.outcome, TripOutcome::SafeTrip);
    }

    #[test]
    fn all_channels_failed_is_always_dangerous_on_demand() {
        let r = simulate_demand(Architecture::OO3, true, &[0, 1, 2]).unwrap();
        assert!(!r.tripped);
        assert_eq!(r.outcome, TripOutcome::DangerousFailure);
    }
}
