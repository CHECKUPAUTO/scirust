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
    pub spurious_channels: Vec<usize>,
    pub votes: Vec<bool>,
    pub tripped: bool,
    pub outcome: TripOutcome,
}

/// Simulates one demand scenario. `failed_channels` lists the (0-indexed)
/// channels that have suffered a dangerous-undetected failure and therefore
/// never vote trip, independent of `demand_present`. Every healthy channel
/// votes trip iff `demand_present`.
///
/// This is a thin convenience wrapper over
/// [`simulate_demand_with_spurious`] with no spurious-trip channels — see
/// that function to also model a channel that votes trip on its own
/// (a different, safe-side failure mode from dangerous-undetected).
pub fn simulate_demand(
    architecture: Architecture,
    demand_present: bool,
    failed_channels: &[usize],
) -> SisResult<TripSimulationResult> {
    simulate_demand_with_spurious(architecture, demand_present, failed_channels, &[])
}

/// Simulates one demand scenario with two independent channel fault modes:
/// `failed_channels` (dangerous-undetected — never vote trip, regardless of
/// the real condition) and `spurious_channels` (votes trip unconditionally,
/// regardless of the real condition — e.g. a stuck sensor reading permanently
/// past its trip threshold). A channel present in both lists is treated as
/// spurious (it always votes trip) — the two fault modes cannot coexist on
/// the same physical channel at the same instant.
pub fn simulate_demand_with_spurious(
    architecture: Architecture,
    demand_present: bool,
    failed_channels: &[usize],
    spurious_channels: &[usize],
) -> SisResult<TripSimulationResult> {
    let n = architecture.n as usize;
    let votes: Vec<bool> = (0..n)
        .map(|i| {
            if spurious_channels.contains(&i)
            {
                true
            }
            else
            {
                demand_present && !failed_channels.contains(&i)
            }
        })
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
        spurious_channels: spurious_channels.to_vec(),
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

    #[test]
    fn spurious_channel_trips_1oo2_with_no_real_demand() {
        // A single stuck-high channel is already enough to trip a 1oo2 group.
        let r = simulate_demand_with_spurious(Architecture::OO2, false, &[], &[0]).unwrap();
        assert!(r.tripped);
        assert_eq!(r.outcome, TripOutcome::SpuriousTrip);
    }

    #[test]
    fn single_spurious_channel_does_not_trip_2oo3() {
        // 2oo3 needs 2 votes; one stuck-high channel alone isn't enough.
        let r = simulate_demand_with_spurious(Architecture::TWO_OO3, false, &[], &[0]).unwrap();
        assert!(!r.tripped);
        assert_eq!(r.outcome, TripOutcome::SafeIdle);
    }

    #[test]
    fn spurious_channel_still_trips_correctly_on_a_real_demand() {
        let r = simulate_demand_with_spurious(Architecture::TWO_OO3, true, &[], &[0]).unwrap();
        assert!(r.tripped);
        assert_eq!(r.outcome, TripOutcome::SafeTrip);
    }

    #[test]
    fn channel_marked_both_failed_and_spurious_is_treated_as_spurious() {
        let r = simulate_demand_with_spurious(Architecture::OO2, false, &[0], &[0]).unwrap();
        assert!(r.votes[0], "a channel in both lists always votes trip");
    }
}
