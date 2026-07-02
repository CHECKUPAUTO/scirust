//! `MooN` voting architectures ("M out of N channels must vote trip") — the
//! logic layer that decides *whether the group trips* given per-channel
//! votes, on top of the quantitative `PFDavg`/`PFH` primitives already in
//! `scirust-reliability`.

use crate::error::{SisError, SisResult};
use scirust_reliability::{pfd_1oo1, pfd_1oo2, pfd_1oo3, pfd_2oo2, pfd_2oo3};
use serde::{Deserialize, Serialize};

/// An `M`-out-of-`N` voting architecture: at least `m` of `n` channels must
/// vote "trip" for the group to trip.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Architecture {
    pub m: u8,
    pub n: u8,
}

impl Architecture {
    pub const OO1: Architecture = Architecture { m: 1, n: 1 };
    pub const OO2: Architecture = Architecture { m: 1, n: 2 };
    pub const TWO_OO2: Architecture = Architecture { m: 2, n: 2 };
    pub const TWO_OO3: Architecture = Architecture { m: 2, n: 3 };
    pub const OO3: Architecture = Architecture { m: 1, n: 3 };

    pub fn new(m: u8, n: u8) -> SisResult<Self> {
        if m == 0 || m > n
        {
            return Err(SisError::InvalidArchitecture { m, n });
        }
        Ok(Self { m, n })
    }

    pub fn label(&self) -> String {
        format!("{}oo{}", self.m, self.n)
    }

    /// `PFDavg` (low-demand mode) of this architecture, dispatching to the
    /// matching `scirust-reliability` formula. `beta` (common-cause
    /// fraction) is ignored by architectures that don't use it (1oo1, 2oo2).
    pub fn pfd_avg(&self, lambda_du: f64, t1: f64, beta: f64) -> SisResult<f64> {
        match (self.m, self.n)
        {
            (1, 1) => Ok(pfd_1oo1(lambda_du, t1)),
            (1, 2) => Ok(pfd_1oo2(lambda_du, t1, beta)),
            (2, 2) => Ok(pfd_2oo2(lambda_du, t1)),
            (2, 3) => Ok(pfd_2oo3(lambda_du, t1, beta)),
            (1, 3) => Ok(pfd_1oo3(lambda_du, t1, beta)),
            (m, n) => Err(SisError::UnsupportedArchitecture { m, n }),
        }
    }

    /// Whether the group trips given one vote per channel (`true` = "this
    /// channel demands trip"). `votes.len()` must equal `n`.
    pub fn evaluate_votes(&self, votes: &[bool]) -> SisResult<bool> {
        if votes.len() != self.n as usize
        {
            return Err(SisError::VoteCountMismatch {
                expected: self.n,
                got: votes.len(),
            });
        }
        let trip_votes = votes.iter().filter(|&&v| v).count() as u8;
        Ok(trip_votes >= self.m)
    }
}

impl std::fmt::Display for Architecture {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.label())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn rejects_invalid_architecture() {
        assert!(Architecture::new(0, 2).is_err());
        assert!(Architecture::new(3, 2).is_err());
        assert!(Architecture::new(1, 1).is_ok());
    }

    #[test]
    fn pfd_avg_dispatches_to_matching_formula() {
        let a = Architecture::OO2; // 1oo2
        let pfd = a.pfd_avg(1e-3, 1000.0, 0.1).unwrap();
        assert_relative_eq!(pfd, scirust_reliability::pfd_1oo2(1e-3, 1000.0, 0.1));
    }

    #[test]
    fn pfd_avg_rejects_unsupported_architecture() {
        let a = Architecture::new(2, 4).unwrap();
        assert!(a.pfd_avg(1e-3, 1000.0, 0.1).is_err());
    }

    #[test]
    fn oo2_trips_on_either_channel() {
        let a = Architecture::OO2;
        assert!(a.evaluate_votes(&[true, false]).unwrap());
        assert!(a.evaluate_votes(&[false, true]).unwrap());
        assert!(a.evaluate_votes(&[true, true]).unwrap());
        assert!(!a.evaluate_votes(&[false, false]).unwrap());
    }

    #[test]
    fn two_oo2_needs_both_channels() {
        let a = Architecture::TWO_OO2;
        assert!(a.evaluate_votes(&[true, true]).unwrap());
        assert!(!a.evaluate_votes(&[true, false]).unwrap());
        assert!(!a.evaluate_votes(&[false, false]).unwrap());
    }

    #[test]
    fn two_oo3_needs_majority() {
        let a = Architecture::TWO_OO3;
        assert!(a.evaluate_votes(&[true, true, false]).unwrap());
        assert!(a.evaluate_votes(&[true, true, true]).unwrap());
        assert!(!a.evaluate_votes(&[true, false, false]).unwrap());
        assert!(!a.evaluate_votes(&[false, false, false]).unwrap());
    }

    #[test]
    fn evaluate_votes_rejects_wrong_channel_count() {
        let a = Architecture::TWO_OO3;
        assert!(a.evaluate_votes(&[true, false]).is_err());
    }

    #[test]
    fn label_matches_moon_notation() {
        assert_eq!(Architecture::OO1.label(), "1oo1");
        assert_eq!(Architecture::TWO_OO3.label(), "2oo3");
    }
}
