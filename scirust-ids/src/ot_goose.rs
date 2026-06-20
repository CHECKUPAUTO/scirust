//! IEC 61850 GOOSE sequence-integrity monitor.
//!
//! GOOSE multicast frames carry a state number `stNum` (incremented on every
//! protection state change) and a sequence number `sqNum` (incremented on each
//! retransmission, reset to 0 on a state change). That structure makes
//! **replay** and **spoofing** attacks — the classic substation threat —
//! detectable from the counters alone: a valid stream either repeats the state
//! with `sqNum+1`, or advances the state with `stNum+1, sqNum=0`. Anything else
//! is a stale replay or an injected frame. Deterministic, stateful.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// A GOOSE frame's integrity-relevant fields.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct GooseFrame {
    /// Publisher / GoCB identifier.
    pub src: u32,
    /// State number.
    pub st_num: u32,
    /// Sequence (retransmission) number.
    pub sq_num: u32,
}

/// Verdict for one GOOSE frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GooseVerdict {
    /// First frame seen from this publisher.
    FirstSeen,
    /// A legitimate retransmission or state change.
    Valid,
    /// A stale frame (old `stNum`, or non-advancing `sqNum`) — replay.
    Replay,
    /// An unexpected `stNum` jump — injected/spoofed frame.
    Spoof,
}

/// Per-publisher GOOSE sequence monitor.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GooseMonitor {
    last: BTreeMap<u32, (u32, u32)>,
}

impl GooseMonitor {
    /// New, empty monitor.
    pub fn new() -> Self {
        Self::default()
    }

    /// Validate the next frame against the publisher's last accepted counters.
    pub fn check(&mut self, f: &GooseFrame) -> GooseVerdict {
        match self.last.get(&f.src).copied()
        {
            None =>
            {
                self.last.insert(f.src, (f.st_num, f.sq_num));
                GooseVerdict::FirstSeen
            },
            Some((st, sq)) =>
            {
                let state_change = f.st_num == st + 1 && f.sq_num == 0;
                let retransmit = f.st_num == st && f.sq_num == sq + 1;
                let verdict = if state_change || retransmit
                {
                    GooseVerdict::Valid
                }
                else if f.st_num < st || (f.st_num == st && f.sq_num <= sq)
                {
                    GooseVerdict::Replay // stale / non-advancing
                }
                else
                {
                    GooseVerdict::Spoof // unexpected stNum jump
                };
                // Only legitimate frames advance the accepted counters, so a
                // replay can't corrupt the reference state.
                if verdict == GooseVerdict::Valid
                {
                    self.last.insert(f.src, (f.st_num, f.sq_num));
                }
                verdict
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame(src: u32, st: u32, sq: u32) -> GooseFrame {
        GooseFrame {
            src,
            st_num: st,
            sq_num: sq,
        }
    }

    #[test]
    fn accepts_a_valid_sequence() {
        let mut m = GooseMonitor::new();
        assert_eq!(m.check(&frame(1, 5, 0)), GooseVerdict::FirstSeen);
        assert_eq!(m.check(&frame(1, 5, 1)), GooseVerdict::Valid); // retransmit
        assert_eq!(m.check(&frame(1, 5, 2)), GooseVerdict::Valid);
        assert_eq!(m.check(&frame(1, 6, 0)), GooseVerdict::Valid); // state change
        assert_eq!(m.check(&frame(1, 6, 1)), GooseVerdict::Valid);
    }

    #[test]
    fn flags_replay_and_spoof() {
        let mut m = GooseMonitor::new();
        m.check(&frame(1, 6, 0));
        m.check(&frame(1, 6, 1));
        // Replay an old state.
        assert_eq!(m.check(&frame(1, 5, 0)), GooseVerdict::Replay);
        // Re-sent same counters (stale).
        assert_eq!(m.check(&frame(1, 6, 1)), GooseVerdict::Replay);
        // Injected stNum jump.
        assert_eq!(m.check(&frame(1, 9, 0)), GooseVerdict::Spoof);
        // The reference state was not corrupted by the attacks: the legitimate
        // next retransmission is still accepted.
        assert_eq!(m.check(&frame(1, 6, 2)), GooseVerdict::Valid);
    }

    #[test]
    fn publishers_are_independent() {
        let mut m = GooseMonitor::new();
        assert_eq!(m.check(&frame(1, 1, 0)), GooseVerdict::FirstSeen);
        assert_eq!(m.check(&frame(2, 7, 3)), GooseVerdict::FirstSeen);
        assert_eq!(m.check(&frame(1, 1, 1)), GooseVerdict::Valid);
        assert_eq!(m.check(&frame(2, 8, 0)), GooseVerdict::Valid);
    }
}
