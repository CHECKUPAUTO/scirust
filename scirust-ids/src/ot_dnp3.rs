//! DNP3 (IEEE 1815) application-layer anomaly detection.
//!
//! DNP3 is the dominant SCADA protocol for electric utilities and water. Normal
//! traffic is overwhelmingly **READ** polling of input points; the dangerous
//! events are **control** operations (SELECT/OPERATE/DIRECT_OPERATE/WRITE) on
//! output points — the unauthorized-actuation threat (Stuxnet-style). This model
//! learns the normal function-code / object-group mix and the set of legitimate
//! control points, scores deviations, and thresholds them by split conformal
//! (reusing [`scirust_pdm::ConformalGuard`]) for a bounded false-alarm rate.

use scirust_pdm::{ConformalGuard, GuardVerdict};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

/// A DNP3 application-layer request on one object point.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Dnp3Event {
    /// Function code (1 READ, 2 WRITE, 3 SELECT, 4 OPERATE, 5/6 DIRECT_OPERATE…).
    pub function: u8,
    /// Object group (1 binary-in, 10/12 binary-out, 20 counter, 30 analog-in,
    /// 40/41 analog-out…).
    pub group: u8,
    /// Point index within the group.
    pub index: u16,
}

impl Dnp3Event {
    /// Whether this is a control/write operation on an output group — the
    /// security-relevant actuation class.
    pub fn is_control(&self) -> bool {
        matches!(self.function, 2..=6) && matches!(self.group, 10 | 12 | 40 | 41)
    }
}

/// Behavioural baseline of normal DNP3 traffic.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Dnp3Profile {
    fn_counts: BTreeMap<u8, u64>,
    group_counts: BTreeMap<u8, u64>,
    total: u64,
    seen_controls: BTreeSet<(u8, u16)>,
}

fn rarity(counts: &BTreeMap<u8, u64>, key: u8, total: u64) -> f32 {
    let c = *counts.get(&key).unwrap_or(&0);
    -(((c + 1) as f32) / ((total + 1) as f32)).log2()
}

impl Dnp3Profile {
    /// Learn from normal traffic.
    pub fn learn(events: &[Dnp3Event]) -> Self {
        let mut p = Dnp3Profile::default();
        for e in events
        {
            *p.fn_counts.entry(e.function).or_insert(0) += 1;
            *p.group_counts.entry(e.group).or_insert(0) += 1;
            p.total += 1;
            if e.is_control()
            {
                p.seen_controls.insert((e.group, e.index));
            }
        }
        p
    }

    /// Anomaly score: function-code rarity + object-group rarity + a heavy
    /// penalty for a control operation on a point never actuated in the baseline.
    pub fn score(&self, e: &Dnp3Event) -> f32 {
        let fr = rarity(&self.fn_counts, e.function, self.total);
        let gr = rarity(&self.group_counts, e.group, self.total);
        let control_penalty = if e.is_control() && !self.seen_controls.contains(&(e.group, e.index))
        {
            12.0
        }
        else
        {
            0.0
        };
        fr + gr + control_penalty
    }
}

/// DNP3 anomaly guard: behavioural profile + conformal false-alarm bound.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Dnp3Guard {
    profile: Dnp3Profile,
    guard: ConformalGuard,
}

impl Dnp3Guard {
    /// Split-conformal calibration (learn on `train`, threshold on `calib`).
    pub fn calibrate(train: &[Dnp3Event], calib: &[Dnp3Event], alpha: f32) -> Self {
        let profile = Dnp3Profile::learn(train);
        let scores: Vec<f32> = calib.iter().map(|e| profile.score(e)).collect();
        let guard = ConformalGuard::calibrate(&scores, alpha);
        Self { profile, guard }
    }

    /// Classify a single DNP3 event.
    pub fn check(&self, e: &Dnp3Event) -> GuardVerdict {
        self.guard.check(self.profile.score(e))
    }

    /// Empirical false-alarm rate over held-out normal traffic.
    pub fn false_alarm_rate(&self, normal: &[Dnp3Event]) -> f32 {
        let scores: Vec<f32> = normal.iter().map(|e| self.profile.score(e)).collect();
        self.guard.false_alarm_rate(&scores)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Rng {
        s: u64,
    }
    impl Rng {
        fn new(seed: u64) -> Self {
            Self { s: seed }
        }
        fn next(&mut self) -> u64 {
            self.s = self.s.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.s;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }
        /// Normal SCADA: READ polling of input groups {1,30,20}, indices 0..50.
        fn normal(&mut self) -> Dnp3Event {
            let groups = [1u8, 30, 30, 20];
            Dnp3Event {
                function: 1, // READ
                group: groups[(self.next() % groups.len() as u64) as usize],
                index: (self.next() % 50) as u16,
            }
        }
        fn batch(&mut self, n: usize) -> Vec<Dnp3Event> {
            (0..n).map(|_| self.normal()).collect()
        }
    }

    #[test]
    fn flags_unauthorized_operate_on_an_output() {
        let mut rng = Rng::new(0xD2B3);
        let guard = Dnp3Guard::calibrate(&rng.batch(4000), &rng.batch(3000), 0.05);

        // OPERATE a control-relay output block never actuated in the baseline.
        let attack = Dnp3Event {
            function: 4,
            group: 12,
            index: 7,
        };
        assert!(attack.is_control());
        assert_eq!(guard.check(&attack), GuardVerdict::Anomaly);

        // Normal read is not flagged.
        let read = Dnp3Event {
            function: 1,
            group: 30,
            index: 12,
        };
        assert_eq!(guard.check(&read), GuardVerdict::Normal);
    }

    #[test]
    fn false_alarm_rate_bounded() {
        let mut rng = Rng::new(0x5CADA);
        let guard = Dnp3Guard::calibrate(&rng.batch(4000), &rng.batch(3000), 0.05);
        let far = guard.false_alarm_rate(&rng.batch(8000));
        assert!(far <= 0.05 + 0.03, "FAR {far}");
    }
}
