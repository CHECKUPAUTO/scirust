//! Industrial (OT/ICS) protocol anomaly detection with a conformal false-alarm
//! guarantee.
//!
//! Plant networks speak Modbus/DNP3/OPC-UA, not HTTP. This module models normal
//! **Modbus** behaviour — the mix of function codes and the set of registers a
//! device touches — and scores each request by how far it departs from that
//! baseline (rare/unknown function code, never-seen register). The alarm
//! threshold is then set by **split conformal** calibration (reusing
//! [`scirust_pdm::ConformalGuard`]), so the false-alarm rate on normal traffic
//! is bounded by `α` with no distributional assumption — the same guarantee the
//! predictive-maintenance guard provides, now on the OT wire.

use scirust_pdm::{ConformalGuard, GuardVerdict};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

/// A minimal Modbus request event.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModbusEvent {
    /// Slave / unit identifier.
    pub unit_id: u8,
    /// Function code (e.g. 1 read-coils, 3 read-holding, 6 write-single, 16 write-multiple).
    pub function: u8,
    /// Starting register / coil address.
    pub address: u16,
    /// Quantity of registers / coils.
    pub quantity: u16,
}

/// Behavioural baseline of normal Modbus traffic.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ModbusProfile {
    fn_counts: BTreeMap<u8, u64>,
    total: u64,
    seen_addr: BTreeSet<u16>,
}

impl ModbusProfile {
    /// Learn the function-code distribution and register-access set from normal
    /// traffic.
    pub fn learn(events: &[ModbusEvent]) -> Self {
        let mut p = ModbusProfile::default();
        for e in events
        {
            *p.fn_counts.entry(e.function).or_insert(0) += 1;
            p.total += 1;
            p.seen_addr.insert(e.address);
        }
        p
    }

    /// Anomaly score (higher = more anomalous): the function code's rarity in
    /// bits, `−log₂((count+1)/(total+1))`, plus a fixed penalty for a register
    /// never accessed in the baseline.
    pub fn score(&self, e: &ModbusEvent) -> f32 {
        let c = *self.fn_counts.get(&e.function).unwrap_or(&0);
        let rarity = -(((c + 1) as f32) / ((self.total + 1) as f32)).log2();
        let addr_penalty = if self.seen_addr.contains(&e.address)
        {
            0.0
        }
        else
        {
            8.0
        };
        rarity + addr_penalty
    }
}

/// OT anomaly guard: a Modbus behavioural profile thresholded by a conformal
/// false-alarm bound.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OtGuard {
    profile: ModbusProfile,
    guard: ConformalGuard,
}

impl OtGuard {
    /// Split-conformal calibration: learn the profile on `train`, set the
    /// threshold from the scores of a separate `calib` set so the false-alarm
    /// rate on future normal traffic is bounded by `alpha`.
    pub fn calibrate(train: &[ModbusEvent], calib: &[ModbusEvent], alpha: f32) -> Self {
        let profile = ModbusProfile::learn(train);
        let scores: Vec<f32> = calib.iter().map(|e| profile.score(e)).collect();
        let guard = ConformalGuard::calibrate(&scores, alpha);
        Self { profile, guard }
    }

    /// The learned behavioural profile.
    pub fn profile(&self) -> &ModbusProfile {
        &self.profile
    }

    /// The conformal alarm threshold (scores above are anomalies).
    pub fn threshold(&self) -> f32 {
        self.guard.threshold()
    }

    /// Classify a single Modbus event.
    pub fn check(&self, e: &ModbusEvent) -> GuardVerdict {
        self.guard.check(self.profile.score(e))
    }

    /// Empirical false-alarm rate over a held-out normal set.
    pub fn false_alarm_rate(&self, normal: &[ModbusEvent]) -> f32 {
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
        /// Normal Modbus traffic: function codes {1,3,6,16} with a fixed mix,
        /// addresses in [0, 100).
        fn normal_event(&mut self) -> ModbusEvent {
            let codes = [1u8, 3, 3, 3, 6, 16]; // read-holding dominates
            let function = codes[(self.next() % codes.len() as u64) as usize];
            let address = (self.next() % 100) as u16;
            ModbusEvent {
                unit_id: 1,
                function,
                address,
                quantity: 1,
            }
        }
        fn normal_batch(&mut self, n: usize) -> Vec<ModbusEvent> {
            (0..n).map(|_| self.normal_event()).collect()
        }
    }

    #[test]
    fn false_alarm_rate_bounded_by_alpha() {
        let mut rng = Rng::new(0x07);
        let train = rng.normal_batch(4000);
        let calib = rng.normal_batch(3000);
        let test = rng.normal_batch(8000);
        let alpha = 0.05;

        let guard = OtGuard::calibrate(&train, &calib, alpha);
        let far = guard.false_alarm_rate(&test);
        assert!(
            far <= alpha + 0.03,
            "FAR {far} exceeds alpha {alpha} + slack"
        );
    }

    #[test]
    fn flags_illegal_function_code_and_unseen_register() {
        let mut rng = Rng::new(99);
        let train = rng.normal_batch(4000);
        let calib = rng.normal_batch(3000);
        let guard = OtGuard::calibrate(&train, &calib, 0.05);

        // Diagnostics function code 8, never seen in normal traffic.
        let illegal_fn = ModbusEvent {
            unit_id: 1,
            function: 8,
            address: 10,
            quantity: 1,
        };
        assert_eq!(guard.check(&illegal_fn), GuardVerdict::Anomaly);

        // Write to a register far outside the normal [0,100) range.
        let unseen_reg = ModbusEvent {
            unit_id: 1,
            function: 6,
            address: 50000,
            quantity: 1,
        };
        assert_eq!(guard.check(&unseen_reg), GuardVerdict::Anomaly);

        // A perfectly normal read is not flagged.
        let normal = ModbusEvent {
            unit_id: 1,
            function: 3,
            address: 42,
            quantity: 1,
        };
        assert_eq!(guard.check(&normal), GuardVerdict::Normal);
    }
}
