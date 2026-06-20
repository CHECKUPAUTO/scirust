//! Voltage-event detection (IEC 61000-4-30): sags, swells, interruptions.
//!
//! A one-cycle sliding RMS, compared to the declared nominal, classifies each
//! window as normal, sag (dip), swell, or interruption — the core power-quality
//! events utilities must log. Deterministic.

use serde::{Deserialize, Serialize};

/// Voltage-event class for one measurement window.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VoltageEvent {
    /// RMS within `[0.9, 1.1]·nominal`.
    Normal,
    /// Dip: `[0.1, 0.9)·nominal`.
    Sag,
    /// `> 1.1·nominal`.
    Swell,
    /// `< 0.1·nominal`.
    Interruption,
}

/// Classify an RMS voltage against the nominal.
pub fn classify_voltage(rms: f64, nominal: f64) -> VoltageEvent {
    if nominal <= 0.0
    {
        return VoltageEvent::Normal;
    }
    let r = rms / nominal;
    if r < 0.1
    {
        VoltageEvent::Interruption
    }
    else if r < 0.9
    {
        VoltageEvent::Sag
    }
    else if r > 1.1
    {
        VoltageEvent::Swell
    }
    else
    {
        VoltageEvent::Normal
    }
}

/// One-cycle sliding RMS of `signal`, `samples_per_cycle` wide.
pub fn cycle_rms(signal: &[f64], samples_per_cycle: usize) -> Vec<f64> {
    let w = samples_per_cycle.max(1);
    let n = signal.len();
    if n < w
    {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(n - w + 1);
    let mut sq_sum: f64 = signal[..w].iter().map(|x| x * x).sum();
    out.push((sq_sum / w as f64).sqrt());
    for i in w..n
    {
        sq_sum += signal[i] * signal[i] - signal[i - w] * signal[i - w];
        out.push((sq_sum / w as f64).sqrt());
    }
    out
}

/// A detected voltage event over a contiguous span of RMS windows.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct EventSpan {
    pub event: VoltageEvent,
    /// First RMS-window index of the event.
    pub start: usize,
    /// One past the last RMS-window index.
    pub end: usize,
    /// Extreme RMS-to-nominal ratio reached during the event (depth/peak).
    pub extreme_ratio: f64,
}

/// Detect contiguous non-normal voltage events from a waveform.
pub fn detect_events(signal: &[f64], nominal: f64, samples_per_cycle: usize) -> Vec<EventSpan> {
    let rms = cycle_rms(signal, samples_per_cycle);
    let mut events = Vec::new();
    let mut i = 0;
    while i < rms.len()
    {
        let cls = classify_voltage(rms[i], nominal);
        if cls == VoltageEvent::Normal
        {
            i += 1;
            continue;
        }
        let start = i;
        let mut extreme = rms[i] / nominal;
        while i < rms.len() && classify_voltage(rms[i], nominal) == cls
        {
            let ratio = rms[i] / nominal;
            // Track the most extreme deviation from 1.0.
            if (ratio - 1.0).abs() > (extreme - 1.0).abs()
            {
                extreme = ratio;
            }
            i += 1;
        }
        events.push(EventSpan {
            event: cls,
            start,
            end: i,
            extreme_ratio: extreme,
        });
    }
    events
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::f64::consts::PI;

    fn make_wave(n: usize, spc: usize, amp_at: impl Fn(usize) -> f64) -> Vec<f64> {
        (0..n)
            .map(|i| amp_at(i) * (2.0 * PI * i as f64 / spc as f64).sin())
            .collect()
    }

    #[test]
    fn classifies_levels() {
        assert_eq!(classify_voltage(1.0, 1.0), VoltageEvent::Normal);
        assert_eq!(classify_voltage(0.5, 1.0), VoltageEvent::Sag);
        assert_eq!(classify_voltage(1.3, 1.0), VoltageEvent::Swell);
        assert_eq!(classify_voltage(0.02, 1.0), VoltageEvent::Interruption);
    }

    #[test]
    fn detects_a_sag_event() {
        let spc = 64; // samples per cycle
        let n = spc * 40;
        // Nominal peak 1.0 -> RMS ~0.707. Drop amplitude to 0.5 for cycles 10..20.
        let nominal_rms = 1.0 / 2.0_f64.sqrt();
        let wave = make_wave(n, spc, |i| {
            let cycle = i / spc;
            if (10..20).contains(&cycle) { 0.5 } else { 1.0 }
        });
        let events = detect_events(&wave, nominal_rms, spc);
        let sags: Vec<_> = events
            .iter()
            .filter(|e| e.event == VoltageEvent::Sag)
            .collect();
        assert_eq!(sags.len(), 1, "events {events:?}");
        assert!(
            (sags[0].extreme_ratio - 0.5).abs() < 0.05,
            "depth {}",
            sags[0].extreme_ratio
        );
    }

    #[test]
    fn clean_supply_has_no_events() {
        let spc = 64;
        let wave = make_wave(spc * 20, spc, |_| 1.0);
        let nominal_rms = 1.0 / 2.0_f64.sqrt();
        assert!(detect_events(&wave, nominal_rms, spc).is_empty());
    }
}
