//! Moving-target indication (MTI): pulse-to-pulse cancellers that reject
//! stationary (zero-Doppler) clutter while passing moving targets, by
//! high-pass filtering along slow-time. Cascaded first differences give the
//! classic 2-pulse (order 1), 3-pulse (order 2), … cancellers with binomial
//! weights and an exact null at DC.

use crate::complex::Complex;

/// An `order`-pulse MTI canceller applied along slow-time: `order` cascaded
/// first differences (`y[m] = x[m] − x[m−1]`), i.e. binomial weights
/// (`[1, −1]`, `[1, −2, 1]`, …).
///
/// Its DC response is exactly zero, so a stationary return is removed while a
/// moving target at normalized Doppler `f` passes with gain
/// `|1 − e^{−j2πf}|^order`. The output is `order` samples shorter than the
/// input; `order = 0` returns a copy, and an input too short to difference
/// `order` times returns an empty vector.
pub fn mti_canceller(slow_time: &[Complex], order: usize) -> Vec<Complex> {
    let mut cur = slow_time.to_vec();
    for _ in 0..order
    {
        if cur.len() < 2
        {
            return Vec::new();
        }
        cur = cur.windows(2).map(|w| w[1] - w[0]).collect();
    }
    cur
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn mti_nulls_stationary_clutter_exactly() {
        let clutter = vec![Complex::new(5.0, -3.0); 16];
        for order in 1..=3
        {
            let out = mti_canceller(&clutter, order);
            assert_eq!(out.len(), 16 - order);
            assert!(
                out.iter().all(|c| c.mag() < 1e-12),
                "clutter leaked at order {order}"
            );
        }
    }

    #[test]
    fn mti_passes_a_moving_tone_with_the_binomial_gain() {
        let (n, f) = (32usize, 0.25_f64);
        let tone: Vec<Complex> = (0..n)
            .map(|m| Complex::cis(2.0 * PI * f * m as f64))
            .collect();
        // 2-pulse: |y| = |1 − e^{−j2πf}| = 2|sin(πf)|, constant across the tone.
        let g1 = 2.0 * (PI * f).sin().abs();
        for c in &mti_canceller(&tone, 1)
        {
            assert!((c.mag() - g1).abs() < 1e-9);
        }
        // 3-pulse: the gain is squared.
        for c in &mti_canceller(&tone, 2)
        {
            assert!((c.mag() - g1 * g1).abs() < 1e-9);
        }
    }

    #[test]
    fn mti_removes_clutter_but_keeps_the_moving_target() {
        let (n, f) = (32usize, 0.3_f64);
        let sig: Vec<Complex> = (0..n)
            .map(|m| Complex::new(10.0, 0.0) + Complex::cis(2.0 * PI * f * m as f64))
            .collect();
        // The DC clutter (amplitude 10) vanishes; only the moving tone's
        // difference (gain 2|sin πf|) survives.
        let g = 2.0 * (PI * f).sin().abs();
        for c in &mti_canceller(&sig, 1)
        {
            assert!((c.mag() - g).abs() < 1e-9);
        }
    }

    #[test]
    fn mti_edge_cases() {
        assert_eq!(mti_canceller(&[Complex::new(1.0, 0.0); 4], 0).len(), 4);
        assert!(mti_canceller(&[Complex::new(1.0, 0.0)], 1).is_empty());
        assert!(mti_canceller(&[], 1).is_empty());
    }
}
