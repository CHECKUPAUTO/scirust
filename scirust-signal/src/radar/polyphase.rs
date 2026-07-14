//! Polyphase and CAZAC pulse-compression codes.
//!
//! Barker codes ([`super::waveform::barker_code`]) are optimal binary codes but
//! exist only up to length 13. **Polyphase codes** lift that ceiling: they use
//! many phase values instead of just `0`/`π`, exist at any length, and — for the
//! Frank and Zadoff-Chu families — have *perfect periodic autocorrelation* (a
//! single impulse, zero sidelobes), the property a pulse-Doppler radar wants when
//! it repeats a code every PRI. The LFM-derived **P3/P4** codes trade a little of
//! that for graceful Doppler tolerance and a low, thumbtack-like response, which
//! together with their noise-like phase progression makes them the canonical
//! **low-probability-of-intercept (LPI)** radar waveforms.
//!
//! - **Frank** — from an `N`-point DFT matrix, length `N²`, phase `2π·i·k/N`.
//! - **P3 / P4** — a sampled linear-FM chirp folded to baseband; any length.
//! - **Zadoff-Chu** — a constant-amplitude zero-autocorrelation (CAZAC) sequence,
//!   perfect at *any* length for a root coprime to it (also used in LTE/5G).
//!
//! Built on the crate's [`Complex`](crate::complex::Complex); dependency-free.

use crate::complex::Complex;
use std::f64::consts::PI;

/// Greatest common divisor (Euclid), for the Zadoff-Chu coprimality test.
fn gcd(mut a: usize, mut b: usize) -> usize {
    while b != 0
    {
        let t = b;
        b = a % b;
        a = t;
    }
    a
}

/// The **Frank code** of order `n`: a length-`N²` polyphase code with element
/// `(i, k)` (row-major, `i, k = 0..n`) at phase `2π·i·k/n`. Has perfect periodic
/// autocorrelation. Empty for `n = 0`.
pub fn frank_code(n: usize) -> Vec<Complex> {
    if n == 0
    {
        return Vec::new();
    }
    let mut code = Vec::with_capacity(n * n);
    for i in 0..n
    {
        for k in 0..n
        {
            let phase = 2.0 * PI * (i * k) as f64 / n as f64;
            code.push(Complex::cis(phase));
        }
    }
    code
}

/// The **P3 code** of the given `length`: a sampled linear-FM chirp with phase
/// `π·n²/L`. A Doppler-tolerant LPI code that exists at any length. Empty for
/// `length = 0`.
pub fn p3_code(length: usize) -> Vec<Complex> {
    if length == 0
    {
        return Vec::new();
    }
    let l = length as f64;
    (0..length)
        .map(|n| {
            let nn = n as f64;
            Complex::cis(PI * nn * nn / l)
        })
        .collect()
}

/// The **P4 code** of the given `length`: the LFM-derived code with phase
/// `π·n²/L − π·n`. Like P3 but with a symmetric frequency sweep, giving lower
/// autocorrelation sidelobes and better tolerance to receiver band-limiting.
/// Empty for `length = 0`.
pub fn p4_code(length: usize) -> Vec<Complex> {
    if length == 0
    {
        return Vec::new();
    }
    let l = length as f64;
    (0..length)
        .map(|n| {
            let nn = n as f64;
            Complex::cis(PI * nn * nn / l - PI * nn)
        })
        .collect()
}

/// The **Zadoff-Chu** sequence of the given `length` and `root`: a CAZAC code
/// `exp(−j·π·u·n·(n + L mod 2)/L)` with perfect periodic autocorrelation for any
/// `root` coprime to `length`. Empty unless `0 < root < length` and
/// `gcd(root, length) = 1`.
pub fn zadoff_chu(length: usize, root: usize) -> Vec<Complex> {
    if length == 0 || root == 0 || root >= length || gcd(root, length) != 1
    {
        return Vec::new();
    }
    let l = length as f64;
    let cf = (length % 2) as f64;
    (0..length)
        .map(|n| {
            let nn = n as f64;
            Complex::cis(-PI * root as f64 * nn * (nn + cf) / l)
        })
        .collect()
}

/// The **periodic autocorrelation** `R[τ] = Σ_n code[n]·conj(code[(n+τ) mod L])`,
/// one value per lag `τ = 0..L`. For a perfect code (Frank, Zadoff-Chu) this is
/// `L` at `τ = 0` and zero at every other lag. Empty for an empty code.
pub fn periodic_autocorrelation(code: &[Complex]) -> Vec<Complex> {
    let l = code.len();
    (0..l)
        .map(|tau| {
            let mut acc = Complex::zero();
            for n in 0..l
            {
                acc += code[n] * code[(n + tau) % l].conj();
            }
            acc
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::super::matched_filter::{cross_correlate, peak_lag};
    use super::super::waveform::barker_code;
    use super::*;

    /// Largest sidelobe magnitude of a periodic autocorrelation (all lags but 0).
    fn max_periodic_sidelobe(code: &[Complex]) -> f64 {
        let r = periodic_autocorrelation(code);
        r.iter().skip(1).map(|c| c.mag()).fold(0.0_f64, f64::max)
    }

    #[test]
    fn frank_code_structure() {
        let n = 4;
        let code = frank_code(n);
        assert_eq!(code.len(), n * n);
        for (idx, c) in code.iter().enumerate()
        {
            assert!((c.mag() - 1.0).abs() < 1e-12);
            let (i, k) = (idx / n, idx % n);
            let expected = Complex::cis(2.0 * PI * (i * k) as f64 / n as f64);
            assert!((c.re - expected.re).abs() < 1e-12 && (c.im - expected.im).abs() < 1e-12);
        }
    }

    #[test]
    fn frank_has_perfect_periodic_autocorrelation() {
        for n in [2usize, 3, 4, 5]
        {
            let code = frank_code(n);
            let r = periodic_autocorrelation(&code);
            assert!((r[0].mag() - (n * n) as f64).abs() < 1e-9);
            for c in r.iter().skip(1)
            {
                assert!(
                    c.mag() < 1e-9,
                    "Frank order {n} periodic sidelobe {}",
                    c.mag()
                );
            }
        }
    }

    #[test]
    fn zadoff_chu_is_a_cazac_sequence() {
        // Perfect periodic autocorrelation at any length, even (12) or prime (7).
        for &(l, u) in &[(7usize, 1usize), (7, 3), (12, 5), (16, 3)]
        {
            let code = zadoff_chu(l, u);
            assert_eq!(code.len(), l);
            assert!(code.iter().all(|c| (c.mag() - 1.0).abs() < 1e-12));
            let r = periodic_autocorrelation(&code);
            assert!((r[0].mag() - l as f64).abs() < 1e-9);
            for c in r.iter().skip(1)
            {
                assert!(c.mag() < 1e-9, "ZC ({l},{u}) periodic sidelobe {}", c.mag());
            }
        }
        // A root sharing a factor with the length is rejected.
        assert!(zadoff_chu(12, 4).is_empty());
        assert!(zadoff_chu(12, 6).is_empty());
    }

    #[test]
    fn p3_and_p4_are_sampled_lfm_phases() {
        let m = 16usize;
        let p3 = p3_code(m);
        let p4 = p4_code(m);
        assert_eq!(p3.len(), m);
        assert_eq!(p4.len(), m);
        let l = m as f64;
        for n in 0..m
        {
            let nn = n as f64;
            assert!((p3[n].mag() - 1.0).abs() < 1e-12);
            assert!((p4[n].mag() - 1.0).abs() < 1e-12);
            let e3 = Complex::cis(PI * nn * nn / l);
            let e4 = Complex::cis(PI * nn * nn / l - PI * nn);
            assert!((p3[n].re - e3.re).abs() < 1e-12 && (p3[n].im - e3.im).abs() < 1e-12);
            assert!((p4[n].re - e4.re).abs() < 1e-12 && (p4[n].im - e4.im).abs() < 1e-12);
        }
    }

    #[test]
    fn aperiodic_autocorrelation_peak_is_the_code_length() {
        let codes = [frank_code(4), p3_code(16), p4_code(16), zadoff_chu(13, 2)];
        for code in &codes
        {
            let r = cross_correlate(code, code);
            // Zero-lag (full overlap) term equals the energy = length (unit mags).
            assert!((r[code.len() - 1].mag() - code.len() as f64).abs() < 1e-9);
            assert_eq!(peak_lag(&r, code.len()), Some(0));
        }
    }

    #[test]
    fn polyphase_periodic_autocorrelation_beats_barker() {
        // Frank (length 16) is periodically perfect; Barker-13, the best binary
        // code, is not — its periodic autocorrelation carries a real sidelobe.
        let frank = frank_code(4);
        assert!(max_periodic_sidelobe(&frank) < 1e-9);
        let barker: Vec<Complex> = barker_code(13)
            .unwrap()
            .iter()
            .map(|&x| Complex::new(x, 0.0))
            .collect();
        assert!(
            max_periodic_sidelobe(&barker) > 0.5,
            "Barker-13 periodic sidelobe unexpectedly small"
        );
    }

    #[test]
    fn degenerate_inputs_are_safe() {
        assert!(frank_code(0).is_empty());
        assert!(p3_code(0).is_empty());
        assert!(p4_code(0).is_empty());
        assert!(zadoff_chu(0, 1).is_empty());
        assert!(zadoff_chu(5, 0).is_empty());
        assert!(zadoff_chu(5, 5).is_empty()); // root not < length
        assert!(periodic_autocorrelation(&[]).is_empty());
    }
}
