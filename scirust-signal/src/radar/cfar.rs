//! Constant-false-alarm-rate (CFAR) detection.
//!
//! Adaptive thresholding that holds the false-alarm probability fixed as the
//! local noise / clutter level varies, by estimating the noise for each cell
//! under test (CUT) from the surrounding reference cells and scaling it by a
//! factor chosen for the desired `P_fa`. Two variants: **cell-averaging**
//! (CA-CFAR — the mean of the reference window; optimal in homogeneous noise)
//! and **ordered-statistic** (OS-CFAR — a rank of the window, robust to clutter
//! edges and interfering targets that would bias the average). Both act on a
//! power (magnitude-squared) series such as a range profile or a range-Doppler
//! cut, and return a per-cell detection mask.

/// The CA-CFAR threshold scaling `α = N·(P_fa^{−1/N} − 1)` for `n_ref` total
/// reference cells targeting false-alarm probability `pfa`. With this scaling
/// the achieved `P_fa` is exactly `pfa` for exponentially distributed noise
/// power, independent of the (unknown) noise level — the CFAR property
/// `(1 + α/N)^{−N} = P_fa`.
pub fn ca_cfar_alpha(n_ref: usize, pfa: f64) -> f64 {
    let n = n_ref as f64;
    n * (pfa.powf(-1.0 / n) - 1.0)
}

/// Cell-averaging CFAR over a 1-D power series.
///
/// For each cell under test the noise level is the mean of `num_train`
/// reference cells on each side, skipping `num_guard` guard cells per side so a
/// target's own energy does not bias its estimate. A detection is flagged when
/// the CUT exceeds `α · mean(reference)` with `α` from [`ca_cfar_alpha`]
/// (`N = 2·num_train`). Cells within `num_train + num_guard` of either edge
/// (no full window) are never flagged.
pub fn ca_cfar(power: &[f64], num_train: usize, num_guard: usize, pfa: f64) -> Vec<bool> {
    let n = power.len();
    let half = num_train + num_guard;
    let mut det = vec![false; n];
    if num_train == 0 || pfa <= 0.0 || pfa >= 1.0 || n < 2 * half + 1
    {
        return det;
    }
    let n_ref = 2 * num_train;
    let alpha = ca_cfar_alpha(n_ref, pfa);
    for cut in half..n - half
    {
        let lead: f64 = power[cut - half..cut - num_guard].iter().sum();
        let trail: f64 = power[cut + num_guard + 1..cut + half + 1].iter().sum();
        let noise = (lead + trail) / n_ref as f64;
        det[cut] = power[cut] > alpha * noise;
    }
    det
}

/// The OS-CFAR threshold scaling for the `k`-th smallest of `n_ref` reference
/// cells targeting `pfa`, found by bisection on the strictly decreasing
/// relation `P_fa(α) = ∏_{i=0}^{k−1} (N − i)/(N − i + α)` (`N = n_ref`, `k`
/// 1-based). Returns `0.0` for an out-of-range `k`.
pub fn os_cfar_alpha(n_ref: usize, k: usize, pfa: f64) -> f64 {
    if k == 0 || k > n_ref
    {
        return 0.0;
    }
    let pfa_of = |alpha: f64| -> f64 {
        (0..k)
            .map(|i| {
                let ni = (n_ref - i) as f64;
                ni / (ni + alpha)
            })
            .product::<f64>()
    };
    let (mut lo, mut hi) = (0.0_f64, 1.0_f64);
    while pfa_of(hi) > pfa && hi < 1e12
    {
        hi *= 2.0;
    }
    for _ in 0..100
    {
        let mid = 0.5 * (lo + hi);
        if pfa_of(mid) > pfa
        {
            lo = mid;
        }
        else
        {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
}

/// Ordered-statistic CFAR over a 1-D power series.
///
/// Like [`ca_cfar`], but the noise estimate is the `k`-th smallest of the
/// `2·num_train` reference cells (1-based `k`) rather than their mean, so a
/// small number of interfering targets or a clutter edge in the window does not
/// inflate the threshold. Scaled by [`os_cfar_alpha`].
pub fn os_cfar(power: &[f64], num_train: usize, num_guard: usize, k: usize, pfa: f64) -> Vec<bool> {
    let n = power.len();
    let half = num_train + num_guard;
    let n_ref = 2 * num_train;
    let mut det = vec![false; n];
    if num_train == 0 || k == 0 || k > n_ref || pfa <= 0.0 || pfa >= 1.0 || n < 2 * half + 1
    {
        return det;
    }
    let alpha = os_cfar_alpha(n_ref, k, pfa);
    let mut window: Vec<f64> = Vec::with_capacity(n_ref);
    for cut in half..n - half
    {
        window.clear();
        window.extend_from_slice(&power[cut - half..cut - num_guard]);
        window.extend_from_slice(&power[cut + num_guard + 1..cut + half + 1]);
        window.sort_by(f64::total_cmp);
        det[cut] = power[cut] > alpha * window[k - 1];
    }
    det
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A deterministic LCG producing unit-mean exponential noise (the model
    /// CFAR is designed for), so the statistical false-alarm test is
    /// reproducible without an ambient RNG.
    struct Lcg(u64);
    impl Lcg {
        fn uniform(&mut self) -> f64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((self.0 >> 11) as f64 + 1.0) / ((1u64 << 53) as f64 + 1.0)
        }
        fn exponential(&mut self) -> f64 {
            -self.uniform().ln()
        }
    }

    #[test]
    fn ca_cfar_alpha_matches_the_closed_form_and_cfar_identity() {
        let a = ca_cfar_alpha(32, 0.01);
        assert!((a - 32.0 * (0.01_f64.powf(-1.0 / 32.0) - 1.0)).abs() < 1e-12);
        // The CFAR identity: (1 + α/N)^{−N} = P_fa.
        assert!(((1.0 + a / 32.0).powf(-32.0) - 0.01).abs() < 1e-12);
    }

    #[test]
    fn ca_cfar_holds_the_design_false_alarm_rate_on_exponential_noise() {
        let (num_train, num_guard, pfa) = (16usize, 2usize, 0.05);
        let mut rng = Lcg(0x00C0_FFEE);
        let n = 20_000;
        let power: Vec<f64> = (0..n).map(|_| rng.exponential()).collect();
        let det = ca_cfar(&power, num_train, num_guard, pfa);
        let tested = n - 2 * (num_train + num_guard);
        let alarms = det.iter().filter(|&&d| d).count();
        let empirical = alarms as f64 / tested as f64;
        // CA-CFAR achieves exactly P_fa in expectation; empirical is within a
        // few σ over 20k cells.
        assert!(
            (empirical - pfa).abs() < 0.015,
            "empirical P_fa {empirical} vs {pfa}"
        );
    }

    #[test]
    fn ca_cfar_detects_a_target_without_flooding_false_alarms() {
        let mut power = vec![1.0; 200];
        power[100] = 100.0;
        let det = ca_cfar(&power, 16, 2, 0.01);
        assert!(det[100], "missed the target");
        // The flat floor stays below threshold, and the target does not leak
        // into neighbours' estimates enough to trip them: exactly one detection.
        assert_eq!(det.iter().filter(|&&d| d).count(), 1);
    }

    #[test]
    fn os_cfar_alpha_inverts_the_false_alarm_formula() {
        let (n_ref, k, pfa) = (32usize, 12usize, 0.01);
        let alpha = os_cfar_alpha(n_ref, k, pfa);
        let recovered: f64 = (0..k)
            .map(|i| {
                let ni = (n_ref - i) as f64;
                ni / (ni + alpha)
            })
            .product();
        assert!(
            (recovered - pfa).abs() < 1e-9,
            "α does not invert P_fa: {recovered}"
        );
    }

    #[test]
    fn os_cfar_survives_an_interferer_that_masks_ca_cfar() {
        // A weak target at 100 and a strong interferer at 112 (inside the CFAR
        // reference window). CA-CFAR averages the interferer in and raises the
        // threshold, masking the weak target; OS-CFAR (low rank) ignores it.
        let mut power = vec![1.0; 200];
        power[100] = 20.0;
        power[112] = 500.0;
        let ca = ca_cfar(&power, 16, 2, 0.01);
        let os = os_cfar(&power, 16, 2, 12, 0.01);
        assert!(!ca[100], "CA-CFAR should be masked here");
        assert!(os[100], "OS-CFAR should still detect the weak target");
        assert!(ca[112] && os[112], "both must detect the strong interferer");
    }

    #[test]
    fn cfar_edge_and_parameter_guards() {
        assert!(ca_cfar(&[1.0; 5], 16, 2, 0.01).iter().all(|&d| !d));
        assert!(os_cfar(&[1.0; 5], 16, 2, 12, 0.01).iter().all(|&d| !d));
        assert!(ca_cfar(&[1.0; 200], 16, 2, 0.0).iter().all(|&d| !d));
        assert!(ca_cfar(&[1.0; 200], 16, 2, 1.5).iter().all(|&d| !d));
        assert!(os_cfar(&[1.0; 200], 16, 2, 0, 0.01).iter().all(|&d| !d));
        assert!(os_cfar(&[1.0; 200], 16, 2, 99, 0.01).iter().all(|&d| !d));
    }
}
