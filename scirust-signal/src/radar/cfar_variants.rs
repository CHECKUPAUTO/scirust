//! Greatest-of, smallest-of and trimmed-mean CFAR detectors.
//!
//! Three sliding-window constant-false-alarm-rate detectors that complement the
//! cell-averaging (CA) and ordered-statistic (OS) variants. Each scans a 1-D
//! power series (a range profile or a range-Doppler cut) with a symmetric
//! reference window — `guard` guard cells then `train` training cells on *each*
//! side of the cell under test (CUT) — estimates the local noise from those two
//! training half-windows, and flags a detection when the CUT exceeds
//! `alpha · noise`. They differ only in how the two half-window statistics are
//! combined:
//!
//! - **greatest-of** ([`go_cfar`]) takes the *larger* half-window mean, so a
//!   clutter edge (a step in the noise floor) inside the window raises the
//!   threshold and suppresses the false alarms that cell-averaging emits there;
//! - **smallest-of** ([`so_cfar`]) takes the *smaller* mean, so an interfering
//!   target sitting in one half-window is ignored (the clean side wins) and a
//!   weak target beside it is still detected;
//! - **trimmed-mean** ([`tm_cfar`]) pools all `2·train` training cells, discards
//!   the `trim_low` smallest and `trim_high` largest, and averages the rest —
//!   censoring a bounded number of interferers while keeping the low variance of
//!   an average in homogeneous noise.
//!
//! All three take the CUT-to-noise scaling `alpha` directly (rather than a design
//! `P_fa`) and return a per-cell boolean mask the length of the input. A cell
//! closer to either edge than `guard + train` has no full window and is never
//! flagged. Real-valued; depends only on `std`.

/// Mean of a non-empty reference half-window.
fn half_mean(cells: &[f64]) -> f64 {
    cells.iter().sum::<f64>() / cells.len() as f64
}

/// **Greatest-of CFAR** (GO-CFAR) over a 1-D power series.
///
/// For each cell under test the noise level is the **larger** of the two
/// training half-window means — the mean of the `train` leading cells (skipping
/// `guard` guard cells) and the mean of the `train` lagging cells — and the cell
/// is detected when `signal[i] > alpha · noise`. Taking the maximum keeps the
/// threshold high when a clutter edge (a step up in the noise floor) falls in one
/// half-window, suppressing the false alarms cell-averaging would raise there, at
/// the cost of some masking when an interfering target occupies a half-window.
///
/// Cells within `guard + train` of either edge have no full window and are
/// returned `false`; a zero `train` or a signal shorter than `2·(guard+train)+1`
/// yields an all-`false` mask.
pub fn go_cfar(signal: &[f64], guard: usize, train: usize, alpha: f64) -> Vec<bool> {
    let n = signal.len();
    let half = guard + train;
    let mut det = vec![false; n];
    if train == 0 || n < 2 * half + 1
    {
        return det;
    }
    for cut in half..n - half
    {
        let lead = half_mean(&signal[cut - half..cut - guard]);
        let lag = half_mean(&signal[cut + guard + 1..cut + half + 1]);
        let noise = lead.max(lag);
        det[cut] = signal[cut] > alpha * noise;
    }
    det
}

/// **Smallest-of CFAR** (SO-CFAR) over a 1-D power series.
///
/// Like [`go_cfar`], but the noise level is the **smaller** of the two training
/// half-window means. Taking the minimum ignores an interfering target (or a
/// clutter edge) that inflates one half-window — the clean side sets the
/// threshold — so a weak target beside a strong interferer is still detected,
/// where greatest-of would mask it. The trade-off is a raised false-alarm rate
/// on the *low* side of a clutter edge.
///
/// Edge and degenerate-parameter handling matches [`go_cfar`].
pub fn so_cfar(signal: &[f64], guard: usize, train: usize, alpha: f64) -> Vec<bool> {
    let n = signal.len();
    let half = guard + train;
    let mut det = vec![false; n];
    if train == 0 || n < 2 * half + 1
    {
        return det;
    }
    for cut in half..n - half
    {
        let lead = half_mean(&signal[cut - half..cut - guard]);
        let lag = half_mean(&signal[cut + guard + 1..cut + half + 1]);
        let noise = lead.min(lag);
        det[cut] = signal[cut] > alpha * noise;
    }
    det
}

/// **Trimmed-mean CFAR** (TM-CFAR) over a 1-D power series.
///
/// The `2·train` training cells (both half-windows, skipping `guard` guard cells
/// per side) are pooled and sorted ascending; the `trim_low` smallest and
/// `trim_high` largest are discarded, and the noise level is the mean of the
/// remaining cells. Trimming the high end censors a bounded number of interfering
/// targets in the window (which would inflate a plain average and mask the CUT),
/// while retaining the low estimator variance of an average in homogeneous noise;
/// `trim_low = trim_high = 0` recovers cell-averaging. The cell is detected when
/// `signal[i] > alpha · noise`.
///
/// Cells without a full window are `false`; so is every cell when `train == 0`,
/// when the signal is shorter than `2·(guard+train)+1`, or when
/// `trim_low + trim_high ≥ 2·train` leaves no cell to average.
pub fn tm_cfar(
    signal: &[f64],
    guard: usize,
    train: usize,
    alpha: f64,
    trim_low: usize,
    trim_high: usize,
) -> Vec<bool> {
    let n = signal.len();
    let half = guard + train;
    let n_ref = 2 * train;
    let mut det = vec![false; n];
    if train == 0 || n < 2 * half + 1 || trim_low + trim_high >= n_ref
    {
        return det;
    }
    let kept = (n_ref - trim_low - trim_high) as f64;
    let mut window: Vec<f64> = Vec::with_capacity(n_ref);
    for cut in half..n - half
    {
        window.clear();
        window.extend_from_slice(&signal[cut - half..cut - guard]);
        window.extend_from_slice(&signal[cut + guard + 1..cut + half + 1]);
        window.sort_by(f64::total_cmp);
        let noise = window[trim_low..n_ref - trim_high].iter().sum::<f64>() / kept;
        det[cut] = signal[cut] > alpha * noise;
    }
    det
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_three_detect_a_lone_spike_and_nothing_else() {
        // Flat unit floor, one strong spike: on the flat floor every half-window
        // mean is 1.0, so max, min and trimmed mean all give noise = 1.0 and the
        // floor (1.0 ≯ α·1.0) never trips; the spike (100 > α) always does.
        let mut signal = vec![1.0_f64; 200];
        signal[100] = 100.0;
        let (guard, train, alpha) = (2usize, 8usize, 5.0_f64);
        for det in [
            go_cfar(&signal, guard, train, alpha),
            so_cfar(&signal, guard, train, alpha),
            tm_cfar(&signal, guard, train, alpha, 0, 0),
        ]
        {
            assert!(det[100], "missed the spike");
            assert_eq!(det.iter().filter(|&&d| d).count(), 1, "spurious detections");
        }
    }

    #[test]
    fn greatest_of_suppresses_a_clutter_edge_false_alarm() {
        // A step up in the noise floor at index 100 (1.0 → 20.0), no target.
        let mut signal = vec![1.0_f64; 200];
        for s in signal.iter_mut().skip(100)
        {
            *s = 20.0;
        }
        let (guard, train, alpha) = (2usize, 8usize, 1.3_f64);
        // At the first clutter cell the leading half-window is all clear and the
        // lagging one all clutter: cell-averaging averages them (10.5) and the
        // 20.0 clutter cell trips the threshold — a false alarm.
        let cut = 100usize;
        let lead: f64 = signal[cut - 10..cut - 2].iter().sum();
        let lag: f64 = signal[cut + 3..cut + 11].iter().sum();
        let ca = (lead + lag) / 16.0;
        assert!((ca - 10.5).abs() < 1e-12);
        assert!(signal[cut] > alpha * ca, "cell-averaging would false-alarm");
        // Greatest-of takes the clutter (20.0) half-window, so the threshold
        // stays above the clutter and nothing false-alarms anywhere.
        let go = go_cfar(&signal, guard, train, alpha);
        assert!(go.iter().all(|&d| !d), "GO-CFAR false-alarmed on clutter");
    }

    #[test]
    fn smallest_of_sees_past_an_interferer_that_greatest_of_masks() {
        // A weak target at 100 with a strong interferer at 105 sitting in the
        // lagging half-window. SO takes the clean leading side (mean 1.0) and
        // detects; GO takes the interferer side (mean 13.375) and masks.
        let mut signal = vec![1.0_f64; 200];
        signal[100] = 8.0;
        signal[105] = 100.0;
        let (guard, train, alpha) = (2usize, 8usize, 5.0_f64);
        let so = so_cfar(&signal, guard, train, alpha);
        let go = go_cfar(&signal, guard, train, alpha);
        assert!(so[100], "SO-CFAR should detect the weak target");
        assert!(!go[100], "GO-CFAR should be masked by the interferer");
        assert!(so[105] && go[105], "both must detect the interferer itself");
    }

    #[test]
    fn trimmed_mean_rejects_interferers_that_the_average_masks() {
        // A target at 100 with three strong interferers in the lagging window.
        let mut signal = vec![1.0_f64; 200];
        signal[100] = 8.0;
        for &i in &[104usize, 105, 106]
        {
            signal[i] = 100.0;
        }
        let (guard, train, alpha) = (2usize, 8usize, 5.0_f64);
        // Plain average (no trim) folds the interferers in (19.5625) and masks.
        let ca = (13.0_f64 + 3.0 * 100.0) / 16.0;
        assert!((ca - 19.5625).abs() < 1e-12);
        assert!(signal[100] < alpha * ca, "an average would mask the target");
        assert!(!tm_cfar(&signal, guard, train, alpha, 0, 0)[100]);
        // Trimming the three highest cells restores a clean 1.0 estimate.
        let tm = tm_cfar(&signal, guard, train, alpha, 0, 3);
        assert!(
            tm[100],
            "TM-CFAR should detect after censoring the interferers"
        );
    }

    #[test]
    fn edge_and_parameter_guards_yield_all_false() {
        // Window wider than the signal.
        assert!(go_cfar(&[1.0; 5], 2, 8, 5.0).iter().all(|&d| !d));
        assert!(so_cfar(&[1.0; 5], 2, 8, 5.0).iter().all(|&d| !d));
        assert!(tm_cfar(&[1.0; 5], 2, 8, 5.0, 0, 3).iter().all(|&d| !d));
        // No training cells.
        assert!(go_cfar(&[1.0; 50], 2, 0, 5.0).iter().all(|&d| !d));
        // Trimming away the entire reference window (8 + 8 == 2·train).
        assert!(tm_cfar(&[1.0; 50], 2, 8, 5.0, 8, 8).iter().all(|&d| !d));
    }

    #[test]
    fn detection_is_invariant_to_a_global_power_scaling() {
        // Both the CUT and its noise estimate scale with the signal, so the
        // threshold test signal[i] > α·noise is unchanged by a global gain.
        let mut signal = vec![1.0_f64; 200];
        signal[100] = 30.0;
        signal[60] = 12.0;
        let (guard, train, alpha) = (2usize, 8usize, 4.0_f64);
        let scaled: Vec<f64> = signal.iter().map(|&s| s * 1000.0).collect();
        assert_eq!(
            go_cfar(&signal, guard, train, alpha),
            go_cfar(&scaled, guard, train, alpha)
        );
        assert_eq!(
            so_cfar(&signal, guard, train, alpha),
            so_cfar(&scaled, guard, train, alpha)
        );
        assert_eq!(
            tm_cfar(&signal, guard, train, alpha, 1, 2),
            tm_cfar(&scaled, guard, train, alpha, 1, 2)
        );
    }

    #[test]
    fn greatest_of_detections_are_a_subset_of_smallest_of() {
        // noise_GO = max(lead, lag) ≥ min(lead, lag) = noise_SO, so the GO
        // threshold is never below the SO threshold: every GO detection is also
        // an SO detection.
        let mut signal = vec![1.0_f64; 200];
        for s in signal.iter_mut().skip(100)
        {
            *s = 5.0;
        }
        signal[60] = 30.0;
        signal[150] = 40.0;
        let (guard, train, alpha) = (2usize, 8usize, 3.0_f64);
        let go = go_cfar(&signal, guard, train, alpha);
        let so = so_cfar(&signal, guard, train, alpha);
        assert!(go.iter().zip(&so).all(|(&g, &s)| !g || s), "GO ⊄ SO");
        assert!(
            go.iter().any(|&d| d),
            "test needs at least one GO detection"
        );
    }
}
