//! **Confidence calibration** — temperature scaling (Guo et al., *On Calibration
//! of Modern Neural Networks*, ICML 2017).
//!
//! A classifier is *calibrated* when its confidence matches its accuracy: among
//! the predictions made with confidence ≈ p, a fraction ≈ p should be correct.
//! Modern networks are typically **over-confident**. Temperature scaling is a
//! post-hoc fix: divide the logits by a single scalar `T > 0` (found by
//! minimising the negative log-likelihood on a held-out set) before the softmax.
//! It does not change the argmax — so **accuracy is unchanged** — but it
//! recalibrates the probabilities, lowering the *expected calibration error*.
//!
//! Everything here is pure `f32` in a fixed order ⇒ **bit-for-bit deterministic**
//! (the temperature search is a deterministic golden-section minimisation).

/// Softmax of one logit row at temperature `t` (`softmax(logits / t)`),
/// computed with the max-subtraction trick for numerical stability.
fn softmax_t(row: &[f32], t: f32) -> Vec<f32> {
    let inv = 1.0 / t;
    let m = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut exps: Vec<f32> = row.iter().map(|&z| ((z - m) * inv).exp()).collect();
    let s: f32 = exps.iter().sum();
    let inv_s = 1.0 / s;
    for e in exps.iter_mut()
    {
        *e *= inv_s;
    }
    exps
}

/// Mean negative log-likelihood of `(logits, labels)` at temperature `t`.
/// `logits` is `n × classes` row-major; `labels[i]` is the true class of row `i`.
pub fn nll(logits: &[f32], labels: &[usize], n: usize, classes: usize, t: f32) -> f32 {
    assert_eq!(logits.len(), n * classes, "nll: logits size");
    assert_eq!(labels.len(), n, "nll: one label per row");
    let mut acc = 0.0f32;
    for i in 0..n
    {
        let p = softmax_t(&logits[i * classes..(i + 1) * classes], t);
        acc -= (p[labels[i]].max(1e-12)).ln();
    }
    acc / n as f32
}

/// **Expected Calibration Error** (ECE) at temperature `t`: bin predictions by
/// confidence (the max softmax probability) into `n_bins` equal-width bins and
/// average `|accuracy − confidence|` over bins, weighted by bin population.
pub fn expected_calibration_error(
    logits: &[f32],
    labels: &[usize],
    n: usize,
    classes: usize,
    t: f32,
    n_bins: usize,
) -> f32 {
    assert!(n_bins >= 1, "ece: need ≥ 1 bin");
    let mut bin_conf = vec![0.0f32; n_bins];
    let mut bin_acc = vec![0.0f32; n_bins];
    let mut bin_cnt = vec![0usize; n_bins];
    for i in 0..n
    {
        let p = softmax_t(&logits[i * classes..(i + 1) * classes], t);
        let (mut arg, mut conf) = (0usize, p[0]);
        for (c, &pc) in p.iter().enumerate()
        {
            if pc > conf
            {
                conf = pc;
                arg = c;
            }
        }
        // Bin index in [0, n_bins); confidence 1.0 lands in the last bin.
        let b = ((conf * n_bins as f32) as usize).min(n_bins - 1);
        bin_conf[b] += conf;
        bin_acc[b] += if arg == labels[i] { 1.0 } else { 0.0 };
        bin_cnt[b] += 1;
    }
    let mut ece = 0.0f32;
    for b in 0..n_bins
    {
        if bin_cnt[b] == 0
        {
            continue;
        }
        let cnt = bin_cnt[b] as f32;
        let acc = bin_acc[b] / cnt;
        let conf = bin_conf[b] / cnt;
        ece += (cnt / n as f32) * (acc - conf).abs();
    }
    ece
}

/// Find the temperature `T > 0` minimising the NLL on `(logits, labels)` by
/// **golden-section search** on `[t_lo, t_hi]` (deterministic, `iters` steps).
/// The NLL is unimodal in `T` for the over-confident regime this targets.
pub fn temperature_scale(logits: &[f32], labels: &[usize], n: usize, classes: usize) -> f32 {
    let (mut a, mut b) = (0.05f32, 10.0f32);
    let inv_phi = (5.0f32.sqrt() - 1.0) / 2.0; // 1/φ ≈ 0.618
    let mut c = b - (b - a) * inv_phi;
    let mut d = a + (b - a) * inv_phi;
    let mut fc = nll(logits, labels, n, classes, c);
    let mut fd = nll(logits, labels, n, classes, d);
    for _ in 0..60
    {
        if fc < fd
        {
            b = d;
            d = c;
            fd = fc;
            c = b - (b - a) * inv_phi;
            fc = nll(logits, labels, n, classes, c);
        }
        else
        {
            a = c;
            c = d;
            fc = fd;
            d = a + (b - a) * inv_phi;
            fd = nll(logits, labels, n, classes, d);
        }
    }
    0.5 * (a + b)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::PcgEngine;

    /// Build an **over-confident** synthetic logit set: the correct class gets a
    /// large logit (scaled up), so softmax-at-T=1 is far more confident than the
    /// model is accurate (some labels are corrupted).
    fn overconfident(rng: &mut PcgEngine, n: usize, classes: usize) -> (Vec<f32>, Vec<usize>) {
        let mut logits = vec![0f32; n * classes];
        let mut labels = vec![0usize; n];
        for i in 0..n
        {
            let true_c = (rng.float() * classes as f32) as usize % classes;
            for c in 0..classes
            {
                logits[i * classes + c] = 0.5 * rng.float_signed();
            }
            // Make the predicted class very peaked (over-confident).
            logits[i * classes + true_c] += 6.0;
            // Corrupt ~30% of labels so confidence overstates accuracy.
            labels[i] = if rng.float() < 0.3
            {
                (true_c + 1) % classes
            }
            else
            {
                true_c
            };
        }
        (logits, labels)
    }

    /// Temperature scaling lowers the expected calibration error and keeps the
    /// temperature > 1 for over-confident logits — without changing accuracy.
    #[test]
    fn temperature_scaling_reduces_ece() {
        let (n, classes) = (2000usize, 5usize);
        let mut rng = PcgEngine::new(1);
        let (logits, labels) = overconfident(&mut rng, n, classes);

        let ece_before = expected_calibration_error(&logits, &labels, n, classes, 1.0, 15);
        let t = temperature_scale(&logits, &labels, n, classes);
        let ece_after = expected_calibration_error(&logits, &labels, n, classes, t, 15);

        assert!(t > 1.0, "over-confident logits should want T > 1, got {t}");
        assert!(
            ece_after < ece_before,
            "ECE did not improve: {ece_before} -> {ece_after}"
        );
        // Accuracy (argmax) is unchanged by a positive temperature.
        let acc = |tt: f32| -> usize {
            (0..n)
                .filter(|&i| {
                    let p = softmax_t(&logits[i * classes..(i + 1) * classes], tt);
                    let arg = (0..classes).max_by(|&x, &y| p[x].total_cmp(&p[y])).unwrap();
                    arg == labels[i]
                })
                .count()
        };
        assert_eq!(acc(1.0), acc(t), "temperature changed the argmax accuracy");
    }

    /// Deterministic: the search returns a bit-identical temperature each run.
    #[test]
    fn temperature_scale_is_deterministic() {
        let (n, classes) = (500usize, 4usize);
        let mut r1 = PcgEngine::new(7);
        let (l1, y1) = overconfident(&mut r1, n, classes);
        let mut r2 = PcgEngine::new(7);
        let (l2, y2) = overconfident(&mut r2, n, classes);
        let t1 = temperature_scale(&l1, &y1, n, classes);
        let t2 = temperature_scale(&l2, &y2, n, classes);
        assert_eq!(t1.to_bits(), t2.to_bits());
    }
}
