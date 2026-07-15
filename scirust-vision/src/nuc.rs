//! Two-point non-uniformity correction (NUC) for an infrared focal-plane array.
//!
//! Every detector in a staring focal-plane array (FPA) responds slightly
//! differently: pixel `i` reads `raw = gain_true·scene + offset_true`, so a
//! perfectly uniform scene still comes out grainy. That fixed-pattern noise
//! (FPN) is the dominant artefact of an uncorrected thermal imager. The classic
//! remedy is a **two-point** calibration: image two spatially uniform blackbody
//! sources at known radiance levels — a *cold* level and a *hot* level — and use
//! the two responses per pixel to invert the affine law. Because a straight line
//! is fixed by two points, the recovered per-pixel `(gain, offset)` maps the raw
//! response back onto the true scene, so any subsequent frame reads uniformly.
//!
//! For pixel `i`, given the cold/hot raw counts `cᵢ`, `hᵢ` at true levels
//! `L_c`, `L_h`, the correction coefficients are
//! `gainᵢ = (L_h − L_c)/(hᵢ − cᵢ)` and `offsetᵢ = L_c − gainᵢ·cᵢ`, and a frame is
//! corrected pixel-wise by `gainᵢ·rawᵢ + offsetᵢ`. A pixel whose cold and hot
//! counts are equal (`hᵢ − cᵢ = 0`) carries no scene information — a *dead*
//! pixel — and the calibration is rejected. The residual FPN of a frame is
//! summarised by its spatial standard deviation. Depends only on `std`.

/// Per-pixel **two-point NUC coefficients** from two uniform calibration frames.
///
/// `cold_frame` / `hot_frame` are the raw FPA responses (row-major, any length)
/// to spatially uniform sources at true levels `cold_level` and `hot_level`.
/// Returns `(gain, offset)` with, per pixel `i`,
/// `gainᵢ = (hot_level − cold_level)/(hotᵢ − coldᵢ)` and
/// `offsetᵢ = cold_level − gainᵢ·coldᵢ`, so that [`apply_nuc`] maps `coldᵢ ↦
/// cold_level` and `hotᵢ ↦ hot_level` exactly. Returns `None` if the two frames
/// differ in length, or if any pixel has `hotᵢ − coldᵢ == 0` (a dead pixel, which
/// carries no scene information).
pub fn two_point_coeffs(
    cold_frame: &[f64],
    hot_frame: &[f64],
    cold_level: f64,
    hot_level: f64,
) -> Option<(Vec<f64>, Vec<f64>)> {
    if cold_frame.len() != hot_frame.len()
    {
        return None;
    }
    let span = hot_level - cold_level;
    let mut gain = Vec::with_capacity(cold_frame.len());
    let mut offset = Vec::with_capacity(cold_frame.len());
    for (&cold, &hot) in cold_frame.iter().zip(hot_frame.iter())
    {
        let delta = hot - cold;
        if delta == 0.0
        {
            return None; // dead pixel: cold and hot responses coincide
        }
        let g = span / delta;
        gain.push(g);
        offset.push(cold_level - g * cold);
    }
    Some((gain, offset))
}

/// Apply per-pixel NUC coefficients to a `frame`: `gainᵢ·frameᵢ + offsetᵢ`.
///
/// The output length is the minimum of the three input lengths, so mismatched
/// slices are simply truncated rather than panicking. With coefficients from
/// [`two_point_coeffs`], this inverts the affine detector response and flattens
/// the fixed-pattern noise of the frame.
pub fn apply_nuc(frame: &[f64], gain: &[f64], offset: &[f64]) -> Vec<f64> {
    let n = frame.len().min(gain.len()).min(offset.len());
    (0..n).map(|i| gain[i] * frame[i] + offset[i]).collect()
}

/// The **fixed-pattern noise** of a frame: its spatial standard deviation.
///
/// This is the population standard deviation over all pixels — a scalar residual
/// metric that is near zero for a well-corrected uniform scene and grows with
/// pixel-to-pixel non-uniformity. Returns `0.0` for an empty frame.
pub fn fixed_pattern_noise(frame: &[f64]) -> f64 {
    let n = frame.len();
    if n == 0
    {
        return 0.0;
    }
    let mean = frame.iter().sum::<f64>() / n as f64;
    let var = frame.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n as f64;
    var.sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A synthetic FPA: per-pixel true gain/offset and the raw response to a
    /// uniform scene at `level`, `rawᵢ = gain_trueᵢ·level + offset_trueᵢ`.
    fn raw_response(gain_true: &[f64], offset_true: &[f64], level: f64) -> Vec<f64> {
        gain_true
            .iter()
            .zip(offset_true.iter())
            .map(|(&g, &o)| g * level + o)
            .collect()
    }

    #[test]
    fn calibration_frames_read_uniform() {
        // The two calibration frames must map back onto their true levels exactly,
        // pixel by pixel, so their post-NUC fixed-pattern noise is ~0.
        let gain_true = [0.90_f64, 1.05, 1.20, 0.95];
        let offset_true = [10.0_f64, -5.0, 3.0, 7.0];
        let (cold_level, hot_level) = (300.0_f64, 500.0);
        let cold = raw_response(&gain_true, &offset_true, cold_level);
        let hot = raw_response(&gain_true, &offset_true, hot_level);

        let (gain, offset) = two_point_coeffs(&cold, &hot, cold_level, hot_level).unwrap();
        let corrected_cold = apply_nuc(&cold, &gain, &offset);
        let corrected_hot = apply_nuc(&hot, &gain, &offset);
        for &v in &corrected_cold
        {
            assert!((v - cold_level).abs() < 1e-9, "{v}");
        }
        for &v in &corrected_hot
        {
            assert!((v - hot_level).abs() < 1e-9, "{v}");
        }
        assert!(fixed_pattern_noise(&corrected_cold) < 1e-9);
        assert!(fixed_pattern_noise(&corrected_hot) < 1e-9);
    }

    #[test]
    fn intermediate_scene_is_flattened() {
        // A THIRD uniform scene, never used for calibration, must also flatten:
        // its raw FPN is large, its corrected FPN is negligible, and it reads
        // the true intermediate level.
        let gain_true = [0.90_f64, 1.05, 1.20, 0.95, 1.10];
        let offset_true = [10.0_f64, -5.0, 3.0, 7.0, -2.0];
        let (cold_level, hot_level) = (300.0_f64, 500.0);
        let cold = raw_response(&gain_true, &offset_true, cold_level);
        let hot = raw_response(&gain_true, &offset_true, hot_level);
        let (gain, offset) = two_point_coeffs(&cold, &hot, cold_level, hot_level).unwrap();

        let mid_level = 380.0_f64;
        let raw_mid = raw_response(&gain_true, &offset_true, mid_level);
        let corrected_mid = apply_nuc(&raw_mid, &gain, &offset);
        let raw_fpn = fixed_pattern_noise(&raw_mid);
        let corrected_fpn = fixed_pattern_noise(&corrected_mid);
        assert!(raw_fpn > 1.0, "raw FPN should be substantial: {raw_fpn}");
        assert!(
            corrected_fpn < 1e-9,
            "corrected FPN should vanish: {corrected_fpn}"
        );
        assert!(corrected_fpn < raw_fpn * 1e-6);
        for &v in &corrected_mid
        {
            assert!((v - mid_level).abs() < 1e-9, "{v}");
        }
    }

    #[test]
    fn apply_nuc_inverts_linear_response() {
        // apply_nuc is exactly the affine map gainᵢ·frameᵢ + offsetᵢ.
        let frame = [2.0, 4.0, 8.0];
        let gain = [1.5, 0.5, 3.0];
        let offset = [1.0, -2.0, 0.5];
        let out = apply_nuc(&frame, &gain, &offset);
        assert_eq!(out, vec![1.5 * 2.0 + 1.0, 0.5 * 4.0 - 2.0, 3.0 * 8.0 + 0.5]);
    }

    #[test]
    fn coeffs_match_closed_form() {
        // gainᵢ = (L_h−L_c)/(hᵢ−cᵢ); offsetᵢ = L_c − gainᵢ·cᵢ.
        let cold = [100.0, 120.0];
        let hot = [300.0, 320.0]; // hᵢ − cᵢ = 200 for both
        let (cold_level, hot_level) = (0.0_f64, 100.0);
        let (gain, offset) = two_point_coeffs(&cold, &hot, cold_level, hot_level).unwrap();
        assert!((gain[0] - 0.5).abs() < 1e-12); // 100/200
        assert!((gain[1] - 0.5).abs() < 1e-12);
        assert!((offset[0] - (-50.0)).abs() < 1e-12); // 0 − 0.5·100
        assert!((offset[1] - (-60.0)).abs() < 1e-12); // 0 − 0.5·120
    }

    #[test]
    fn fixed_pattern_noise_closed_form() {
        // Population std of [1,2,3,4,5]: mean 3, variance 2, std √2. Empty ⇒ 0;
        // a perfectly uniform frame ⇒ 0.
        let sqrt2 = 2.0_f64.sqrt();
        assert!((fixed_pattern_noise(&[1.0, 2.0, 3.0, 4.0, 5.0]) - sqrt2).abs() < 1e-12);
        assert_eq!(fixed_pattern_noise(&[]), 0.0);
        assert_eq!(fixed_pattern_noise(&[7.0, 7.0, 7.0]), 0.0);
    }

    #[test]
    fn guards_reject_mismatch_and_dead_pixels() {
        // Mismatched calibration-frame lengths ⇒ None.
        assert!(two_point_coeffs(&[1.0, 2.0], &[3.0], 0.0, 1.0).is_none());
        // A dead pixel (hᵢ − cᵢ = 0) ⇒ None.
        assert!(two_point_coeffs(&[1.0, 5.0], &[3.0, 5.0], 0.0, 1.0).is_none());
        // A healthy pair still calibrates.
        assert!(two_point_coeffs(&[1.0, 5.0], &[3.0, 9.0], 0.0, 1.0).is_some());
    }

    #[test]
    fn apply_nuc_length_is_min_of_inputs() {
        // Mismatched slices truncate to the shortest length rather than panicking.
        let out = apply_nuc(&[1.0, 2.0, 3.0], &[2.0, 2.0], &[0.0, 1.0, 1.0, 1.0]);
        assert_eq!(out, vec![2.0, 5.0]);
    }
}
