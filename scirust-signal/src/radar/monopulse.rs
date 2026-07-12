//! Amplitude-comparison monopulse angle estimation.
//!
//! A scanning beam finds a target's angle by peaking up the return, which is
//! slow and only as precise as the beamwidth. A **monopulse** tracker instead
//! measures the off-boresight angle from a *single* dwell: two beams squinted to
//! either side of boresight are combined into a **sum** channel `Σ = A + B` and a
//! **difference** channel `Δ = A − B`, and the **monopulse ratio** `Δ/Σ` encodes
//! which side of boresight the target is on and how far — a sharp, signed error
//! signal that is zero on boresight and (near boresight) linear in the angle,
//! giving angle accuracy far finer than the beamwidth.
//!
//! For Gaussian beams of width `σ` squinted by `±θ_s`, the ratio is exactly
//! `tanh(θ·θ_s/σ²)`, with monopulse slope `k_m = θ_s/σ²` at boresight; inverting
//! it recovers the angle. Dependency-free.

/// The **voltage gain** of a Gaussian beam of width `sigma` pointed at
/// `boresight`, evaluated at angle `theta`: `e^{−(θ−θ₀)²/2σ²}`.
pub fn beam_voltage(theta: f64, boresight: f64, sigma: f64) -> f64 {
    if sigma <= 0.0
    {
        return 0.0;
    }
    let d = theta - boresight;
    (-d * d / (2.0 * sigma * sigma)).exp()
}

/// The **monopulse ratio** `Δ/Σ = (A − B)/(A + B)` for a target at `theta`, from
/// two Gaussian beams of width `sigma` squinted to `±squint`. Zero on boresight,
/// odd in `theta`, and bounded to `(−1, 1)`; equal to `tanh(θ·θ_s/σ²)`.
pub fn monopulse_ratio(theta: f64, squint: f64, sigma: f64) -> f64 {
    let a = beam_voltage(theta, squint, sigma);
    let b = beam_voltage(theta, -squint, sigma);
    let sum = a + b;
    if sum == 0.0
    {
        return 0.0;
    }
    (a - b) / sum
}

/// The **monopulse slope** `k_m = θ_s/σ²` — the discriminator gain `d(Δ/Σ)/dθ` at
/// boresight, where the ratio is locally `k_m·θ`.
pub fn monopulse_slope(squint: f64, sigma: f64) -> f64 {
    if sigma <= 0.0
    {
        return 0.0;
    }
    squint / (sigma * sigma)
}

/// Recover the off-boresight **angle** from a measured monopulse `ratio`,
/// inverting `ratio = tanh(θ·θ_s/σ²)`: `θ = atanh(ratio)·σ²/θ_s`. The ratio is
/// clamped just inside `(−1, 1)`.
pub fn estimate_angle(ratio: f64, squint: f64, sigma: f64) -> f64 {
    if squint == 0.0
    {
        return 0.0;
    }
    let r = ratio.clamp(-1.0 + 1e-15, 1.0 - 1e-15);
    r.atanh() * sigma * sigma / squint
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ratio_is_zero_on_boresight_and_odd() {
        let (squint, sigma) = (0.05, 0.1);
        assert!(monopulse_ratio(0.0, squint, sigma).abs() < 1e-15);
        // Odd in angle: r(−θ) = −r(θ), so the sign gives the side of boresight.
        let r = monopulse_ratio(0.03, squint, sigma);
        assert!((monopulse_ratio(-0.03, squint, sigma) + r).abs() < 1e-12);
        assert!(r > 0.0);
    }

    #[test]
    fn ratio_is_monotonic_and_bounded() {
        let (squint, sigma) = (0.05, 0.1);
        let mut last = f64::NEG_INFINITY;
        for i in -10..=10
        {
            let theta = i as f64 * 0.01;
            let r = monopulse_ratio(theta, squint, sigma);
            assert!(r > last, "ratio must increase with angle");
            assert!(r.abs() < 1.0);
            last = r;
        }
    }

    #[test]
    fn ratio_equals_the_tanh_closed_form() {
        let (squint, sigma) = (0.04_f64, 0.12_f64);
        for &theta in &[-0.08, -0.02, 0.01, 0.05, 0.1]
        {
            let closed = (theta * squint / (sigma * sigma)).tanh();
            assert!((monopulse_ratio(theta, squint, sigma) - closed).abs() < 1e-12);
        }
    }

    #[test]
    fn estimate_inverts_the_ratio_exactly() {
        // The headline: forming the ratio for a known angle and inverting it
        // recovers the angle, well beyond the beamwidth's resolution.
        let (squint, sigma) = (0.05, 0.15);
        for &theta in &[-0.09, -0.03, 0.0, 0.02, 0.07]
        {
            let r = monopulse_ratio(theta, squint, sigma);
            let est = estimate_angle(r, squint, sigma);
            assert!((est - theta).abs() < 1e-9, "{est} vs {theta}");
        }
    }

    #[test]
    fn slope_is_the_boresight_linearisation() {
        let (squint, sigma) = (0.05, 0.15);
        let km = monopulse_slope(squint, sigma);
        assert!((km - squint / (sigma * sigma)).abs() < 1e-12);
        // Near boresight the ratio ≈ k_m·θ, and the linear estimate ratio/k_m
        // recovers a small angle.
        let theta = 0.005;
        let r = monopulse_ratio(theta, squint, sigma);
        assert!(
            (r - km * theta).abs() / (km * theta) < 1e-3,
            "linear near boresight"
        );
        assert!((r / km - theta).abs() / theta < 1e-3);
    }

    #[test]
    fn degenerate_inputs_are_safe() {
        assert_eq!(beam_voltage(0.1, 0.0, 0.0), 0.0);
        assert_eq!(estimate_angle(0.5, 0.0, 0.1), 0.0);
        // A ratio at the ±1 rail clamps to a large but finite angle.
        assert!(estimate_angle(1.0, 0.05, 0.1).is_finite());
    }
}
