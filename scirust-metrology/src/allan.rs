//! Allan variance / deviation — sensor and clock stability.
//!
//! The Allan deviation `σ_y(τ)` characterises how a sensor's noise averages down
//! with integration time `τ`: white noise falls as `τ^{-1/2}`, while drift rises
//! at long `τ`, so the curve's minimum sets the optimal averaging time. Computed
//! here non-overlapping over a set of averaging factors.

/// Non-overlapping Allan deviation at averaging factor `m` (`τ = m·τ0`).
/// `None` if there are fewer than two `m`-sized bins.
pub fn allan_deviation(data: &[f64], m: usize) -> Option<f64> {
    let m = m.max(1);
    let nb = data.len() / m;
    if nb < 2
    {
        return None;
    }
    // Bin averages.
    let means: Vec<f64> = (0..nb)
        .map(|b| data[b * m..(b + 1) * m].iter().sum::<f64>() / m as f64)
        .collect();
    let diffs: f64 = means.windows(2).map(|w| (w[1] - w[0]).powi(2)).sum();
    let var = diffs / (2.0 * (nb - 1) as f64);
    Some(var.sqrt())
}

/// Allan deviation over a list of averaging factors, returning `(m, σ)` pairs
/// where computable.
pub fn allan_curve(data: &[f64], ms: &[usize]) -> Vec<(usize, f64)> {
    ms.iter()
        .filter_map(|&m| allan_deviation(data, m).map(|s| (m, s)))
        .collect()
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
        fn normal(&mut self) -> f64 {
            let u1 = {
                self.s = self.s.wrapping_add(0x9E37_79B9_7F4A_7C15);
                let mut z = self.s;
                z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
                z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
                z ^= z >> 31;
                ((z >> 11) as f64 + 0.5) / ((1u64 << 53) as f64)
            };
            let u2 = {
                self.s = self.s.wrapping_add(0x9E37_79B9_7F4A_7C15);
                let mut z = self.s;
                z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
                z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
                z ^= z >> 31;
                ((z >> 11) as f64 + 0.5) / ((1u64 << 53) as f64)
            };
            (-2.0 * u1.max(1e-12).ln()).sqrt() * (2.0 * core::f64::consts::PI * u2).cos()
        }
    }

    #[test]
    fn constant_signal_has_zero_allan_deviation() {
        let data = vec![3.3; 1000];
        assert!(allan_deviation(&data, 1).unwrap() < 1e-12);
        assert!(allan_deviation(&data, 8).unwrap() < 1e-12);
    }

    #[test]
    fn white_noise_averages_down_with_tau() {
        // White noise: σ(τ) ∝ τ^{-1/2}, so σ decreases as m grows.
        let mut rng = Rng::new(0xA11A2);
        let data: Vec<f64> = (0..16384).map(|_| rng.normal()).collect();
        let s1 = allan_deviation(&data, 1).unwrap();
        let s4 = allan_deviation(&data, 4).unwrap();
        let s16 = allan_deviation(&data, 16).unwrap();
        assert!(s4 < s1 && s16 < s4, "σ should fall: {s1} {s4} {s16}");
        // Slope close to -1/2: σ(4)/σ(1) ≈ 1/2.
        assert!((s4 / s1 - 0.5).abs() < 0.15, "ratio {}", s4 / s1);
    }
}
