use crate::model::TraceRow;

#[derive(Debug, Clone)]
struct SplitMix64(u64);

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut value = self.0;
        value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        value ^ (value >> 31)
    }

    fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    fn normal(&mut self) -> f64 {
        let first = self.uniform().max(1e-12);
        let second = self.uniform();
        (-2.0 * first.ln()).sqrt() * (std::f64::consts::TAU * second).cos()
    }
}

/// Nonlinear oracle where cosine similarity is deliberately not sufficient.
/// This validates discovery machinery only, never a real-model performance claim.
pub fn synthetic_trace(
    trajectories: usize,
    rows_per_trajectory: usize,
    seed: u64,
) -> Vec<TraceRow> {
    let mut rng = SplitMix64::new(seed);
    let mut rows = Vec::with_capacity(trajectories * rows_per_trajectory);
    for trajectory in 0..trajectories {
        let regime = rng.uniform();
        for step in 0..rows_per_trajectory {
            let layer_id = (step % 32) as u32;
            let layer_fraction = layer_id as f64 / 31.0;
            let cache_age = ((step / 32) % 16) as f64 / 15.0;
            let similarity = (0.74 + 0.25 * rng.uniform()).clamp(0.0, 1.0);
            let similarity_delta =
                rng.uniform() * 0.08 - 0.04 - 0.035 * (regime - 0.5);
            let head_variance =
                (0.01 + 0.22 * rng.uniform() + 0.12 * (regime - 0.5).abs())
                    .clamp(0.0, 1.0);
            let attention_mass = (0.35 + 0.60 * rng.uniform()).clamp(0.0, 1.0);
            let refresh_cost = (0.2 + 0.8 * (1.0 - layer_fraction)).clamp(0.0, 1.0);
            let drift = 1.0 - similarity;
            let worsening = (-similarity_delta).max(0.0);
            let latent_risk = 1.8 * drift
                + 1.25 * worsening
                + 0.95 * head_variance.sqrt()
                + 0.55 * cache_age
                + 0.75 * (1.0 - attention_mass)
                + 0.65 * drift * cache_age
                + 0.25 * regime
                - 1.25;
            let stale_loss = (latent_risk + 0.025 * rng.normal())
                .max(0.0)
                .powi(2)
                .min(1.0);
            rows.push(TraceRow {
                trajectory_id: trajectory as u64,
                step: step as u32,
                layer_id,
                similarity,
                similarity_delta,
                head_variance,
                cache_age,
                attention_mass,
                layer_fraction,
                refresh_cost,
                stale_loss,
            });
        }
    }
    rows
}
