use scirust_core::nn::rng::PcgEngine;

use crate::pinn::collocation::CollocationPoints;
use crate::pinn::domain::Domain1D;

#[derive(Debug, Clone)]
pub struct AdaptiveSamplingConfig {
    pub initial_points: usize,
    pub candidate_pool_size: usize,
    pub add_per_round: usize,
    pub max_points: usize,
    pub refinement_rounds: usize,
    pub seed: u64,
}

impl Default for AdaptiveSamplingConfig {
    fn default() -> Self {
        Self {
            initial_points: 100,
            candidate_pool_size: 1000,
            add_per_round: 50,
            max_points: 500,
            refinement_rounds: 5,
            seed: 42,
        }
    }
}

#[derive(Debug, Clone)]
pub struct AdaptiveRefinementHistory {
    pub rounds: Vec<RoundInfo>,
}

#[derive(Debug, Clone)]
pub struct RoundInfo {
    pub round: usize,
    pub n_points_before: usize,
    pub n_points_after: usize,
    pub max_residual: f32,
    pub mean_residual: f32,
}

pub struct AdaptiveSampler;

impl AdaptiveSampler {
    pub fn refine<R>(
        domain: &Domain1D,
        residual_fn: R,
        config: &AdaptiveSamplingConfig,
    ) -> (CollocationPoints, AdaptiveRefinementHistory)
    where
        R: Fn(f32) -> f32,
    {
        let mut rng = PcgEngine::new(config.seed);
        let mut points =
            CollocationPoints::from_random_uniform(domain, config.initial_points, config.seed);
        let mut history = AdaptiveRefinementHistory { rounds: Vec::new() };

        for round in 0..config.refinement_rounds
        {
            if points.len() >= config.max_points
            {
                break;
            }

            let residuals: Vec<f32> = points
                .points
                .iter()
                .map(|pt| residual_fn(pt[0]).abs())
                .collect();
            let max_res = residuals.iter().copied().fold(0.0, f32::max);
            let mean_res = residuals.iter().sum::<f32>() / residuals.len().max(1) as f32;

            let before = points.len();

            let n_candidates = config.candidate_pool_size;
            let mut candidates: Vec<(f32, f32)> = (0..n_candidates)
                .map(|_| {
                    let x = domain.start + rng.float() * (domain.end - domain.start);
                    let r = residual_fn(x).abs();
                    (x, r)
                })
                .collect();

            candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

            let mut to_add = 0;
            for &(x, _) in &candidates
            {
                if to_add >= config.add_per_round
                {
                    break;
                }
                if points.len() >= config.max_points
                {
                    break;
                }
                let is_duplicate = points.points.iter().any(|pt| (pt[0] - x).abs() < 1e-4);
                if !is_duplicate
                {
                    points.points.push(vec![x]);
                    to_add += 1;
                }
            }

            history.rounds.push(RoundInfo {
                round,
                n_points_before: before,
                n_points_after: points.len(),
                max_residual: max_res,
                mean_residual: mean_res,
            });
        }

        (points, history)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adaptive_refinement() {
        let domain = Domain1D::new(0.0, 1.0).unwrap();
        let config = AdaptiveSamplingConfig {
            initial_points: 20,
            candidate_pool_size: 100,
            add_per_round: 10,
            max_points: 60,
            refinement_rounds: 3,
            seed: 42,
        };
        let (points, history) = AdaptiveSampler::refine(&domain, |x| (x - 0.5).abs(), &config);
        assert!(points.len() >= 20);
        assert!(!history.rounds.is_empty());
    }
}
