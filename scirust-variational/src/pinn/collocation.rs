use scirust_core::nn::rng::PcgEngine;

use crate::pinn::domain::Domain1D;

#[derive(Debug, Clone)]
pub struct CollocationPoints {
    pub points: Vec<Vec<f32>>,
    pub ndim: usize,
}

impl CollocationPoints {
    pub fn from_uniform_grid(domain: &Domain1D, n_points: usize) -> Self {
        let xs = domain.uniform_points(n_points);
        let points: Vec<Vec<f32>> = xs.into_iter().map(|x| vec![x]).collect();
        Self { points, ndim: 1 }
    }

    pub fn from_random_uniform(domain: &Domain1D, n_points: usize, seed: u64) -> Self {
        let mut rng = PcgEngine::new(seed);
        let mut points = Vec::with_capacity(n_points);
        for _ in 0..n_points
        {
            let x = domain.start + rng.float() * (domain.end - domain.start);
            points.push(vec![x]);
        }
        Self { points, ndim: 1 }
    }

    pub fn from_latin_hypercube(bounds: &[(f32, f32)], n_points: usize, seed: u64) -> Self {
        let ndim = bounds.len();
        let mut rng = PcgEngine::new(seed);

        let mut points: Vec<Vec<f32>> = (0..n_points)
            .map(|_| {
                bounds
                    .iter()
                    .map(|&(lo, hi)| {
                        let bin_width = (hi - lo) / n_points as f32;
                        lo + rng.float() * bin_width
                    })
                    .collect()
            })
            .collect();

        for d in 0..ndim
        {
            let mut perm: Vec<usize> = (0..n_points).collect();
            for i in (1..perm.len()).rev()
            {
                let j = (rng.float() * (i as f32 + 1.0)) as usize;
                perm.swap(i, j.min(i));
            }
            for i in 0..n_points
            {
                let bin_width = (bounds[d].1 - bounds[d].0) / n_points as f32;
                let bin_start = bounds[d].0 + perm[i] as f32 * bin_width;
                points[i][d] = bin_start + rng.float() * bin_width;
            }
        }

        Self { points, ndim }
    }

    pub fn len(&self) -> usize {
        self.points.len()
    }

    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }

    pub fn to_flat(&self) -> Vec<f32> {
        let mut flat = Vec::with_capacity(self.points.len() * self.ndim);
        for pt in &self.points
        {
            flat.extend_from_slice(pt);
        }
        flat
    }

    pub fn to_batched_tensor(&self) -> (Vec<f32>, Vec<usize>) {
        (self.to_flat(), vec![self.points.len(), self.ndim])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pinn::domain::Domain1D;

    #[test]
    fn test_uniform_collocation() {
        let domain = Domain1D::new(0.0, 1.0).unwrap();
        let pts = CollocationPoints::from_uniform_grid(&domain, 10);
        assert_eq!(pts.len(), 10);
    }

    #[test]
    fn test_deterministic_sampling() {
        let domain = Domain1D::new(0.0, 1.0).unwrap();
        let a = CollocationPoints::from_random_uniform(&domain, 100, 42);
        let b = CollocationPoints::from_random_uniform(&domain, 100, 42);
        assert_eq!(a.to_flat(), b.to_flat());
    }

    #[test]
    fn test_latin_hypercube() {
        let bounds = vec![(0.0, 1.0), (0.0, 2.0)];
        let pts = CollocationPoints::from_latin_hypercube(&bounds, 10, 42);
        assert_eq!(pts.len(), 10);
        assert_eq!(pts.ndim, 2);
    }
}
