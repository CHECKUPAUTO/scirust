use crate::error::{Result, VariationalError};

#[derive(Debug, Clone)]
pub struct Domain1D {
    pub start: f32,
    pub end: f32,
}

impl Domain1D {
    pub fn new(start: f32, end: f32) -> Result<Self> {
        if start >= end
        {
            return Err(VariationalError::InvalidInterval { start, end });
        }
        Ok(Self { start, end })
    }

    pub fn contains(&self, x: f32) -> bool {
        x >= self.start && x <= self.end
    }

    pub fn uniform_points(&self, n: usize) -> Vec<f32> {
        if n == 0
        {
            return Vec::new();
        }
        if n == 1
        {
            return vec![0.5 * (self.start + self.end)];
        }
        let dx = (self.end - self.start) / (n - 1) as f32;
        (0..n).map(|i| self.start + i as f32 * dx).collect()
    }
}

#[derive(Debug, Clone)]
pub struct DomainRect {
    pub ndim: usize,
    pub bounds: Vec<(f32, f32)>,
}

impl DomainRect {
    pub fn new(bounds: Vec<(f32, f32)>) -> Result<Self> {
        if bounds.is_empty()
        {
            return Err(VariationalError::UnsupportedOperation {
                details: "empty domain bounds".into(),
            });
        }
        for &(lo, hi) in &bounds
        {
            if lo >= hi
            {
                return Err(VariationalError::InvalidInterval { start: lo, end: hi });
            }
        }
        Ok(Self {
            ndim: bounds.len(),
            bounds,
        })
    }

    pub fn uniform_grid(&self, points_per_dim: &[usize]) -> Vec<Vec<f32>> {
        assert_eq!(points_per_dim.len(), self.ndim);
        let mut grid = Vec::new();
        let total: usize = points_per_dim.iter().product();
        for idx in 0..total
        {
            let mut point = Vec::with_capacity(self.ndim);
            let mut remaining = idx;
            for d in (0..self.ndim).rev()
            {
                let dim_count = points_per_dim[d];
                let coord_idx = remaining % dim_count;
                remaining /= dim_count;
                let (lo, hi) = self.bounds[d];
                let x = if dim_count == 1
                {
                    0.5 * (lo + hi)
                }
                else
                {
                    lo + coord_idx as f32 * (hi - lo) / (dim_count - 1) as f32
                };
                point.push(x);
            }
            point.reverse();
            grid.push(point);
        }
        grid
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_domain1d() {
        let d = Domain1D::new(0.0, 1.0).unwrap();
        assert!(d.contains(0.5));
        assert!(!d.contains(-0.1));
        let pts = d.uniform_points(5);
        assert_eq!(pts.len(), 5);
        assert!((pts[0] - 0.0).abs() < 1e-6);
        assert!((pts[4] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_domain_rect() {
        let bounds = vec![(0.0, 1.0), (0.0, 2.0)];
        let d = DomainRect::new(bounds).unwrap();
        let grid = d.uniform_grid(&[3, 3]);
        assert_eq!(grid.len(), 9);
    }

    #[test]
    fn test_invalid_interval() {
        assert!(Domain1D::new(1.0, 0.0).is_err());
    }
}
