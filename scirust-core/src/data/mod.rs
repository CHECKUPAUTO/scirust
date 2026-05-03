/// Dataset minimal pour V10A
#[derive(Clone, Debug)]
pub struct InMemoryDataset {
    pub x: Vec<f32>,
    pub y: Vec<f32>,
    pub x_dim: usize,
    pub y_dim: usize,
    pub n: usize,
}

impl InMemoryDataset {
    pub fn new(x: Vec<f32>, y: Vec<f32>, x_dim: usize, y_dim: usize) -> Self {
        let n = x.len() / x_dim;
        Self { x, y, x_dim, y_dim, n }
    }
    pub fn sample(&self, idx: usize) -> (&[f32], &[f32]) {
        let x_start = idx * self.x_dim;
        let y_start = idx * self.y_dim;
        (&self.x[x_start..x_start + self.x_dim],
         &self.y[y_start..y_start + self.y_dim])
    }
    pub fn n_samples(&self) -> usize { self.n }
    pub fn len(&self) -> usize { self.n }
    pub fn is_empty(&self) -> bool { self.n == 0 }
    pub fn get(&self, idx: usize) -> (&[f32], &[f32]) { self.sample(idx) }
    pub fn x_features(&self) -> usize { self.x_dim }
    pub fn subsample(&self, n: usize, _seed: u64) -> Self {
        let n = n.min(self.n);
        let mut x = Vec::with_capacity(n * self.x_dim);
        let mut y = Vec::with_capacity(n * self.y_dim);
        for i in 0..n {
            let (xi, yi) = self.sample(i);
            x.extend_from_slice(xi);
            y.extend_from_slice(yi);
        }
        Self::new(x, y, self.x_dim, self.y_dim)
    }

}

pub trait Dataset {
    fn sample(&self, idx: usize) -> (&[f32], &[f32]);
    fn n_samples(&self) -> usize;
    fn len(&self) -> usize { self.n_samples() }
    fn is_empty(&self) -> bool { self.len() == 0 }
}

impl Dataset for InMemoryDataset {
    fn sample(&self, idx: usize) -> (&[f32], &[f32]) { self.sample(idx) }
    fn n_samples(&self) -> usize { self.n }
}

pub mod mnist;
pub mod cifar10;
pub mod augment;
pub mod loader;

pub use loader::DataLoader;
pub use mnist::MnistDataset;
pub use cifar10::Cifar10Dataset;
