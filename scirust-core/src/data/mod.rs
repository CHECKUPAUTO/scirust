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
        Self {
            x,
            y,
            x_dim,
            y_dim,
            n,
        }
    }
    pub fn sample(&self, idx: usize) -> (&[f32], &[f32]) {
        let x_start = idx * self.x_dim;
        let y_start = idx * self.y_dim;
        (
            &self.x[x_start..x_start + self.x_dim],
            &self.y[y_start..y_start + self.y_dim],
        )
    }
    pub fn n_samples(&self) -> usize {
        self.n
    }
    pub fn len(&self) -> usize {
        self.n
    }
    pub fn is_empty(&self) -> bool {
        self.n == 0
    }
    pub fn get(&self, idx: usize) -> (&[f32], &[f32]) {
        self.sample(idx)
    }
    pub fn x_features(&self) -> usize {
        self.x_dim
    }
    pub fn subsample(&self, n: usize, _seed: u64) -> Self {
        let n = n.min(self.n);
        let mut x = Vec::with_capacity(n * self.x_dim);
        let mut y = Vec::with_capacity(n * self.y_dim);
        for i in 0..n
        {
            let (xi, yi) = self.sample(i);
            x.extend_from_slice(xi);
            y.extend_from_slice(yi);
        }
        Self::new(x, y, self.x_dim, self.y_dim)
    }

    /// Split aleatoire train / val. `train_ratio` dans [0, 1].
    /// Ex: `split_train_val(0.8, 42)` renvoie 80% train, 20% val.
    pub fn split_train_val(&self, train_ratio: f32, seed: u64) -> (Self, Self) {
        assert!(
            (0.0..=1.0).contains(&train_ratio),
            "train_ratio must be in [0, 1]"
        );
        let n_train = (self.n as f32 * train_ratio).round() as usize;
        let n_train = n_train.min(self.n).max(1);

        // Shuffle deterministe via PCG
        let mut indices: Vec<usize> = (0..self.n).collect();
        let mut rng = crate::nn::rng::PcgEngine::new(seed);
        for i in (1..indices.len()).rev()
        {
            let j = (rng.next_u32() as usize) % (i + 1);
            indices.swap(i, j);
        }

        let mut x_train = Vec::with_capacity(n_train * self.x_dim);
        let mut y_train = Vec::with_capacity(n_train * self.y_dim);
        let mut x_val = Vec::with_capacity((self.n - n_train) * self.x_dim);
        let mut y_val = Vec::with_capacity((self.n - n_train) * self.y_dim);

        for (pos, &idx) in indices.iter().enumerate()
        {
            let (xi, yi) = self.sample(idx);
            if pos < n_train
            {
                x_train.extend_from_slice(xi);
                y_train.extend_from_slice(yi);
            }
            else
            {
                x_val.extend_from_slice(xi);
                y_val.extend_from_slice(yi);
            }
        }

        (
            Self::new(x_train, y_train, self.x_dim, self.y_dim),
            Self::new(x_val, y_val, self.x_dim, self.y_dim),
        )
    }
}

pub trait Dataset {
    fn sample(&self, idx: usize) -> (&[f32], &[f32]);
    fn n_samples(&self) -> usize;
    fn len(&self) -> usize {
        self.n_samples()
    }
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl Dataset for InMemoryDataset {
    fn sample(&self, idx: usize) -> (&[f32], &[f32]) {
        self.sample(idx)
    }
    fn n_samples(&self) -> usize {
        self.n
    }
}

pub mod augment;
pub mod cifar10;
pub mod dataloader;
pub mod loader;
pub mod mnist;

pub use cifar10::Cifar10Dataset;
pub use loader::DataLoader;
pub use mnist::MnistDataset;
