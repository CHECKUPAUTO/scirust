use crate::autodiff::reverse::Tensor;
use crate::data::Dataset;
use rand::{SeedableRng, RngCore};

pub struct DataLoader<D: Dataset> {
    dataset: D,
    batch_size: usize,
    shuffle: bool,
    indices: Vec<usize>,
    seed: u64,
    epoch: u64,
}

impl<D: Dataset> DataLoader<D> {
    pub fn new(dataset: D, batch_size: usize, shuffle: bool, seed: u64) -> Self {
        let n = dataset.n_samples();
        Self {
            dataset,
            batch_size,
            shuffle,
            indices: (0..n).collect(),
            seed,
            epoch: 0,
        }
    }

    pub fn shuffle_epoch(&mut self, epoch: u64) {
        self.epoch = epoch;
        if self.shuffle {
            let mut rng = rand::rngs::StdRng::seed_from_u64(self.seed.wrapping_add(epoch));
            for i in (1..self.indices.len()).rev() {
                let j = (rng.next_u32() as usize) % (i + 1);
                self.indices.swap(i, j);
            }
        }
    }

    pub fn iter(&mut self) -> DataLoaderIter<'_, D> {
        DataLoaderIter { loader: self, pos: 0 }
    }
}

pub struct DataLoaderIter<'a, D: Dataset> {
    loader: &'a mut DataLoader<D>,
    pos: usize,
}

impl<'a, D: Dataset> Iterator for DataLoaderIter<'a, D> {
    type Item = (Tensor, Tensor);

    fn next(&mut self) -> Option<Self::Item> {
        let n = self.loader.dataset.n_samples();
        if self.pos >= n { return None; }
        let end = (self.pos + self.loader.batch_size).min(n);
        let bs = end - self.pos;
        let x_dim = self.loader.dataset.sample(0).0.len();
        let y_dim = self.loader.dataset.sample(0).1.len();
        let mut x_data = Vec::with_capacity(bs * x_dim);
        let mut y_data = Vec::with_capacity(bs * y_dim);
        for i in self.pos..end {
            let idx = self.loader.indices[i];
            let (x, y) = self.loader.dataset.sample(idx);
            x_data.extend_from_slice(x);
            y_data.extend_from_slice(y);
        }
        self.pos = end;
        Some((Tensor::from_vec(x_data, bs, x_dim), Tensor::from_vec(y_data, bs, y_dim)))
    }
}
