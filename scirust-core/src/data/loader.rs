use crate::autodiff::reverse::Tensor;
use crate::data::Dataset;
use crate::tensor::tensor_nd::TensorND;
use rand::{RngCore, SeedableRng};

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
        if self.shuffle
        {
            let mut rng = rand::rngs::StdRng::seed_from_u64(self.seed.wrapping_add(epoch));
            for i in (1..self.indices.len()).rev()
            {
                let j = (rng.next_u32() as usize) % (i + 1);
                self.indices.swap(i, j);
            }
        }
    }

    pub fn iter(&mut self) -> DataLoaderIter<'_, D> {
        DataLoaderIter {
            loader: self,
            pos: 0,
        }
    }

    pub fn iter_nd(
        &mut self,
        x_shape: &[usize],
        y_shape: &[usize],
    ) -> impl Iterator<Item = (TensorND, TensorND)> + '_ {
        let x_shape = x_shape.to_vec();
        let y_shape = y_shape.to_vec();
        self.iter().map(move |(x, y)| {
            let bs = x.rows;
            let mut full_x = vec![bs];
            full_x.extend_from_slice(&x_shape);
            let x_nd = TensorND::from_vec(x.data, full_x);
            let mut full_y = vec![bs];
            full_y.extend_from_slice(&y_shape);
            let y_nd = TensorND::from_vec(y.data, full_y);
            (x_nd, y_nd)
        })
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
        if self.pos >= n
        {
            return None;
        }
        let end = (self.pos + self.loader.batch_size).min(n);
        let bs = end - self.pos;
        let x_dim = self.loader.dataset.sample(0).0.len();
        let y_dim = self.loader.dataset.sample(0).1.len();
        let mut x_data = Vec::with_capacity(bs * x_dim);
        let mut y_data = Vec::with_capacity(bs * y_dim);
        for i in self.pos..end
        {
            let idx = self.loader.indices[i];
            let (x, y) = self.loader.dataset.sample(idx);
            x_data.extend_from_slice(x);
            y_data.extend_from_slice(y);
        }
        self.pos = end;
        Some((
            Tensor::from_vec(x_data, bs, x_dim),
            Tensor::from_vec(y_data, bs, y_dim),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::InMemoryDataset;

    #[test]
    fn loader_nd_batch_shape_correct() {
        // Dataset : 8 samples, chaque input = 3*4*5 = 60 elements, target = 2 elements
        let n = 8;
        let x_dim = 60;
        let y_dim = 2;
        let x_data: Vec<f32> = (0..n * x_dim).map(|i| i as f32).collect();
        let y_data: Vec<f32> = (0..n * y_dim).map(|i| i as f32).collect();
        let dataset = InMemoryDataset::new(x_data, y_data, x_dim, y_dim);

        let mut loader = DataLoader::new(dataset, 4, false, 42);
        let (x_batch, y_batch) = loader.iter_nd(&[3, 4, 5], &[2]).next().unwrap();

        assert_eq!(x_batch.shape(), &[4, 3, 4, 5]);
        assert_eq!(y_batch.shape(), &[4, 2]);

        // Verifier que les valeurs sont correctes
        // x_batch[0, 0, 0, 0] = x_data[0] = 0.0
        assert_eq!(x_batch.get(&[0, 0, 0, 0]), 0.0);
        // x_batch[1, 0, 0, 0] = x_data[60] = 60.0
        assert_eq!(x_batch.get(&[1, 0, 0, 0]), 60.0);
    }

    #[test]
    fn loader_nd_shuffle_deterministic() {
        let n = 10;
        let x_dim = 6;
        let y_dim = 1;
        let x_data: Vec<f32> = (0..n * x_dim).map(|i| i as f32).collect();
        let y_data: Vec<f32> = (0..n * y_dim).map(|i| i as f32).collect();
        let dataset = InMemoryDataset::new(x_data.clone(), y_data.clone(), x_dim, y_dim);

        let mut loader1 = DataLoader::new(dataset.clone(), 4, true, 99);
        loader1.shuffle_epoch(0);
        let (x1, _) = loader1.iter_nd(&[2, 3], &[1]).next().unwrap();

        let mut loader2 = DataLoader::new(dataset.clone(), 4, true, 99);
        loader2.shuffle_epoch(0);
        let (x2, _) = loader2.iter_nd(&[2, 3], &[1]).next().unwrap();

        assert_eq!(
            x1.data, x2.data,
            "Same seed should produce identical shuffled batches"
        );
    }

    #[test]
    fn loader_nd_multiple_batches() {
        let n = 7;
        let x_dim = 4;
        let y_dim = 1;
        let x_data: Vec<f32> = (0..n * x_dim).map(|i| i as f32).collect();
        let y_data: Vec<f32> = (0..n * y_dim).map(|i| i as f32).collect();
        let dataset = InMemoryDataset::new(x_data, y_data, x_dim, y_dim);

        let mut loader = DataLoader::new(dataset, 3, false, 42);
        let mut batches = loader.iter_nd(&[2, 2], &[1]);

        let (x1, _) = batches.next().unwrap();
        assert_eq!(x1.shape(), &[3, 2, 2]);

        let (x2, _) = batches.next().unwrap();
        assert_eq!(x2.shape(), &[3, 2, 2]);

        let (x3, _) = batches.next().unwrap();
        assert_eq!(x3.shape(), &[1, 2, 2]); // dernier batch partiel

        assert!(batches.next().is_none());
    }
}
