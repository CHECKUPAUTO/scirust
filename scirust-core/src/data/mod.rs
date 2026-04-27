// scirust-core/src/data/mod.rs
//
// DataLoader — chargement et mini-batching pour l'entraînement.
//
// Architecture en deux niveaux :
//
//   1. Trait Dataset      : abstraction sur N échantillons indexables (i → (x, y))
//      ├─ InMemoryDataset : tout chargé en RAM (pour datasets petits)
//      └─ StreamingDataset: lecture à la demande depuis disque (MNIST réel)
//
//   2. DataLoader         : itérateur sur des mini-batchs
//      - shuffle déterministe par graine + epoch
//      - batch_size + drop_last
//      - parallèle-friendly : peut être consommé en multi-thread
//
// USAGE TYPIQUE :
//
//   let ds = MnistDataset::load_idx("train-images.idx3-ubyte",
//                                    "train-labels.idx1-ubyte")?;
//   let mut loader = DataLoader::new(ds, 64, true, 42);
//   for epoch in 0..10 {
//       loader.shuffle_epoch(epoch);
//       for (x_batch, y_batch) in loader.iter() {
//           // ... train step ...
//       }
//   }

pub mod mnist;

use crate::autodiff::reverse::Tensor;
use crate::nn::rng::PcgEngine;

// ================================================================== //
//  Trait Dataset                                                      //
// ================================================================== //

pub trait Dataset {
    /// Nombre total d'échantillons.
    fn len(&self) -> usize;

    /// Récupère un échantillon par indice. Renvoie (x, y) — chacun
    /// est un Tensor (1, n_features) pour une image plate, ou (1, n_classes)
    /// pour un one-hot.
    fn get(&self, idx: usize) -> (Tensor, Tensor);

    /// Dimensions des features (utile pour pré-allouer le batch).
    fn x_features(&self) -> usize;
    fn y_features(&self) -> usize;

    fn is_empty(&self) -> bool { self.len() == 0 }
}

// ================================================================== //
//  InMemoryDataset — tout en RAM                                      //
// ================================================================== //

pub struct InMemoryDataset {
    /// Buffer (N, x_features) row-major
    pub x_data: Vec<f32>,
    /// Buffer (N, y_features) row-major
    pub y_data: Vec<f32>,
    pub n:      usize,
    pub x_dim:  usize,
    pub y_dim:  usize,
}

impl InMemoryDataset {
    pub fn new(x_data: Vec<f32>, y_data: Vec<f32>, x_dim: usize, y_dim: usize) -> Self {
        let n = x_data.len() / x_dim;
        assert_eq!(x_data.len(), n * x_dim, "x_data taille incohérente");
        assert_eq!(y_data.len(), n * y_dim, "y_data taille incohérente");
        Self { x_data, y_data, n, x_dim, y_dim }
    }
}

impl Dataset for InMemoryDataset {
    fn len(&self) -> usize { self.n }
    fn x_features(&self) -> usize { self.x_dim }
    fn y_features(&self) -> usize { self.y_dim }

    fn get(&self, idx: usize) -> (Tensor, Tensor) {
        assert!(idx < self.n);
        let x = Tensor::from_vec(
            self.x_data[idx * self.x_dim .. (idx + 1) * self.x_dim].to_vec(),
            1, self.x_dim,
        );
        let y = Tensor::from_vec(
            self.y_data[idx * self.y_dim .. (idx + 1) * self.y_dim].to_vec(),
            1, self.y_dim,
        );
        (x, y)
    }
}

// ================================================================== //
//  DataLoader — itérateur de mini-batchs                              //
// ================================================================== //

pub struct DataLoader<D: Dataset> {
    pub dataset:    D,
    pub batch_size: usize,
    pub shuffle:    bool,
    pub drop_last:  bool,
    /// Permutation des indices pour cette époque
    indices:        Vec<usize>,
    seed:           u64,
}

impl<D: Dataset> DataLoader<D> {
    pub fn new(dataset: D, batch_size: usize, shuffle: bool, seed: u64) -> Self {
        let n = dataset.len();
        let indices: Vec<usize> = (0..n).collect();
        Self { dataset, batch_size, shuffle, drop_last: false, indices, seed }
    }

    pub fn drop_last(mut self, b: bool) -> Self { self.drop_last = b; self }

    /// Réorganise les indices pour cette époque (Fisher-Yates seedable).
    pub fn shuffle_epoch(&mut self, epoch: u64) {
        if !self.shuffle { return; }
        let mut rng = PcgEngine::new(self.seed.wrapping_add(epoch));
        let n = self.indices.len();
        for i in (1..n).rev() {
            let j = (rng.next_u32() as usize) % (i + 1);
            self.indices.swap(i, j);
        }
    }

    pub fn n_batches(&self) -> usize {
        let n = self.dataset.len();
        if self.drop_last { n / self.batch_size }
        else              { (n + self.batch_size - 1) / self.batch_size }
    }

    /// Renvoie un itérateur sur les batchs (x_batch, y_batch) de l'époque actuelle.
    pub fn iter(&self) -> BatchIter<'_, D> {
        BatchIter { loader: self, pos: 0 }
    }

    /// Récupère un batch spécifique par indice (utile pour data parallelism).
    pub fn batch(&self, batch_idx: usize) -> Option<(Tensor, Tensor)> {
        let start = batch_idx * self.batch_size;
        if start >= self.dataset.len() { return None; }
        let end = (start + self.batch_size).min(self.dataset.len());
        if self.drop_last && end - start < self.batch_size { return None; }

        let actual_size = end - start;
        let x_dim = self.dataset.x_features();
        let y_dim = self.dataset.y_features();
        let mut x_buf = Vec::with_capacity(actual_size * x_dim);
        let mut y_buf = Vec::with_capacity(actual_size * y_dim);

        for i in start..end {
            let real_idx = self.indices[i];
            let (x, y) = self.dataset.get(real_idx);
            x_buf.extend(x.data);
            y_buf.extend(y.data);
        }

        Some((
            Tensor::from_vec(x_buf, actual_size, x_dim),
            Tensor::from_vec(y_buf, actual_size, y_dim),
        ))
    }
}

pub struct BatchIter<'a, D: Dataset> {
    loader: &'a DataLoader<D>,
    pos:    usize,
}

impl<'a, D: Dataset> Iterator for BatchIter<'a, D> {
    type Item = (Tensor, Tensor);
    fn next(&mut self) -> Option<Self::Item> {
        let b = self.loader.batch(self.pos)?;
        self.pos += 1;
        Some(b)
    }
}

// ================================================================== //
//  Tests                                                              //
// ================================================================== //
#[cfg(test)]
mod tests {
    use super::*;

    fn make_toy() -> InMemoryDataset {
        // 10 échantillons : x = [i, i*2], y = [i % 3 == 0]
        let x_data: Vec<f32> = (0..10).flat_map(|i| vec![i as f32, i as f32 * 2.0]).collect();
        let y_data: Vec<f32> = (0..10).map(|i| if i % 3 == 0 { 1.0 } else { 0.0 }).collect();
        InMemoryDataset::new(x_data, y_data, 2, 1)
    }

    #[test]
    fn dataset_basic() {
        let ds = make_toy();
        assert_eq!(ds.len(), 10);
        let (x, y) = ds.get(3);
        assert_eq!(x.data, vec![3.0, 6.0]);
        assert_eq!(y.data, vec![1.0]); // 3 % 3 == 0
    }

    #[test]
    fn batch_iteration() {
        let loader = DataLoader::new(make_toy(), 4, false, 0);
        let batches: Vec<_> = loader.iter().collect();
        assert_eq!(batches.len(), 3); // 4+4+2
        assert_eq!(batches[0].0.shape(), (4, 2));
        assert_eq!(batches[0].1.shape(), (4, 1));
        assert_eq!(batches[2].0.shape(), (2, 2)); // dernier batch partiel
    }

    #[test]
    fn drop_last_works() {
        let loader = DataLoader::new(make_toy(), 4, false, 0).drop_last(true);
        let batches: Vec<_> = loader.iter().collect();
        assert_eq!(batches.len(), 2); // 4+4 (le dernier de 2 est jeté)
    }

    #[test]
    fn shuffle_changes_order() {
        let mut loader = DataLoader::new(make_toy(), 10, true, 42);
        let initial_indices = loader.indices.clone();
        loader.shuffle_epoch(0);
        assert_ne!(loader.indices, initial_indices);
    }

    #[test]
    fn shuffle_deterministic_per_seed_and_epoch() {
        let mut a = DataLoader::new(make_toy(), 10, true, 42);
        let mut b = DataLoader::new(make_toy(), 10, true, 42);
        a.shuffle_epoch(7);
        b.shuffle_epoch(7);
        assert_eq!(a.indices, b.indices);
    }

    #[test]
    fn shuffle_different_per_epoch() {
        let mut loader = DataLoader::new(make_toy(), 10, true, 42);
        loader.shuffle_epoch(0);
        let e0 = loader.indices.clone();
        loader.shuffle_epoch(1);
        assert_ne!(loader.indices, e0);
    }
}
