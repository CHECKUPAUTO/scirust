// scirust-core/src/data/cifar10.rs
//
// Lecteur du format binaire CIFAR-10.
//
// FORMAT :
//   Chaque fichier contient 10000 records de 3073 bytes :
//     [0]     : label (0-9)
//     [1..3073] : image RGB 32×32 (1024 rouge, 1024 vert, 1024 bleu)
//   Les pixels sont uint8 [0,255], stockés row-major par canal.
//
// FICHIERS (à télécharger depuis https://www.cs.toronto.edu/~kriz/cifar.html) :
//   data_batch_1.bin … data_batch_5.bin  (train)
//   test_batch.bin                        (test)

use crate::data::InMemoryDataset;
use std::fs::File;
use std::io::{self, Read};
use std::path::Path;

pub const CIFAR10_IMAGE_SIZE: usize = 32 * 32 * 3;
pub const CIFAR10_N_CLASSES: usize = 10;

/// Charge un seul fichier batch CIFAR-10.
/// Renvoie (images, labels_raw) où images est un Vec<f32> normalisé [0,1].
pub fn load_cifar10_batch<P: AsRef<Path>>(path: P) -> io::Result<(Vec<f32>, Vec<u8>)> {
    let mut f = File::open(path)?;
    let mut raw = Vec::new();
    f.read_to_end(&mut raw)?;

    let record_size = 3073;
    let n = raw.len() / record_size;
    if raw.len() % record_size != 0
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "taille fichier CIFAR-10 invalide : {} non divisible par {}",
                raw.len(),
                record_size
            ),
        ));
    }

    let mut images = vec![0.0f32; n * CIFAR10_IMAGE_SIZE];
    let mut labels = vec![0u8; n];

    for i in 0..n
    {
        let off = i * record_size;
        labels[i] = raw[off];
        // Normalisation : uint8 [0,255] → f32 [0,1]
        for j in 0..CIFAR10_IMAGE_SIZE
        {
            images[i * CIFAR10_IMAGE_SIZE + j] = raw[off + 1 + j] as f32 / 255.0;
        }
    }
    Ok((images, labels))
}

/// Dataset CIFAR-10 complet (train + test).
pub struct Cifar10Dataset {
    pub n_train: usize,
    pub n_test: usize,
    pub images_train: Vec<f32>,
    pub labels_train_one_hot: Vec<f32>,
    pub labels_train_raw: Vec<u8>,
    pub images_test: Vec<f32>,
    pub labels_test_one_hot: Vec<f32>,
    pub labels_test_raw: Vec<u8>,
}

impl Cifar10Dataset {
    /// Charge les 5 batches d'entraînement + 1 batch de test.
    pub fn load<P: AsRef<Path>>(data_dir: P) -> io::Result<Self> {
        let dir = data_dir.as_ref();
        let mut train_images = Vec::new();
        let mut train_labels_raw = Vec::new();

        for i in 1..=5
        {
            let path = dir.join(format!("data_batch_{}.bin", i));
            let (imgs, lbls) = load_cifar10_batch(&path)?;
            train_images.extend_from_slice(&imgs);
            train_labels_raw.extend_from_slice(&lbls);
        }

        let n_train = train_labels_raw.len();
        let mut train_one_hot = vec![0.0f32; n_train * CIFAR10_N_CLASSES];
        for (i, &lbl) in train_labels_raw.iter().enumerate()
        {
            train_one_hot[i * CIFAR10_N_CLASSES + lbl as usize] = 1.0;
        }

        let test_path = dir.join("test_batch.bin");
        let (test_images, test_labels_raw) = load_cifar10_batch(&test_path)?;
        let n_test = test_labels_raw.len();
        let mut test_one_hot = vec![0.0f32; n_test * CIFAR10_N_CLASSES];
        for (i, &lbl) in test_labels_raw.iter().enumerate()
        {
            test_one_hot[i * CIFAR10_N_CLASSES + lbl as usize] = 1.0;
        }

        Ok(Self {
            n_train,
            n_test,
            images_train: train_images,
            labels_train_one_hot: train_one_hot,
            labels_train_raw: train_labels_raw,
            images_test: test_images,
            labels_test_one_hot: test_one_hot,
            labels_test_raw: test_labels_raw,
        })
    }

    pub fn train_in_memory(&self) -> InMemoryDataset {
        InMemoryDataset::new(
            self.images_train.clone(),
            self.labels_train_one_hot.clone(),
            CIFAR10_IMAGE_SIZE,
            CIFAR10_N_CLASSES,
        )
    }

    pub fn test_in_memory(&self) -> InMemoryDataset {
        InMemoryDataset::new(
            self.images_test.clone(),
            self.labels_test_one_hot.clone(),
            CIFAR10_IMAGE_SIZE,
            CIFAR10_N_CLASSES,
        )
    }

    pub fn subsample_train(&self, max_n: usize) -> InMemoryDataset {
        let n = max_n.min(self.n_train);
        InMemoryDataset::new(
            self.images_train[..n * CIFAR10_IMAGE_SIZE].to_vec(),
            self.labels_train_one_hot[..n * CIFAR10_N_CLASSES].to_vec(),
            CIFAR10_IMAGE_SIZE,
            CIFAR10_N_CLASSES,
        )
    }

    pub fn subsample_test(&self, max_n: usize) -> InMemoryDataset {
        let n = max_n.min(self.n_test);
        InMemoryDataset::new(
            self.images_test[..n * CIFAR10_IMAGE_SIZE].to_vec(),
            self.labels_test_one_hot[..n * CIFAR10_N_CLASSES].to_vec(),
            CIFAR10_IMAGE_SIZE,
            CIFAR10_N_CLASSES,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn make_synthetic_batch(n: usize, labels: &[u8]) -> Vec<u8> {
        let mut data = Vec::with_capacity(n * 3073);
        for i in 0..n
        {
            data.push(labels[i % labels.len()]);
            for _ in 0..3072
            {
                data.push(((i * 7) % 256) as u8);
            }
        }
        data
    }

    #[test]
    fn load_synthetic_batch() {
        let dir = std::env::temp_dir().join(format!("scirust_cifar10_test_{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("data_batch_1.bin");
        {
            let mut f = File::create(&path).unwrap();
            f.write_all(&make_synthetic_batch(3, &[0, 5, 9])).unwrap();
        }
        let (imgs, lbls) = load_cifar10_batch(&path).unwrap();
        assert_eq!(lbls, vec![0, 5, 9]);
        assert_eq!(imgs.len(), 3 * 3072);
        // Pixel normalisé : valeur / 255
        assert!(imgs[0].abs() < 1e-6);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn load_full_dataset_synthetic() {
        let dir = std::env::temp_dir().join(format!("scirust_cifar10_full_{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);

        for i in 1..=5
        {
            let path = dir.join(format!("data_batch_{}.bin", i));
            let mut f = File::create(&path).unwrap();
            f.write_all(&make_synthetic_batch(10, &[i as u8])).unwrap();
        }
        {
            let path = dir.join("test_batch.bin");
            let mut f = File::create(&path).unwrap();
            f.write_all(&make_synthetic_batch(5, &[7])).unwrap();
        }

        let ds = Cifar10Dataset::load(&dir).unwrap();
        assert_eq!(ds.n_train, 50); // 5 × 10
        assert_eq!(ds.n_test, 5);
        assert_eq!(ds.labels_train_raw[0], 1);
        assert_eq!(ds.labels_test_raw[0], 7);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
