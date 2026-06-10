// scirust-core/src/data/mnist.rs
//
// Lecteur du format IDX utilisé par MNIST (et d'autres datasets historiques).
//
// FORMAT IDX (big-endian, 100% spec) :
//
//   [0..4]  : magic number 0x00000803 (images) ou 0x00000801 (labels)
//   [4..8]  : nombre d'éléments (uint32 BE)
//   [8..12] : (images uniquement) nombre de lignes
//   [12..16]: (images uniquement) nombre de colonnes
//   [16..] / [8..] : données uint8
//
// Pour MNIST :
//   train-images.idx3-ubyte : 60000 images 28×28 (uint8 = pixel intensity)
//   train-labels.idx1-ubyte : 60000 labels (uint8 = chiffre 0-9)
//
// Cette implémentation supporte aussi les fichiers gzippés (.gz) si présents,
// mais via l'extension uniquement (pas de décompression intégrée — l'utilisateur
// doit gunzip avant).

use crate::autodiff::reverse::Tensor;
use crate::data::{Dataset, InMemoryDataset};
use std::fs::File;
use std::io::{self, Read};
use std::path::Path;

const MAGIC_IMAGES: u32 = 0x00000803;
const MAGIC_LABELS: u32 = 0x00000801;

// ================================================================== //
//  Lecteurs bas niveau                                                //
// ================================================================== //

fn read_be_u32<R: Read>(r: &mut R) -> io::Result<u32> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)?;
    Ok(u32::from_be_bytes(buf))
}

/// Charge les images IDX. Renvoie (Vec<f32> normalisé [0,1], n, h, w).
pub fn load_idx_images<P: AsRef<Path>>(path: P) -> io::Result<(Vec<f32>, usize, usize, usize)> {
    let mut f = File::open(path)?;
    let magic = read_be_u32(&mut f)?;
    if magic != MAGIC_IMAGES
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("magic image incorrect : 0x{magic:08x}, attendu 0x{MAGIC_IMAGES:08x}"),
        ));
    }
    let n = read_be_u32(&mut f)? as usize;
    let h = read_be_u32(&mut f)? as usize;
    let w = read_be_u32(&mut f)? as usize;

    let total = n * h * w;
    let mut raw = vec![0u8; total];
    f.read_exact(&mut raw)?;

    // Normalisation : pixels uint8 [0,255] → f32 [0,1]
    let data: Vec<f32> = raw.iter().map(|&b| b as f32 / 255.0).collect();
    Ok((data, n, h, w))
}

/// Charge les labels IDX. Renvoie Vec<u8> (chiffres 0..K-1).
pub fn load_idx_labels<P: AsRef<Path>>(path: P) -> io::Result<Vec<u8>> {
    let mut f = File::open(path)?;
    let magic = read_be_u32(&mut f)?;
    if magic != MAGIC_LABELS
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("magic label incorrect : 0x{magic:08x}, attendu 0x{MAGIC_LABELS:08x}"),
        ));
    }
    let n = read_be_u32(&mut f)? as usize;
    let mut labels = vec![0u8; n];
    f.read_exact(&mut labels)?;
    Ok(labels)
}

// ================================================================== //
//  MnistDataset — wrapper pratique                                    //
// ================================================================== //

pub struct MnistDataset {
    pub n: usize,
    pub h: usize,
    pub w: usize,
    pub n_classes: usize,
    /// Images aplaties en (N, H·W) row-major, normalisées
    pub images: Vec<f32>,
    /// Labels en one-hot (N, n_classes)
    pub labels_one_hot: Vec<f32>,
    /// Labels bruts (N,) pour eval / accuracy
    pub labels_raw: Vec<u8>,
}

impl MnistDataset {
    /// Charge MNIST depuis 2 fichiers IDX. Convertit les labels en one-hot.
    pub fn load_idx<P: AsRef<Path>>(images_path: P, labels_path: P) -> io::Result<Self> {
        let (images, n, h, w) = load_idx_images(images_path)?;
        let labels_raw = load_idx_labels(labels_path)?;
        if labels_raw.len() != n
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("incohérence : {} images vs {} labels", n, labels_raw.len()),
            ));
        }

        let n_classes = (*labels_raw.iter().max().unwrap_or(&0) as usize) + 1;
        let n_classes = n_classes.max(10); // MNIST = 10 classes même si max < 9
        let mut one_hot = vec![0.0f32; n * n_classes];
        for (i, &lbl) in labels_raw.iter().enumerate()
        {
            one_hot[i * n_classes + lbl as usize] = 1.0;
        }

        Ok(Self {
            n,
            h,
            w,
            n_classes,
            images,
            labels_one_hot: one_hot,
            labels_raw,
        })
    }

    /// Conversion en InMemoryDataset standard (pour DataLoader).
    pub fn into_in_memory(self) -> InMemoryDataset {
        InMemoryDataset::new(
            self.images,
            self.labels_one_hot,
            self.h * self.w,
            self.n_classes,
        )
    }

    /// Sous-échantillonnage : utile pour iter rapide ou test rapide.
    pub fn subsample(&self, max_n: usize) -> InMemoryDataset {
        let actual = max_n.min(self.n);
        let xdim = self.h * self.w;
        let ydim = self.n_classes;
        InMemoryDataset::new(
            self.images[..actual * xdim].to_vec(),
            self.labels_one_hot[..actual * ydim].to_vec(),
            xdim,
            ydim,
        )
    }
}

impl MnistDataset {
    pub fn get(&self, idx: usize) -> (Tensor, Tensor) {
        let xdim = self.h * self.w;
        let x = Tensor::from_vec(self.images[idx * xdim..(idx + 1) * xdim].to_vec(), 1, xdim);
        let y = Tensor::from_vec(
            self.labels_one_hot[idx * self.n_classes..(idx + 1) * self.n_classes].to_vec(),
            1,
            self.n_classes,
        );
        (x, y)
    }
}

impl Dataset for MnistDataset {
    fn sample(&self, idx: usize) -> (&[f32], &[f32]) {
        let xdim = self.h * self.w;
        let x = &self.images[idx * xdim..(idx + 1) * xdim];
        let y = &self.labels_one_hot[idx * self.n_classes..(idx + 1) * self.n_classes];
        (x, y)
    }
    fn n_samples(&self) -> usize {
        self.n
    }
}

// ================================================================== //
//  Tests — utilisent un fichier IDX synthétique                       //
// ================================================================== //
#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn write_synthetic_images(path: &Path, n: u32, h: u32, w: u32, value: u8) -> io::Result<()> {
        let mut f = File::create(path)?;
        f.write_all(&MAGIC_IMAGES.to_be_bytes())?;
        f.write_all(&n.to_be_bytes())?;
        f.write_all(&h.to_be_bytes())?;
        f.write_all(&w.to_be_bytes())?;
        f.write_all(&vec![value; (n * h * w) as usize])?;
        Ok(())
    }

    fn write_synthetic_labels(path: &Path, labels: &[u8]) -> io::Result<()> {
        let mut f = File::create(path)?;
        f.write_all(&MAGIC_LABELS.to_be_bytes())?;
        f.write_all(&(labels.len() as u32).to_be_bytes())?;
        f.write_all(labels)?;
        Ok(())
    }

    #[test]
    fn load_synthetic_idx() {
        let dir = std::env::temp_dir().join(format!("scirust_idx_test_{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        let img_path = dir.join("img.idx");
        let lbl_path = dir.join("lbl.idx");

        // 5 images 4×4 toutes à 128, labels = [0, 1, 2, 3, 4]
        write_synthetic_images(&img_path, 5, 4, 4, 128).unwrap();
        write_synthetic_labels(&lbl_path, &[0, 1, 2, 3, 4]).unwrap();

        let mnist = MnistDataset::load_idx(&img_path, &lbl_path).unwrap();
        assert_eq!(mnist.n, 5);
        assert_eq!((mnist.h, mnist.w), (4, 4));
        // Pixels normalisés : 128 / 255 ≈ 0.502
        assert!((mnist.images[0] - 128.0 / 255.0).abs() < 1e-5);

        // One-hot : ligne 2 (label 2) doit avoir 1.0 en index 2, 0 ailleurs
        assert_eq!(mnist.labels_one_hot[2 * 10 + 2], 1.0);
        assert_eq!(mnist.labels_one_hot[2 * 10], 0.0);

        // Cleanup
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn rejects_wrong_magic() {
        let dir = std::env::temp_dir().join(format!("scirust_idx_bad_{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("bad.idx");
        let mut f = File::create(&path).unwrap();
        f.write_all(&[0xDE, 0xAD, 0xBE, 0xEF]).unwrap();

        let result = load_idx_images(&path);
        assert!(result.is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn dataset_get_returns_normalized_tensor() {
        let dir = std::env::temp_dir().join(format!("scirust_idx_norm_{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        let img_path = dir.join("img.idx");
        let lbl_path = dir.join("lbl.idx");
        write_synthetic_images(&img_path, 3, 2, 2, 255).unwrap();
        write_synthetic_labels(&lbl_path, &[7, 8, 9]).unwrap();

        let mnist = MnistDataset::load_idx(&img_path, &lbl_path).unwrap();
        let (x, y) = mnist.get(1);
        assert_eq!(x.shape(), (1, 4));
        assert_eq!(y.shape(), (1, 10));
        // Pixel 255 → 1.0
        assert!((x.data[0] - 1.0).abs() < 1e-6);
        // Label 8 → one-hot[8] = 1
        assert_eq!(y.data[8], 1.0);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
