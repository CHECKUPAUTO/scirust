// scirust-core/src/data/augment.rs
//
// Pipeline d'augmentation pour le DataLoader.
//
// Architecture :
//
//   Trait Transform           — opération sur un Tensor (mut), seedable via PcgEngine
//   ├─ RandomFlipH            — symétrie horizontale (flip gauche/droite)
//   ├─ RandomFlipV            — symétrie verticale (flip haut/bas)
//   ├─ RandomCrop             — pad puis crop aléatoire
//   ├─ Normalize              — (x - mean) / std par canal
//   ├─ AddGaussianNoise       — bruit gaussien additif
//   └─ Compose                — chaîne plusieurs Transform
//
//   Trait IntoTransform       — pour faciliter la conversion en Box
//   AugmentedDataset<D, T>    — wrap un Dataset avec un Transform
//
// USAGE :
//
//   let aug = Compose::new()
//       .add(RandomFlipH::new(0.5))
//       .add(RandomCrop::new(28, 28, 4))
//       .add(Normalize::with_per_channel(&[0.1307], &[0.3081]));   // MNIST stats
//
//   let augmented = AugmentedDataset::new(train_ds, aug, 1, 28, 28);
//   let mut loader = DataLoader::new(augmented, 64, true, 42);
//
// CONVENTION : les images sont passées au Transform sous forme (1, C·H·W)
// row-major (cohérent avec notre stockage). Le Transform connaît C, H, W
// via les méthodes `dims_required` ou via la struct AugmentedDataset.

use crate::autodiff::reverse::Tensor;
use crate::nn::rng::PcgEngine;
use crate::data::{Dataset, InMemoryDataset};

// ================================================================== //
//  Trait Transform                                                    //
// ================================================================== //

/// Une transformation in-place sur un Tensor d'image.
/// Le Tensor est de shape (1, C·H·W) row-major.
/// L'implémenteur doit lire C, H, W depuis la struct ImageDims associée.
pub trait Transform: Send + Sync {
    /// Applique la transformation sur l'image.
    fn apply(&self, x: &mut Tensor, dims: ImageDims, rng: &mut PcgEngine);

    /// Permet le clone via Box pour AugmentedDataset (Clone n'est pas
    /// object-safe, alors on a notre propre method).
    fn box_clone(&self) -> Box<dyn Transform>;
}

#[derive(Clone, Copy, Debug)]
pub struct ImageDims {
    pub c: usize,
    pub h: usize,
    pub w: usize,
}

impl ImageDims {
    pub fn new(c: usize, h: usize, w: usize) -> Self { Self { c, h, w } }
    pub fn total(&self) -> usize { self.c * self.h * self.w }
}

// ================================================================== //
//  RandomFlipH — symétrie horizontale (flip left-right)               //
// ================================================================== //

#[derive(Clone)]
pub struct RandomFlipH { pub prob: f32 }

impl RandomFlipH {
    pub fn new(prob: f32) -> Self {
        assert!(prob >= 0.0 && prob <= 1.0);
        Self { prob }
    }
}

impl Transform for RandomFlipH {
    fn apply(&self, x: &mut Tensor, dims: ImageDims, rng: &mut PcgEngine) {
        if rng.float() >= self.prob { return; }
        // Flip chaque ligne de chaque canal
        for c in 0..dims.c {
            for h in 0..dims.h {
                let row_start = c * dims.h * dims.w + h * dims.w;
                for w_idx in 0..dims.w / 2 {
                    let left  = row_start + w_idx;
                    let right = row_start + dims.w - 1 - w_idx;
                    x.data.swap(left, right);
                }
            }
        }
    }
    fn box_clone(&self) -> Box<dyn Transform> { Box::new(self.clone()) }
}

// ================================================================== //
//  RandomFlipV — symétrie verticale (flip up-down)                    //
// ================================================================== //

#[derive(Clone)]
pub struct RandomFlipV { pub prob: f32 }

impl RandomFlipV {
    pub fn new(prob: f32) -> Self {
        assert!(prob >= 0.0 && prob <= 1.0);
        Self { prob }
    }
}

impl Transform for RandomFlipV {
    fn apply(&self, x: &mut Tensor, dims: ImageDims, rng: &mut PcgEngine) {
        if rng.float() >= self.prob { return; }
        // Échange ligne r et (h-1-r) pour chaque canal
        for c in 0..dims.c {
            let plane_start = c * dims.h * dims.w;
            for r in 0..dims.h / 2 {
                let top    = plane_start + r * dims.w;
                let bottom = plane_start + (dims.h - 1 - r) * dims.w;
                for w_idx in 0..dims.w {
                    x.data.swap(top + w_idx, bottom + w_idx);
                }
            }
        }
    }
    fn box_clone(&self) -> Box<dyn Transform> { Box::new(self.clone()) }
}

// ================================================================== //
//  RandomCrop — pad puis crop aléatoire                               //
// ================================================================== //

/// Padding constant 0, puis crop d'une fenêtre `(target_h, target_w)`
/// aléatoirement positionnée.
///
/// Le résultat conserve les dimensions originales si target == input.
/// `padding` ajoute des bords avant le crop : avec padding=4 et target=32
/// sur un input 32×32, le crop sample dans une zone 40×40.
#[derive(Clone)]
pub struct RandomCrop {
    pub target_h: usize,
    pub target_w: usize,
    pub padding:  usize,
}

impl RandomCrop {
    pub fn new(target_h: usize, target_w: usize, padding: usize) -> Self {
        Self { target_h, target_w, padding }
    }
}

impl Transform for RandomCrop {
    fn apply(&self, x: &mut Tensor, dims: ImageDims, rng: &mut PcgEngine) {
        let pad = self.padding;
        let padded_h = dims.h + 2 * pad;
        let padded_w = dims.w + 2 * pad;

        // 1. Pad (alloue un buffer paddé temporaire)
        let mut padded = vec![0.0f32; dims.c * padded_h * padded_w];
        for c in 0..dims.c {
            for r in 0..dims.h {
                for col in 0..dims.w {
                    let src = c * dims.h * dims.w + r * dims.w + col;
                    let dst = c * padded_h * padded_w
                              + (r + pad) * padded_w + (col + pad);
                    padded[dst] = x.data[src];
                }
            }
        }

        // 2. Crop aléatoire
        let max_top  = padded_h.saturating_sub(self.target_h);
        let max_left = padded_w.saturating_sub(self.target_w);
        let top  = if max_top  == 0 { 0 } else { (rng.next_u32() as usize) % (max_top  + 1) };
        let left = if max_left == 0 { 0 } else { (rng.next_u32() as usize) % (max_left + 1) };

        // 3. Réécrit dans x.data (qui doit faire la nouvelle taille,
        //    sinon on alloue une nouvelle Tensor — pas in-place strict)
        if dims.h == self.target_h && dims.w == self.target_w {
            // Cas commun : taille identique → in-place
            for c in 0..dims.c {
                for r in 0..self.target_h {
                    for col in 0..self.target_w {
                        let src = c * padded_h * padded_w
                                  + (r + top) * padded_w + (col + left);
                        let dst = c * self.target_h * self.target_w
                                  + r * self.target_w + col;
                        x.data[dst] = padded[src];
                    }
                }
            }
        } else {
            // Cas où on change la taille : on remplace x.data
            let mut out = vec![0.0f32; dims.c * self.target_h * self.target_w];
            for c in 0..dims.c {
                for r in 0..self.target_h {
                    for col in 0..self.target_w {
                        let src = c * padded_h * padded_w
                                  + (r + top) * padded_w + (col + left);
                        let dst = c * self.target_h * self.target_w
                                  + r * self.target_w + col;
                        out[dst] = padded[src];
                    }
                }
            }
            x.data = out;
            x.cols = dims.c * self.target_h * self.target_w;
            // x.rows reste 1 (notre convention)
        }
    }
    fn box_clone(&self) -> Box<dyn Transform> { Box::new(self.clone()) }
}

// ================================================================== //
//  Normalize — (x - mean) / std par canal                             //
// ================================================================== //

#[derive(Clone)]
pub struct Normalize {
    pub mean: Vec<f32>,
    pub std:  Vec<f32>,
}

impl Normalize {
    pub fn new(mean: Vec<f32>, std: Vec<f32>) -> Self {
        assert_eq!(mean.len(), std.len(),
                   "Normalize: mean et std doivent avoir la même taille (= n_channels)");
        Self { mean, std }
    }

    /// Helper : valeurs typiques pour MNIST (image en niveaux de gris)
    pub fn mnist() -> Self {
        Self::new(vec![0.1307], vec![0.3081])
    }

    /// Helper : valeurs typiques pour CIFAR-10 (RGB)
    pub fn cifar10() -> Self {
        Self::new(
            vec![0.4914, 0.4822, 0.4465],
            vec![0.2470, 0.2435, 0.2616],
        )
    }
}

impl Transform for Normalize {
    fn apply(&self, x: &mut Tensor, dims: ImageDims, _rng: &mut PcgEngine) {
        assert_eq!(self.mean.len(), dims.c,
                   "Normalize: nb canaux mean ({}) ≠ dims.c ({})",
                   self.mean.len(), dims.c);
        for c in 0..dims.c {
            let plane = c * dims.h * dims.w;
            for i in 0..dims.h * dims.w {
                x.data[plane + i] = (x.data[plane + i] - self.mean[c]) / self.std[c];
            }
        }
    }
    fn box_clone(&self) -> Box<dyn Transform> { Box::new(self.clone()) }
}

// ================================================================== //
//  AddGaussianNoise — bruit additif gaussien                          //
// ================================================================== //

#[derive(Clone)]
pub struct AddGaussianNoise { pub stddev: f32 }

impl AddGaussianNoise {
    pub fn new(stddev: f32) -> Self {
        assert!(stddev >= 0.0);
        Self { stddev }
    }
}

impl Transform for AddGaussianNoise {
    fn apply(&self, x: &mut Tensor, _dims: ImageDims, rng: &mut PcgEngine) {
        if self.stddev == 0.0 { return; }
        // Box-Muller : à partir de 2 uniformes, génère 2 gaussiennes N(0, 1)
        // On les multiplie par stddev avant ajout.
        let n = x.data.len();
        let mut i = 0;
        while i + 1 < n {
            let u1 = rng.float().max(1e-12);
            let u2 = rng.float();
            let r = (-2.0 * u1.ln()).sqrt();
            let theta = 2.0 * std::f32::consts::PI * u2;
            x.data[i]     += self.stddev * r * theta.cos();
            x.data[i + 1] += self.stddev * r * theta.sin();
            i += 2;
        }
        // Si n est impair, on traite le dernier élément séparément
        if i < n {
            let u1 = rng.float().max(1e-12);
            let u2 = rng.float();
            let r = (-2.0 * u1.ln()).sqrt();
            x.data[i] += self.stddev * r * (2.0 * std::f32::consts::PI * u2).cos();
        }
    }
    fn box_clone(&self) -> Box<dyn Transform> { Box::new(self.clone()) }
}

// ================================================================== //
//  Compose — chaîne plusieurs Transforms                              //
// ================================================================== //

pub struct Compose {
    transforms: Vec<Box<dyn Transform>>,
}

impl Compose {
    pub fn new() -> Self { Self { transforms: Vec::new() } }

    pub fn add<T: Transform + 'static>(mut self, t: T) -> Self {
        self.transforms.push(Box::new(t));
        self
    }

    pub fn add_box(mut self, t: Box<dyn Transform>) -> Self {
        self.transforms.push(t);
        self
    }

    pub fn len(&self) -> usize { self.transforms.len() }
    pub fn is_empty(&self) -> bool { self.transforms.is_empty() }
}

impl Default for Compose {
    fn default() -> Self { Self::new() }
}

impl Transform for Compose {
    fn apply(&self, x: &mut Tensor, dims: ImageDims, rng: &mut PcgEngine) {
        // ATTENTION : si une transformation modifie les dimensions
        // (ex: RandomCrop avec target ≠ input), les transformations
        // suivantes verront ces nouvelles dimensions. Pour cette PR
        // on suppose que toutes les transformations préservent (c, h, w),
        // ce qui est le cas commun (RandomCrop avec target == input + padding).
        for t in &self.transforms {
            t.apply(x, dims, rng);
        }
    }
    fn box_clone(&self) -> Box<dyn Transform> {
        let mut c = Compose::new();
        for t in &self.transforms { c.transforms.push(t.box_clone()); }
        Box::new(c)
    }
}

// ================================================================== //
//  AugmentedDataset — wrap un Dataset avec un Transform               //
// ================================================================== //

pub struct AugmentedDataset {
    base:       InMemoryDataset,    // pour simplicité, on prend un type concret
    transform:  Box<dyn Transform>,
    dims:       ImageDims,
    rng_seed:   u64,
}

impl AugmentedDataset {
    pub fn new<T: Transform + 'static>(
        base: InMemoryDataset,
        transform: T,
        c: usize, h: usize, w: usize,
    ) -> Self {
        assert_eq!(base.x_dim, c * h * w,
                   "AugmentedDataset: x_dim={} != c·h·w={}",
                   base.x_dim, c * h * w);
        Self {
            base,
            transform: Box::new(transform),
            dims: ImageDims::new(c, h, w),
            rng_seed: 0,
        }
    }

    pub fn with_seed(mut self, seed: u64) -> Self {
        self.rng_seed = seed;
        self
    }
}

impl Dataset for AugmentedDataset {
    fn len(&self) -> usize { self.base.len() }
    fn x_features(&self) -> usize { self.base.x_features() }
    fn y_features(&self) -> usize { self.base.y_features() }

    fn get(&self, idx: usize) -> (Tensor, Tensor) {
        let (mut x, y) = self.base.get(idx);
        // RNG seedé par (rng_seed, idx) → reproductible mais varié par échantillon
        let mut rng = PcgEngine::new(self.rng_seed.wrapping_add(idx as u64));
        self.transform.apply(&mut x, self.dims, &mut rng);
        (x, y)
    }
}

// ================================================================== //
//  Tests                                                              //
// ================================================================== //
#[cfg(test)]
mod tests {
    use super::*;

    fn checkered_image() -> (Tensor, ImageDims) {
        // Image 1 canal 4×4 avec damier 0/1
        // .  X  .  X
        // X  .  X  .
        // .  X  .  X
        // X  .  X  .
        let data = vec![
            0.0, 1.0, 0.0, 1.0,
            1.0, 0.0, 1.0, 0.0,
            0.0, 1.0, 0.0, 1.0,
            1.0, 0.0, 1.0, 0.0,
        ];
        (Tensor::from_vec(data, 1, 16), ImageDims::new(1, 4, 4))
    }

    #[test]
    fn flip_h_swaps_columns() {
        let (mut img, dims) = checkered_image();
        let mut rng = PcgEngine::new(0);
        let flip = RandomFlipH::new(1.0);  // proba 1 → toujours
        flip.apply(&mut img, dims, &mut rng);
        // Première ligne après flip : [1, 0, 1, 0]
        assert_eq!(&img.data[0..4], &[1.0, 0.0, 1.0, 0.0]);
    }

    #[test]
    fn flip_h_zero_proba_no_op() {
        let (mut img, dims) = checkered_image();
        let original = img.data.clone();
        let mut rng = PcgEngine::new(0);
        let flip = RandomFlipH::new(0.0);
        flip.apply(&mut img, dims, &mut rng);
        assert_eq!(img.data, original);
    }

    #[test]
    fn flip_v_swaps_rows() {
        let (mut img, dims) = checkered_image();
        let mut rng = PcgEngine::new(0);
        let flip = RandomFlipV::new(1.0);
        flip.apply(&mut img, dims, &mut rng);
        // Première ligne après flip = ancienne dernière ligne
        assert_eq!(&img.data[0..4], &[1.0, 0.0, 1.0, 0.0]);
        // Dernière ligne = ancienne première
        assert_eq!(&img.data[12..16], &[0.0, 1.0, 0.0, 1.0]);
    }

    #[test]
    fn random_crop_same_size_in_place() {
        let (mut img, dims) = checkered_image();
        let mut rng = PcgEngine::new(42);
        let crop = RandomCrop::new(4, 4, 1);   // pad 1 puis crop 4×4
        crop.apply(&mut img, dims, &mut rng);
        // Le résultat fait toujours 16 éléments (1 canal × 4 × 4)
        assert_eq!(img.data.len(), 16);
    }

    #[test]
    fn normalize_centers_data() {
        let mut img = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], 1, 4);
        let dims = ImageDims::new(1, 2, 2);
        let mut rng = PcgEngine::new(0);
        let norm = Normalize::new(vec![2.5], vec![1.0]);
        norm.apply(&mut img, dims, &mut rng);
        // x - 2.5 : [-1.5, -0.5, 0.5, 1.5]
        assert_eq!(img.data, vec![-1.5, -0.5, 0.5, 1.5]);
    }

    #[test]
    fn normalize_per_channel() {
        // 2 canaux, 2x2 chacun
        let mut img = Tensor::from_vec(vec![
            10.0, 20.0, 30.0, 40.0,    // canal 0
            100.0, 200.0, 300.0, 400.0, // canal 1
        ], 1, 8);
        let dims = ImageDims::new(2, 2, 2);
        let mut rng = PcgEngine::new(0);
        let norm = Normalize::new(vec![25.0, 250.0], vec![10.0, 100.0]);
        norm.apply(&mut img, dims, &mut rng);
        // Canal 0 : (x - 25) / 10
        assert!((img.data[0] - (-1.5)).abs() < 1e-5);
        // Canal 1 : (x - 250) / 100
        assert!((img.data[4] - (-1.5)).abs() < 1e-5);
        assert!((img.data[7] -  1.5).abs() < 1e-5);
    }

    #[test]
    fn gaussian_noise_changes_data() {
        let mut img = Tensor::from_vec(vec![0.0; 1000], 1, 1000);
        let dims = ImageDims::new(1, 100, 10);
        let mut rng = PcgEngine::new(42);
        let noise = AddGaussianNoise::new(0.1);
        noise.apply(&mut img, dims, &mut rng);
        // Au moins certaines valeurs ne sont plus 0
        let nonzero = img.data.iter().filter(|&&v| v != 0.0).count();
        assert!(nonzero > 950, "got {nonzero} nonzero values");
        // Stats grossières : moyenne ~0, std ~0.1
        let mean: f32 = img.data.iter().sum::<f32>() / 1000.0;
        assert!(mean.abs() < 0.05, "mean = {mean}");
    }

    #[test]
    fn compose_chains_transforms() {
        let (img, dims) = checkered_image();
        let mut img1 = img.clone();
        let mut img2 = img.clone();
        let mut rng = PcgEngine::new(0);

        // Direct : flip H puis Normalize
        RandomFlipH::new(1.0).apply(&mut img1, dims, &mut rng);
        Normalize::new(vec![0.5], vec![1.0]).apply(&mut img1, dims, &mut rng);

        // Compose : devrait donner le même résultat
        let mut rng2 = PcgEngine::new(0);
        let comp = Compose::new()
            .add(RandomFlipH::new(1.0))
            .add(Normalize::new(vec![0.5], vec![1.0]));
        comp.apply(&mut img2, dims, &mut rng2);

        assert_eq!(img1.data, img2.data);
    }

    #[test]
    fn augmented_dataset_applies_per_get() {
        let raw_x: Vec<f32> = vec![
            // 3 images 1x2x2
            1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
            9.0, 10.0, 11.0, 12.0,
        ];
        let raw_y: Vec<f32> = vec![1.0, 0.0, 0.0, 1.0, 0.0, 1.0];
        let base = InMemoryDataset::new(raw_x, raw_y, 4, 2);
        // Normalize centré : (x - 5) / 1
        let aug = AugmentedDataset::new(
            base,
            Normalize::new(vec![5.0], vec![1.0]),
            1, 2, 2,
        );
        let (x, _) = aug.get(0);
        // Première image : [1, 2, 3, 4] - 5 = [-4, -3, -2, -1]
        assert_eq!(x.data, vec![-4.0, -3.0, -2.0, -1.0]);
    }

    #[test]
    fn augmented_dataset_reproducible_with_seed() {
        let base = InMemoryDataset::new(
            vec![1.0; 16], vec![1.0, 0.0, 0.0, 1.0], 16, 2,
        );
        let aug1 = AugmentedDataset::new(
            base.clone(),
            AddGaussianNoise::new(0.1),
            1, 4, 4,
        ).with_seed(42);
        let aug2 = AugmentedDataset::new(
            base,
            AddGaussianNoise::new(0.1),
            1, 4, 4,
        ).with_seed(42);

        let (x1, _) = aug1.get(0);
        let (x2, _) = aug2.get(0);
        assert_eq!(x1.data, x2.data);
    }
}

// Note : InMemoryDataset doit être Clone pour le test ci-dessus.
// Si ce n'est pas le cas, l'agent ajoute une impl Clone trivialement
// (tous les champs sont déjà Clone).
