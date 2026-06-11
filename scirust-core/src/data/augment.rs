//! Data augmentation pipeline — V10A

/// Dimensions d'une image (channels, height, width)
#[derive(Clone, Copy, Debug)]
pub struct ImageDims {
    pub c: usize,
    pub h: usize,
    pub w: usize,
}

impl ImageDims {
    pub fn new(c: usize, h: usize, w: usize) -> Self {
        Self { c, h, w }
    }
    pub fn n_pixels(&self) -> usize {
        self.c * self.h * self.w
    }
}

/// Trait pour une transformation d'augmentation.
pub trait Transform: Send + Sync + TransformClone {
    /// Applique la transformation en-place sur le buffer.
    fn apply(&self, img: &mut [f32], dims: ImageDims);
}

/// Clone spécialisé pour Box<dyn Transform>.
pub trait TransformClone {
    fn clone_box(&self) -> Box<dyn Transform>;
}

impl<T> TransformClone for T
where
    T: Transform + Clone + 'static,
{
    fn clone_box(&self) -> Box<dyn Transform> {
        Box::new(self.clone())
    }
}

impl Clone for Box<dyn Transform> {
    fn clone(&self) -> Self {
        self.clone_box()
    }
}

// ──────────────────────────────────────────
// Transforms concrets
// ──────────────────────────────────────────

/// Flip horizontal aléatoire.
#[derive(Clone, Debug)]
pub struct RandomFlipH {
    pub p: f32,
}

impl RandomFlipH {
    pub fn new(p: f32) -> Self {
        Self { p }
    }
}

impl Transform for RandomFlipH {
    fn apply(&self, img: &mut [f32], dims: ImageDims) {
        use rand::random;
        if random::<f32>() >= self.p
        {
            return;
        }
        for c in 0..dims.c
        {
            let c_off = c * dims.h * dims.w;
            for y in 0..dims.h
            {
                let row_off = c_off + y * dims.w;
                for x in 0..dims.w / 2
                {
                    let a = row_off + x;
                    let b = row_off + (dims.w - 1 - x);
                    img.swap(a, b);
                }
            }
        }
    }
}

/// Flip vertical aléatoire.
#[derive(Clone, Debug)]
pub struct RandomFlipV {
    pub p: f32,
}

impl RandomFlipV {
    pub fn new(p: f32) -> Self {
        Self { p }
    }
}

impl Transform for RandomFlipV {
    fn apply(&self, img: &mut [f32], dims: ImageDims) {
        use rand::random;
        if random::<f32>() >= self.p
        {
            return;
        }
        for c in 0..dims.c
        {
            let c_off = c * dims.h * dims.w;
            for y in 0..dims.h / 2
            {
                let row_a = c_off + y * dims.w;
                let row_b = c_off + (dims.h - 1 - y) * dims.w;
                for x in 0..dims.w
                {
                    img.swap(row_a + x, row_b + x);
                }
            }
        }
    }
}

/// Random crop avec padding (reflect).
#[derive(Clone, Debug)]
pub struct RandomCrop {
    pub crop_h: usize,
    pub crop_w: usize,
    pub pad: usize,
}

impl RandomCrop {
    pub fn new(crop_h: usize, crop_w: usize, pad: usize) -> Self {
        Self {
            crop_h,
            crop_w,
            pad,
        }
    }
}

impl Transform for RandomCrop {
    fn apply(&self, img: &mut [f32], dims: ImageDims) {
        use rand::{Rng, thread_rng};
        let padded_h = dims.h + 2 * self.pad;
        let padded_w = dims.w + 2 * self.pad;
        let mut padded = vec![0.0f32; dims.c * padded_h * padded_w];

        for c in 0..dims.c
        {
            let c_in = c * dims.h * dims.w;
            let c_out = c * padded_h * padded_w;
            for y in 0..dims.h
            {
                for x in 0..dims.w
                {
                    let py = y + self.pad;
                    let px = x + self.pad;
                    padded[c_out + py * padded_w + px] = img[c_in + y * dims.w + x];
                }
            }
        }

        let top = thread_rng().gen_range(0..=padded_h - self.crop_h);
        let left = thread_rng().gen_range(0..=padded_w - self.crop_w);

        let mut out = vec![0.0f32; dims.c * self.crop_h * self.crop_w];
        for c in 0..dims.c
        {
            let c_out = c * self.crop_h * self.crop_w;
            let c_pad = c * padded_h * padded_w;
            for y in 0..self.crop_h
            {
                for x in 0..self.crop_w
                {
                    let src = c_pad + (top + y) * padded_w + (left + x);
                    out[c_out + y * self.crop_w + x] = padded[src];
                }
            }
        }
    }
}

/// Normalisation par canal.
#[derive(Clone, Debug)]
pub struct Normalize {
    pub mean: Vec<f32>,
    pub std: Vec<f32>,
}

impl Normalize {
    pub fn new(mean: Vec<f32>, std: Vec<f32>) -> Self {
        Self { mean, std }
    }
    pub fn mnist() -> Self {
        Self::new(vec![0.1307], vec![0.3081])
    }
    pub fn cifar10() -> Self {
        Self::new(vec![0.4914, 0.4822, 0.4465], vec![0.2470, 0.2435, 0.2616])
    }
}

impl Transform for Normalize {
    fn apply(&self, img: &mut [f32], dims: ImageDims) {
        assert_eq!(self.mean.len(), dims.c);
        assert_eq!(self.std.len(), dims.c);
        for c in 0..dims.c
        {
            let c_off = c * dims.h * dims.w;
            for i in 0..(dims.h * dims.w)
            {
                img[c_off + i] = (img[c_off + i] - self.mean[c]) / self.std[c];
            }
        }
    }
}

/// Bruit gaussien additif.
#[derive(Clone, Debug)]
pub struct AddGaussianNoise {
    pub std: f32,
}

impl AddGaussianNoise {
    pub fn new(std: f32) -> Self {
        Self { std }
    }
}

impl Transform for AddGaussianNoise {
    fn apply(&self, img: &mut [f32], _dims: ImageDims) {
        use rand::random;
        for x in img.iter_mut()
        {
            let noise: f32 = random::<f32>() * 2.0 - 1.0;
            *x += noise * self.std;
        }
    }
}

// ──────────────────────────────────────────
// Compose
// ──────────────────────────────────────────

/// Chaîne de transformations.
#[derive(Clone)]
pub struct Compose {
    pub transforms: Vec<Box<dyn Transform>>,
}

impl Compose {
    pub fn new() -> Self {
        Self {
            transforms: Vec::new(),
        }
    }
    #[must_use]
    #[allow(clippy::should_implement_trait)]
    pub fn add(mut self, t: impl Transform + 'static) -> Self {
        self.transforms.push(Box::new(t));
        self
    }
}

impl Default for Compose {
    fn default() -> Self {
        Self::new()
    }
}

impl Transform for Compose {
    fn apply(&self, img: &mut [f32], dims: ImageDims) {
        for t in &self.transforms
        {
            t.apply(img, dims);
        }
    }
}

// ──────────────────────────────────────────
// AugmentedDataset
// ──────────────────────────────────────────

use super::InMemoryDataset;

/// Wrapper Dataset qui applique des transforms.
///
/// L'implémentation du trait `Dataset` doit renvoyer `&[f32]` (durée de vie liée
/// à `&self`), ce qui interdit d'augmenter à la volée dans `sample()` (la donnée
/// augmentée serait locale). On **précalcule** donc les entrées augmentées une
/// fois à la construction (`aug_x`) ; `Dataset::sample` renvoie une référence
/// vers cette donnée augmentée. Pour une augmentation fraîche à chaque appel,
/// utiliser la méthode inhérente `sample()` (qui renvoie un `Vec` possédé) ou
/// `augment_batch()`.
pub struct AugmentedDataset {
    pub base: InMemoryDataset,
    pub transforms: Vec<Box<dyn Transform>>,
    pub dims: ImageDims,
    /// Entrées augmentées précalculées (une par échantillon de `base`).
    aug_x: Vec<Vec<f32>>,
}

impl AugmentedDataset {
    pub fn new(
        base: InMemoryDataset,
        transforms: Vec<Box<dyn Transform>>,
        dims: ImageDims,
    ) -> Self {
        let aug_x = Self::precompute(&base, &transforms, dims);
        Self {
            base,
            transforms,
            dims,
            aug_x,
        }
    }
    pub fn from_pipeline(
        base: InMemoryDataset,
        pipeline: Compose,
        _channels: usize,
        h: usize,
        w: usize,
    ) -> Self {
        let dims = ImageDims::new(_channels, h, w);
        let transforms = pipeline.transforms;
        let aug_x = Self::precompute(&base, &transforms, dims);
        Self {
            base,
            transforms,
            dims,
            aug_x,
        }
    }

    /// Applique les transforms à chaque échantillon de `base` une fois.
    fn precompute(
        base: &InMemoryDataset,
        transforms: &[Box<dyn Transform>],
        dims: ImageDims,
    ) -> Vec<Vec<f32>> {
        (0..base.n_samples())
            .map(|i| {
                let (x, _y) = base.sample(i);
                let mut x_aug = x.to_vec();
                for t in transforms
                {
                    t.apply(&mut x_aug, dims);
                }
                x_aug
            })
            .collect()
    }
    pub fn with_seed(self, _seed: u64) -> Self {
        self
    }

    pub fn sample(&self, idx: usize) -> (Vec<f32>, &[f32]) {
        let (x, y) = self.base.sample(idx);
        let mut x_aug = x.to_vec();
        for t in &self.transforms
        {
            t.apply(&mut x_aug, self.dims);
        }
        (x_aug, y)
    }

    pub fn n_samples(&self) -> usize {
        self.base.n_samples()
    }
    pub fn len(&self) -> usize {
        self.n_samples()
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl super::Dataset for AugmentedDataset {
    fn sample(&self, idx: usize) -> (&[f32], &[f32]) {
        // Renvoie l'entrée augmentée précalculée (durée de vie liée à `&self`),
        // et le label d'origine inchangé.
        let (_x, y) = self.base.sample(idx);
        (&self.aug_x[idx], y)
    }
    fn n_samples(&self) -> usize {
        self.base.n_samples()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn image_dims_n_pixels() {
        let d = ImageDims::new(3, 32, 32);
        assert_eq!(d.n_pixels(), 3072);
    }

    #[test]
    fn random_flip_h_swaps_pixels() {
        let mut img = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let dims = ImageDims::new(1, 2, 3);
        let flip = RandomFlipH { p: 1.0 }; // always flip
        flip.apply(&mut img, dims);
        // Row 0: [1,2,3] -> [3,2,1]
        // Row 1: [4,5,6] -> [6,5,4]
        assert_eq!(img, vec![3.0, 2.0, 1.0, 6.0, 5.0, 4.0]);
    }

    #[test]
    fn dataset_trait_sample_returns_augmented_data() {
        // Regression: the `Dataset::sample` impl used to return the *unaugmented*
        // base sample. It must now return augmented data.
        use crate::data::Dataset;
        let base = InMemoryDataset::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![0.0], 6, 1);
        let dims = ImageDims::new(1, 2, 3);
        let transforms: Vec<Box<dyn Transform>> = vec![Box::new(RandomFlipH { p: 1.0 })];
        let ds = AugmentedDataset::new(base, transforms, dims);
        let ds_ref: &dyn Dataset = &ds;
        let (x, _y) = ds_ref.sample(0);
        assert_eq!(
            x,
            &[3.0, 2.0, 1.0, 6.0, 5.0, 4.0],
            "Dataset::sample must return augmented (flipped) data"
        );
    }

    #[test]
    fn random_flip_v_swaps_rows() {
        let mut img = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let dims = ImageDims::new(1, 2, 3);
        let flip = RandomFlipV { p: 1.0 }; // always flip
        flip.apply(&mut img, dims);
        // Row 0 and Row 1 swapped
        assert_eq!(img, vec![4.0, 5.0, 6.0, 1.0, 2.0, 3.0]);
    }

    #[test]
    fn normalize_scales_correctly() {
        let mut img = vec![0.1307, 0.1307, 0.1307];
        let dims = ImageDims::new(1, 1, 3);
        let norm = Normalize::mnist();
        norm.apply(&mut img, dims);
        // (x - 0.1307) / 0.3081 = 0.0 for all
        assert!(img.iter().all(|v| v.abs() < 1e-5));
    }

    #[test]
    fn compose_chains_transforms() {
        let mut img = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let dims = ImageDims::new(1, 2, 3);
        let pipeline = Compose::new()
            .add(RandomFlipH { p: 1.0 })
            .add(RandomFlipV { p: 1.0 });
        pipeline.apply(&mut img, dims);
        // H then V: [1,2,3,4,5,6] -> H -> [3,2,1,6,5,4] -> V -> [6,5,4,3,2,1]
        assert_eq!(img, vec![6.0, 5.0, 4.0, 3.0, 2.0, 1.0]);
    }
}
