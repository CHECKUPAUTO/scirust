//! Data augmentation pipeline — déterministe par construction.
//!
//! Toute transformation aléatoire tire ses décisions d'un [`PcgEngine`]
//! fourni par l'appelant : même seed ⇒ mêmes augmentations, bit pour bit.
//! [`AugmentedDataset`] dérive un RNG **par échantillon** à partir de sa
//! seed et de l'index (`seed ^ splitmix(idx)`), si bien que l'augmentation
//! d'un échantillon ne dépend ni de l'ordre de visite ni des autres
//! échantillons — reproductible et parallélisable.

use crate::nn::rng::PcgEngine;

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
///
/// Le RNG est injecté : aucune transformation ne possède de source
/// d'aléa propre, c'est la condition du déterminisme bout-en-bout.
pub trait Transform: Send + Sync + TransformClone {
    /// Applique la transformation en-place sur le buffer.
    fn apply(&self, img: &mut [f32], dims: ImageDims, rng: &mut PcgEngine);
}

/// Clone spécialisé pour `Box<dyn Transform>`.
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
    fn apply(&self, img: &mut [f32], dims: ImageDims, rng: &mut PcgEngine) {
        if rng.float() >= self.p
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
    fn apply(&self, img: &mut [f32], dims: ImageDims, rng: &mut PcgEngine) {
        if rng.float() >= self.p
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

/// Random crop avec padding zéro : l'image est plongée dans un canevas
/// agrandi de `pad` de chaque côté, puis une fenêtre `crop_h × crop_w`
/// est découpée à une position aléatoire et **réécrite dans `img`**.
///
/// Contrainte : `crop_h == dims.h` et `crop_w == dims.w` (le buffer de
/// sortie a la taille de l'entrée — c'est la translation aléatoire
/// classique type CIFAR-10 `pad=4`).
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
    fn apply(&self, img: &mut [f32], dims: ImageDims, rng: &mut PcgEngine) {
        assert_eq!(
            (self.crop_h, self.crop_w),
            (dims.h, dims.w),
            "RandomCrop: crop dims must equal image dims (in-place buffer)"
        );
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

        let top = (rng.next_u32() as usize) % (padded_h - self.crop_h + 1);
        let left = (rng.next_u32() as usize) % (padded_w - self.crop_w + 1);

        for c in 0..dims.c
        {
            let c_img = c * self.crop_h * self.crop_w;
            let c_pad = c * padded_h * padded_w;
            for y in 0..self.crop_h
            {
                for x in 0..self.crop_w
                {
                    let src = c_pad + (top + y) * padded_w + (left + x);
                    img[c_img + y * self.crop_w + x] = padded[src];
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
    fn apply(&self, img: &mut [f32], dims: ImageDims, _rng: &mut PcgEngine) {
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

/// Bruit gaussien additif N(0, std) — un vrai gaussien (Box-Muller du
/// [`PcgEngine`]), pas un uniforme déguisé.
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
    fn apply(&self, img: &mut [f32], _dims: ImageDims, rng: &mut PcgEngine) {
        for x in img.iter_mut()
        {
            *x += rng.normal(0.0, self.std);
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
    fn apply(&self, img: &mut [f32], dims: ImageDims, rng: &mut PcgEngine) {
        for t in &self.transforms
        {
            t.apply(img, dims, rng);
        }
    }
}

// ──────────────────────────────────────────
// AugmentedDataset
// ──────────────────────────────────────────

use super::InMemoryDataset;

/// Mélangeur d'index (SplitMix64) : dérive un flux RNG indépendant par
/// échantillon, stable quel que soit l'ordre de parcours.
fn per_sample_seed(seed: u64, idx: usize) -> u64 {
    let mut z = (idx as u64).wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    seed ^ (z ^ (z >> 31))
}

/// Wrapper Dataset qui applique des transforms, déterministe par seed.
///
/// L'implémentation du trait `Dataset` doit renvoyer `&[f32]` (durée de vie
/// liée à `&self`), ce qui interdit d'augmenter à la volée dans `sample()`.
/// Les entrées augmentées sont donc **précalculées** à la construction
/// (`aug_x`) à partir de la seed ; `with_seed` recalcule ce cache pour une
/// nouvelle seed (utile pour ré-augmenter à chaque époque, de façon
/// reproductible).
pub struct AugmentedDataset {
    pub base: InMemoryDataset,
    pub transforms: Vec<Box<dyn Transform>>,
    pub dims: ImageDims,
    pub seed: u64,
    /// Entrées augmentées précalculées (une par échantillon de `base`).
    aug_x: Vec<Vec<f32>>,
}

impl AugmentedDataset {
    pub fn new(
        base: InMemoryDataset,
        transforms: Vec<Box<dyn Transform>>,
        dims: ImageDims,
    ) -> Self {
        let seed = 42;
        let aug_x = Self::precompute(&base, &transforms, dims, seed);
        Self {
            base,
            transforms,
            dims,
            seed,
            aug_x,
        }
    }
    pub fn from_pipeline(
        base: InMemoryDataset,
        pipeline: Compose,
        channels: usize,
        h: usize,
        w: usize,
    ) -> Self {
        let dims = ImageDims::new(channels, h, w);
        Self::new(base, pipeline.transforms, dims)
    }

    /// Applique les transforms à chaque échantillon, RNG dérivé par index.
    fn precompute(
        base: &InMemoryDataset,
        transforms: &[Box<dyn Transform>],
        dims: ImageDims,
        seed: u64,
    ) -> Vec<Vec<f32>> {
        (0..base.n_samples())
            .map(|i| {
                let (x, _y) = base.sample(i);
                let mut x_aug = x.to_vec();
                let mut rng = PcgEngine::new(per_sample_seed(seed, i));
                for t in transforms
                {
                    t.apply(&mut x_aug, dims, &mut rng);
                }
                x_aug
            })
            .collect()
    }

    /// Rejoue l'augmentation avec une nouvelle seed (reproductible).
    #[must_use]
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self.aug_x = Self::precompute(&self.base, &self.transforms, self.dims, seed);
        self
    }

    /// Augmentation fraîche et possédée d'un échantillon (même flux RNG
    /// que le cache : déterministe pour (seed, idx)).
    pub fn sample(&self, idx: usize) -> (Vec<f32>, &[f32]) {
        let (x, y) = self.base.sample(idx);
        let mut x_aug = x.to_vec();
        let mut rng = PcgEngine::new(per_sample_seed(self.seed, idx));
        for t in &self.transforms
        {
            t.apply(&mut x_aug, self.dims, &mut rng);
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

    fn rng() -> PcgEngine {
        PcgEngine::new(7)
    }

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
        flip.apply(&mut img, dims, &mut rng());
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
        flip.apply(&mut img, dims, &mut rng());
        // Row 0 and Row 1 swapped
        assert_eq!(img, vec![4.0, 5.0, 6.0, 1.0, 2.0, 3.0]);
    }

    #[test]
    fn normalize_scales_correctly() {
        let mut img = vec![0.1307, 0.1307, 0.1307];
        let dims = ImageDims::new(1, 1, 3);
        let norm = Normalize::mnist();
        norm.apply(&mut img, dims, &mut rng());
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
        pipeline.apply(&mut img, dims, &mut rng());
        // H then V: [1,2,3,4,5,6] -> H -> [3,2,1,6,5,4] -> V -> [6,5,4,3,2,1]
        assert_eq!(img, vec![6.0, 5.0, 4.0, 3.0, 2.0, 1.0]);
    }

    #[test]
    fn random_crop_actually_writes_the_image() {
        // Regression: RandomCrop used to compute the crop and silently
        // discard it. With pad>0 some random window must come from the
        // zero padding or shift the content — either way, for a non-zero
        // image and a translating window, the buffer must change for at
        // least one seed among several.
        let dims = ImageDims::new(1, 2, 2);
        let crop = RandomCrop::new(2, 2, 1);
        let base = vec![1.0, 2.0, 3.0, 4.0];
        let mut changed = false;
        for seed in 0..8
        {
            let mut img = base.clone();
            let mut r = PcgEngine::new(seed);
            crop.apply(&mut img, dims, &mut r);
            if img != base
            {
                changed = true;
            }
        }
        assert!(changed, "RandomCrop must write its result into the buffer");
    }

    #[test]
    fn augmentation_is_deterministic_per_seed() {
        let dims = ImageDims::new(1, 2, 3);
        let pipeline: Vec<Box<dyn Transform>> = vec![
            Box::new(RandomFlipH { p: 0.5 }),
            Box::new(AddGaussianNoise { std: 0.1 }),
        ];
        let mk = |seed: u64| {
            let base = InMemoryDataset::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![0.0], 6, 1);
            AugmentedDataset::new(base, pipeline.clone(), dims).with_seed(seed)
        };
        let a = mk(11).sample(0).0;
        let b = mk(11).sample(0).0;
        let c = mk(12).sample(0).0;
        let bits = |v: &Vec<f32>| v.iter().map(|f| f.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(&a), bits(&b), "same seed ⇒ bit-identical augmentation");
        assert_ne!(
            bits(&a),
            bits(&c),
            "different seed ⇒ different augmentation"
        );
    }

    #[test]
    fn per_sample_rng_is_order_independent() {
        let dims = ImageDims::new(1, 1, 4);
        let pipeline: Vec<Box<dyn Transform>> = vec![Box::new(AddGaussianNoise { std: 1.0 })];
        let base = InMemoryDataset::new((0..8).map(|i| i as f32).collect(), vec![0.0, 0.0], 4, 1);
        let ds = AugmentedDataset::new(base, pipeline, dims).with_seed(5);
        // Sampling 1 then 0 gives the same values as 0 then 1.
        let s1_first = ds.sample(1).0;
        let s0_after = ds.sample(0).0;
        let s0_first = ds.sample(0).0;
        let s1_after = ds.sample(1).0;
        assert_eq!(s0_first, s0_after);
        assert_eq!(s1_first, s1_after);
    }
}
