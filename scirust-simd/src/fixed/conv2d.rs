// scirust-simd/src/fixed/conv2d.rs
//
// # Convolution 2D quantifiée déterministe
//
// [`conv2d`] : extension bidimensionnelle de [`super::conv::conv1d`], pour les
// données à deux dimensions spatiales (images, spectrogrammes temps-fréquence)
// d'un CNN léger. Même technique **im2col + GEMM** : les fenêtres glissantes
// 2D de l'entrée sont dépliées en colonnes contiguës, puis
// [`super::linalg::matmul`] calcule tous les canaux de sortie d'un coup — le
// déterminisme bit-à-bit du GEMM se transmet donc sans travail supplémentaire.
//
// ## Disposition mémoire
//
// * `x` : `in_channels × height × width`, row-major (canal, puis ligne, puis
//   colonne).
// * `weights` : `out_channels × in_channels × kernel_h × kernel_w`, row-major
//   (même convention que PyTorch `Conv2d`).
// * `bias` : `out_channels` éléments, un par canal de sortie.
// * Sortie : `out_channels × height_out × width_out`, row-major, avec
//   `height_out`/`width_out` donnés par [`Conv2dShape::height_out`] /
//   [`Conv2dShape::width_out`] (convolution valide, sans remplissage).
//
// ## im2col
//
// La matrice dépliée a pour forme `(in_channels·kernel_h·kernel_w) ×
// (height_out·width_out)` : la ligne `(ci·kernel_h·kernel_w + kh·kernel_w +
// kw)`, colonne `(oh·width_out + ow)`, vaut `x[ci, oh·stride_h + kh,
// ow·stride_w + kw]` — exactement l'ordre attendu par `weights`.

use super::reductions::FixedReducible;

/// Dimensions d'une convolution 2D valide (sans remplissage).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Conv2dShape {
    /// Nombre de canaux d'entrée.
    pub in_channels: usize,
    /// Hauteur de l'entrée (par canal).
    pub height: usize,
    /// Largeur de l'entrée (par canal).
    pub width: usize,
    /// Nombre de canaux de sortie (filtres).
    pub out_channels: usize,
    /// Hauteur du noyau de convolution.
    pub kernel_h: usize,
    /// Largeur du noyau de convolution.
    pub kernel_w: usize,
    /// Pas de déplacement vertical de la fenêtre glissante.
    pub stride_h: usize,
    /// Pas de déplacement horizontal de la fenêtre glissante.
    pub stride_w: usize,
}

impl Conv2dShape {
    /// Hauteur de sortie `(height − kernel_h) / stride_h + 1`.
    ///
    /// Panique si `stride_h == 0` ou `height < kernel_h`.
    #[must_use]
    pub fn height_out(&self) -> usize {
        assert!(
            self.stride_h >= 1,
            "Conv2dShape::height_out : stride_h doit être ≥ 1"
        );
        assert!(
            self.height >= self.kernel_h,
            "Conv2dShape::height_out : hauteur {} < noyau {}",
            self.height,
            self.kernel_h
        );
        (self.height - self.kernel_h) / self.stride_h + 1
    }

    /// Largeur de sortie `(width − kernel_w) / stride_w + 1`.
    ///
    /// Panique si `stride_w == 0` ou `width < kernel_w`.
    #[must_use]
    pub fn width_out(&self) -> usize {
        assert!(
            self.stride_w >= 1,
            "Conv2dShape::width_out : stride_w doit être ≥ 1"
        );
        assert!(
            self.width >= self.kernel_w,
            "Conv2dShape::width_out : largeur {} < noyau {}",
            self.width,
            self.kernel_w
        );
        (self.width - self.kernel_w) / self.stride_w + 1
    }
}

/// Déplie les fenêtres glissantes 2D de `x` en colonnes contiguës : forme
/// `(in_channels·kernel_h·kernel_w) × (height_out·width_out)`, row-major.
fn im2col2d<T: Copy>(x: &[T], shape: Conv2dShape, height_out: usize, width_out: usize) -> Vec<T> {
    let mut col = Vec::with_capacity(
        shape.in_channels * shape.kernel_h * shape.kernel_w * height_out * width_out,
    );
    for ci in 0..shape.in_channels
    {
        for kh in 0..shape.kernel_h
        {
            for kw in 0..shape.kernel_w
            {
                for oh in 0..height_out
                {
                    for ow in 0..width_out
                    {
                        let h = oh * shape.stride_h + kh;
                        let w = ow * shape.stride_w + kw;
                        col.push(x[ci * (shape.height * shape.width) + h * shape.width + w]);
                    }
                }
            }
        }
    }
    col
}

/// Convolution 2D **valide** (sans remplissage), multi-canaux, déterministe.
///
/// `x` : `shape.in_channels × shape.height × shape.width` ; `weights` :
/// `shape.out_channels × shape.in_channels × shape.kernel_h × shape.kernel_w` ;
/// `bias` : `shape.out_channels`. Retourne `shape.out_channels ×
/// shape.height_out() × shape.width_out()`.
///
/// Panique si les longueurs de slice ne correspondent pas aux dimensions
/// annoncées, ou selon les préconditions de [`Conv2dShape::height_out`] /
/// [`Conv2dShape::width_out`] — incohérence d'appelant.
#[must_use]
pub fn conv2d<T: FixedReducible>(x: &[T], weights: &[T], bias: &[T], shape: Conv2dShape) -> Vec<T> {
    let height_out = shape.height_out();
    let width_out = shape.width_out();
    assert_eq!(
        x.len(),
        shape.in_channels * shape.height * shape.width,
        "conv2d : x de longueur {} ≠ {}×{}×{}",
        x.len(),
        shape.in_channels,
        shape.height,
        shape.width
    );
    assert_eq!(
        weights.len(),
        shape.out_channels * shape.in_channels * shape.kernel_h * shape.kernel_w,
        "conv2d : poids de longueur {} ≠ {}×{}×{}×{}",
        weights.len(),
        shape.out_channels,
        shape.in_channels,
        shape.kernel_h,
        shape.kernel_w
    );
    assert_eq!(
        bias.len(),
        shape.out_channels,
        "conv2d : biais de longueur {} ≠ {}",
        bias.len(),
        shape.out_channels
    );

    let col = im2col2d(x, shape, height_out, width_out);
    let spatial_out = height_out * width_out;
    let mut y = super::linalg::matmul(
        weights,
        &col,
        shape.out_channels,
        shape.in_channels * shape.kernel_h * shape.kernel_w,
        spatial_out,
    );
    for (co, &b) in bias.iter().enumerate()
    {
        for pos in 0..spatial_out
        {
            let idx = co * spatial_out + pos;
            y[idx] = y[idx].wrapping_add(b);
        }
    }
    y
}
