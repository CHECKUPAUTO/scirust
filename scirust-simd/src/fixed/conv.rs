// scirust-simd/src/fixed/conv.rs
//
// # Convolution 1D quantifiée déterministe
//
// [`conv1d`] : convolution « valide » (sans remplissage) multi-canaux, telle
// qu'utilisée par les couches convolutives d'un CNN léger (audio, séries
// temporelles). Implémentée par **im2col + GEMM** : les fenêtres glissantes de
// l'entrée sont dépliées en colonnes contiguës, puis [`super::linalg::matmul`]
// calcule tous les produits de convolution d'un coup. Le déterminisme
// bit-à-bit du GEMM (cf. [`super::linalg`]) se transmet donc **sans travail
// supplémentaire** : `conv1d` hérite de la même reproductibilité que `matmul`.
//
// ## Disposition mémoire
//
// * `x` : `in_channels × length`, row-major (canal, puis position).
// * `weights` : `out_channels × in_channels × kernel_size`, row-major (même
//   convention que PyTorch `Conv1d`) — canal de sortie, canal d'entrée, puis
//   décalage du noyau.
// * `bias` : `out_channels` éléments, un par canal de sortie.
// * Sortie : `out_channels × length_out`, row-major, avec `length_out` donné
//   par [`Conv1dShape::length_out`] (convolution valide : panique si
//   `length < kernel_size`).
//
// ## im2col
//
// La matrice dépliée a pour forme `(in_channels·kernel_size) × length_out` :
// la ligne `(ci·kernel_size + k)`, colonne `j`, vaut `x[ci, j·stride + k]`.
// C'est exactement l'ordre de ligne attendu par `weights` (canal d'entrée puis
// décalage), donc `matmul(weights, col, out_channels, in_channels·kernel_size,
// length_out)` calcule directement tous les canaux de sortie.

use super::reductions::FixedReducible;

/// Dimensions d'une convolution 1D valide (sans remplissage).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Conv1dShape {
    /// Nombre de canaux d'entrée.
    pub in_channels: usize,
    /// Longueur de la séquence d'entrée (par canal).
    pub length: usize,
    /// Nombre de canaux de sortie (filtres).
    pub out_channels: usize,
    /// Taille du noyau de convolution.
    pub kernel_size: usize,
    /// Pas de déplacement de la fenêtre glissante.
    pub stride: usize,
}

impl Conv1dShape {
    /// Longueur de sortie `(length − kernel_size) / stride + 1`.
    ///
    /// Panique si `stride == 0` ou `length < kernel_size` (aucune fenêtre ne
    /// tient dans l'entrée).
    #[must_use]
    pub fn length_out(&self) -> usize {
        assert!(
            self.stride >= 1,
            "Conv1dShape::length_out : stride doit être ≥ 1"
        );
        assert!(
            self.length >= self.kernel_size,
            "Conv1dShape::length_out : longueur {} < taille de noyau {}",
            self.length,
            self.kernel_size
        );
        (self.length - self.kernel_size) / self.stride + 1
    }
}

/// Déplie les fenêtres glissantes de `x` en colonnes contiguës : forme
/// `(in_channels·kernel_size) × length_out`, row-major.
fn im2col<T: Copy>(
    x: &[T],
    in_channels: usize,
    length: usize,
    kernel_size: usize,
    stride: usize,
    length_out: usize,
) -> Vec<T> {
    let mut col = Vec::with_capacity(in_channels * kernel_size * length_out);
    for ci in 0..in_channels
    {
        for k in 0..kernel_size
        {
            for j in 0..length_out
            {
                col.push(x[ci * length + j * stride + k]);
            }
        }
    }
    col
}

/// Convolution 1D **valide** (sans remplissage), multi-canaux, déterministe.
///
/// `x` : `shape.in_channels × shape.length` ; `weights` : `shape.out_channels
/// × shape.in_channels × shape.kernel_size` ; `bias` : `shape.out_channels`.
/// Retourne `shape.out_channels × shape.length_out()`.
///
/// Panique si les longueurs de slice ne correspondent pas aux dimensions
/// annoncées, ou selon les préconditions de [`Conv1dShape::length_out`] —
/// incohérence d'appelant.
#[must_use]
pub fn conv1d<T: FixedReducible>(x: &[T], weights: &[T], bias: &[T], shape: Conv1dShape) -> Vec<T> {
    let length_out = shape.length_out();
    assert_eq!(
        x.len(),
        shape.in_channels * shape.length,
        "conv1d : x de longueur {} ≠ {}×{}",
        x.len(),
        shape.in_channels,
        shape.length
    );
    assert_eq!(
        weights.len(),
        shape.out_channels * shape.in_channels * shape.kernel_size,
        "conv1d : poids de longueur {} ≠ {}×{}×{}",
        weights.len(),
        shape.out_channels,
        shape.in_channels,
        shape.kernel_size
    );
    assert_eq!(
        bias.len(),
        shape.out_channels,
        "conv1d : biais de longueur {} ≠ {}",
        bias.len(),
        shape.out_channels
    );

    let col = im2col(
        x,
        shape.in_channels,
        shape.length,
        shape.kernel_size,
        shape.stride,
        length_out,
    );
    let mut y = super::linalg::matmul(
        weights,
        &col,
        shape.out_channels,
        shape.in_channels * shape.kernel_size,
        length_out,
    );
    for (co, &b) in bias.iter().enumerate()
    {
        for j in 0..length_out
        {
            let idx = co * length_out + j;
            y[idx] = y[idx].wrapping_add(b);
        }
    }
    y
}
