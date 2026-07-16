// scirust-simd/src/fixed/pool.rs
//
// # Pooling 1D quantifié déterministe
//
// [`max_pool1d`] et [`avg_pool1d`] : sous-échantillonnage par fenêtre
// glissante, multi-canaux, tel qu'utilisé après une couche convolutive d'un
// CNN léger. Avec [`super::conv::conv1d`] et [`super::activation`], ils
// complètent la chaîne **convolution → pooling → activation** pour
// l'inférence quantifiée.
//
// ## Déterminisme
//
// * `max_pool1d` : maximum par fenêtre — un ordre total exact sur les entiers
//   signés, donc **exact et déterministe**, indépendant de tout ordre de
//   parcours (comme [`super::reductions::max`]).
// * `avg_pool1d` : moyenne par fenêtre — **somme entière exacte** (comme
//   [`super::reductions::sum`]) divisée par la taille de la fenêtre
//   (troncature vers zéro, même politique que l'opérateur `/`). Aucune erreur
//   d'arrondi flottant à accumuler : le résultat est bit-à-bit reproductible.
//
// Les deux partagent la même disposition mémoire que [`super::conv::conv1d`]
// (`channels × length`, row-major) pour s'enchaîner directement à sa sortie.

use super::reductions::FixedReducible;
use super::traits::NumericScalar;

/// Dimensions d'un pooling 1D valide (sans remplissage).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Pool1dShape {
    /// Nombre de canaux (indépendants, chacun poolé séparément).
    pub channels: usize,
    /// Longueur de la séquence d'entrée (par canal).
    pub length: usize,
    /// Taille de la fenêtre de pooling.
    pub window: usize,
    /// Pas de déplacement de la fenêtre glissante.
    pub stride: usize,
}

impl Pool1dShape {
    /// Longueur de sortie `(length − window) / stride + 1`.
    ///
    /// Panique si `stride == 0` ou `length < window` (aucune fenêtre ne tient
    /// dans l'entrée).
    #[must_use]
    pub fn length_out(&self) -> usize {
        assert!(
            self.stride >= 1,
            "Pool1dShape::length_out : stride doit être ≥ 1"
        );
        assert!(
            self.length >= self.window,
            "Pool1dShape::length_out : longueur {} < fenêtre {}",
            self.length,
            self.window
        );
        (self.length - self.window) / self.stride + 1
    }
}

fn check_input_len<T>(x: &[T], shape: Pool1dShape, caller: &str) {
    assert_eq!(
        x.len(),
        shape.channels * shape.length,
        "{caller} : x de longueur {} ≠ {}×{}",
        x.len(),
        shape.channels,
        shape.length
    );
}

/// Max-pooling 1D **valide** (sans remplissage), multi-canaux, déterministe.
///
/// `x` : `shape.channels × shape.length`, row-major. Retourne `shape.channels
/// × shape.length_out()` : `y[c, j] = max_{k<window} x[c, j·stride + k]`.
///
/// Panique si `x.len() != shape.channels·shape.length`, ou selon les
/// préconditions de [`Pool1dShape::length_out`].
#[must_use]
pub fn max_pool1d<T: FixedReducible>(x: &[T], shape: Pool1dShape) -> Vec<T> {
    check_input_len(x, shape, "max_pool1d");
    let length_out = shape.length_out();
    let mut y = Vec::with_capacity(shape.channels * length_out);
    for c in 0..shape.channels
    {
        for j in 0..length_out
        {
            let start = c * shape.length + j * shape.stride;
            let window = &x[start..start + shape.window];
            let m = super::reductions::max(window).expect("fenêtre non vide (window ≥ 1)");
            y.push(m);
        }
    }
    y
}

/// Average-pooling 1D **valide** (sans remplissage), multi-canaux,
/// déterministe.
///
/// `x` : `shape.channels × shape.length`, row-major. Retourne `shape.channels
/// × shape.length_out()` : `y[c, j] = (Σ_{k<window} x[c, j·stride + k]) /
/// window` (division tronquée vers zéro, somme entière exacte).
///
/// Requiert [`NumericScalar`] en plus de [`FixedReducible`] (satisfait par
/// `FixedI32<FRAC>` et `FixedI64<FRAC>`) pour convertir la taille de fenêtre
/// en scalaire diviseur.
///
/// Panique si `x.len() != shape.channels·shape.length`, selon les
/// préconditions de [`Pool1dShape::length_out`], ou si la division par
/// `window` déborde (overflow de `T::MIN / -1`, cas extrême documenté par
/// l'opérateur `/` de [`super::types::Fixed`]).
#[must_use]
pub fn avg_pool1d<T: FixedReducible + NumericScalar>(x: &[T], shape: Pool1dShape) -> Vec<T> {
    check_input_len(x, shape, "avg_pool1d");
    let length_out = shape.length_out();
    let divisor = T::from_i32(shape.window as i32);
    let mut y = Vec::with_capacity(shape.channels * length_out);
    for c in 0..shape.channels
    {
        for j in 0..length_out
        {
            let start = c * shape.length + j * shape.stride;
            let window = &x[start..start + shape.window];
            let total = super::reductions::sum(window);
            y.push(
                total
                    .checked_div(divisor)
                    .expect("division par la taille de fenêtre (≥ 1) ne déborde pas"),
            );
        }
    }
    y
}
