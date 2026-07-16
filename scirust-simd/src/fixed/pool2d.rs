// scirust-simd/src/fixed/pool2d.rs
//
// # Pooling 2D quantifié déterministe
//
// [`max_pool2d`] et [`avg_pool2d`] : extension bidimensionnelle de
// [`super::pool::max_pool1d`]/[`super::pool::avg_pool1d`], pour les données à
// deux dimensions spatiales (images, spectrogrammes) d'un CNN léger. Avec
// [`super::conv2d::conv2d`] et [`super::activation`], ils complètent la chaîne
// **convolution 2D → pooling 2D → activation**.
//
// ## Déterminisme
//
// Mêmes garanties que le pooling 1D : `max_pool2d` est un maximum (ordre total
// exact sur les entiers signés) et `avg_pool2d` une somme entière exacte
// divisée par la taille de la fenêtre (`window_h·window_w`) — aucune erreur
// d'arrondi flottant à accumuler, bit-à-bit reproductible.
//
// ## Disposition mémoire
//
// Même convention que [`super::conv2d`] : `channels × height × width`,
// row-major, pour s'enchaîner directement à la sortie de `conv2d`.
//
// ## Implémentation
//
// Contrairement au pooling 1D, une fenêtre 2D n'est **pas contiguë** en
// mémoire (les lignes sont séparées de `width` éléments) : chaque ligne de la
// fenêtre est copiée (contiguë horizontalement) dans un tampon réutilisé, puis
// [`super::reductions::max`]/[`super::reductions::sum`] s'appliquent dessus —
// mêmes primitives exactes que le pooling 1D.

use super::reductions::FixedReducible;
use super::traits::NumericScalar;

/// Dimensions d'un pooling 2D valide (sans remplissage).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Pool2dShape {
    /// Nombre de canaux (indépendants, chacun poolé séparément).
    pub channels: usize,
    /// Hauteur de l'entrée (par canal).
    pub height: usize,
    /// Largeur de l'entrée (par canal).
    pub width: usize,
    /// Hauteur de la fenêtre de pooling.
    pub window_h: usize,
    /// Largeur de la fenêtre de pooling.
    pub window_w: usize,
    /// Pas de déplacement vertical de la fenêtre glissante.
    pub stride_h: usize,
    /// Pas de déplacement horizontal de la fenêtre glissante.
    pub stride_w: usize,
}

impl Pool2dShape {
    /// Hauteur de sortie `(height − window_h) / stride_h + 1`.
    ///
    /// Panique si `stride_h == 0` ou `height < window_h`.
    #[must_use]
    pub fn height_out(&self) -> usize {
        assert!(
            self.stride_h >= 1,
            "Pool2dShape::height_out : stride_h doit être ≥ 1"
        );
        assert!(
            self.height >= self.window_h,
            "Pool2dShape::height_out : hauteur {} < fenêtre {}",
            self.height,
            self.window_h
        );
        (self.height - self.window_h) / self.stride_h + 1
    }

    /// Largeur de sortie `(width − window_w) / stride_w + 1`.
    ///
    /// Panique si `stride_w == 0` ou `width < window_w`.
    #[must_use]
    pub fn width_out(&self) -> usize {
        assert!(
            self.stride_w >= 1,
            "Pool2dShape::width_out : stride_w doit être ≥ 1"
        );
        assert!(
            self.width >= self.window_w,
            "Pool2dShape::width_out : largeur {} < fenêtre {}",
            self.width,
            self.window_w
        );
        (self.width - self.window_w) / self.stride_w + 1
    }
}

fn check_input_len<T>(x: &[T], shape: Pool2dShape, caller: &str) {
    assert_eq!(
        x.len(),
        shape.channels * shape.height * shape.width,
        "{caller} : x de longueur {} ≠ {}×{}×{}",
        x.len(),
        shape.channels,
        shape.height,
        shape.width
    );
}

/// Copie la fenêtre 2D `(oh, ow)` du canal `c` dans `buf` (ligne par ligne,
/// chacune contiguë horizontalement), en remplaçant son contenu.
fn gather_window<T: Copy>(
    x: &[T],
    shape: Pool2dShape,
    c: usize,
    oh: usize,
    ow: usize,
    buf: &mut Vec<T>,
) {
    buf.clear();
    let base = c * (shape.height * shape.width);
    for kh in 0..shape.window_h
    {
        let h = oh * shape.stride_h + kh;
        let row_start = base + h * shape.width + ow * shape.stride_w;
        buf.extend_from_slice(&x[row_start..row_start + shape.window_w]);
    }
}

/// Max-pooling 2D **valide** (sans remplissage), multi-canaux, déterministe.
///
/// `x` : `shape.channels × shape.height × shape.width`, row-major. Retourne
/// `shape.channels × shape.height_out() × shape.width_out()`.
///
/// Panique si `x.len() != shape.channels·shape.height·shape.width`, ou selon
/// les préconditions de [`Pool2dShape::height_out`]/[`Pool2dShape::width_out`].
#[must_use]
pub fn max_pool2d<T: FixedReducible>(x: &[T], shape: Pool2dShape) -> Vec<T> {
    check_input_len(x, shape, "max_pool2d");
    let height_out = shape.height_out();
    let width_out = shape.width_out();
    let mut y = Vec::with_capacity(shape.channels * height_out * width_out);
    let mut buf = Vec::with_capacity(shape.window_h * shape.window_w);
    for c in 0..shape.channels
    {
        for oh in 0..height_out
        {
            for ow in 0..width_out
            {
                gather_window(x, shape, c, oh, ow, &mut buf);
                let m = super::reductions::max(&buf).expect("fenêtre non vide (window ≥ 1)");
                y.push(m);
            }
        }
    }
    y
}

/// Average-pooling 2D **valide** (sans remplissage), multi-canaux,
/// déterministe.
///
/// `x` : `shape.channels × shape.height × shape.width`, row-major. Retourne
/// `shape.channels × shape.height_out() × shape.width_out()` : moyenne de
/// chaque fenêtre (somme entière exacte divisée par `window_h·window_w`,
/// troncature vers zéro — même politique que l'opérateur `/`).
///
/// Requiert [`NumericScalar`] en plus de [`FixedReducible`] (satisfait par
/// `FixedI32<FRAC>` et `FixedI64<FRAC>`) pour convertir la taille de fenêtre
/// en scalaire diviseur.
///
/// Panique si `x.len() != shape.channels·shape.height·shape.width`, selon les
/// préconditions de [`Pool2dShape::height_out`]/[`Pool2dShape::width_out`], ou
/// si la division par la taille de fenêtre déborde.
#[must_use]
pub fn avg_pool2d<T: FixedReducible + NumericScalar>(x: &[T], shape: Pool2dShape) -> Vec<T> {
    check_input_len(x, shape, "avg_pool2d");
    let height_out = shape.height_out();
    let width_out = shape.width_out();
    let divisor = T::from_i32((shape.window_h * shape.window_w) as i32);
    let mut y = Vec::with_capacity(shape.channels * height_out * width_out);
    let mut buf = Vec::with_capacity(shape.window_h * shape.window_w);
    for c in 0..shape.channels
    {
        for oh in 0..height_out
        {
            for ow in 0..width_out
            {
                gather_window(x, shape, c, oh, ow, &mut buf);
                let total = super::reductions::sum(&buf);
                y.push(
                    total
                        .checked_div(divisor)
                        .expect("division par la taille de fenêtre (≥ 1) ne déborde pas"),
                );
            }
        }
    }
    y
}
