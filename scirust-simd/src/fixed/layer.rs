// scirust-simd/src/fixed/layer.rs
//
// # Couche linéaire quantifiée déterministe
//
// [`Linear<T>`] regroupe des poids (`out × in` row-major) et un biais
// (`out` éléments) pour calculer `y = W·x + b`, éventuellement suivi d'une
// activation — l'opération complète d'une couche dense quantifiée. C'est
// l'assemblage naturel de [`super::linalg::matvec`] (produit matrice-vecteur
// déterministe) et de [`super::activation`] (activations ponctuelles) en une
// seule structure réutilisable.
//
// ## Déterminisme
//
// `matvec` est déjà bit-à-bit reproductible (cf. [`super::linalg`]) ; l'ajout
// du biais est une addition virgule fixe **exacte** (entière, enveloppante) ;
// l'activation est **ponctuelle**. La composition des trois reste donc
// intégralement déterministe, indépendante de l'ordre de parcours, du nombre
// de lanes SIMD, de l'architecture et du nombre de threads.

use super::reductions::FixedReducible;
use super::traits::NumericScalar;

/// Couche linéaire quantifiée `y = W·x + b` (`W` : `out_features × in_features`
/// row-major, `b` : `out_features` éléments).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Linear<T> {
    weights: Vec<T>,
    bias: Vec<T>,
    out_features: usize,
    in_features: usize,
}

impl<T: FixedReducible> Linear<T> {
    /// Construit une couche à partir des poids et du biais.
    ///
    /// Panique si `weights.len() != out_features·in_features` ou
    /// `bias.len() != out_features` — incohérence d'appelant.
    #[must_use]
    pub fn new(weights: Vec<T>, bias: Vec<T>, out_features: usize, in_features: usize) -> Self {
        assert_eq!(
            weights.len(),
            out_features * in_features,
            "Linear::new : poids de longueur {} ≠ {out_features}×{in_features}",
            weights.len()
        );
        assert_eq!(
            bias.len(),
            out_features,
            "Linear::new : biais de longueur {} ≠ {out_features}",
            bias.len()
        );
        Self {
            weights,
            bias,
            out_features,
            in_features,
        }
    }

    /// Nombre de features en sortie.
    #[inline(always)]
    #[must_use]
    pub fn out_features(&self) -> usize {
        self.out_features
    }
    /// Nombre de features en entrée.
    #[inline(always)]
    #[must_use]
    pub fn in_features(&self) -> usize {
        self.in_features
    }

    /// Propagation avant `y = W·x + b`, sans activation.
    ///
    /// Panique si `x.len() != in_features`.
    #[must_use]
    pub fn forward(&self, x: &[T]) -> Vec<T> {
        assert_eq!(
            x.len(),
            self.in_features,
            "Linear::forward : entrée de longueur {} ≠ {}",
            x.len(),
            self.in_features
        );
        let mut y = super::linalg::matvec(&self.weights, x, self.out_features, self.in_features);
        for (yi, &bi) in y.iter_mut().zip(&self.bias)
        {
            *yi = yi.wrapping_add(bi);
        }
        y
    }

    /// Propagation avant suivie d'une activation ponctuelle : `f(W·x + b)`.
    ///
    /// `f` est appliquée élément par élément (déterministe, cf.
    /// [`super::activation::apply_inplace`]). Requiert [`NumericScalar`] en
    /// plus de [`FixedReducible`] (satisfait par `FixedI32<FRAC>` et
    /// `FixedI64<FRAC>`), pour permettre les activations de
    /// [`super::activation`].
    #[must_use]
    pub fn forward_activated(&self, x: &[T], f: impl Fn(T) -> T) -> Vec<T>
    where
        T: NumericScalar,
    {
        let mut y = self.forward(x);
        super::activation::apply_inplace(&mut y, f);
        y
    }
}
