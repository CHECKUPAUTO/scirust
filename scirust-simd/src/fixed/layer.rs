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
// Pour une tête de classification, [`Linear::predict_class`] (`argmax` du
// logit, tout stockage) et [`Linear::predict_proba`] (`softmax`, `i32`
// uniquement) complètent la chaîne poids → biais → décision.
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
use super::types::Fixed;

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

    /// Classe prédite : indice du **premier** logit maximal de `W·x + b`.
    ///
    /// `argmax` suffit — pas besoin de softmax : la softmax est une fonction
    /// strictement croissante de chaque logit (à valeur des autres fixée), donc
    /// `argmax(softmax(z)) == argmax(z)` **toujours**. Éviter l'exponentielle
    /// garde ce classement disponible pour tout stockage
    /// ([`FixedReducible`] : `i32` **et** `i64`), pas seulement `i32`.
    ///
    /// `None` si `out_features == 0`.
    #[must_use]
    pub fn predict_class(&self, x: &[T]) -> Option<usize> {
        super::reductions::argmax(&self.forward(x))
    }
}

impl<const FRAC: u32> Linear<Fixed<i32, FRAC>> {
    /// Probabilités de classe : `softmax(W·x + b)`, numériquement stable et
    /// déterministe bit-à-bit (cf. [`super::transcendental::softmax_into`]).
    ///
    /// Réservé au stockage `i32` : la softmax passe par l'exponentielle
    /// virgule fixe, elle-même réservée à `FixedI32<FRAC>` (précision interne
    /// Q32, cf. [`super::traits::RealScalar`]). Pour ne classer qu'une entrée
    /// (sans les probabilités), préférer [`Linear::predict_class`], qui
    /// fonctionne aussi pour le stockage `i64` sans calculer d'exponentielle.
    #[must_use]
    pub fn predict_proba(&self, x: &[Fixed<i32, FRAC>]) -> Vec<Fixed<i32, FRAC>> {
        let logits = self.forward(x);
        let mut proba = vec![Fixed::zero(); logits.len()];
        super::transcendental::softmax_into(&logits, &mut proba);
        proba
    }
}
