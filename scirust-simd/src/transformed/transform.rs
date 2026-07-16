// scirust-simd/src/transformed/transform.rs
//
// # `ScalarTransform` — abstraction d'une transformation scalaire φ : D → C
//
// Une transformation encode une valeur *latente* `x ∈ D` en une valeur
// *encodée* `y = φ(x) ∈ C`, et sait (quand c'est possible) revenir en arrière.
// Le point délicat, traité **explicitement** ici, est que φ n'est pas
// nécessairement globalement injective : le décodage est donc **faillible** et
// paramétré par une **branche** ([`ScalarTransform::Branch`]).
//
// La transformation ne suppose rien qu'elle ne garantisse : un domaine restreint
// est signalé par [`DomainError`], une valeur hors image par [`InverseError`].

/// Erreur de domaine : l'argument latent `x` est hors du domaine `D` de φ.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DomainError {
    /// `x` est en deçà (ou égal) de la borne inférieure exclue du domaine.
    BelowDomain {
        /// La valeur fautive.
        value: f64,
        /// La borne inférieure **exclue** du domaine.
        lower_bound: f64,
    },
    /// `x` n'est pas fini (NaN ou ±∞).
    NotFinite {
        /// La valeur fautive.
        value: f64,
    },
}

/// Erreur d'inversion : la valeur encodée `y` n'admet pas d'antécédent
/// (représentable) dans la branche demandée.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum InverseError {
    /// `y` est hors de l'image `C = φ(D)` de la transformation.
    OutOfRange {
        /// La valeur fautive.
        value: f64,
    },
    /// `y` n'est pas fini (NaN ou ±∞).
    NotFinite {
        /// La valeur fautive.
        value: f64,
    },
    /// Aucun antécédent trouvé dans la branche (encadrement épuisé) — la valeur
    /// est en principe atteignable mais au-delà de la plage explorée.
    NoSolutionInBranch,
}

/// Une transformation scalaire `φ : D → C` sur le type `T`.
///
/// Contrat :
/// * [`encode`](ScalarTransform::encode) applique φ ; échoue si `x ∉ D`.
/// * [`derivative`](ScalarTransform::derivative) donne `φ'(x)` ; même domaine.
/// * [`decode`](ScalarTransform::decode) applique φ⁻¹ **sur une branche donnée** ;
///   échoue si `y ∉ C` ou si la branche ne contient pas d'antécédent.
///
/// Pour une transformation **globalement injective**, [`Branch`](ScalarTransform::Branch)
/// vaut `()` et le décodage est non ambigu. Sinon, `Branch` sélectionne l'un des
/// intervalles monotones — **l'ambiguïté n'est jamais masquée**.
pub trait ScalarTransform<T> {
    /// Sélecteur de branche pour le décodage (`()` si globalement inversible).
    type Branch: Copy + Default;

    /// Nom lisible de la transformation (documentation, expériences, CSV).
    const NAME: &'static str;

    /// Vrai si φ est globalement injective (décodage non ambigu, une seule branche).
    fn is_globally_invertible() -> bool;

    /// `y = φ(x)`. Échoue si `x` est hors du domaine `D`.
    fn encode(x: T) -> Result<T, DomainError>;

    /// `φ'(x)`. Échoue si `x` est hors du domaine `D`.
    fn derivative(x: T) -> Result<T, DomainError>;

    /// `x = φ⁻¹(y)` restreint à `branch`. Échoue si `y ∉ C` ou si la branche est vide.
    fn decode(y: T, branch: Self::Branch) -> Result<T, InverseError>;

    /// Décodage sur la branche **principale** (`Branch::default()`).
    #[inline]
    fn decode_principal(y: T) -> Result<T, InverseError> {
        Self::decode(y, Self::Branch::default())
    }
}
