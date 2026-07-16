// scirust-simd/src/fixed/mod.rs
//
// # Virgule fixe SIMD — calcul déterministe & reproductible
//
// Sous-système de nombres en virgule fixe pour SciRust : calcul scientifique,
// IA déterministe, traitement du signal et algèbres hypercomplexes, en
// complément (jamais en remplacement) des chemins flottants.
//
// ## Pourquoi la virgule fixe ?
//
// Un flottant a une résolution **relative** (le pas croît avec la magnitude) et
// une somme **non associative** : le résultat dépend de l'ordre, donc du nombre
// de threads et de la vectorisation. Un nombre en virgule fixe a une résolution
// **absolue et constante** et une addition **exacte** (tant qu'il n'y a pas
// d'overflow) : `a + b + c` donne le même bit quel que soit l'ordre. C'est la
// base d'un déterminisme bit-à-bit indépendant de l'architecture et du nombre
// de threads.
//
// ## Représentation
//
// [`Fixed<I, FRAC>`] enveloppe un entier signé `I` ; la valeur réelle est
// `raw / 2^FRAC`. Le type est générique sur le stockage ([`repr::FixedStorage`],
// implémenté pour `i32` et `i64`) : **tout l'algorithme est écrit une seule
// fois**. Ajouter le stockage `i16` (audio) n'a demandé que deux lignes ; un
// `i128` (très haute précision) s'ajouterait de même, sans réécriture.
//
// ## Alias fournis
//
// | Alias | Type | Plage approximative | Résolution |
// |---|---|---|---|
// | [`Q1_15`]  | `FixedI16<15>` | ±1 | 3.1e-5 |
// | [`Q8_8`]   | `FixedI16<8>`  | ±128 | 3.9e-3 |
// | [`Q16_16`] | `FixedI32<16>` | ±32 768 | 1.5e-5 |
// | [`Q8_24`]  | `FixedI32<24>` | ±128 | 6.0e-8 |
// | [`Q24_8`]  | `FixedI32<8>`  | ±8.4e6 | 3.9e-3 |
// | [`Q32_32`] | `FixedI64<32>` | ±2.1e9 | 2.3e-10 |
//
// [`Q1_15`] (audio 16 bits) valide la **généricité du stockage** : ajouter
// `i16` n'a demandé que deux lignes ([`repr`]), aucun algorithme réécrit.
//
// ## Politiques explicites (jamais cachées)
//
// * **Arrondi** ([`RoundingMode`]) : `TowardZero` (défaut), `Floor`, `Ceil`,
//   `NearestEven`.
// * **Overflow** ([`OverflowMode`]) : `Wrap` (défaut des opérateurs), `Checked`,
//   `Saturate`.
//
// Les opérateurs `+ − * / -x` **enveloppent et tronquent** (déterministe quel
// que soit le profil debug/release, contrairement à l'entier Rust). Les
// méthodes `checked_*` / `saturating_*` / `*_rounded` donnent le contrôle total.
//
// ## Généricité
//
// [`NumericScalar`] (anneau ordonné + `abs`) est implémenté pour `f32`, `f64`
// et tout [`Fixed`], si bien qu'un futur `Quaternion<T: NumericScalar>` couvrira
// aussi bien `Quaternion<f32>` que `Quaternion<FixedI32<16>>` sans réécriture.
//
// ## Différences avec le flottant (résumé)
//
// * Pas de NaN ni d'infini : l'overflow est géré par politique explicite.
// * Résolution constante, addition exacte, reproductibilité bit-à-bit.
// * Division/multiplication plus lentes (accumulateur élargi) ; add/sub plus
//   rapides (une instruction entière).
//
// ## Modules
//
// * [`repr`] — plomberie entière générique (`FixedStorage`, `WideInt`).
// * [`rounding`] / [`overflow`] — politiques.
// * [`types`] — le type [`Fixed`] et son arithmétique.
// * [`ops`] — surcharge d'opérateurs.
// * [`convert`] — conversions.
// * [`traits`] — [`NumericScalar`] et [`RealScalar`].
// * [`simd`] — vecteurs [`FixedI16x8`], [`FixedI32x8`], [`FixedI64x4`].
// * [`reductions`] — sommes, `dot`, normes, extrema, cosinus.
// * [`linalg`] — GEMM déterministe (`matmul`, `matvec`, `transpose`).
// * [`activation`] — activations quantifiées (`relu`, `relu6`, `hardswish`…).
// * [`math`] — `sqrt`, `rsqrt`, `reciprocal` (Newton entier exact).
// * [`transcendental`] — `exp`/`ln`/`sin`/`cos`/`tanh`/`sigmoid`/`softmax`
//   (minimax + réduction d'argument, bornes ULP prouvées ; `FixedI32<FRAC>`).

pub mod activation;
pub mod convert;
pub mod linalg;
pub mod math;
pub mod ops;
pub mod overflow;
pub mod reductions;
pub mod repr;
pub mod rounding;
pub mod simd;
pub mod traits;
pub mod transcendental;
pub mod types;

#[cfg(test)]
mod tests;

pub use convert::TryFromFloatError;
pub use overflow::OverflowMode;
pub use repr::{FixedStorage, WideInt};
pub use rounding::RoundingMode;
pub use simd::{FixedI16x8, FixedI32x8, FixedI64x4};
pub use traits::{NumericScalar, RealScalar};
pub use types::Fixed;

/// Virgule fixe sur `i16` : `FixedI16<FRAC>` = `raw / 2^FRAC` (audio, embarqué).
///
/// Fournit l'algèbre d'anneau ([`NumericScalar`]) — donc filtres DSP, produit
/// hypercomplexe générique, etc. Les transcendantes ([`RealScalar`]) restent
/// réservées au stockage `i32` (précision interne Q32).
pub type FixedI16<const FRAC: u32> = Fixed<i16, FRAC>;
/// Virgule fixe sur `i32` : `FixedI32<FRAC>` = `raw / 2^FRAC`.
pub type FixedI32<const FRAC: u32> = Fixed<i32, FRAC>;
/// Virgule fixe sur `i64` : `FixedI64<FRAC>` = `raw / 2^FRAC`.
pub type FixedI64<const FRAC: u32> = Fixed<i64, FRAC>;

/// Q1.15 — format audio 16 bits canonique (échantillons dans `[−1, 1)`).
///
/// **Attention** : `1.0` n'est **pas** représentable (`FRAC = BITS − 1`), donc
/// [`Fixed::one`] enveloppe vers `−1.0`. Ce format sert aux **échantillons**
/// dans `[−1, 1)`, pas à une algèbre nécessitant l'unité (coefficients de
/// filtre, quaternions…) — pour cela, utiliser [`Q8_8`] (`FRAC = 8`).
pub type Q1_15 = FixedI16<15>;
/// Q8.8 — 16 bits, plage modérée (±128), résolution 3.9e-3.
pub type Q8_8 = FixedI16<8>;
/// Q8.24 — 8 bits entiers, 24 fractionnaires (haute résolution, faible plage).
pub type Q8_24 = FixedI32<24>;
/// Q16.16 — équilibre plage/résolution le plus courant en DSP.
pub type Q16_16 = FixedI32<16>;
/// Q24.8 — large plage, résolution modérée.
pub type Q24_8 = FixedI32<8>;
/// Q32.32 — 64 bits, large plage et très haute résolution.
pub type Q32_32 = FixedI64<32>;
