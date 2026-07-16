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
// fois**. De nouveaux stockages (`i16`, `i128`) s'ajouteraient sans réécriture.
//
// ## Alias fournis
//
// | Alias | Type | Plage approximative | Résolution |
// |---|---|---|---|
// | [`Q16_16`] | `FixedI32<16>` | ±32 768 | 1.5e-5 |
// | [`Q8_24`]  | `FixedI32<24>` | ±128 | 6.0e-8 |
// | [`Q24_8`]  | `FixedI32<8>`  | ±8.4e6 | 3.9e-3 |
// | [`Q32_32`] | `FixedI64<32>` | ±2.1e9 | 2.3e-10 |
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
// * [`simd`] — vecteurs [`FixedI32x8`], [`FixedI64x4`].
// * [`reductions`] — sommes, `dot`, normes, extrema, cosinus.
// * [`math`] — `sqrt`, `rsqrt`, `reciprocal` (Newton entier exact).
// * [`transcendental`] — `exp`/`ln`/`sin`/`cos`/`tanh`/`sigmoid`/`softmax`
//   (minimax + réduction d'argument, bornes ULP prouvées ; `FixedI32<FRAC>`).

pub mod convert;
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
pub use simd::{FixedI32x8, FixedI64x4};
pub use traits::{NumericScalar, RealScalar};
pub use types::Fixed;

/// Virgule fixe sur `i32` : `FixedI32<FRAC>` = `raw / 2^FRAC`.
pub type FixedI32<const FRAC: u32> = Fixed<i32, FRAC>;
/// Virgule fixe sur `i64` : `FixedI64<FRAC>` = `raw / 2^FRAC`.
pub type FixedI64<const FRAC: u32> = Fixed<i64, FRAC>;

/// Q8.24 — 8 bits entiers, 24 fractionnaires (haute résolution, faible plage).
pub type Q8_24 = FixedI32<24>;
/// Q16.16 — équilibre plage/résolution le plus courant en DSP.
pub type Q16_16 = FixedI32<16>;
/// Q24.8 — large plage, résolution modérée.
pub type Q24_8 = FixedI32<8>;
/// Q32.32 — 64 bits, large plage et très haute résolution.
pub type Q32_32 = FixedI64<32>;
