// scirust-simd/src/geometry/mod.rs
//
// # Géométrie générique — orientation & rotations 3D
//
// Types géométriques **génériques sur le scalaire**, construits sur les traits
// [`NumericScalar`](crate::fixed::NumericScalar) /
// [`RealScalar`](crate::fixed::RealScalar) du sous-système virgule fixe. La
// même implémentation sert le flottant (`f32`/`f64`) **et** la virgule fixe
// déterministe (`FixedI32<FRAC>`) — aucune duplication.
//
// ## Contenu
//
// * [`Quaternion`] — quaternion de Hamilton générique : produit, conjugaison,
//   norme, normalisation, inverse, construction angle-axe, rotation de vecteur,
//   matrice de rotation, interpolation `nlerp`.
//
// ## Pourquoi générique ?
//
// Un quaternion en `Quaternion<FixedI32<16>>` fait tourner un vecteur avec un
// résultat **reproductible bit-à-bit** sur toute architecture — utile pour la
// robotique déterministe, la simulation rejouable et l'embarqué sans FPU — tout
// en réutilisant le **même code** que `Quaternion<f32>`. C'est la validation de
// bout en bout du trait [`RealScalar`](crate::fixed::RealScalar).

pub mod quaternion;

pub use quaternion::Quaternion;

#[cfg(test)]
mod tests;
