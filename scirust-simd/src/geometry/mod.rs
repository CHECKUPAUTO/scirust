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
//   matrice de rotation (aller-retour), angles d'Euler (aller-retour),
//   interpolation `nlerp`/`slerp`.
// * [`Transform`] — déplacement rigide `SE(3)` (rotation + translation) :
//   composition, inverse, matrice homogène 4×4 (aller-retour).
//
// ## Pourquoi générique ?
//
// Un quaternion en `Quaternion<FixedI32<16>>` fait tourner un vecteur avec un
// résultat **reproductible bit-à-bit** sur toute architecture — utile pour la
// robotique déterministe, la simulation rejouable et l'embarqué sans FPU — tout
// en réutilisant le **même code** que `Quaternion<f32>`. C'est la validation de
// bout en bout du trait [`RealScalar`](crate::fixed::RealScalar). `Transform`
// hérite de cette généricité : composer des poses `SE(3)` en virgule fixe
// donne la même trajectoire, bit pour bit, sur toute plateforme.

pub mod quaternion;
pub mod transform;

pub use quaternion::Quaternion;
pub use transform::Transform;

#[cfg(test)]
mod tests;
