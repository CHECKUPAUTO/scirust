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
// * [`DualQuaternion`] — le même déplacement `SE(3)`, encodé en un seul
//   quaternion dual (`qᵣ + ε·q_d`) plutôt qu'une paire ; permet
//   [`DualQuaternion::sclerp`] (*screw linear interpolation*), la
//   généralisation exacte de `slerp` à `SE(3)` entier (vitesse angulaire
//   **et** linéaire constantes), là où interpoler séparément rotation et
//   translation de deux `Transform` ne suit pas la trajectoire physique
//   réelle si l'axe de rotation ne passe pas par l'origine.
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

pub mod dual_quaternion;
pub mod quaternion;
pub mod transform;

pub use dual_quaternion::DualQuaternion;
pub use quaternion::Quaternion;
pub use transform::Transform;

#[cfg(test)]
mod tests;
