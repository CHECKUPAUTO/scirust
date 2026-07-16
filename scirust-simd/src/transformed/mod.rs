// scirust-simd/src/transformed/mod.rs
//
// # Transformed-Scalar Hypercomplex Algebra (TSHA) — sous-système de recherche
//
// Cadre **expérimental** permettant aux algèbres hypercomplexes (quaternions,
// octonions, sédénions, et toute algèbre de Cayley–Dickson future) d'opérer sur
// des **représentations scalaires transformées** plutôt que sur des flottants
// IEEE bruts. L'algèbre elle-même est inchangée ; seule la *représentation du
// scalaire* change.
//
// > **Statut : recherche.** La rigueur mathématique prime sur la performance.
// > Aucune mathématique n'est inventée ni revendiquée comme nouvelle ; chaque
// > propriété incertaine est documentée puis vérifiée expérimentalement.
//
// ## Idée centrale
//
// ```text
//   Quaternion<f32>                              (actuel)
//   Quaternion<TransformedScalar<f64, Identity>> (contrôle, ≡ à l'algèbre brute)
//   Quaternion<TransformedScalar<f64, ReciprocalGamma>>
//   Quaternion<TransformedScalar<f64, LogGamma>>
// ```
//
// Une transformation `φ : D → C` ([`ScalarTransform`]) encode une valeur latente
// en valeur encodée. Le latent est **autoritatif** ; l'encodé est calculé.
//
// ## Deux modèles d'exécution (fondamentalement distincts)
//
// * **Modèle A — transport latent** : `decode → opération → encode`, soit
//   `φ(A ⋆ B)`. **Préserve** les lois algébriques (transportées par φ).
// * **Modèle B — algèbre transformée directe** : `φ(A) ⋆ φ(B)`. En général
//   **non** équivalent. Leur écart `Δ = φ(A⋆B) − φ(A)⋆φ(B)`, le *défaut de
//   transformation*, est l'objet de recherche de premier plan
//   ([`metrics`], [`experiments`]).
//
// ## Non-inversibilité (rigueur)
//
// `ReciprocalGamma` et `LogGamma` **ne sont pas** globalement inversibles :
// `Γ(x+1)` a un extremum en `x* ≈ 0.4616`, d'où deux branches monotones. Le
// décodage est **faillible** et **paramétré par branche** ([`branch::GammaBranch`])
// — l'ambiguïté n'est jamais masquée.
//
// ## Contenu
//
// * [`transform`] — trait [`ScalarTransform`] + erreurs de domaine/inversion.
// * [`special`] — `Γ`, `ln Γ`, `ψ` (Lanczos, 100 % Rust, zéro FFI).
// * [`branch`] — branches d'inversion + bissection déterministe.
// * [`identity`], [`reciprocal_gamma`], [`log_gamma`] — transformations fournies.
// * [`scalar`] — [`TransformedScalar`] (latent autoritatif).
// * [`hypercomplex`] — algèbre de Cayley–Dickson générique + Modèles A/B.
// * [`metrics`] — défaut, distorsion de norme, commutateur, associateur.
// * [`experiments`] — expériences déterministes + export CSV.
//
// ## Contraintes tenues
//
// 100 % Rust, zéro FFI, zéro `unsafe`, déterministe, entièrement testé et
// documenté, générique et extensible. Aucune vectorisation ici (par choix : le
// cadre cible d'abord des scalaires ; la SIMD relève d'un lot ultérieur).

pub mod branch;
pub mod experiments;
pub mod hypercomplex;
pub mod identity;
pub mod log_gamma;
pub mod metrics;
pub mod reciprocal_gamma;
pub mod scalar;
pub mod special;
pub mod transform;

pub use branch::GammaBranch;
pub use hypercomplex::{
    Complex, Hypercomplex, Octonion, Quaternion, Sedenion, model_a_product, model_b_product,
};
pub use identity::Identity;
pub use log_gamma::LogGamma;
pub use metrics::DefectReport;
pub use reciprocal_gamma::ReciprocalGamma;
pub use scalar::TransformedScalar;
pub use transform::{DomainError, InverseError, ScalarTransform};

#[cfg(test)]
mod tests;
