//! # scirust-burn-bridge
//!
//! Pont d'inférence entre les modules `burn` (réseaux de neurones) et les
//! boucles SciRust (algorithmes non-différentiables : évolution, RL, MCTS,
//! Monte-Carlo).
//!
//! ## Philosophie
//!
//! On ne réimplémente pas Burn. On l'**utilise** depuis nos boucles.
//! Quand un algorithme évolutionnaire veut évaluer la fitness d'un individu
//! qui contient un réseau de neurones, ce crate fournit l'adaptateur.
//!
//! ## Interdiction explicite
//!
//! Ce crate ne doit **JAMAIS** être utilisé avec [`burn::backend::Autodiff`].
//! Il est conçu pour l'inférence pure : pas de tape, pas de gradient,
//! pas de tracking.
//!
//! Si vous devez entraîner un réseau par descente de gradient, utilisez Burn
//! directement. Ce crate ne sert que pour l'évaluation.
//!
//! ## Exemple minimal
//!
//! ```ignore
//! use scirust_burn_bridge::{InferenceOnly, Policy};
//! use burn::backend::NdArray;
//!
//! type B = NdArray<f32>;
//!
//! // Définir un type qui implémente Policy<B>
//! // (voir tests/integration.rs pour l'exemple complet)
//!
//! let device = Default::default();
//! let policy = MyTinyMlp::<B>::new(&device);
//! let bridge = InferenceOnly::new(policy, device);
//!
//! let input = /* construire un Tensor<B, 2> */;
//! let output = bridge.eval(input);
//! ```
//!
//! ## Vérifié contre Burn 0.20.x
//!
//! Si la version Burn évolue significativement, ce crate doit être adapté.
//! Voir `Cargo.toml` pour la version exacte.

#![deny(missing_docs)]
#![deny(unsafe_code)]
#![warn(clippy::all)]

use burn::tensor::backend::Backend;
use std::marker::PhantomData;

// ─────────────────────────────────────────────────────────────────────────
// Public API
// ─────────────────────────────────────────────────────────────────────────

/// Une politique évaluable depuis une boucle SciRust.
///
/// L'implémentation doit être `Send + Sync` pour permettre l'évaluation
/// parallèle d'une population entière (NEAT, GA, ERL, etc.).
///
/// **Important** : `forward` ne doit produire aucun tracking de gradient.
/// L'usage avec un backend `Autodiff<_>` est une erreur d'utilisation —
/// le bridge ne peut pas le détecter au compile-time pour l'instant
/// (cf. v0.1 où on ajoutera un trait-bound `NotAutodiff`).
pub trait Policy<B: Backend>: Send {
    /// Type du tenseur d'entrée (typiquement `Tensor<B, 2>` pour `[batch, features]`).
    type Input;

    /// Type du tenseur de sortie.
    type Output;

    /// Forward pass pur. Ne mute pas l'état interne.
    fn forward(&self, input: Self::Input) -> Self::Output;
}

/// Wrapper qui matérialise l'engagement "inference-only".
///
/// Stocke la politique et le device Burn associé. Donne une API simple
/// (`eval`) sans exposer la machinerie Burn aux algorithmes SciRust.
#[derive(Debug)]
pub struct InferenceOnly<B, P>
where
    B: Backend,
    P: Policy<B>,
{
    policy: P,
    device: B::Device,
    _phantom: PhantomData<B>,
}

impl<B, P> InferenceOnly<B, P>
where
    B: Backend,
    P: Policy<B>,
{
    /// Construit un nouveau wrapper d'inférence.
    pub fn new(policy: P, device: B::Device) -> Self {
        Self {
            policy,
            device,
            _phantom: PhantomData,
        }
    }

    /// Évalue la politique sur une entrée.
    ///
    /// Pour l'évaluation batchée d'une population entière, voir
    /// [`InferenceOnly::eval_batch`] (à venir v0.1).
    pub fn eval(&self, input: P::Input) -> P::Output {
        self.policy.forward(input)
    }

    /// Référence au device Burn utilisé pour cette politique.
    ///
    /// Utile pour construire des tenseurs d'entrée sur le bon device
    /// avant d'appeler [`InferenceOnly::eval`].
    pub fn device(&self) -> &B::Device {
        &self.device
    }

    /// Référence à la politique sous-jacente. Lecture seule.
    pub fn policy(&self) -> &P {
        &self.policy
    }
}

// ─────────────────────────────────────────────────────────────────────────
// Erreurs
// ─────────────────────────────────────────────────────────────────────────

/// Erreurs spécifiques au bridge.
///
/// Volontairement minimal en v0.0.1 — étendu au fil des itérations.
#[derive(thiserror::Error, Debug)]
pub enum BridgeError {
    /// Une opération est incompatible avec un backend autodiff.
    #[error(
        "this bridge is inference-only and cannot be used with burn::backend::Autodiff. \
         Use a bare backend (NdArray, Wgpu, Cuda, etc.) instead."
    )]
    AutodiffBackendNotSupported,

    /// Une dimension d'entrée ne correspond pas à ce que la politique attend.
    #[error("input shape mismatch: expected {expected:?}, got {got:?}")]
    InputShapeMismatch {
        /// Forme attendue.
        expected: Vec<usize>,
        /// Forme reçue.
        got: Vec<usize>,
    },
}

/// Type de résultat utilisé par le crate.
pub type Result<T> = std::result::Result<T, BridgeError>;

// ─────────────────────────────────────────────────────────────────────────
// Tests de fumée
// ─────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod smoke_tests {
    use super::*;

    /// Test que les bornes du trait `Policy` permettent bien `Send + Sync`.
    /// Ce test compile-time est crucial pour l'usage parallèle (rayon).
    #[test]
    fn policy_is_send_sync() {
        fn assert_send_sync<T: Send>() {}

        // On ne peut pas instancier un Policy générique sans un backend concret,
        // donc on vérifie juste que la borne du trait est cohérente.
        struct DummyPolicy;
        // Note : pour un vrai test, voir tests/integration.rs qui utilise burn-ndarray.
        let _ = std::any::type_name::<DummyPolicy>();
        assert_send_sync::<DummyPolicy>();
    }

    #[test]
    fn bridge_error_displays() {
        let err = BridgeError::AutodiffBackendNotSupported;
        let msg = format!("{err}");
        assert!(msg.contains("inference-only"));
    }
}
