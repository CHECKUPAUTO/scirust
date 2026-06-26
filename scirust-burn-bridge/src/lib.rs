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
/// La borne est volontairement `Send` (et **pas** `Sync`) : un
/// [`burn::module::Module`] réel stocke ses poids dans des `Param`, dont
/// l'initialisation paresseuse repose sur un `OnceCell` — `Send` mais
/// `!Sync`. L'évaluation parallèle d'une population entière (NEAT, GA, ERL,
/// etc.) se fait donc en **déplaçant** un individu possédé par worker
/// (`rayon::into_par_iter`), jamais en partageant `&self` entre threads.
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

    use burn::backend::NdArray;
    use burn::tensor::{Tensor, TensorData};

    type B = NdArray<f32>;

    /// Politique concrète minimale : recopie l'entrée (identité).
    /// Sert à exercer réellement la borne `Send` du trait et le chemin
    /// `InferenceOnly::eval` avec un backend Burn concret.
    struct IdentityPolicy;

    impl Policy<B> for IdentityPolicy {
        type Input = Tensor<B, 2>;
        type Output = Tensor<B, 2>;

        fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
            input
        }
    }

    /// Vérifie au compile-time que le bridge enveloppant une vraie `Policy`
    /// est `Send` — c'est la borne exacte qu'expose le trait et la seule qui
    /// soit tenable : un `burn::Module` réel est `Send` mais **pas** `Sync`
    /// (ses `Param` utilisent un `OnceCell`). L'évaluation parallèle déplace
    /// donc des individus possédés, elle ne partage pas `&self`.
    #[test]
    fn inference_only_is_send() {
        fn assert_send<T: Send>() {}

        assert_send::<IdentityPolicy>();
        assert_send::<InferenceOnly<B, IdentityPolicy>>();
    }

    /// `eval` doit déléguer à `forward` sans altérer la donnée : ici l'identité
    /// doit ressortir bit-pour-bit, et le device exposé doit être le bon.
    #[test]
    fn eval_delegates_to_forward_unchanged() {
        let device = Default::default();
        let bridge = InferenceOnly::<B, _>::new(IdentityPolicy, device);

        let input = Tensor::<B, 2>::from_data(
            TensorData::new(vec![1.0f32, -2.0, 3.5, 4.0], [2, 2]),
            bridge.device(),
        );
        let out = bridge.eval(input);

        assert_eq!(out.dims(), [2, 2]);
        let got: Vec<f32> = out.into_data().to_vec().expect("to_vec");
        assert_eq!(got, vec![1.0f32, -2.0, 3.5, 4.0]);
    }

    #[test]
    fn bridge_error_displays() {
        let autodiff = BridgeError::AutodiffBackendNotSupported;
        assert!(format!("{autodiff}").contains("inference-only"));

        let mismatch = BridgeError::InputShapeMismatch {
            expected: vec![1, 4],
            got: vec![1, 3],
        };
        let msg = format!("{mismatch}");
        assert!(msg.contains("[1, 4]"), "expected shape in message: {msg}");
        assert!(msg.contains("[1, 3]"), "got shape in message: {msg}");
    }
}
