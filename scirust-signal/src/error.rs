//! Erreur unifiée pour `scirust-signal`.
//!
//! Les fonctions publiques qui ont un espace de paramètres invalides (taille
//! de fenêtre, ordre de filtre, coupure hors bande, etc.) renvoient
//! [`SignalResult`] plutôt que de paniquer via `assert!` — l'appelant décide
//! alors comment réagir à une entrée invalide (log, message utilisateur, échec
//! propre) au lieu de faire crasher tout le processus.
//!
//! Migration en cours, fonction par fonction (voir les modules eux-mêmes pour
//! l'état actuel) — toute l'API historique de ce crate ne renvoie pas encore
//! `SignalResult`.

use thiserror::Error;

/// Erreur unifiée pour `scirust-signal`.
#[derive(Debug, Clone, PartialEq, Error)]
pub enum SignalError {
    /// Une précondition de paramètre a été violée (taille, plage, forme...).
    /// Le message décrit précisément quel paramètre et pourquoi.
    #[error("invalid input: {0}")]
    InvalidInput(String),
}

pub type SignalResult<T> = Result<T, SignalError>;
