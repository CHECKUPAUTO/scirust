//! Boucle de dosage en boucle fermée : brique PID générique
//! ([`pid`]), suivi d'insuline active ([`iob`]), et deux couches de
//! sécurité indépendantes — supervision par seuils ([`insulin_safety`])
//! et filtre par fonction barrière de contrôle ([`barrier`]).
//!
//! **Avertissement global** : voir l'avertissement de non-usage clinique
//! en tête de chaque sous-module — ceci démontre des *techniques* de
//! contrôle certifiable (PID, CBF-QP, supervision par seuils), pas un
//! algorithme de dosage validé pour un dispositif réel.

pub mod barrier;
pub mod insulin_safety;
pub mod iob;
pub mod pid;

pub use barrier::{GlucoseModel, SafeDose, cbf_safe_dose};
pub use insulin_safety::{AutoModeMonitor, max_safe_bolus, predictive_suspend, suspend_on_low};
pub use iob::InsulinOnBoard;
pub use pid::{PidController, PidGains};
