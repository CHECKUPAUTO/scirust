//! Vote de protection de réacteur avec canaux en dérivation (bypass) —
//! l'extension de la logique `MooN` générique ([`crate::voting::Architecture`])
//! au cas opérationnel standard de la protection nucléaire : un canal mis
//! en dérivation pour maintenance ou surveillance périodique (IEC 61513
//! §6.2.3.5) est retiré du vote, ce qui réduit `N` sans changer `M` — un
//! 2oo4 devient un 2oo3 tant que le canal reste en dérivation, **pas** un
//! 1oo3 : le seuil de déclenchement `M` ne change pas parce qu'un canal
//! disparaît, c'est le nombre de canaux disponibles qui diminue.
//!
//! **Limite honnête** : ce module modélise seulement la reconfiguration
//! de l'architecture de vote pendant une dérivation, en réutilisant les
//! primitives déjà vérifiées de [`crate::voting::Architecture`] et
//! [`scirust_reliability::pfd_moon`]. Il n'implémente **pas** la
//! méthodologie de calcul de seuil ISA-67.04 (Analytical Limit → Trip Set
//! Point via SRSS → Nominal Trip Set Point → Limiting Trip Setpoint), ni
//! les exigences de repli sur défaillance en mode commun de NUREG-0800
//! BTP 7-19 — ces méthodologies restent documentées dans
//! `docs/DOMAIN_ROADMAP.md` (D8) mais non portées en code faute d'une
//! vérification jugée suffisante pour du code de sécurité dans cette
//! passe.

use crate::error::SisResult;
use crate::voting::Architecture;

/// Architecture de vote après mise en dérivation de `bypassed_channels`
/// canaux sur une architecture nominale `m`-parmi-`n`. Le seuil `m` reste
/// inchangé ; seul `n` diminue — pratique standard documentée pour la
/// protection nucléaire (IEC 61513 §6.2.3.5) : un canal en dérivation ne
/// vote plus, il n'abaisse pas le nombre de votes requis pour déclencher.
///
/// Erreur ([`crate::error::SisError::InvalidArchitecture`]) si la
/// dérivation rend l'architecture insatisfaisable (`n - bypassed_channels
/// < m`) — le cas où même tous les canaux restants votant trip ne
/// suffiraient plus, signe qu'une action administrative (arrêt du
/// réacteur, ou limite technique de spécification atteinte) est requise
/// plutôt qu'un vote dégradé silencieux.
pub fn architecture_with_bypass(
    nominal: Architecture,
    bypassed_channels: u8,
) -> SisResult<Architecture> {
    let reduced_n = nominal.n.saturating_sub(bypassed_channels);
    Architecture::new(nominal.m, reduced_n)
}

/// `PFDavg` de l'architecture nominale pendant qu'un ou plusieurs canaux
/// sont en dérivation — délègue à [`Architecture::pfd_avg`] sur
/// l'architecture réduite d'[`architecture_with_bypass`]. Utile pour
/// vérifier qu'une dérivation planifiée ne fait pas sortir le système du
/// SIL requis pendant sa durée.
pub fn pfd_avg_during_bypass(
    nominal: Architecture,
    bypassed_channels: u8,
    lambda_du: f64,
    t1: f64,
    beta: f64,
) -> SisResult<f64> {
    architecture_with_bypass(nominal, bypassed_channels)?.pfd_avg(lambda_du, t1, beta)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn bypassing_one_channel_reduces_n_not_m() {
        let nominal = Architecture::new(2, 4).unwrap();
        let reduced = architecture_with_bypass(nominal, 1).unwrap();
        assert_eq!(reduced, Architecture::new(2, 3).unwrap());
    }

    #[test]
    fn bypassing_down_to_the_threshold_still_works() {
        let nominal = Architecture::new(2, 4).unwrap();
        let reduced = architecture_with_bypass(nominal, 2).unwrap();
        assert_eq!(reduced, Architecture::TWO_OO2);
    }

    #[test]
    fn bypassing_below_the_threshold_is_rejected() {
        let nominal = Architecture::new(2, 4).unwrap();
        assert!(architecture_with_bypass(nominal, 3).is_err());
    }

    #[test]
    fn bypassing_more_channels_than_exist_is_rejected_not_a_panic() {
        let nominal = Architecture::new(2, 4).unwrap();
        assert!(architecture_with_bypass(nominal, 10).is_err());
    }

    #[test]
    fn reduced_architecture_votes_on_the_remaining_channels_only() {
        // 2oo4 with one channel bypassed becomes 2oo3: needs 2 of the 3
        // remaining live channels, not 2 of the original 4.
        let nominal = Architecture::new(2, 4).unwrap();
        let reduced = architecture_with_bypass(nominal, 1).unwrap();
        assert!(reduced.evaluate_votes(&[true, true, false]).unwrap());
        assert!(!reduced.evaluate_votes(&[true, false, false]).unwrap());
    }

    #[test]
    fn pfd_avg_during_bypass_matches_the_reduced_architectures_pfd_avg() {
        let nominal = Architecture::new(2, 4).unwrap();
        let direct = architecture_with_bypass(nominal, 1)
            .unwrap()
            .pfd_avg(1e-6, 8760.0, 0.05)
            .unwrap();
        let via_helper = pfd_avg_during_bypass(nominal, 1, 1e-6, 8760.0, 0.05).unwrap();
        assert_relative_eq!(direct, via_helper);
    }
}
