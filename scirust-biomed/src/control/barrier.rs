//! Filtre de sécurité par fonction barrière de contrôle (Control Barrier
//! Function, CBF-QP) — Ames, Grizzle & Tabuada, "Control Barrier Function
//! Based Quadratic Programs for Safety Critical Systems", IEEE Trans.
//! Automatic Control 62(8), 2017; voir aussi Ames et al., "Control
//! Barrier Functions: Theory and Applications", ECC 2019, pour la
//! formulation générale à fonction de classe K.
//!
//! Filtre toute commande désirée `u_desired` (venant par ex. d'un PID,
//! [`super::pid::PidController`]) en la commande sûre la plus proche qui
//! respecte une contrainte de sécurité formelle — une alternative
//! certifiable au réglage ad hoc de garde-fous statiques
//! ([`super::insulin_safety`]).
//!
//! ## Modèle
//! Dynamique de glycémie affine en la commande (modèle mono-compartimental
//! simplifié — voir avertissement) :
//! `dG/dt = -a·(G - G_b) - k·u`, avec `u ≥ 0` (débit d'insuline, jamais
//! négatif : on ne peut que suspendre, pas injecter de "anti-insuline").
//!
//! Contrainte de sécurité (ensemble sûr) : `h(G) = G - G_min ≥ 0` (ne
//! jamais passer sous le seuil hypoglycémique `G_min`). Condition CBF à
//! fonction de classe K linéaire de gain `alpha > 0` :
//! `ḣ(G) ≥ -alpha·h(G)`, soit `-a·(G-G_b) - k·u ≥ -alpha·(G-G_min)`.
//!
//! Cette inégalité est affine en l'unique variable de décision `u` : le
//! programme quadratique `min (u-u_desired)² s.c. contrainte CBF,
//! 0 ≤ u ≤ u_max` se résout donc en forme close (une borne supérieure sur
//! `u`), sans solveur QP général :
//! `u ≤ u_cbf_max = [alpha·(G-G_min) - a·(G-G_b)] / k`.
//!
//! **Avertissement** : le modèle de dynamique glycémique ci-dessus est une
//! simplification pédagogique (linéaire, un seul compartiment, pas de
//! retard d'absorption sous-cutanée). Un filtre de sécurité réel
//! utiliserait un modèle physiologique validé (modèle minimal de Bergman,
//! simulateur UVA/Padova) et une identification patient-spécifique des
//! paramètres `a`/`k` — ce module démontre la *technique* (CBF-QP résolu
//! en forme close), pas un modèle prêt pour un usage clinique.

/// Paramètres du modèle de dynamique glycémique affine utilisé par le filtre.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GlucoseModel {
    /// Taux de retour vers la cible basale, `a > 0` (1/min).
    pub reversion_rate: f64,
    /// Cible basale `G_b` (même unité que `glucose`).
    pub basal_target: f64,
    /// Sensibilité à l'insuline `k > 0` : chute de glycémie par unité de
    /// débit (même unité que `glucose`, par unité de débit).
    pub insulin_sensitivity: f64,
}

/// Résultat du filtre CBF.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SafeDose {
    /// La dose effectivement autorisée.
    pub units_per_hour: f64,
    /// `true` si le filtre a dû s'écarter de `u_desired` (borne `u_max`
    /// ou contrainte CBF active).
    pub constrained: bool,
    /// `true` si même `u = 0` ne suffit pas à satisfaire la condition CBF
    /// sous ce modèle — la dérive naturelle seule violerait déjà la marge
    /// de sécurité. Signal à faire remonter à une supervision de plus
    /// haut niveau (par ex. [`super::insulin_safety::suspend_on_low`] ou
    /// une recommandation de resucrage), pas une commande d'insuline : le
    /// filtre ne peut pas "retirer" de l'insuline déjà administrée.
    pub barrier_violated_at_zero_dose: bool,
}

/// Filtre CBF-QP : renvoie la dose la plus proche de `u_desired` (`≥ 0`)
/// qui respecte `u ≤ u_max` et la condition CBF pour ne pas approcher
/// `glucose_floor` plus vite que le gain `alpha` ne le permet.
///
/// Précondition (non validée à l'exécution — modèle fourni par
/// l'appelant) : `model.insulin_sensitivity > 0` et `alpha > 0`.
pub fn cbf_safe_dose(
    model: GlucoseModel,
    glucose: f64,
    glucose_floor: f64,
    alpha: f64,
    u_desired: f64,
    u_max: f64,
) -> SafeDose {
    let drift = model.reversion_rate * (glucose - model.basal_target);
    let margin = alpha * (glucose - glucose_floor);
    let u_cbf_max = (margin - drift) / model.insulin_sensitivity;

    let upper = u_max.min(u_cbf_max).max(0.0);
    let units_per_hour = u_desired.max(0.0).min(upper);
    SafeDose {
        units_per_hour,
        constrained: (units_per_hour - u_desired).abs() > 1e-9,
        barrier_violated_at_zero_dose: u_cbf_max < 0.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    /// Valeurs vérifiées indépendamment (calcul numérique direct) avant
    /// portage : u_cbf_max = [alpha(G-Gmin) - a(G-Gb)] / k.
    #[test]
    fn caps_an_aggressive_dose_that_would_overshoot_toward_hypo() {
        let model = GlucoseModel {
            reversion_rate: 0.02,
            basal_target: 100.0,
            insulin_sensitivity: 3.0,
        };
        let out = cbf_safe_dose(model, 180.0, 70.0, 0.05, 4.0, 10.0);
        assert_relative_eq!(out.units_per_hour, 1.3, epsilon = 1e-9);
        assert!(out.constrained);
        assert!(!out.barrier_violated_at_zero_dose);
    }

    #[test]
    fn leaves_an_already_safe_dose_untouched() {
        let model = GlucoseModel {
            reversion_rate: 0.0,
            basal_target: 100.0,
            insulin_sensitivity: 1.0,
        };
        let out = cbf_safe_dose(model, 150.0, 70.0, 0.1, 2.0, 10.0);
        assert_relative_eq!(out.units_per_hour, 2.0, epsilon = 1e-9);
        assert!(!out.constrained);
    }

    #[test]
    fn pump_max_rate_binds_before_the_barrier_does() {
        let model = GlucoseModel {
            reversion_rate: 0.0,
            basal_target: 100.0,
            insulin_sensitivity: 1.0,
        };
        // u_cbf_max = 80 here, but the pump itself caps at 5 U/h.
        let out = cbf_safe_dose(model, 150.0, 70.0, 1.0, 20.0, 5.0);
        assert_relative_eq!(out.units_per_hour, 5.0, epsilon = 1e-9);
        assert!(out.constrained);
    }

    #[test]
    fn flags_when_even_zero_dose_cannot_satisfy_the_barrier() {
        let model = GlucoseModel {
            reversion_rate: 0.1,
            basal_target: 100.0,
            insulin_sensitivity: 3.0,
        };
        // Fast reversion + weak barrier gain far above target: natural
        // dynamics alone already fall faster than alpha tolerates.
        let out = cbf_safe_dose(model, 200.0, 70.0, 0.01, 0.5, 10.0);
        assert_relative_eq!(out.units_per_hour, 0.0, epsilon = 1e-9);
        assert!(out.constrained);
        assert!(out.barrier_violated_at_zero_dose);
    }

    #[test]
    fn negative_desired_dose_is_clamped_to_zero() {
        let model = GlucoseModel {
            reversion_rate: 0.0,
            basal_target: 100.0,
            insulin_sensitivity: 1.0,
        };
        let out = cbf_safe_dose(model, 150.0, 70.0, 0.1, -3.0, 10.0);
        assert_relative_eq!(out.units_per_hour, 0.0, epsilon = 1e-9);
        assert!(out.constrained);
    }
}
