//! Stabilité au basculement d'un engin de levage — moment de renversement,
//! moment stabilisateur, facteur de stabilité et charge maximale admissible à un
//! rayon donné.
//!
//! ```text
//! moment de renversement   M_tip  = W·r
//! moment stabilisateur     M_stab = W_cw·r_cw + W_m·r_m
//! facteur de stabilité     n      = M_stab / M_tip
//! charge max au rayon      W_max  = M_stab / (r·s)
//! ```
//!
//! `W` poids de la charge levée (N), `r` rayon de la charge = bras de levier
//! autour de l'arête de basculement (m), `M_tip` moment de renversement (N·m),
//! `W_cw` poids du contrepoids (N), `r_cw` bras de levier du contrepoids (m),
//! `W_m` poids propre de la machine (N), `r_m` bras de levier du centre de gravité
//! machine (m), `M_stab` moment stabilisateur (N·m), `n` facteur de stabilité
//! (sans dimension, `n > 1` requis), `W_max` charge maximale admissible au rayon
//! `r` (N), `s` facteur de sécurité réglementaire (sans dimension).
//!
//! **Convention** : SI cohérent — poids/forces en N, bras de levier en m, moments
//! en N·m ; les rayons/bras sont mesurés horizontalement depuis l'arête de
//! basculement (contrepoids et machine du côté stabilisant, charge du côté
//! renversant).
//!
//! **Limite honnête** : équilibre **statique 2D** autour d'une arête de
//! basculement unique. Les **poids** et **bras de levier** (`W`, `W_cw`, `W_m`,
//! `r`, `r_cw`, `r_m`) sont **fournis par l'appelant** — aucune géométrie ni masse
//! « par défaut » n'est inventée ici. Le modèle **néglige les effets dynamiques**
//! (balancement de la charge, vent, accélérations de levage/rotation, freinage,
//! inclinaison du sol) : ceux-ci sont couverts par le **facteur de sécurité `s`**
//! imposé par la **réglementation**, lui aussi **fourni** et jamais présumé.

/// Moment de renversement `M_tip = W·r` autour de l'arête de basculement.
///
/// `load` = `W` poids de la charge (N), `load_radius` = `r` bras de levier (m) ;
/// renvoie un moment (N·m).
///
/// Panique si `load < 0` ou `load_radius < 0`.
pub fn crane_tipping_moment(load: f64, load_radius: f64) -> f64 {
    assert!(load >= 0.0 && load_radius >= 0.0, "W ≥ 0 et r ≥ 0 requis");
    load * load_radius
}

/// Moment stabilisateur `M_stab = W_cw·r_cw + W_m·r_m` (contrepoids + machine).
///
/// `counterweight` = `W_cw` (N), `counterweight_radius` = `r_cw` (m),
/// `machine_weight` = `W_m` (N), `machine_cg_radius` = `r_m` (m) ; renvoie un
/// moment (N·m).
///
/// Panique si l'un des quatre arguments est négatif.
pub fn crane_stabilizing_moment(
    counterweight: f64,
    counterweight_radius: f64,
    machine_weight: f64,
    machine_cg_radius: f64,
) -> f64 {
    assert!(
        counterweight >= 0.0
            && counterweight_radius >= 0.0
            && machine_weight >= 0.0
            && machine_cg_radius >= 0.0,
        "W_cw ≥ 0, r_cw ≥ 0, W_m ≥ 0 et r_m ≥ 0 requis"
    );
    counterweight * counterweight_radius + machine_weight * machine_cg_radius
}

/// Facteur de stabilité `n = M_stab / M_tip` (marge au basculement, `n > 1` requis).
///
/// `stabilizing_moment` = `M_stab` (N·m), `tipping_moment` = `M_tip` (N·m) ;
/// renvoie un facteur sans dimension. La marge minimale exigée est **fournie** par
/// la réglementation, elle n'est pas comparée ici.
///
/// Panique si `stabilizing_moment < 0` ou `tipping_moment <= 0`.
pub fn crane_stability_factor(stabilizing_moment: f64, tipping_moment: f64) -> f64 {
    assert!(
        stabilizing_moment >= 0.0 && tipping_moment > 0.0,
        "M_stab ≥ 0 et M_tip > 0 requis"
    );
    stabilizing_moment / tipping_moment
}

/// Charge maximale admissible au rayon `r` : `W_max = M_stab / (r·s)`.
///
/// `stabilizing_moment` = `M_stab` (N·m), `load_radius` = `r` (m),
/// `safety_factor` = `s` facteur de sécurité réglementaire (sans dimension) ;
/// renvoie un poids (N).
///
/// Panique si `stabilizing_moment < 0`, `load_radius <= 0` ou `safety_factor <= 0`.
pub fn crane_max_load_at_radius(
    stabilizing_moment: f64,
    load_radius: f64,
    safety_factor: f64,
) -> f64 {
    assert!(
        stabilizing_moment >= 0.0 && load_radius > 0.0 && safety_factor > 0.0,
        "M_stab ≥ 0, r > 0 et s > 0 requis"
    );
    stabilizing_moment / (load_radius * safety_factor)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn tipping_moment_is_bilinear() {
        // M_tip ∝ W et ∝ r : doubler chacun quadruple le moment de renversement.
        let base = crane_tipping_moment(20_000.0, 5.0);
        let quad = crane_tipping_moment(40_000.0, 10.0);
        assert_relative_eq!(quad, 4.0 * base, epsilon = 1e-6);
    }

    #[test]
    fn stabilizing_moment_is_additive() {
        // Le moment stabilisateur est la somme des contributions contrepoids et
        // machine, chacune valant W·r.
        let cw = crane_tipping_moment(30_000.0, 3.0);
        let machine = crane_tipping_moment(50_000.0, 1.0);
        let total = crane_stabilizing_moment(30_000.0, 3.0, 50_000.0, 1.0);
        assert_relative_eq!(total, cw + machine, epsilon = 1e-6);
    }

    #[test]
    fn max_load_produces_expected_stability_factor() {
        // À la charge maximale W_max = M_stab/(r·s), le moment de renversement vaut
        // M_stab/s, donc le facteur de stabilité vaut exactement s (réciprocité).
        let m_stab = 140_000.0_f64;
        let (r, s) = (5.0_f64, 1.5_f64);
        let w_max = crane_max_load_at_radius(m_stab, r, s);
        let m_tip = crane_tipping_moment(w_max, r);
        assert_relative_eq!(crane_stability_factor(m_stab, m_tip), s, epsilon = 1e-9);
    }

    #[test]
    fn max_load_at_unit_safety_recovers_moment() {
        // Avec s = 1, la charge max au rayon r crée un moment de renversement égal
        // au moment stabilisateur (basculement imminent, n = 1).
        let m_stab = 90_000.0_f64;
        let r = 4.0_f64;
        let w_max = crane_max_load_at_radius(m_stab, r, 1.0);
        assert_relative_eq!(crane_tipping_moment(w_max, r), m_stab, epsilon = 1e-6);
    }

    #[test]
    fn realistic_mobile_crane() {
        // Charge 20 kN à 5 m → M_tip = 100 kN·m.
        let m_tip = crane_tipping_moment(20_000.0, 5.0);
        assert_relative_eq!(m_tip, 100_000.0, epsilon = 1e-6);
        // Contrepoids 30 kN à 3 m (90 kN·m) + machine 50 kN à 1 m (50 kN·m)
        // → M_stab = 140 kN·m.
        let m_stab = crane_stabilizing_moment(30_000.0, 3.0, 50_000.0, 1.0);
        assert_relative_eq!(m_stab, 140_000.0, epsilon = 1e-6);
        // Facteur de stabilité 140/100 = 1.4.
        assert_relative_eq!(crane_stability_factor(m_stab, m_tip), 1.4, epsilon = 1e-9);
        // Charge max au rayon 5 m avec s = 1.5 : 140000/(5·1.5) = 18666.66… N.
        assert_relative_eq!(
            crane_max_load_at_radius(m_stab, 5.0, 1.5),
            140_000.0 / 7.5,
            epsilon = 1e-6
        );
    }

    #[test]
    fn stability_factor_unity_at_balance() {
        // Quand le moment stabilisateur égale le moment de renversement, n = 1
        // (limite de basculement).
        let m = 120_000.0_f64;
        assert_relative_eq!(crane_stability_factor(m, m), 1.0, epsilon = 1e-12);
    }

    #[test]
    #[should_panic(expected = "M_tip > 0")]
    fn zero_tipping_moment_panics() {
        crane_stability_factor(140_000.0, 0.0);
    }
}
