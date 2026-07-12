//! Cycles thermodynamiques — rendements théoriques : **Carnot**, **Otto**
//! (allumage commandé), **Diesel**, et coefficients de performance des machines
//! frigorifiques et pompes à chaleur.
//!
//! ```text
//! Carnot          η = 1 − T_froid/T_chaud
//! Otto            η = 1 − 1/r^{γ−1}                     (r = taux de compression)
//! Diesel          η = 1 − (1/r^{γ−1})·(rc^γ − 1)/(γ·(rc − 1))
//! rendement       η = W_net/Q_entrée
//! COP froid       COP_f = T_froid/(T_chaud − T_froid)   (Carnot inverse)
//! COP PAC         COP_c = T_chaud/(T_chaud − T_froid)
//! ```
//!
//! `T` températures **absolues** (K), `r` taux de compression volumétrique, `rc`
//! rapport de détente à pression constante (Diesel), `γ = cp/cv` (≈ 1,4 pour
//! l'air), `W_net` travail net, `Q_entrée` chaleur fournie.
//!
//! **Convention** : températures en **kelvin**. **Limite honnête** : cycles à
//! **air standard** idéaux (gaz parfait, γ constant, transformations
//! réversibles) ; les rendements réels sont inférieurs. Carnot est la borne
//! supérieure entre deux sources.

/// Rendement de Carnot `η = 1 − T_froid/T_chaud`.
///
/// Panique si `t_hot <= 0` ou si `t_cold > t_hot`.
pub fn carnot_efficiency(t_cold_k: f64, t_hot_k: f64) -> f64 {
    assert!(
        t_hot_k > 0.0,
        "la température chaude (K) doit être strictement positive"
    );
    assert!(t_cold_k <= t_hot_k, "T_froid ne peut pas dépasser T_chaud");
    1.0 - t_cold_k / t_hot_k
}

/// Rendement du cycle d'**Otto** `η = 1 − 1/r^{γ−1}`.
///
/// Panique si `r <= 1` ou `gamma <= 1`.
pub fn otto_efficiency(compression_ratio: f64, gamma: f64) -> f64 {
    assert!(
        compression_ratio > 1.0 && gamma > 1.0,
        "r > 1 et γ > 1 requis"
    );
    1.0 - 1.0 / compression_ratio.powf(gamma - 1.0)
}

/// Rendement du cycle **Diesel**
/// `η = 1 − (1/r^{γ−1})·(rc^γ − 1)/(γ·(rc − 1))`.
///
/// Panique si `r <= 1`, `cutoff_ratio <= 1` ou `gamma <= 1`.
pub fn diesel_efficiency(compression_ratio: f64, cutoff_ratio: f64, gamma: f64) -> f64 {
    assert!(
        compression_ratio > 1.0 && cutoff_ratio > 1.0 && gamma > 1.0,
        "r > 1, rc > 1 et γ > 1 requis"
    );
    let head = 1.0 / compression_ratio.powf(gamma - 1.0);
    let bracket = (cutoff_ratio.powf(gamma) - 1.0) / (gamma * (cutoff_ratio - 1.0));
    1.0 - head * bracket
}

/// Rendement thermique `η = W_net/Q_entrée`.
///
/// Panique si `heat_in <= 0`.
pub fn thermal_efficiency(net_work: f64, heat_in: f64) -> f64 {
    assert!(
        heat_in > 0.0,
        "la chaleur fournie doit être strictement positive"
    );
    net_work / heat_in
}

/// COP d'une machine **frigorifique** de Carnot `COP_f = T_froid/(T_chaud − T_froid)`.
///
/// Panique si `t_hot <= t_cold`.
pub fn cop_refrigerator_carnot(t_cold_k: f64, t_hot_k: f64) -> f64 {
    assert!(
        t_hot_k > t_cold_k,
        "T_chaud doit dépasser strictement T_froid"
    );
    t_cold_k / (t_hot_k - t_cold_k)
}

/// COP d'une **pompe à chaleur** de Carnot `COP_c = T_chaud/(T_chaud − T_froid)`.
///
/// Panique si `t_hot <= t_cold`.
pub fn cop_heat_pump_carnot(t_cold_k: f64, t_hot_k: f64) -> f64 {
    assert!(
        t_hot_k > t_cold_k,
        "T_chaud doit dépasser strictement T_froid"
    );
    t_hot_k / (t_hot_k - t_cold_k)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn carnot_between_two_reservoirs() {
        // 300 K / 600 K → η = 0,5.
        assert_relative_eq!(carnot_efficiency(300.0, 600.0), 0.5, epsilon = 1e-12);
    }

    #[test]
    fn otto_efficiency_for_petrol_engine() {
        // r=8, γ=1,4 → η = 1 − 1/8^0,4 ≈ 0,565.
        let eta = otto_efficiency(8.0, 1.4);
        assert_relative_eq!(eta, 1.0 - 1.0 / 8.0f64.powf(0.4), epsilon = 1e-12);
        assert!(eta > 0.56 && eta < 0.57);
    }

    #[test]
    fn diesel_below_otto_at_same_ratio() {
        // À même taux de compression, le Diesel (rc>1) rend un peu moins que
        // l'Otto idéal (à cause du terme de détente).
        let (r, gamma) = (18.0, 1.4);
        let diesel = diesel_efficiency(r, 2.0, gamma);
        let otto = otto_efficiency(r, gamma);
        assert!(diesel < otto);
        assert!(diesel > 0.0 && diesel < 1.0);
    }

    #[test]
    fn cop_relations() {
        // COP_PAC = COP_froid + 1 (identité thermodynamique).
        let (tc, th) = (270.0, 300.0);
        let cf = cop_refrigerator_carnot(tc, th);
        let cc = cop_heat_pump_carnot(tc, th);
        assert_relative_eq!(cc, cf + 1.0, epsilon = 1e-9);
        // COP_froid = 270/30 = 9.
        assert_relative_eq!(cf, 9.0, epsilon = 1e-9);
    }

    #[test]
    fn thermal_efficiency_ratio() {
        assert_relative_eq!(thermal_efficiency(400.0, 1000.0), 0.4, epsilon = 1e-12);
    }

    #[test]
    #[should_panic(expected = "T_froid ne peut pas dépasser")]
    fn cold_above_hot_panics() {
        carnot_efficiency(700.0, 600.0);
    }
}
