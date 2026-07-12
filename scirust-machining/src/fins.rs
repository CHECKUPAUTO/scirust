//! Ailettes (surfaces étendues) — paramètre d'ailette, efficacité, efficience et
//! flux évacué d'une ailette droite à bout adiabatique.
//!
//! ```text
//! paramètre       m = √(h·P/(k·A_c))
//! efficacité      η = tanh(m·L)/(m·L)              (rendement de l'ailette)
//! flux évacué     Q = √(h·P·k·A_c)·θb·tanh(m·L)    (bout adiabatique)
//! efficience      ε = Q / (h·A_c·θb)               (gain vs surface nue)
//! ```
//!
//! `h` coefficient de convection (W/(m²·K)), `P` périmètre de l'ailette (m),
//! `A_c` aire de section (m²), `k` conductivité (W/(m·K)), `L` longueur (m), `θb`
//! écart de température base-fluide (K), `m` paramètre d'ailette (1/m).
//!
//! **Convention** : SI cohérent. **Limite honnête** : ailette à **section
//! constante**, régime **permanent** 1D, `h` uniforme, **bout adiabatique**
//! (utiliser une longueur corrigée `Lc = L + A_c/P` pour approcher un bout
//! convectif). Pas de rayonnement ni de contact base-ailette imparfait.

/// Paramètre d'ailette `m = √(h·P/(k·A_c))` (1/m).
///
/// Panique si `k·A_c <= 0`.
pub fn fin_parameter(h: f64, perimeter: f64, k: f64, cross_area: f64) -> f64 {
    assert!(k * cross_area > 0.0, "k·A_c doit être strictement positif");
    (h * perimeter / (k * cross_area)).sqrt()
}

/// Efficacité (rendement) d'une ailette à bout adiabatique `η = tanh(m·L)/(m·L)`.
///
/// Panique si `m*length <= 0`.
pub fn fin_efficiency(fin_parameter: f64, length: f64) -> f64 {
    let ml = fin_parameter * length;
    assert!(ml > 0.0, "m·L doit être strictement positif");
    ml.tanh() / ml
}

/// Flux évacué par l'ailette `Q = √(h·P·k·A_c)·θb·tanh(m·L)` (W).
///
/// Panique si `k·A_c <= 0`.
pub fn fin_heat_rate(
    h: f64,
    perimeter: f64,
    k: f64,
    cross_area: f64,
    base_temp_excess: f64,
    length: f64,
) -> f64 {
    assert!(k * cross_area > 0.0, "k·A_c doit être strictement positif");
    let m = fin_parameter(h, perimeter, k, cross_area);
    (h * perimeter * k * cross_area).sqrt() * base_temp_excess * (m * length).tanh()
}

/// Efficience de l'ailette `ε = Q/(h·A_c·θb)` (gain par rapport à la surface nue).
///
/// Panique si `h·A_c·θb <= 0`.
pub fn fin_effectiveness(
    fin_heat_rate: f64,
    h: f64,
    cross_area: f64,
    base_temp_excess: f64,
) -> f64 {
    let denom = h * cross_area * base_temp_excess;
    assert!(denom > 0.0, "h·A_c·θb doit être strictement positif");
    fin_heat_rate / denom
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn fin_parameter_definition() {
        // h=100, P=0,1, k=200, A_c=1e-4 → m = √(100·0,1/(200·1e-4)) = √500 ≈ 22,36.
        assert_relative_eq!(
            fin_parameter(100.0, 0.1, 200.0, 1e-4),
            500.0f64.sqrt(),
            epsilon = 1e-9
        );
    }

    #[test]
    fn efficiency_below_one_and_falls_with_length() {
        // η = tanh(mL)/(mL) < 1, décroissante en L.
        let m = fin_parameter(100.0, 0.1, 200.0, 1e-4);
        let short = fin_efficiency(m, 0.02);
        let long = fin_efficiency(m, 0.08);
        assert!(short < 1.0 && long < short);
    }

    #[test]
    fn heat_rate_and_effectiveness_consistent() {
        // Une ailette utile a une efficience ≫ 1.
        let (h, p, k, ac, tb, l) = (100.0, 0.1, 200.0, 1e-4, 50.0, 0.05);
        let q = fin_heat_rate(h, p, k, ac, tb, l);
        assert!(q > 0.0);
        let eff = fin_effectiveness(q, h, ac, tb);
        assert!(eff > 2.0); // ajouter l'ailette augmente nettement l'échange
    }

    #[test]
    #[should_panic(expected = "k·A_c")]
    fn zero_conductivity_panics() {
        fin_parameter(100.0, 0.1, 0.0, 1e-4);
    }
}
