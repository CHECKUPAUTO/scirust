//! Point de fonctionnement **pompe-réseau** — intersection de la caractéristique
//! de la pompe et de la courbe de charge du réseau.
//!
//! ```text
//! charge réseau     Hr = H_stat + k·Q²          (statique + pertes)
//! charge pompe      Hp = H0 − a·Q²              (parabole descendante)
//! débit de fonct.   Q_op = √( (H0 − H_stat)/(a + k) )
//! charge de fonct.  H_op = H_stat + k·Q_op²
//! ```
//!
//! `H_stat` hauteur géométrique + pression statique (m), `k` coefficient de perte
//! du réseau (s²/m⁵), `Q` débit (m³/s), `H0` hauteur à débit nul de la pompe
//! (shut-off, m), `a` coefficient de la caractéristique pompe (s²/m⁵). Le point
//! de fonctionnement est l'unique intersection des deux courbes.
//!
//! **Convention** : SI cohérent, charges en mètres de colonne. **Limite
//! honnête** : caractéristiques modélisées par des **paraboles** (`H0 − a·Q²` et
//! `H_stat + k·Q²`) ; les coefficients `H0`, `a`, `k` proviennent d'un ajustement
//! des courbes constructeur/réseau fourni par l'appelant. Une seule pompe, régime
//! permanent.

/// Charge du réseau `Hr = H_stat + k·Q²` (m).
pub fn system_head(static_head: f64, resistance_coeff: f64, flow: f64) -> f64 {
    static_head + resistance_coeff * flow * flow
}

/// Charge fournie par la pompe `Hp = H0 − a·Q²` (m).
pub fn pump_head(shutoff_head: f64, pump_coeff: f64, flow: f64) -> f64 {
    shutoff_head - pump_coeff * flow * flow
}

/// Débit au **point de fonctionnement** `Q_op = √((H0 − H_stat)/(a + k))` (m³/s).
///
/// Panique si `a + k <= 0` ou si `H0 < H_stat` (la pompe ne peut pas vaincre la
/// hauteur statique).
pub fn operating_flow(
    shutoff_head: f64,
    pump_coeff: f64,
    static_head: f64,
    resistance_coeff: f64,
) -> f64 {
    let denom = pump_coeff + resistance_coeff;
    assert!(denom > 0.0, "a + k doit être strictement positif");
    assert!(
        shutoff_head >= static_head,
        "la pompe ne peut pas vaincre la hauteur statique (H0 < H_stat)"
    );
    ((shutoff_head - static_head) / denom).sqrt()
}

/// Charge au point de fonctionnement `H_op = H_stat + k·Q_op²` (m).
pub fn operating_head(
    shutoff_head: f64,
    pump_coeff: f64,
    static_head: f64,
    resistance_coeff: f64,
) -> f64 {
    let q = operating_flow(shutoff_head, pump_coeff, static_head, resistance_coeff);
    system_head(static_head, resistance_coeff, q)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn operating_point_is_where_curves_cross() {
        // Au débit de fonctionnement, charge pompe = charge réseau.
        let (h0, a, hstat, k) = (50.0, 2000.0, 20.0, 1000.0);
        let q = operating_flow(h0, a, hstat, k);
        assert_relative_eq!(
            pump_head(h0, a, q),
            system_head(hstat, k, q),
            max_relative = 1e-9
        );
        // et operating_head coïncide avec pump_head à ce débit.
        assert_relative_eq!(
            operating_head(h0, a, hstat, k),
            pump_head(h0, a, q),
            max_relative = 1e-9
        );
    }

    #[test]
    fn more_resistance_reduces_flow() {
        // Fermer une vanne (k↑) déplace le point vers un débit plus faible.
        let q1 = operating_flow(50.0, 2000.0, 20.0, 1000.0);
        let q2 = operating_flow(50.0, 2000.0, 20.0, 4000.0);
        assert!(q2 < q1);
    }

    #[test]
    fn known_operating_flow() {
        // (H0−H_stat)/(a+k) = (50−20)/3000 = 0,01 → Q_op = 0,1 m³/s.
        assert_relative_eq!(
            operating_flow(50.0, 2000.0, 20.0, 1000.0),
            0.1,
            epsilon = 1e-9
        );
    }

    #[test]
    #[should_panic(expected = "hauteur statique")]
    fn pump_too_weak_panics() {
        // H0 < H_stat : pas de fonctionnement possible.
        operating_flow(15.0, 2000.0, 20.0, 1000.0);
    }
}
