//! **Canal jaugeur Parshall** — mesure d'un débit à surface libre à partir de la
//! charge amont dans un canal Parshall en écoulement libre (non noyé).
//!
//! ```text
//! débit libre (Q = C·W·Ha^n)   Q  = C · W · Ha^n
//! charge amont (réciproque)    Ha = ( Q / (C · W) )^(1/n)
//! taux de noyage               S  = Hb / Ha
//! ```
//!
//! `Q` débit volumique (m³/s), `C` coefficient de débit du canal (unité dépendant
//! de la norme et de la largeur de col), `W` largeur du col (m), `Ha` charge amont
//! mesurée au point de jauge amont (m), `n` exposant de la loi de débit (sans
//! dimension), `Hb` charge aval au point de jauge de col (m), `S` taux de noyage
//! (sans dimension).
//!
//! **Convention** : SI ; le couple `(C, n)` doit être exprimé dans un système
//! d'unités cohérent avec `W` et `Ha`. **Limite honnête** : canal Parshall en
//! écoulement **libre** (non noyé, l'aval reste sous le seuil de submersion). Le
//! **coefficient de débit `C` et l'exposant `n` sont fournis par l'appelant**
//! d'après la norme, selon la **largeur de col** ; ils ne sont jamais supposés.
//! Au-delà du **taux de noyage critique** (lui aussi fourni par la norme, typ.
//! `S ≈ 0,6`–`0,8` selon la taille du canal) l'écoulement est noyé et une
//! **correction de noyage fournie par l'appelant** est nécessaire : ces formules
//! ne s'appliquent alors plus.

/// Débit en **écoulement libre** d'un canal Parshall `Q = C·W·Ha^n` (m³/s).
///
/// Loi de débit à surface libre : le débit croît comme la puissance `n` de la
/// charge amont `Ha`, à coefficient `C` et largeur de col `W` fixés.
///
/// Panique si `discharge_coefficient <= 0`, `throat_width <= 0`, `head < 0`
/// ou `exponent <= 0`.
pub fn flume_parshall_free_flow(
    discharge_coefficient: f64,
    throat_width: f64,
    head: f64,
    exponent: f64,
) -> f64 {
    assert!(
        discharge_coefficient > 0.0,
        "le coefficient de débit C doit être > 0"
    );
    assert!(throat_width > 0.0, "la largeur de col W doit être > 0");
    assert!(head >= 0.0, "la charge amont Ha doit être ≥ 0");
    assert!(exponent > 0.0, "l'exposant n doit être > 0");
    discharge_coefficient * throat_width * head.powf(exponent)
}

/// Charge amont d'un canal Parshall à partir de son débit libre
/// `Ha = ( Q / (C·W) )^(1/n)` (m) — réciproque de [`flume_parshall_free_flow`].
///
/// Panique si `discharge_coefficient <= 0`, `throat_width <= 0`, `flow_rate < 0`
/// ou `exponent <= 0`.
pub fn flume_head_from_free_flow(
    flow_rate: f64,
    discharge_coefficient: f64,
    throat_width: f64,
    exponent: f64,
) -> f64 {
    assert!(
        discharge_coefficient > 0.0,
        "le coefficient de débit C doit être > 0"
    );
    assert!(throat_width > 0.0, "la largeur de col W doit être > 0");
    assert!(flow_rate >= 0.0, "le débit Q doit être ≥ 0");
    assert!(exponent > 0.0, "l'exposant n doit être > 0");
    (flow_rate / (discharge_coefficient * throat_width)).powf(1.0 / exponent)
}

/// Taux de **noyage** d'un canal Parshall `S = Hb/Ha` (sans dimension).
///
/// Rapport de la charge aval `Hb` (point de jauge de col) à la charge amont `Ha`.
/// Comparé au taux de noyage critique fourni par la norme, il indique si
/// l'écoulement reste libre (`S` sous le seuil) ou devient noyé.
///
/// Panique si `upstream_head <= 0` ou si `downstream_head < 0`.
pub fn flume_submergence_ratio(downstream_head: f64, upstream_head: f64) -> f64 {
    assert!(upstream_head > 0.0, "la charge amont Ha doit être > 0");
    assert!(downstream_head >= 0.0, "la charge aval Hb doit être ≥ 0");
    downstream_head / upstream_head
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn free_flow_and_head_are_reciprocal() {
        // Aller-retour : Ha → Q → Ha doit redonner la charge initiale.
        let c = 2.4_f64;
        let w = 1.0_f64;
        let ha = 0.5_f64;
        let n = 1.522_f64;
        let q = flume_parshall_free_flow(c, w, ha, n);
        let ha_back = flume_head_from_free_flow(q, c, w, n);
        assert_relative_eq!(ha_back, ha, epsilon = 1e-12);
    }

    #[test]
    fn free_flow_is_linear_in_throat_width() {
        // Q ∝ W à charge, coefficient et exposant fixés : doubler la largeur
        // de col double le débit.
        let c = 2.0_f64;
        let ha = 0.3_f64;
        let n = 1.55_f64;
        let q1 = flume_parshall_free_flow(c, 0.5, ha, n);
        let q2 = flume_parshall_free_flow(c, 1.0, ha, n);
        assert_relative_eq!(q2, 2.0 * q1, epsilon = 1e-12);
    }

    #[test]
    fn free_flow_scales_as_head_power_n() {
        // Q ∝ Ha^n : doubler la charge multiplie le débit par 2^n.
        let c = 2.4_f64;
        let w = 0.75_f64;
        let n = 1.55_f64;
        let q1 = flume_parshall_free_flow(c, w, 0.2, n);
        let q2 = flume_parshall_free_flow(c, w, 0.4, n);
        assert_relative_eq!(q2 / q1, 2.0_f64.powf(n), epsilon = 1e-12);
    }

    #[test]
    fn free_flow_realistic_case() {
        // C = 2,4 ; W = 1,0 m ; Ha = 0,5 m ; n = 1,522.
        // Ha^n = 0,5^1,522 = 0,348203 ; Q = 2,4·1,0·0,348203 = 0,835687 m³/s.
        let q = flume_parshall_free_flow(2.4, 1.0, 0.5, 1.522);
        assert_relative_eq!(q, 0.835_686_888, epsilon = 1e-6);
    }

    #[test]
    fn submergence_ratio_is_dimensionless_ratio() {
        // S = Hb/Ha : cas chiffré Hb = 0,36 m, Ha = 0,60 m → S = 0,60.
        let s = flume_submergence_ratio(0.36, 0.60);
        assert_relative_eq!(s, 0.60, epsilon = 1e-12);
    }

    #[test]
    fn submergence_ratio_equals_one_when_heads_equal() {
        // Charges amont et aval égales : le taux de noyage vaut exactement 1.
        let s = flume_submergence_ratio(0.42, 0.42);
        assert_relative_eq!(s, 1.0, epsilon = 1e-12);
    }

    #[test]
    #[should_panic(expected = "le coefficient de débit C doit être > 0")]
    fn non_positive_discharge_coefficient_panics() {
        flume_parshall_free_flow(0.0, 1.0, 0.5, 1.522);
    }
}
