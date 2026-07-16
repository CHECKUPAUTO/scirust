//! **Flèche d'une ligne aérienne (câble tendu)** — flèche par approximation
//! parabolique, longueur développée du conducteur, tension horizontale visée et
//! poids apparent résultant sous surcharge de givre et de vent, pour des appuis
//! de même niveau en régime permanent.
//!
//! ```text
//! flèche parabolique             s = w·L² / (8·H)
//! longueur du conducteur         ℓ = L + 8·s² / (3·L)
//! tension pour une flèche visée  H = w·L² / (8·s_cible)     (réciproque de s)
//! poids apparent (givre + vent)  w_a = √((w + w_g)² + w_v²)
//! ```
//!
//! `s` flèche au milieu de la portée (m), `w` poids linéique du conducteur
//! (N/m), `L` portée / longueur horizontale entre appuis (m), `H` tension
//! horizontale du conducteur (N), `ℓ` longueur développée du conducteur le long
//! de la parabole (m), `s_cible` flèche visée (m), `w_g` surcharge linéique de
//! givre (N/m), `w_v` surcharge linéique de vent, supposée horizontale (N/m),
//! `w_a` poids apparent linéique résultant (N/m).
//!
//! **Convention** : SI ; longueurs en m, poids et tensions linéiques en N/m,
//! tensions en N. **Limite honnête** : approximation **parabolique** (flèche
//! **faible devant la portée**, conducteur uniformément chargé, appuis de
//! **même niveau**) ; le poids linéique `w` et la tension horizontale `H` sont
//! **fournis par l'appelant** d'après une fiche de câble ou une mesure ; les
//! surcharges de **givre** `w_g` et de **vent** `w_v` (linéiques) sont
//! **fournies** et composées **vectoriellement** (vent horizontal, poids +
//! givre verticaux). Ce module ne traite **ni les portées dénivelées ni le
//! fluage** ; la caténaire exacte (`cosh`) reste à la charge de l'appelant.

/// Flèche au milieu de la portée par approximation parabolique
/// `s = w·L² / (8·H)` (m), appuis de même niveau.
///
/// Panique si `weight_per_length < 0`, si `span <= 0` ou si
/// `horizontal_tension <= 0`.
pub fn sag_parabolic(weight_per_length: f64, span: f64, horizontal_tension: f64) -> f64 {
    assert!(
        weight_per_length >= 0.0,
        "le poids linéique w doit être ≥ 0"
    );
    assert!(span > 0.0, "la portée L doit être > 0");
    assert!(
        horizontal_tension > 0.0,
        "la tension horizontale H doit être > 0"
    );
    weight_per_length * span * span / (8.0 * horizontal_tension)
}

/// Longueur développée du conducteur le long de la parabole
/// `ℓ = L + 8·s² / (3·L)` (m).
///
/// Panique si `span <= 0` ou si `sag < 0`.
pub fn sag_conductor_length_parabolic(span: f64, sag: f64) -> f64 {
    assert!(span > 0.0, "la portée L doit être > 0");
    assert!(sag >= 0.0, "la flèche s doit être ≥ 0");
    span + 8.0 * sag * sag / (3.0 * span)
}

/// Tension horizontale nécessaire pour obtenir une flèche visée
/// `H = w·L² / (8·s_cible)` (N), réciproque de [`sag_parabolic`].
///
/// Panique si `weight_per_length < 0`, si `span <= 0` ou si `target_sag <= 0`.
pub fn sag_tension_for_sag(weight_per_length: f64, span: f64, target_sag: f64) -> f64 {
    assert!(
        weight_per_length >= 0.0,
        "le poids linéique w doit être ≥ 0"
    );
    assert!(span > 0.0, "la portée L doit être > 0");
    assert!(target_sag > 0.0, "la flèche visée s_cible doit être > 0");
    weight_per_length * span * span / (8.0 * target_sag)
}

/// Poids apparent linéique résultant sous givre et vent
/// `w_a = √((w + w_g)² + w_v²)` (N/m), le vent étant supposé horizontal et le
/// poids ainsi que le givre verticaux.
///
/// Panique si `bare_weight_per_length < 0`, si `ice_weight_per_length < 0` ou si
/// `wind_load_per_length < 0`.
pub fn sag_with_ice_wind(
    bare_weight_per_length: f64,
    ice_weight_per_length: f64,
    wind_load_per_length: f64,
) -> f64 {
    assert!(
        bare_weight_per_length >= 0.0,
        "le poids linéique nu w doit être ≥ 0"
    );
    assert!(
        ice_weight_per_length >= 0.0,
        "la surcharge de givre w_g doit être ≥ 0"
    );
    assert!(
        wind_load_per_length >= 0.0,
        "la surcharge de vent w_v doit être ≥ 0"
    );
    ((bare_weight_per_length + ice_weight_per_length).powi(2)
        + wind_load_per_length * wind_load_per_length)
        .sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn sag_scales_with_span_squared_and_inverse_tension() {
        // s ∝ L² : doubler la portée quadruple la flèche.
        let s1 = sag_parabolic(10.0, 100.0, 8000.0);
        let s2 = sag_parabolic(10.0, 200.0, 8000.0);
        assert_relative_eq!(s2 / s1, 4.0, epsilon = 1e-12);
        // s ∝ 1/H : doubler la tension divise la flèche par deux.
        let s3 = sag_parabolic(10.0, 100.0, 16000.0);
        assert_relative_eq!(s1 / s3, 2.0, epsilon = 1e-12);
    }

    #[test]
    fn sag_realistic_case() {
        // Cas chiffré : w = 10 N/m, L = 100 m, H = 8000 N.
        //   s = 10·100² / (8·8000) = 100000 / 64000 = 1,5625 m
        let s = sag_parabolic(10.0, 100.0, 8000.0);
        assert_relative_eq!(s, 1.5625, epsilon = 1e-9);
    }

    #[test]
    fn tension_for_sag_is_reciprocal_of_sag() {
        // H → s → H doit boucler : la tension retrouvée redonne la flèche visée.
        let w = 12.5_f64;
        let span = 120.0_f64;
        let h = 9500.0_f64;
        let s = sag_parabolic(w, span, h);
        let h_back = sag_tension_for_sag(w, span, s);
        assert_relative_eq!(h_back, h, epsilon = 1e-6);
    }

    #[test]
    fn conductor_length_exceeds_span_by_parabolic_term() {
        // ℓ = L + 8·s²/(3·L) ; avec L = 100 m et s = 1,5625 m :
        //   8·1,5625² / (3·100) = 8·2,44140625 / 300 = 19,53125 / 300
        //                       = 0,065104166666...
        //   ℓ = 100,065104166666... m
        let span = 100.0_f64;
        let sag = 1.5625_f64;
        let length = sag_conductor_length_parabolic(span, sag);
        assert_relative_eq!(length, 100.065_104_166_666_67, epsilon = 1e-9);
        // Sans flèche, la longueur développée vaut exactement la portée.
        assert_relative_eq!(
            sag_conductor_length_parabolic(span, 0.0),
            span,
            epsilon = 1e-12
        );
    }

    #[test]
    fn apparent_weight_is_pythagorean_resultant() {
        // w_a = √((w + w_g)² + w_v²) ; avec w = 10, w_g = 5, w_v = 8 N/m :
        //   √((15)² + 8²) = √(225 + 64) = √289 = 17 N/m
        let w_a = sag_with_ice_wind(10.0, 5.0, 8.0);
        assert_relative_eq!(w_a, 17.0, epsilon = 1e-9);
        // Sans vent ni givre, le poids apparent se réduit au poids nu.
        assert_relative_eq!(sag_with_ice_wind(10.0, 0.0, 0.0), 10.0, epsilon = 1e-12);
    }

    #[test]
    #[should_panic(expected = "la tension horizontale H doit être > 0")]
    fn sag_rejects_non_positive_tension() {
        let _ = sag_parabolic(10.0, 100.0, 0.0);
    }
}
