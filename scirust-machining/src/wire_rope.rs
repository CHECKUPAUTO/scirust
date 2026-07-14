//! Câbles métalliques de levage — effort de rupture minimal, charge de travail
//! et diamètre de poulie minimal selon la classe et la norme du fabricant.
//!
//! ```text
//! effort de rupture minimal   MBF = K·d²
//! charge de travail           WLL = MBF / SF
//! diamètre de poulie minimal  D   = (D/d)·d
//! ```
//!
//! `d` diamètre nominal du câble (m), `K` constante de classe/construction
//! (N/m², dépend du grade d'acier et du type de toronnage, fournie par la
//! norme ou le fabricant), `MBF` effort de rupture minimal garanti (N), `SF`
//! facteur de sécurité (sans dimension, ≥ 1), `WLL` charge maximale
//! d'utilisation (N), `D/d` rapport minimal poulie/câble prescrit (sans
//! dimension), `D` diamètre primitif de poulie ou de tambour (m).
//!
//! **Convention** : SI cohérent (mètres, newtons). **Limite honnête** : le
//! modèle `MBF = K·d²` est une corrélation d'ingénierie ; la constante `K`, le
//! facteur de sécurité `SF` et le rapport `D/d` **proviennent de la norme
//! applicable ou du fabricant** (p. ex. ISO 2408, réglementation levage) et ne
//! sont **jamais** inventés ici. Aucune tolérance, usure, fatigue de flexion,
//! angle de déviation ni effet de terminaison n'est pris en compte.

/// Effort de rupture minimal `MBF = K·d²` (N).
///
/// `grade_constant` = `K` en N/m², fourni par la norme/le fabricant selon la
/// classe et la construction du câble.
///
/// Panique si `diameter <= 0` ou `grade_constant <= 0`.
pub fn minimum_breaking_force(diameter: f64, grade_constant: f64) -> f64 {
    assert!(
        diameter > 0.0,
        "le diamètre du câble doit être strictement positif"
    );
    assert!(
        grade_constant > 0.0,
        "la constante de classe K doit être strictement positive"
    );
    grade_constant * diameter * diameter
}

/// Charge maximale d'utilisation `WLL = MBF / SF` (N).
///
/// Panique si `minimum_breaking_force < 0` ou `safety_factor <= 0`.
pub fn wire_rope_working_load(minimum_breaking_force: f64, safety_factor: f64) -> f64 {
    assert!(
        minimum_breaking_force >= 0.0,
        "l'effort de rupture minimal ne peut pas être négatif"
    );
    assert!(
        safety_factor > 0.0,
        "le facteur de sécurité doit être strictement positif"
    );
    minimum_breaking_force / safety_factor
}

/// Diamètre primitif minimal de poulie `D = (D/d)·d` (m).
///
/// `d_over_d_ratio` = rapport minimal `D/d` prescrit par la norme/le fabricant.
///
/// Panique si `rope_diameter <= 0` ou `d_over_d_ratio <= 0`.
pub fn minimum_sheave_diameter(rope_diameter: f64, d_over_d_ratio: f64) -> f64 {
    assert!(
        rope_diameter > 0.0,
        "le diamètre du câble doit être strictement positif"
    );
    assert!(
        d_over_d_ratio > 0.0,
        "le rapport D/d doit être strictement positif"
    );
    d_over_d_ratio * rope_diameter
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn breaking_force_scales_as_diameter_squared() {
        // MBF ∝ d² : doubler le diamètre quadruple l'effort de rupture.
        let k = 5.0e8_f64;
        let f1 = minimum_breaking_force(0.010, k);
        let f2 = minimum_breaking_force(0.020, k);
        assert_relative_eq!(f2 / f1, 4.0, epsilon = 1e-12);
    }

    #[test]
    fn breaking_force_realistic_case() {
        // Câble d=16 mm, K=540 MPa (classe 6×19, ordre de grandeur fourni) :
        // MBF = 540e6 · 0,016² = 138 240 N.
        let mbf = minimum_breaking_force(0.016, 540.0e6);
        assert_relative_eq!(mbf, 138_240.0, epsilon = 1e-6);
    }

    #[test]
    fn working_load_is_breaking_force_over_safety_factor() {
        // WLL = MBF/SF ; réciproquement MBF = WLL·SF.
        let mbf = 138_240.0_f64;
        let sf = 5.0;
        let wll = wire_rope_working_load(mbf, sf);
        assert_relative_eq!(wll, mbf / sf, epsilon = 1e-9);
        assert_relative_eq!(wll * sf, mbf, epsilon = 1e-6);
    }

    #[test]
    fn larger_safety_factor_lowers_working_load() {
        // À MBF fixé, WLL décroît quand SF augmente (WLL ∝ 1/SF).
        let mbf = 100_000.0_f64;
        let low = wire_rope_working_load(mbf, 4.0);
        let high = wire_rope_working_load(mbf, 8.0);
        assert!(high < low);
        assert_relative_eq!(low / high, 2.0, epsilon = 1e-12);
    }

    #[test]
    fn sheave_diameter_is_ratio_times_rope() {
        // D = (D/d)·d : avec d=12 mm et D/d=25 → D = 0,300 m.
        let d = minimum_sheave_diameter(0.012, 25.0);
        assert_relative_eq!(d, 0.300, epsilon = 1e-12);
        // Le rapport se retrouve exactement : D/d = 25.
        assert_relative_eq!(d / 0.012, 25.0, epsilon = 1e-12);
    }

    #[test]
    #[should_panic(expected = "facteur de sécurité")]
    fn zero_safety_factor_panics() {
        wire_rope_working_load(138_240.0, 0.0);
    }
}
