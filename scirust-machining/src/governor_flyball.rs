//! **Régulateur centrifuge à boules** (Watt / Porter) — asservissement mécanique
//! de la vitesse d'une machine par la remontée de masses tournantes.
//!
//! ```text
//! hauteur du cône (Watt)     h = g/ω²
//! effort centrifuge          F_c = m·r·ω²
//! régime du Porter           ω = √[ (g/h)·(m + M)/m ]
//! hauteur du Porter          h = (g/ω²)·(m + M)/m
//! ```
//!
//! `h` hauteur du cône du régulateur — distance verticale du point de suspension
//! au plan des boules (m), `ω` vitesse angulaire de l'arbre (rad/s), `g`
//! accélération de la pesanteur (m/s²), `m` masse d'une boule volante (kg), `r`
//! rayon de giration de la boule (m), `M` masse de la charge centrale (manchon)
//! du Porter (kg), `F_c` effort centrifuge sur une boule (N). Le régulateur de
//! Watt est le cas particulier `M = 0` du Porter.
//!
//! **Convention** : SI ; angles/vitesses angulaires en rad et rad/s. **Limite
//! honnête** : régime **permanent** (équilibre statique du cône), bras
//! **sans masse**, **frottement négligé** et manchon guidé sans effort ; ces
//! modèles idéaux ignorent l'amortissement, l'insensibilité et l'effort de
//! ressort d'un régulateur de Hartnell réel. L'accélération de la pesanteur est
//! prise égale à [`GRAVITY`] = 9,81 m/s² ; l'appelant qui travaille sous une
//! autre gravité doit adapter ses données en conséquence.

/// Accélération de la pesanteur retenue par ces modèles (m/s²).
pub const GRAVITY: f64 = 9.81;

/// Hauteur du cône d'un régulateur de **Watt** `h = g/ω²` (m).
///
/// C'est la distance verticale entre le point de suspension des bras et le plan
/// des boules à l'équilibre : elle décroît en `1/ω²` quand le régime monte.
///
/// Panique si `rotational_speed_rad <= 0` (dénominateur nul, régime non défini).
pub fn watt_governor_height(rotational_speed_rad: f64) -> f64 {
    assert!(
        rotational_speed_rad > 0.0,
        "ω doit être > 0 (dénominateur ω² non nul)"
    );
    GRAVITY / (rotational_speed_rad * rotational_speed_rad)
}

/// Effort centrifuge sur une boule volante `F_c = m·r·ω²` (N).
///
/// Force radiale sortante qui tend à écarter la boule et à relever le manchon.
///
/// Panique si `ball_mass < 0` ou `radius < 0`.
pub fn flyball_centrifugal_force(ball_mass: f64, radius: f64, rotational_speed_rad: f64) -> f64 {
    assert!(ball_mass >= 0.0, "la masse d'une boule doit être ≥ 0");
    assert!(radius >= 0.0, "le rayon de giration doit être ≥ 0");
    ball_mass * radius * rotational_speed_rad * rotational_speed_rad
}

/// Régime d'équilibre d'un régulateur de **Porter**
/// `ω = √[ (g/h)·(m + M)/m ]` (rad/s).
///
/// La charge centrale `M` (manchon lesté) augmente le régime nécessaire pour
/// tenir une hauteur `h` donnée par le facteur `√[(m + M)/m]`. À `M = 0` on
/// retrouve le régulateur de Watt.
///
/// Panique si `height <= 0`, `ball_mass <= 0` ou `central_load_mass < 0`.
pub fn porter_governor_speed_rad(height: f64, ball_mass: f64, central_load_mass: f64) -> f64 {
    assert!(height > 0.0, "la hauteur h doit être > 0");
    assert!(ball_mass > 0.0, "la masse d'une boule m doit être > 0");
    assert!(
        central_load_mass >= 0.0,
        "la masse de la charge centrale M doit être ≥ 0"
    );
    ((GRAVITY / height) * (ball_mass + central_load_mass) / ball_mass).sqrt()
}

/// Hauteur du cône d'un régulateur de **Porter**
/// `h = (g/ω²)·(m + M)/m` (m) — réciproque de [`porter_governor_speed_rad`].
///
/// Panique si `rotational_speed_rad <= 0`, `ball_mass <= 0` ou
/// `central_load_mass < 0`.
pub fn porter_governor_height(
    rotational_speed_rad: f64,
    ball_mass: f64,
    central_load_mass: f64,
) -> f64 {
    assert!(
        rotational_speed_rad > 0.0,
        "ω doit être > 0 (dénominateur ω² non nul)"
    );
    assert!(ball_mass > 0.0, "la masse d'une boule m doit être > 0");
    assert!(
        central_load_mass >= 0.0,
        "la masse de la charge centrale M doit être ≥ 0"
    );
    (GRAVITY / (rotational_speed_rad * rotational_speed_rad)) * (ball_mass + central_load_mass)
        / ball_mass
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn watt_is_the_zero_load_porter() {
        // À M = 0 le Porter dégénère en Watt : mêmes hauteur et régime.
        let omega = 8.0_f64;
        let m = 0.5_f64;
        assert_relative_eq!(
            porter_governor_height(omega, m, 0.0),
            watt_governor_height(omega),
            epsilon = 1e-12
        );
        let h = watt_governor_height(omega);
        assert_relative_eq!(porter_governor_speed_rad(h, m, 0.0), omega, epsilon = 1e-12);
    }

    #[test]
    fn porter_speed_and_height_are_reciprocal() {
        // ω → h → ω doit boucler pour toute charge centrale.
        let m = 0.4_f64;
        let big_m = 3.0_f64;
        for &omega in &[3.0_f64, core::f64::consts::TAU, 12.0, 25.0]
        {
            let h = porter_governor_height(omega, m, big_m);
            assert_relative_eq!(
                porter_governor_speed_rad(h, m, big_m),
                omega,
                epsilon = 1e-9
            );
        }
    }

    #[test]
    fn centrifugal_force_balances_gravity_at_watt_height() {
        // À l'équilibre de Watt : m·r·ω² = m·g·(r/h) avec h = g/ω².
        // Donc F_c = m·g·r/h — identité indépendante de ω.
        let m = 0.6_f64;
        let r = 0.08_f64;
        let omega = 10.0_f64;
        let h = watt_governor_height(omega);
        assert_relative_eq!(
            flyball_centrifugal_force(m, r, omega),
            m * GRAVITY * r / h,
            epsilon = 1e-9
        );
    }

    #[test]
    fn centrifugal_force_scales_with_speed_squared() {
        // Doubler ω quadruple l'effort centrifuge (F_c ∝ ω²).
        let m = 0.5_f64;
        let r = 0.1_f64;
        let f1 = flyball_centrifugal_force(m, r, 7.0);
        let f2 = flyball_centrifugal_force(m, r, 14.0);
        assert_relative_eq!(f2, 4.0 * f1, epsilon = 1e-9);
    }

    #[test]
    fn watt_governor_realistic_case() {
        // Arbre à 60 tr/min → ω = 2π rad/s ; h = g/ω² ≈ 0,2485 m (≈ 248 mm).
        let omega = 2.0_f64 * core::f64::consts::PI;
        let h = watt_governor_height(omega);
        assert_relative_eq!(h, 9.81 / (omega * omega), epsilon = 1e-12);
        assert_relative_eq!(h, 0.248_479, epsilon = 1e-4);
    }

    #[test]
    fn central_load_raises_the_required_speed() {
        // Pour une hauteur donnée, la charge centrale augmente le régime
        // du facteur √[(m + M)/m].
        let h = 0.2_f64;
        let m = 0.5_f64;
        let big_m = 4.5_f64;
        let ratio = porter_governor_speed_rad(h, m, big_m) / porter_governor_speed_rad(h, m, 0.0);
        assert_relative_eq!(ratio, ((m + big_m) / m).sqrt(), epsilon = 1e-9);
    }

    #[test]
    #[should_panic(expected = "ω doit être > 0")]
    fn zero_speed_panics() {
        watt_governor_height(0.0);
    }
}
