//! **Viscosité** des lubrifiants — conversion dynamique ↔ cinématique, unités
//! usuelles et dépendance en température (**Andrade**).
//!
//! ```text
//! dynamique         μ = ν·ρ
//! cinématique       ν = μ/ρ
//! unités            1 cP = 10⁻³ Pa·s     1 cSt = 10⁻⁶ m²/s
//! Andrade           μ(T) = A·exp(B/T)                (T absolue)
//! ```
//!
//! `μ` viscosité dynamique (Pa·s), `ν` viscosité cinématique (m²/s), `ρ` masse
//! volumique (kg/m³), `cP` centipoise, `cSt` centistokes, `A`/`B` constantes
//! d'Andrade du fluide, `T` température **absolue** (K).
//!
//! **Convention** : SI. **Limite honnête** : la loi d'**Andrade** (`μ = A·e^{B/T}`)
//! est un modèle à deux paramètres valable sur une plage modérée ; `A` et `B`
//! proviennent d'un ajustement fourni par l'appelant. La masse volumique varie
//! elle aussi (peu) avec la température, ce qui n'est pas pris en compte ici.

/// Viscosité dynamique `μ = ν·ρ`.
///
/// Panique si `kinematic < 0` ou `density <= 0`.
pub fn dynamic_from_kinematic(kinematic: f64, density: f64) -> f64 {
    assert!(kinematic >= 0.0 && density > 0.0, "ν ≥ 0 et ρ > 0 requis");
    kinematic * density
}

/// Viscosité cinématique `ν = μ/ρ`.
///
/// Panique si `dynamic < 0` ou `density <= 0`.
pub fn kinematic_from_dynamic(dynamic: f64, density: f64) -> f64 {
    assert!(dynamic >= 0.0 && density > 0.0, "μ ≥ 0 et ρ > 0 requis");
    dynamic / density
}

/// Conversion centipoise → Pa·s `μ = cP·10⁻³`.
pub fn pa_s_from_centipoise(centipoise: f64) -> f64 {
    centipoise * 1e-3
}

/// Viscosité dynamique par la loi d'**Andrade** `μ(T) = A·exp(B/T)`.
///
/// Panique si `temperature <= 0`.
pub fn andrade_viscosity(a: f64, b: f64, temperature: f64) -> f64 {
    assert!(
        temperature > 0.0,
        "la température absolue doit être strictement positive"
    );
    a * (b / temperature).exp()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn dynamic_kinematic_round_trip() {
        // μ = ν·ρ et ν = μ/ρ sont réciproques.
        let (nu, rho) = (46e-6, 870.0); // ISO VG 46 typique
        let mu = dynamic_from_kinematic(nu, rho);
        assert_relative_eq!(kinematic_from_dynamic(mu, rho), nu, epsilon = 1e-15);
    }

    #[test]
    fn water_at_20c_is_about_one_centipoise() {
        // Eau ≈ 1 cP = 10⁻³ Pa·s.
        assert_relative_eq!(pa_s_from_centipoise(1.0), 1e-3, epsilon = 1e-15);
    }

    #[test]
    fn iso_vg46_dynamic_viscosity() {
        // ν=46 cSt, ρ=870 → μ = 46e-6·870 ≈ 0,040 Pa·s.
        let mu = dynamic_from_kinematic(46e-6, 870.0);
        assert!(mu > 0.039 && mu < 0.041);
    }

    #[test]
    fn andrade_decreases_with_temperature() {
        // B>0 : la viscosité chute quand T augmente.
        let (a, b) = (1e-5, 1500.0);
        assert!(andrade_viscosity(a, b, 350.0) < andrade_viscosity(a, b, 300.0));
        // vérifie la formule.
        assert_relative_eq!(
            andrade_viscosity(a, b, 300.0),
            a * (b / 300.0).exp(),
            max_relative = 1e-12
        );
    }

    #[test]
    #[should_panic(expected = "température absolue")]
    fn zero_temperature_panics() {
        andrade_viscosity(1e-5, 1500.0, 0.0);
    }
}
