//! Séparation centrifuge : force centrifuge relative (RCF, multiple de `g`) et
//! vitesse de sédimentation radiale d'une particule sphérique en régime de Stokes.
//!
//! ```text
//! RCF (depuis ω)   = r·ω² / g                         (sans dimension, multiple de g)
//! RCF (depuis rpm) = r·(2·π·rpm/60)² / g              (idem)
//! vitesse radiale  v = (ρp - ρf)·d²·r·ω² / (18·μ)     (m/s, Stokes en champ centrifuge)
//! ```
//!
//! `r` rayon considéré (m), `ω` vitesse angulaire (rad/s), `rpm` vitesse de
//! rotation (tours/min), `g` pesanteur standard (m/s²), `d` diamètre de la
//! particule (m), `ρp` masse volumique de la particule (kg/m³), `ρf` masse
//! volumique du fluide (kg/m³), `μ` viscosité dynamique (Pa·s), `v` vitesse
//! radiale de migration (m/s). L'accélération centrifuge vaut `a = r·ω²`, donc
//! `RCF = a/g` et la vitesse radiale est celle de Stokes sous pesanteur avec `g`
//! remplacé par `r·ω²` : `v = v∞(g→r·ω²)`.
//!
//! **Convention** : SI cohérent (m, rad/s, kg/m³, Pa·s, m/s). Le facteur RCF est
//! exprimé comme un multiple de la pesanteur standard
//! `CENTRIFUGAL_STANDARD_GRAVITY = 9.80665 m/s²`.
//!
//! **Limite honnête** : champ centrifuge supposé **uniforme au rayon considéré**
//! (on néglige la variation de `r·ω²` sur l'épaisseur de la particule et le long
//! de la colonne de fluide), particule **sphérique, rigide, isolée** en **régime
//! de Stokes** (`Re < ~1`, écoulement rampant), fluide newtonien. Les masses
//! volumiques, la viscosité et la géométrie sont des **données** fournies par
//! l'appelant (tables fluide/matériau, conditions du procédé) — aucune valeur
//! « par défaut » n'est inventée ici, hormis la pesanteur standard normalisée.

use core::f64::consts::PI;

/// Pesanteur standard normalisée `g₀ = 9.80665 m/s²` (valeur conventionnelle SI).
///
/// Sert de référence pour exprimer la force centrifuge relative (RCF) comme un
/// multiple de `g`.
pub const CENTRIFUGAL_STANDARD_GRAVITY: f64 = 9.80665;

/// Force centrifuge relative `RCF = r·ω²/g₀` (sans dimension, multiple de `g`).
///
/// Rapport de l'accélération centrifuge `r·ω²` à la pesanteur standard.
///
/// Panique si `radius_m < 0`.
pub fn centrifugal_rcf(radius_m: f64, angular_speed_rad: f64) -> f64 {
    assert!(radius_m >= 0.0, "le rayon ne peut pas être négatif");
    radius_m * angular_speed_rad * angular_speed_rad / CENTRIFUGAL_STANDARD_GRAVITY
}

/// Force centrifuge relative depuis la vitesse de rotation
/// `RCF = r·(2·π·rpm/60)²/g₀` (sans dimension, multiple de `g`).
///
/// Convertit les tours par minute en vitesse angulaire `ω = 2·π·rpm/60` puis
/// applique [`centrifugal_rcf`].
///
/// Panique si `radius_m < 0`.
pub fn centrifugal_rcf_from_rpm(radius_m: f64, rpm: f64) -> f64 {
    assert!(radius_m >= 0.0, "le rayon ne peut pas être négatif");
    let angular_speed_rad = 2.0 * PI * rpm / 60.0;
    centrifugal_rcf(radius_m, angular_speed_rad)
}

/// Vitesse de sédimentation radiale de Stokes en champ centrifuge
/// `v = (ρp - ρf)·d²·r·ω²/(18·μ)` (m/s).
///
/// Positive (migration vers l'extérieur) si la particule est plus dense que le
/// fluide, négative (flottation vers l'axe) sinon. C'est la vitesse limite de
/// Stokes sous pesanteur où `g` est remplacé par l'accélération centrifuge
/// `r·ω²`.
///
/// Panique si `particle_diameter <= 0`, `viscosity <= 0` ou `radius < 0`.
pub fn centrifugal_sedimentation_velocity(
    particle_diameter: f64,
    particle_density: f64,
    fluid_density: f64,
    viscosity: f64,
    radius: f64,
    angular_speed: f64,
) -> f64 {
    assert!(
        particle_diameter > 0.0,
        "le diamètre de la particule doit être strictement positif"
    );
    assert!(
        viscosity > 0.0,
        "la viscosité dynamique doit être strictement positive"
    );
    assert!(radius >= 0.0, "le rayon ne peut pas être négatif");
    (particle_density - fluid_density)
        * particle_diameter
        * particle_diameter
        * radius
        * angular_speed
        * angular_speed
        / (18.0 * viscosity)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn rcf_matches_explicit_formula() {
        // Rotor r = 0,1 m tournant à ω = 1047,197551 rad/s (≈ 10000 tr/min).
        let r = 0.1_f64;
        let omega = 1047.197551_f64;
        let rcf = centrifugal_rcf(r, omega);
        let expected = r * omega * omega / 9.80665;
        assert_relative_eq!(rcf, expected, epsilon = 1e-12);
        // Valeur chiffrée attendue ≈ 11182,4 g.
        assert_relative_eq!(rcf, 11_182.4, max_relative = 1e-4);
    }

    #[test]
    fn rcf_from_rpm_matches_rcf_from_omega() {
        // Réciprocité de conversion : ω = 2·π·rpm/60 doit redonner le même RCF.
        let r = 0.1_f64;
        let rpm = 10_000.0_f64;
        let omega = 2.0 * PI * rpm / 60.0;
        assert_relative_eq!(
            centrifugal_rcf_from_rpm(r, rpm),
            centrifugal_rcf(r, omega),
            epsilon = 1e-9
        );
    }

    #[test]
    fn rcf_scales_with_angular_speed_squared() {
        // RCF ∝ ω² : doubler la vitesse angulaire quadruple le facteur.
        let base = centrifugal_rcf(0.1, 500.0);
        let fast = centrifugal_rcf(0.1, 1000.0);
        assert_relative_eq!(fast / base, 4.0, epsilon = 1e-9);
    }

    #[test]
    fn sedimentation_velocity_is_stokes_scaled_by_centrifugal_acceleration() {
        // v = (ρp-ρf)·d²·(r·ω²)/(18·μ) : identique à Stokes sous g remplacé par r·ω².
        let d = 1.0e-6_f64;
        let (rho_p, rho_f, mu, r, omega) = (1050.0, 1000.0, 1.0e-3, 0.1, 1047.197551);
        let v = centrifugal_sedimentation_velocity(d, rho_p, rho_f, mu, r, omega);
        let centrifugal_accel = r * omega * omega;
        let stokes_like = (rho_p - rho_f) * d * d * centrifugal_accel / (18.0 * mu);
        assert_relative_eq!(v, stokes_like, epsilon = 1e-15);
        // Valeur chiffrée attendue ≈ 3,0462e-4 m/s.
        assert_relative_eq!(v, 3.046_2e-4, max_relative = 1e-3);
    }

    #[test]
    fn sedimentation_velocity_equals_gravity_stokes_times_rcf() {
        // Identité physique : v_centrifuge = v_stokes(g) · RCF, car r·ω² = RCF·g.
        let d = 1.0e-6_f64;
        let (rho_p, rho_f, mu, r, omega) = (1050.0, 1000.0, 1.0e-3, 0.1, 1047.197551);
        let v = centrifugal_sedimentation_velocity(d, rho_p, rho_f, mu, r, omega);
        let v_gravity = (rho_p - rho_f) * d * d * CENTRIFUGAL_STANDARD_GRAVITY / (18.0 * mu);
        let rcf = centrifugal_rcf(r, omega);
        assert_relative_eq!(v, v_gravity * rcf, max_relative = 1e-12);
    }

    #[test]
    fn lighter_particle_migrates_inward() {
        // ρp < ρf → vitesse radiale négative (flottation vers l'axe).
        let v = centrifugal_sedimentation_velocity(1.0e-6, 900.0, 1000.0, 1.0e-3, 0.1, 1000.0);
        assert!(v < 0.0, "une particule plus légère doit migrer vers l'axe");
    }

    #[test]
    #[should_panic(expected = "diamètre")]
    fn zero_diameter_panics() {
        centrifugal_sedimentation_velocity(0.0, 1050.0, 1000.0, 1.0e-3, 0.1, 1000.0);
    }
}
