//! Mécanique des fluides — statique et dynamique incompressible : pression
//! hydrostatique, pression dynamique, charge de **Bernoulli**, continuité,
//! vitesse de **Torricelli** et nombre de **Reynolds**.
//!
//! ```text
//! pression hydrostatique  p = ρ·g·h
//! pression dynamique      q = ½·ρ·v²
//! charge totale (m)       H = p/(ρ·g) + v²/(2g) + z
//! continuité              A1·v1 = A2·v2
//! Torricelli              v = √(2·g·h)
//! Reynolds                Re = ρ·v·D/µ
//! ```
//!
//! `ρ` masse volumique (kg/m³), `g` pesanteur (m/s²), `h`/`z` hauteur (m), `v`
//! vitesse (m/s), `p` pression (Pa), `A` section (m²), `D` diamètre (m), `µ`
//! viscosité dynamique (Pa·s). La charge `H` est exprimée en mètres de colonne
//! de fluide.
//!
//! **Convention** : SI cohérent. **Limite honnête** : fluide **incompressible**,
//! écoulement **permanent** le long d'une ligne de courant, sans pertes (le terme
//! de perte de charge est traité dans [`crate::pipe_flow`]) ; `g`, `ρ`, `µ`
//! fournis par l'appelant.

/// Pression hydrostatique `p = ρ·g·h` (Pa).
pub fn hydrostatic_pressure(rho: f64, g: f64, depth: f64) -> f64 {
    rho * g * depth
}

/// Pression dynamique `q = ½·ρ·v²` (Pa).
pub fn dynamic_pressure(rho: f64, velocity: f64) -> f64 {
    0.5 * rho * velocity * velocity
}

/// Charge totale de Bernoulli `H = p/(ρ·g) + v²/(2g) + z` (m de colonne).
///
/// Panique si `ρ·g <= 0`.
pub fn total_head(pressure: f64, rho: f64, g: f64, velocity: f64, elevation: f64) -> f64 {
    assert!(rho * g > 0.0, "ρ·g doit être strictement positif");
    pressure / (rho * g) + velocity * velocity / (2.0 * g) + elevation
}

/// Vitesse en aval par conservation du débit `v2 = A1·v1/A2`.
///
/// Panique si `area2 <= 0`.
pub fn continuity_velocity(area1: f64, velocity1: f64, area2: f64) -> f64 {
    assert!(
        area2 > 0.0,
        "la section aval doit être strictement positive"
    );
    area1 * velocity1 / area2
}

/// Vitesse d'écoulement par un orifice sous charge `h` (Torricelli)
/// `v = √(2·g·h)` (m/s).
///
/// Panique si `g·h < 0`.
pub fn torricelli_velocity(g: f64, head: f64) -> f64 {
    assert!(g * head >= 0.0, "g·h doit être positif");
    (2.0 * g * head).sqrt()
}

/// Nombre de Reynolds `Re = ρ·v·D/µ` (sans dimension).
///
/// Panique si `µ <= 0`.
pub fn reynolds_number(rho: f64, velocity: f64, diameter: f64, mu: f64) -> f64 {
    assert!(
        mu > 0.0,
        "la viscosité dynamique doit être strictement positive"
    );
    rho * velocity * diameter / mu
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn hydrostatic_pressure_of_ten_meters_water() {
        // ρ=1000, g=9,81, h=10 → p = 98,1 kPa.
        assert_relative_eq!(
            hydrostatic_pressure(1000.0, 9.81, 10.0),
            98_100.0,
            epsilon = 1e-6
        );
    }

    #[test]
    fn dynamic_pressure_of_air() {
        // ρ=1,225, v=20 → q = 0,5·1,225·400 = 245 Pa.
        assert_relative_eq!(dynamic_pressure(1.225, 20.0), 245.0, epsilon = 1e-9);
    }

    #[test]
    fn continuity_speeds_up_in_a_contraction() {
        // Section divisée par 4 → vitesse ×4.
        assert_relative_eq!(continuity_velocity(4e-4, 2.0, 1e-4), 8.0, epsilon = 1e-9);
    }

    #[test]
    fn bernoulli_head_conserved_in_ideal_flow() {
        // Sans perte, la charge totale est identique en deux points d'une ligne
        // de courant : point 1 (p1, v1, z1) et point 2 (p2, v2, z2).
        // Prenons un tube horizontal : z égal, contraction v1→v2, p ajusté.
        let (rho, g) = (1000.0, 9.81);
        let (v1, v2, z) = (2.0, 8.0, 0.0);
        // p1 tel que H soit un repère ; on vérifie que p2 déduit conserve H.
        let p1 = 200_000.0;
        let h1 = total_head(p1, rho, g, v1, z);
        // p2 = p1 + ½ρ(v1²−v2²) (Bernoulli horizontal).
        let p2 = p1 + 0.5 * rho * (v1 * v1 - v2 * v2);
        let h2 = total_head(p2, rho, g, v2, z);
        assert_relative_eq!(h1, h2, epsilon = 1e-6);
    }

    #[test]
    fn torricelli_and_reynolds() {
        // v = √(2·9,81·5) ≈ 9,9 m/s.
        assert_relative_eq!(
            torricelli_velocity(9.81, 5.0),
            (2.0f64 * 9.81 * 5.0).sqrt(),
            epsilon = 1e-9
        );
        // Re = 1000·2·0,05/1e-3 = 100000 (turbulent).
        assert_relative_eq!(
            reynolds_number(1000.0, 2.0, 0.05, 1e-3),
            100_000.0,
            epsilon = 1e-6
        );
    }

    #[test]
    #[should_panic(expected = "viscosité")]
    fn zero_viscosity_panics() {
        reynolds_number(1000.0, 2.0, 0.05, 0.0);
    }
}
