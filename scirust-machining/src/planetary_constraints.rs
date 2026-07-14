//! Trains épicycloïdaux — **contraintes géométriques** de conception (coaxialité,
//! calcul des dentures, condition d'assemblage) ; complément cinématique de
//! [`epicyclic`](crate::epicyclic) qui traite les vitesses.
//!
//! ```text
//! coaxialité (module commun)     N_R = N_S + 2·N_P
//! denture satellite déduite      N_P = (N_R − N_S)/2
//! condition d'assemblage         (N_S + N_R) mod n_p = 0   (n_p satellites)
//! entraxe soleil-satellite       a = m·(N_S + N_P)/2
//! ```
//!
//! `N_S` dents du soleil, `N_P` dents d'un satellite, `N_R` dents de la couronne
//! (nombres entiers, sans unité) ; `n_p` nombre de satellites ; `m` module (mm) ;
//! `a` entraxe (mm). La coaxialité impose que soleil, satellites et couronne
//! partagent le même axe : le rayon primitif de la couronne vaut la somme du
//! rayon soleil et du diamètre satellite, d'où `N_R = N_S + 2·N_P`.
//!
//! **Limite honnête** : géométrie **exacte** d'un train planétaire simple à un
//! étage, satellites **régulièrement espacés** et **module commun** à toutes les
//! dentures. Ne traite ni les satellites inégalement répartis (assemblage par
//! séquençage angulaire), ni les corrections de denture, ni les interférences de
//! pied/tête, ni les trains composés. Le module `m` et les nombres de dents sont
//! **fournis par l'appelant** : aucune valeur « par défaut » n'est inventée.

/// Nombre de dents de la couronne imposé par la **coaxialité** d'un train
/// planétaire à module commun : `N_R = N_S + 2·N_P`.
///
/// Panique si `sun_teeth == 0` ou `planet_teeth == 0`.
pub fn ring_teeth_from_sun_planet(sun_teeth: u32, planet_teeth: u32) -> u32 {
    assert!(sun_teeth > 0, "le soleil doit avoir au moins une dent");
    assert!(
        planet_teeth > 0,
        "un satellite doit avoir au moins une dent"
    );
    sun_teeth + 2 * planet_teeth
}

/// Nombre de dents d'un satellite déduit du soleil et de la couronne par
/// coaxialité : `N_P = (N_R − N_S)/2`.
///
/// Panique si `ring_teeth <= sun_teeth` (couronne pas assez grande) ou si
/// `(ring_teeth − sun_teeth)` est impair (coaxialité géométriquement impossible
/// à module commun).
pub fn planet_teeth_from_sun_ring(sun_teeth: u32, ring_teeth: u32) -> u32 {
    assert!(sun_teeth > 0, "le soleil doit avoir au moins une dent");
    assert!(
        ring_teeth > sun_teeth,
        "la couronne doit avoir plus de dents que le soleil"
    );
    let diff = ring_teeth - sun_teeth;
    assert!(
        diff.is_multiple_of(2),
        "coaxialité impossible : (N_R − N_S) doit être pair"
    );
    diff / 2
}

/// Condition d'assemblage de `n_planets` satellites **régulièrement espacés** :
/// `(N_S + N_R) mod n_p == 0`. Renvoie `true` si le train est assemblable.
///
/// Panique si `n_planets == 0` ou `sun_teeth == 0`.
pub fn assembly_condition(sun_teeth: u32, ring_teeth: u32, n_planets: u32) -> bool {
    assert!(sun_teeth > 0, "le soleil doit avoir au moins une dent");
    assert!(n_planets > 0, "il faut au moins un satellite");
    (sun_teeth + ring_teeth).is_multiple_of(n_planets)
}

/// Entraxe soleil-satellite (mm) à module commun : `a = m·(N_S + N_P)/2`.
///
/// C'est aussi le rayon de la trajectoire circulaire des axes de satellites
/// (rayon du porte-satellites). Panique si `module_mm <= 0` ou si une denture
/// est nulle.
pub fn sun_planet_center_distance(module_mm: f64, sun_teeth: u32, planet_teeth: u32) -> f64 {
    assert!(module_mm > 0.0, "le module doit être strictement positif");
    assert!(sun_teeth > 0, "le soleil doit avoir au moins une dent");
    assert!(
        planet_teeth > 0,
        "un satellite doit avoir au moins une dent"
    );
    module_mm * (sun_teeth + planet_teeth) as f64 / 2.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn coaxial_round_trip() {
        // Réciprocité : déduire la couronne puis retrouver le satellite.
        let (sun, planet) = (20u32, 15u32);
        let ring = ring_teeth_from_sun_planet(sun, planet);
        assert_eq!(ring, 50);
        assert_eq!(planet_teeth_from_sun_ring(sun, ring), planet);
    }

    #[test]
    fn coaxiality_forces_even_difference() {
        // Pour tout N_S, N_P : (N_R − N_S) = 2·N_P est pair par construction.
        for sun in 1..40u32
        {
            for planet in 1..40u32
            {
                let ring = ring_teeth_from_sun_planet(sun, planet);
                assert_eq!((ring - sun) % 2, 0);
            }
        }
    }

    #[test]
    fn assembly_condition_realistic_case() {
        // Cas chiffré : N_S=24, N_R=72 → somme 96, divisible par 3 et 4 satellites,
        // pas par 5. (72 = 24 + 2·24 → satellites de 24 dents.)
        assert_eq!(planet_teeth_from_sun_ring(24, 72), 24);
        assert!(assembly_condition(24, 72, 3));
        assert!(assembly_condition(24, 72, 4));
        assert!(!assembly_condition(24, 72, 5));
    }

    #[test]
    fn center_distance_scales_with_module() {
        // Proportionnalité stricte à l'entraxe vs le module.
        let a1 = sun_planet_center_distance(1.0, 20, 15);
        let a2 = sun_planet_center_distance(2.0, 20, 15);
        assert_relative_eq!(a2, 2.0 * a1, max_relative = 1e-12);
        // Valeur : m=2, (20+15)/2 = 17.5 → 35 mm.
        assert_relative_eq!(a2, 35.0, max_relative = 1e-12);
    }

    #[test]
    fn center_distance_equals_ring_minus_sun_radii() {
        // Identité géométrique : a = m·(N_S+N_P)/2 et le rayon primitif couronne
        // r_R = m·N_R/2 = m·(N_S+2·N_P)/2 doit valoir r_S + 2·r_P soleil+diam sat.
        let m = 3.0_f64;
        let (sun, planet) = (18u32, 21u32);
        let ring = ring_teeth_from_sun_planet(sun, planet);
        let r_sun = m * sun as f64 / 2.0;
        let r_planet = m * planet as f64 / 2.0;
        let r_ring = m * ring as f64 / 2.0;
        assert_relative_eq!(r_ring, r_sun + 2.0 * r_planet, max_relative = 1e-12);
        // Et l'entraxe = r_sun + r_planet.
        let a = sun_planet_center_distance(m, sun, planet);
        assert_relative_eq!(a, r_sun + r_planet, max_relative = 1e-12);
    }

    #[test]
    #[should_panic(expected = "(N_R − N_S) doit être pair")]
    fn planet_teeth_rejects_odd_difference() {
        // N_R − N_S = 51 − 20 = 31 impair → coaxialité impossible.
        let _ = planet_teeth_from_sun_ring(20, 51);
    }
}
