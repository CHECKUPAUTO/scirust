//! Ressort de compression **conique** (à raideur variable) — raideurs aux deux
//! extrémités du cône et hauteur bloquée d'un empilement de spires.
//!
//! Un ressort conique enroule le fil sur un rayon d'enroulement qui varie du
//! grand rayon `R_large` (base) au petit rayon `R_small` (sommet). Chaque spire
//! se comporte localement comme une spire hélicoïdale de rayon `R`, donc de
//! raideur d'autant plus grande que `R` est petit. En compression les grandes
//! spires (souples) s'assoient les premières, puis les petites : la raideur
//! globale **croît** au fur et à mesure de l'écrasement — la loi effort-flèche
//! est **non linéaire**. On borne ici ce comportement par les deux raideurs
//! extrêmes.
//!
//! ```text
//! raideur mini (grand rayon)  k_min = G·d⁴ / (64·R_large³·n)   (N/m)
//! raideur maxi (petit rayon)  k_max = G·d⁴ / (64·R_small³·n)   (N/m)
//! rapport de raideur          k_max/k_min = (R_large/R_small)³
//! hauteur bloquée             L_solid = d·n                    (m)
//! ```
//!
//! `d` diamètre de fil (m), `G` module de cisaillement du matériau (Pa),
//! `R_large`/`R_small` rayons d'enroulement extrêmes (m, `R_small ≤ R_large`),
//! `n` nombre de spires actives, `k` raideur (N/m), `L_solid` hauteur à spires
//! jointives (m). Le rayon d'enroulement `R` remplace `D/2` du fil rond ; avec
//! `D = 2R` on retrouve `k = G·d⁴/(8·D³·n)` de [`crate::springs`].
//!
//! **Convention** : SI cohérent (m, Pa, N/m). **Limite honnête** : les spires
//! s'assoient progressivement (raideur croissante) et la loi complète est **non
//! linéaire** ; ce module ne fournit que les raideurs aux deux extrêmes. Le
//! module de cisaillement `G`, les rayons d'enroulement et le nombre de spires
//! sont des données **fournies par l'appelant** — aucune valeur matériau ou
//! géométrique n'est supposée par défaut. La hauteur bloquée suppose un
//! **empilement** de spires (pas de télescopage des spires les unes dans les
//! autres).

/// Raideur **initiale** (mini) d'un ressort conique, côté **grand** rayon
/// `k_min = G·d⁴ / (64·R_large³·n)` (N/m).
///
/// Panique si un argument est négatif ou nul (`d, G, R_large, n > 0` requis).
pub fn conical_spring_min_rate(
    wire_diameter: f64,
    shear_modulus: f64,
    large_coil_radius: f64,
    active_coils: f64,
) -> f64 {
    assert!(
        wire_diameter > 0.0,
        "le diamètre de fil doit être strictement positif"
    );
    assert!(
        shear_modulus > 0.0,
        "le module de cisaillement doit être strictement positif"
    );
    assert!(
        large_coil_radius > 0.0,
        "le grand rayon d'enroulement doit être strictement positif"
    );
    assert!(
        active_coils > 0.0,
        "le nombre de spires actives doit être strictement positif"
    );
    shear_modulus * wire_diameter.powi(4) / (64.0 * large_coil_radius.powi(3) * active_coils)
}

/// Raideur **finale** (maxi) d'un ressort conique, côté **petit** rayon
/// `k_max = G·d⁴ / (64·R_small³·n)` (N/m).
///
/// Panique si un argument est négatif ou nul (`d, G, R_small, n > 0` requis).
pub fn conical_spring_max_rate(
    wire_diameter: f64,
    shear_modulus: f64,
    small_coil_radius: f64,
    active_coils: f64,
) -> f64 {
    assert!(
        wire_diameter > 0.0,
        "le diamètre de fil doit être strictement positif"
    );
    assert!(
        shear_modulus > 0.0,
        "le module de cisaillement doit être strictement positif"
    );
    assert!(
        small_coil_radius > 0.0,
        "le petit rayon d'enroulement doit être strictement positif"
    );
    assert!(
        active_coils > 0.0,
        "le nombre de spires actives doit être strictement positif"
    );
    shear_modulus * wire_diameter.powi(4) / (64.0 * small_coil_radius.powi(3) * active_coils)
}

/// Hauteur **bloquée** (spires jointives) d'un empilement `L_solid = d·n` (m).
///
/// Panique si `wire_diameter <= 0` ou `active_coils <= 0`.
pub fn conical_spring_solid_height(wire_diameter: f64, active_coils: f64) -> f64 {
    assert!(
        wire_diameter > 0.0,
        "le diamètre de fil doit être strictement positif"
    );
    assert!(
        active_coils > 0.0,
        "le nombre de spires actives doit être strictement positif"
    );
    wire_diameter * active_coils
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    // Jeu de paramètres réaliste : fil d = 3 mm, acier à ressort G = 81,5 GPa,
    // grand rayon 30 mm, petit rayon 15 mm, 6 spires actives.
    const D: f64 = 0.003;
    const G: f64 = 81.5e9;
    const R_LARGE: f64 = 0.030;
    const R_SMALL: f64 = 0.015;
    const N: f64 = 6.0;

    #[test]
    fn min_rate_realistic_value() {
        // k_min = 81,5e9·(0,003)⁴ / (64·(0,03)³·6)
        //       = 6,6015 / 0,010368 ≈ 636,71 N/m.
        let k = conical_spring_min_rate(D, G, R_LARGE, N);
        assert_relative_eq!(k, 636.71, max_relative = 1e-4);
    }

    #[test]
    fn rate_ratio_is_radius_cube() {
        // k_max / k_min = (R_large / R_small)³, ici (2)³ = 8.
        let k_min = conical_spring_min_rate(D, G, R_LARGE, N);
        let k_max = conical_spring_max_rate(D, G, R_SMALL, N);
        assert_relative_eq!(
            k_max / k_min,
            (R_LARGE / R_SMALL).powi(3),
            max_relative = 1e-12
        );
        assert_relative_eq!(k_max / k_min, 8.0, max_relative = 1e-12);
    }

    #[test]
    fn rate_scales_as_wire_diameter_pow4() {
        // Doubler le diamètre de fil multiplie la raideur par 2⁴ = 16.
        let k1 = conical_spring_min_rate(D, G, R_LARGE, N);
        let k2 = conical_spring_min_rate(2.0 * D, G, R_LARGE, N);
        assert_relative_eq!(k2 / k1, 16.0, max_relative = 1e-12);
    }

    #[test]
    fn rate_inverse_in_active_coils() {
        // La raideur varie en 1/n : tripler les spires divise k par 3.
        let k1 = conical_spring_min_rate(D, G, R_LARGE, N);
        let k3 = conical_spring_min_rate(D, G, R_LARGE, 3.0 * N);
        assert_relative_eq!(k1 / k3, 3.0, max_relative = 1e-12);
    }

    #[test]
    fn equal_radii_give_equal_rates() {
        // Cas limite : un rayon d'enroulement constant (cylindre) rend les deux
        // raideurs extrêmes identiques.
        let k_min = conical_spring_min_rate(D, G, R_LARGE, N);
        let k_max = conical_spring_max_rate(D, G, R_LARGE, N);
        assert_relative_eq!(k_min, k_max, max_relative = 1e-12);
    }

    #[test]
    fn solid_height_is_stack_of_coils() {
        // Empilement de 6 spires de fil de 3 mm → 18 mm.
        assert_relative_eq!(
            conical_spring_solid_height(D, N),
            0.018,
            max_relative = 1e-12
        );
    }

    #[test]
    #[should_panic(expected = "le grand rayon d'enroulement doit être strictement positif")]
    fn nonpositive_radius_panics() {
        conical_spring_min_rate(D, G, 0.0, N);
    }
}
