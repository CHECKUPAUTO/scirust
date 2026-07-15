//! Ressaut hydraulique en **canal rectangulaire** : profondeur conjuguée aval,
//! perte d'énergie dissipée et nombre de Froude de l'écoulement.
//!
//! ```text
//! nombre de Froude        Fr = v / √(g·y)
//! profondeur conjuguée    y2 = (y1/2)·(√(1 + 8·Fr1²) − 1)
//! perte d'énergie (m)     ΔE = (y2 − y1)³ / (4·y1·y2)
//! ```
//!
//! `v` vitesse moyenne (m/s), `y`/`y1`/`y2` profondeur d'eau amont/aval (m),
//! `g` pesanteur (m/s²), `Fr`/`Fr1` nombre de Froude (sans dimension), `ΔE`
//! perte de charge exprimée en mètres de colonne d'eau. `y1` est la profondeur
//! rapide (torrentielle) amont, `y2` la profondeur lente (fluviale) aval.
//!
//! **Convention** : SI cohérent. **Limite honnête** : canal rectangulaire
//! **horizontal**, écoulement **permanent**, ressaut classique (`Fr1 > 1`),
//! pente du radier et frottement de paroi **négligés** ; la pesanteur `g` et les
//! grandeurs de l'écoulement sont **fournies par l'appelant** (aucune valeur de
//! `g`, de section ou de matériau « par défaut » n'est inventée ici).

/// Nombre de Froude de l'écoulement `Fr = v / √(g·y)` (sans dimension).
///
/// Panique si `g <= 0` ou `depth <= 0`.
pub fn hydraulic_jump_froude(velocity: f64, depth: f64, gravity: f64) -> f64 {
    assert!(gravity > 0.0, "la pesanteur doit être strictement positive");
    assert!(depth > 0.0, "la profondeur doit être strictement positive");
    velocity / (gravity * depth).sqrt()
}

/// Profondeur conjuguée aval d'un ressaut classique (équation de Bélanger)
/// `y2 = (y1/2)·(√(1 + 8·Fr1²) − 1)` (m).
///
/// Panique si `upstream_depth <= 0` ou `upstream_froude <= 1` (le ressaut
/// suppose un écoulement amont torrentiel).
pub fn hydraulic_jump_conjugate_depth(upstream_depth: f64, upstream_froude: f64) -> f64 {
    assert!(
        upstream_depth > 0.0,
        "la profondeur amont doit être strictement positive"
    );
    assert!(
        upstream_froude > 1.0,
        "le nombre de Froude amont doit être supérieur à 1 (écoulement torrentiel)"
    );
    (upstream_depth / 2.0) * ((1.0 + 8.0 * upstream_froude * upstream_froude).sqrt() - 1.0)
}

/// Perte d'énergie dissipée par le ressaut
/// `ΔE = (y2 − y1)³ / (4·y1·y2)` (m de colonne d'eau).
///
/// Panique si `upstream_depth <= 0`, `downstream_depth <= 0` ou si la profondeur
/// aval n'est pas supérieure à la profondeur amont (un ressaut dissipe toujours
/// de l'énergie, donc `y2 > y1`).
pub fn hydraulic_jump_energy_loss(upstream_depth: f64, downstream_depth: f64) -> f64 {
    assert!(
        upstream_depth > 0.0,
        "la profondeur amont doit être strictement positive"
    );
    assert!(
        downstream_depth > 0.0,
        "la profondeur aval doit être strictement positive"
    );
    assert!(
        downstream_depth > upstream_depth,
        "la profondeur aval doit être supérieure à la profondeur amont"
    );
    let drop = downstream_depth - upstream_depth;
    drop * drop * drop / (4.0 * upstream_depth * downstream_depth)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn froude_of_critical_flow_is_unity() {
        // À la profondeur critique, v = √(g·y) donc Fr = 1 exactement.
        let g = 9.81_f64;
        let y = 0.4_f64;
        let v = (g * y).sqrt();
        assert_relative_eq!(hydraulic_jump_froude(v, y, g), 1.0, epsilon = 1e-12);
    }

    #[test]
    fn conjugate_depth_numeric_case() {
        // y1 = 0,5 m, Fr1 = 3 → y2 = 0,25·(√73 − 1).
        // √73 = 8,544003745… → y2 = 0,25·7,544003745 = 1,886000936 m.
        let y2 = hydraulic_jump_conjugate_depth(0.5, 3.0);
        assert_relative_eq!(y2, 0.25 * (73.0_f64.sqrt() - 1.0), epsilon = 1e-12);
        assert_relative_eq!(y2, 1.886_000_936, epsilon = 1e-9);
    }

    #[test]
    fn conjugate_depth_satisfies_belanger_momentum_identity() {
        // La profondeur conjuguée de Bélanger vérifie l'identité de quantité de
        // mouvement r·(r + 1) = 2·Fr1² avec r = y2/y1 (déduite de
        // y2/y1 = ½·(√(1 + 8·Fr1²) − 1)). C'est la forme intrinsèque du ressaut :
        // en aval l'écoulement est fluvial (Fr2 < 1), la fonction refuse donc à
        // juste titre un Froude subcritique — on vérifie l'identité directement.
        for &(y1, fr1) in &[(0.5_f64, 3.0_f64), (0.3, 1.5), (1.2, 6.0)]
        {
            let y2 = hydraulic_jump_conjugate_depth(y1, fr1);
            let r = y2 / y1;
            assert_relative_eq!(r * (r + 1.0), 2.0 * fr1 * fr1, epsilon = 1e-9);
        }
    }

    #[test]
    fn energy_loss_equals_specific_energy_difference() {
        // Identité physique : ΔE = E1 − E2 avec l'énergie spécifique
        // E = y + v²/(2·g) et la continuité v·y = constante.
        let g = 9.81_f64;
        let (y1, fr1) = (0.5_f64, 3.0_f64);
        let y2 = hydraulic_jump_conjugate_depth(y1, fr1);
        let v1 = fr1 * (g * y1).sqrt();
        let v2 = v1 * y1 / y2;
        let e1 = y1 + v1 * v1 / (2.0 * g);
        let e2 = y2 + v2 * v2 / (2.0 * g);
        let loss = hydraulic_jump_energy_loss(y1, y2);
        assert_relative_eq!(loss, e1 - e2, epsilon = 1e-9);
    }

    #[test]
    fn energy_loss_scales_with_cube_of_length() {
        // À rapport y2/y1 fixé, un canal homothétique de facteur k multiplie
        // (y2−y1)³ par k³ et 4·y1·y2 par k², donc ΔE est multiplié par k.
        let base = hydraulic_jump_energy_loss(0.5, 1.5);
        let scaled = hydraulic_jump_energy_loss(1.0, 3.0);
        assert_relative_eq!(scaled, 2.0 * base, epsilon = 1e-12);
    }

    #[test]
    #[should_panic(expected = "torrentiel")]
    fn subcritical_conjugate_depth_panics() {
        // Fr1 <= 1 : pas de ressaut, l'entrée est invalide.
        hydraulic_jump_conjugate_depth(0.5, 0.8);
    }
}
