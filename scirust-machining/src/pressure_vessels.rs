//! RDM — **réservoirs sous pression** : contraintes de membrane des enveloppes
//! minces (cylindre, sphère) et distribution de **Lamé** des cylindres épais.
//!
//! ```text
//! cylindre mince   σ_θ = p·r/t      σ_l = p·r/(2t)      (σ_θ = 2·σ_l)
//! sphère mince     σ = p·r/(2t)
//! cylindre épais (Lamé, pression interne p)
//!   σ_θ(alésage)  = p·(Ro² + Ri²)/(Ro² − Ri²)
//!   σ_θ(extérieur) = 2·p·Ri²/(Ro² − Ri²)
//! ```
//!
//! `p` pression interne (Pa), `r` rayon moyen (m), `t` épaisseur (m), `Ri`/`Ro`
//! rayons intérieur/extérieur (m), `σ_θ` contrainte circonférentielle (hoop),
//! `σ_l` contrainte longitudinale. La contrainte circonférentielle est
//! dimensionnante.
//!
//! **Convention** : SI cohérent. **Limite honnête** : théorie de **membrane**
//! pour les parois **minces** (`t/r ≲ 0,1`, contrainte uniforme dans l'épaisseur) ;
//! au-delà, utiliser les formules de **Lamé** (cylindre épais) fournies ici. Pas
//! d'effet de fond, de discontinuité, ni de pression externe.

/// Contrainte circonférentielle d'un **cylindre mince** `σ_θ = p·r/t` (Pa).
///
/// Panique si `thickness <= 0`.
pub fn thin_cylinder_hoop(pressure: f64, radius: f64, thickness: f64) -> f64 {
    assert!(
        thickness > 0.0,
        "l'épaisseur doit être strictement positive"
    );
    pressure * radius / thickness
}

/// Contrainte longitudinale d'un **cylindre mince** `σ_l = p·r/(2t)` (Pa).
///
/// Panique si `thickness <= 0`.
pub fn thin_cylinder_longitudinal(pressure: f64, radius: f64, thickness: f64) -> f64 {
    assert!(
        thickness > 0.0,
        "l'épaisseur doit être strictement positive"
    );
    pressure * radius / (2.0 * thickness)
}

/// Contrainte de membrane d'une **sphère mince** `σ = p·r/(2t)` (Pa).
///
/// Panique si `thickness <= 0`.
pub fn thin_sphere(pressure: f64, radius: f64, thickness: f64) -> f64 {
    assert!(
        thickness > 0.0,
        "l'épaisseur doit être strictement positive"
    );
    pressure * radius / (2.0 * thickness)
}

/// Contrainte circonférentielle à l'**alésage** d'un cylindre épais (Lamé,
/// pression interne) `σ_θ = p·(Ro² + Ri²)/(Ro² − Ri²)` (Pa).
///
/// Panique si `outer_radius <= inner_radius`.
pub fn thick_cylinder_hoop_inner(pressure: f64, inner_radius: f64, outer_radius: f64) -> f64 {
    assert!(outer_radius > inner_radius, "Ro doit dépasser Ri");
    pressure * (outer_radius * outer_radius + inner_radius * inner_radius)
        / (outer_radius * outer_radius - inner_radius * inner_radius)
}

/// Contrainte circonférentielle à la **surface extérieure** d'un cylindre épais
/// `σ_θ = 2·p·Ri²/(Ro² − Ri²)` (Pa).
///
/// Panique si `outer_radius <= inner_radius`.
pub fn thick_cylinder_hoop_outer(pressure: f64, inner_radius: f64, outer_radius: f64) -> f64 {
    assert!(outer_radius > inner_radius, "Ro doit dépasser Ri");
    2.0 * pressure * inner_radius * inner_radius
        / (outer_radius * outer_radius - inner_radius * inner_radius)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn hoop_is_twice_longitudinal() {
        // Règle « 2:1 » des cylindres minces.
        let p = 2e6;
        assert_relative_eq!(
            thin_cylinder_hoop(p, 0.5, 0.005),
            2.0 * thin_cylinder_longitudinal(p, 0.5, 0.005),
            epsilon = 1e-3
        );
    }

    #[test]
    fn sphere_is_half_cylinder_hoop() {
        // À r, t, p égaux : σ_sphère = ½·σ_θ,cylindre.
        let (p, r, t) = (2e6, 0.5, 0.005);
        assert_relative_eq!(
            thin_sphere(p, r, t),
            0.5 * thin_cylinder_hoop(p, r, t),
            epsilon = 1e-3
        );
    }

    #[test]
    fn thick_wall_hoop_exceeds_thin_estimate() {
        // À l'alésage, Lamé donne une contrainte supérieure à l'estimation mince.
        let (p, ri, ro) = (50e6, 0.050, 0.080);
        let lame = thick_cylinder_hoop_inner(p, ri, ro);
        // Do=2·Ri → facteur (Ro²+Ri²)/(Ro²−Ri²) = (6400+2500)/(6400−2500) mm² > 1.
        assert!(lame > p);
        assert_relative_eq!(
            lame,
            p * (ro * ro + ri * ri) / (ro * ro - ri * ri),
            epsilon = 1e-3
        );
    }

    #[test]
    fn thick_wall_hoop_decreases_outward() {
        // La contrainte circonférentielle est maximale à l'alésage.
        let (p, ri, ro) = (50e6, 0.050, 0.080);
        assert!(thick_cylinder_hoop_inner(p, ri, ro) > thick_cylinder_hoop_outer(p, ri, ro));
    }

    #[test]
    #[should_panic(expected = "Ro doit dépasser Ri")]
    fn inverted_radii_panic() {
        thick_cylinder_hoop_inner(50e6, 0.080, 0.050);
    }
}
