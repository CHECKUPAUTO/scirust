//! Centre de cisaillement des sections **ouvertes à parois minces** — cas du
//! profilé en **U** (channel) et rappel documentaire pour les sections à
//! double symétrie.
//!
//! ```text
//! moment quadratique (axe de symétrie)  Ix = tw·h³/12 + b·tf·h²/2
//! excentricité du centre de cisaillement e  = b²·h²·tf / (4·Ix)
//! forme réduite équivalente               e  = 3·b²·tf / (h·tw + 6·b·tf)
//! ```
//!
//! `b` largeur d'une semelle, `h` hauteur de l'âme, `tf` épaisseur de semelle,
//! `tw` épaisseur d'âme, `Ix` moment quadratique d'aire autour de l'axe de
//! symétrie (horizontal, dans le plan des semelles), `e` excentricité du centre
//! de cisaillement mesurée depuis le plan moyen de l'âme (du côté opposé aux
//! semelles). Toutes les longueurs dans une **unité SI cohérente** (m) ; `Ix`
//! s'exprime alors en m⁴ et `e` en m.
//!
//! **Limite honnête** : théorie des sections ouvertes à **parois minces**
//! (flux de cisaillement, épaisseurs faibles devant les autres dimensions),
//! profilé en U **symétrique par rapport à l'axe de l'âme**, matériau élastique
//! linéaire. Les dimensions de section sont **fournies par l'appelant** : aucune
//! valeur « par défaut » de géométrie, de matériau ou de procédé n'est inventée
//! ici.

/// Moment quadratique d'aire d'un profilé en U mince autour de l'axe de symétrie
/// `Ix = tw·h³/12 + b·tf·h²/2` (semelles idéalisées comme aires concentrées à
/// `±h/2`, propre inertie négligée).
///
/// Panique si l'une des dimensions `flange_width`, `web_height`,
/// `flange_thickness`, `web_thickness` est négative ou nulle.
pub fn shear_center_channel_axis_inertia(
    flange_width: f64,
    web_height: f64,
    flange_thickness: f64,
    web_thickness: f64,
) -> f64 {
    assert!(
        flange_width > 0.0 && web_height > 0.0 && flange_thickness > 0.0 && web_thickness > 0.0,
        "dimensions de section strictement positives requises (b, h, tf, tw > 0)"
    );
    web_thickness * web_height.powi(3) / 12.0
        + flange_width * flange_thickness * web_height.powi(2) / 2.0
}

/// Excentricité du centre de cisaillement d'un profilé en U mince
/// `e = b²·h²·tf / (4·Ix)`, avec `Ix` calculé en interne
/// ([`shear_center_channel_axis_inertia`]). `e` est mesurée depuis le plan moyen
/// de l'âme, du côté opposé à l'ouverture des semelles.
///
/// Panique si l'une des dimensions `flange_width`, `web_height`,
/// `flange_thickness`, `web_thickness` est négative ou nulle.
pub fn shear_center_channel_offset(
    flange_width: f64,
    web_height: f64,
    flange_thickness: f64,
    web_thickness: f64,
) -> f64 {
    let ix = shear_center_channel_axis_inertia(
        flange_width,
        web_height,
        flange_thickness,
        web_thickness,
    );
    flange_width.powi(2) * web_height.powi(2) * flange_thickness / (4.0 * ix)
}

/// Rappel documentaire : pour une section à **double symétrie**, le centre de
/// cisaillement coïncide avec le centroïde, donc l'excentricité est nulle
/// (`e = 0`). Renvoie toujours `true`.
pub fn shear_center_is_on_centroid_for_double_symmetry() -> bool {
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    // Cas de référence commun : b=50 mm, h=100 mm, tf=5 mm, tw=5 mm (en m).
    const B: f64 = 0.05;
    const H: f64 = 0.1;
    const TF: f64 = 0.005;
    const TW: f64 = 0.005;

    #[test]
    fn axis_inertia_matches_hand_calculation() {
        // Ix = tw·h³/12 + b·tf·h²/2
        //    = 0.005·1e-3/12 + 0.05·0.005·0.01/2
        //    = 4.166667e-7 + 1.25e-6 = 1.666667e-6 m⁴.
        let ix = shear_center_channel_axis_inertia(B, H, TF, TW);
        assert_relative_eq!(ix, 1.666_666_666_666_667e-6, epsilon = 1e-15);
    }

    #[test]
    fn channel_offset_realistic_value() {
        // e = b²·h²·tf/(4·Ix) = 1.25e-7 / 6.666667e-6 = 0.01875 m (18.75 mm).
        let e = shear_center_channel_offset(B, H, TF, TW);
        assert_relative_eq!(e, 0.01875, epsilon = 1e-12);
    }

    #[test]
    fn offset_equals_reduced_closed_form() {
        // Identité : e = b²h²tf/(4Ix) ≡ 3·b²·tf/(h·tw + 6·b·tf).
        let e = shear_center_channel_offset(B, H, TF, TW);
        let reduced = 3.0 * B.powi(2) * TF / (H * TW + 6.0 * B * TF);
        assert_relative_eq!(e, reduced, epsilon = 1e-14);
    }

    #[test]
    fn offset_scales_linearly_with_geometry() {
        // e a la dimension d'une longueur : homothétie de rapport k → e·k.
        let k = 2.0_f64;
        let e = shear_center_channel_offset(B, H, TF, TW);
        let e_scaled = shear_center_channel_offset(k * B, k * H, k * TF, k * TW);
        assert_relative_eq!(e_scaled, k * e, epsilon = 1e-14);
    }

    #[test]
    fn wider_flange_pushes_shear_center_further_out() {
        // À âme fixée, une semelle plus large éloigne le centre de cisaillement.
        let narrow = shear_center_channel_offset(B, H, TF, TW);
        let wide = shear_center_channel_offset(1.5 * B, H, TF, TW);
        assert!(wide > narrow);
    }

    #[test]
    fn double_symmetry_offset_is_zero() {
        // Section à double symétrie : centre de cisaillement sur le centroïde.
        assert!(shear_center_is_on_centroid_for_double_symmetry());
    }

    #[test]
    #[should_panic(expected = "dimensions de section strictement positives")]
    fn zero_web_height_panics() {
        shear_center_channel_offset(B, 0.0, TF, TW);
    }
}
