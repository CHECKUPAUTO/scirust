//! RDM — **flux de cisaillement** et **contrainte de cisaillement transverse**
//! dans une poutre fléchie (formule de Jourawski).
//!
//! ```text
//! contrainte transverse (Jourawski)   τ = V·Q / (I·t)
//! flux de cisaillement                q = V·Q / I
//! max au centre d'un rectangle        τmax = 1.5 · V / (b·h)
//! pas des connecteurs                 s = F / q
//! ```
//!
//! `V` effort tranchant (N), `Q` moment statique de l'aire au-dessus de la fibre
//! considérée (m³), `I` moment quadratique de la section entière (m⁴), `t`
//! largeur de la section à la fibre considérée (m), `b` largeur et `h` hauteur
//! du rectangle plein (m), `τ` contrainte de cisaillement (Pa = N/m²), `q` flux
//! de cisaillement (N/m), `F` capacité d'un connecteur (N), `s` pas entre
//! connecteurs (m).
//!
//! **Convention** : SI cohérent. **Limite honnête** : poutre **élastique
//! linéaire**, section **prismatique**, cisaillement **transverse** (formule
//! `V·Q/(I·t)`). `Q` (moment statique) et `I` (inertie de section) sont
//! **fournis ou calculés par l'appelant** ; aucune valeur de matériau, de
//! section ou de procédé n'est inventée ici. La répartition **parabolique** et
//! le facteur `1.5` supposent une **section rectangulaire pleine homogène**.

/// Contrainte de cisaillement transverse de Jourawski `τ = V·Q/(I·t)` (Pa).
///
/// Panique si `second_moment_i <= 0` ou si `thickness <= 0`.
pub fn shearflow_transverse_shear_stress(
    shear_force: f64,
    first_moment_q: f64,
    second_moment_i: f64,
    thickness: f64,
) -> f64 {
    assert!(
        second_moment_i > 0.0,
        "le moment quadratique I doit être strictement positif"
    );
    assert!(
        thickness > 0.0,
        "la largeur t à la fibre doit être strictement positive"
    );
    shear_force * first_moment_q / (second_moment_i * thickness)
}

/// Flux de cisaillement `q = V·Q/I` (N/m), utile pour dimensionner les
/// assemblages cloués, boulonnés ou soudés d'une section composée.
///
/// Panique si `second_moment_i <= 0`.
pub fn shearflow_shear_flow(shear_force: f64, first_moment_q: f64, second_moment_i: f64) -> f64 {
    assert!(
        second_moment_i > 0.0,
        "le moment quadratique I doit être strictement positif"
    );
    shear_force * first_moment_q / second_moment_i
}

/// Contrainte de cisaillement **maximale** au centre d'une section
/// **rectangulaire pleine** `τmax = 1.5·V/(b·h)` (Pa), soit 1,5 fois la
/// contrainte moyenne `V/(b·h)`.
///
/// Panique si `width <= 0` ou si `height <= 0`.
pub fn shearflow_max_rectangular(shear_force: f64, width: f64, height: f64) -> f64 {
    assert!(width > 0.0, "la largeur b doit être strictement positive");
    assert!(height > 0.0, "la hauteur h doit être strictement positive");
    1.5 * shear_force / (width * height)
}

/// Pas entre connecteurs `s = F/q` (m) : espacement admissible pour un flux de
/// cisaillement `q` donné et une capacité `F` par connecteur.
///
/// Panique si `shear_flow <= 0` ou si `fastener_capacity < 0`.
pub fn shearflow_fastener_spacing(fastener_capacity: f64, shear_flow: f64) -> f64 {
    assert!(
        fastener_capacity >= 0.0,
        "la capacité du connecteur doit être positive ou nulle"
    );
    assert!(
        shear_flow > 0.0,
        "le flux de cisaillement doit être strictement positif"
    );
    fastener_capacity / shear_flow
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn stress_is_shear_flow_divided_by_thickness() {
        // Identité : τ = q/t, car τ = V·Q/(I·t) et q = V·Q/I.
        let (v, q, i, t) = (12.0e3, 5.0e-5, 3.0e-6, 8.0e-3);
        let flow = shearflow_shear_flow(v, q, i);
        let stress = shearflow_transverse_shear_stress(v, q, i, t);
        assert_relative_eq!(stress, flow / t, epsilon = 1e-9);
    }

    #[test]
    fn rectangular_max_matches_jourawski_at_neutral_axis() {
        // Pour un rectangle b×h : I = b·h³/12, Q = b·h²/8, t = b au centre.
        // Alors V·Q/(I·t) = 1.5·V/(b·h) : les deux formules coïncident.
        let (v, b, h) = (10.0e3_f64, 0.05_f64, 0.10_f64);
        let i = b * h.powi(3) / 12.0;
        let q_na = b * h.powi(2) / 8.0;
        let jourawski = shearflow_transverse_shear_stress(v, q_na, i, b);
        let direct = shearflow_max_rectangular(v, b, h);
        assert_relative_eq!(jourawski, direct, epsilon = 1e-6);
    }

    #[test]
    fn realistic_rectangular_case() {
        // V = 10 kN, b = 50 mm, h = 100 mm → τmax = 1.5·10000/(0.05·0.10)
        //                                          = 15000/0.005 = 3.0 MPa.
        let tau = shearflow_max_rectangular(10.0e3, 0.05, 0.10);
        assert_relative_eq!(tau, 3.0e6, epsilon = 1.0);
    }

    #[test]
    fn shear_flow_is_linear_in_shear_force() {
        // Proportionnalité : doubler V double le flux (Q et I fixés).
        let (q, i) = (6.25e-5, 4.166_666_666_667e-6);
        let q1 = shearflow_shear_flow(5.0e3, q, i);
        let q2 = shearflow_shear_flow(10.0e3, q, i);
        assert_relative_eq!(q2, 2.0 * q1, epsilon = 1e-9);
    }

    #[test]
    fn spacing_and_flow_are_reciprocal() {
        // Réciprocité : q·s = F (le pas transporte exactement la capacité).
        let (capacity, flow) = (8.0e3, 2.5e5);
        let s = shearflow_fastener_spacing(capacity, flow);
        assert_relative_eq!(flow * s, capacity, epsilon = 1e-6);
    }

    #[test]
    #[should_panic(expected = "le moment quadratique I doit être strictement positif")]
    fn zero_inertia_panics() {
        shearflow_transverse_shear_stress(1.0e3, 1.0e-5, 0.0, 5.0e-3);
    }
}
