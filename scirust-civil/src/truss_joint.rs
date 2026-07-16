//! **Treillis plan — méthode des nœuds** (statique des structures
//! isostatiques) : résolution de l'équilibre d'un **nœud articulé** relié à
//! **deux barres**, contrainte axiale, allongement d'une barre et charge
//! critique de flambement d'Euler d'une barre comprimée bi-articulée.
//!
//! ```text
//! équilibre du nœud (2 barres)   F1·cos(θ1) + F2·cos(θ2) + Px = 0
//!                                F1·sin(θ1) + F2·sin(θ2) + Py = 0
//! déterminant du système         D = cos(θ1)·sin(θ2) − cos(θ2)·sin(θ1)
//!                                  = sin(θ2 − θ1)
//! efforts (règle de Cramer)      F1 = (−Px·sin(θ2) + Py·cos(θ2)) / D
//!                                F2 = ( Px·sin(θ1) − Py·cos(θ1)) / D
//! contrainte axiale              σ  = N / A
//! allongement d'une barre        ΔL = N·L / (E·A)
//! charge critique d'Euler        Ncr = π²·E·I / L²
//! ```
//!
//! `Px` = `applied_load_x` et `Py` = `applied_load_y` composantes de la charge
//! **appliquée au nœud** (N), `θ1` = `angle_1` et `θ2` = `angle_2` angles des
//! deux barres mesurés depuis l'axe `x` (**radians**), `F1`/`F2` = efforts
//! **axiaux** des barres (N, **traction > 0**, compression < 0), `N` =
//! `member_force` effort axial d'une barre (N), `A` = `cross_section_area`
//! aire de la section (m²), `σ` = contrainte axiale (Pa), `L` = `length`
//! longueur de la barre (m), `E` = `elastic_modulus` module de Young (Pa),
//! `ΔL` = allongement (m, > 0 en traction), `I` = `inertia` moment
//! quadratique de la section (m⁴) et `Ncr` = charge critique de flambement
//! (N).
//!
//! **Convention** : unités **SI** cohérentes **N, m, Pa** (avec
//! `1 Pa = 1 N/m²`, donc `N/A` est en Pa, `N·L/(E·A)` en m et `π²·E·I/L²` en
//! N) ; angles en **radians**. Types `f64`.
//!
//! **Limite honnête** : ce module traite un **treillis plan à nœuds
//! articulés**, chargé **aux nœuds**, dont les barres ne reprennent qu'un
//! **effort axial** (ni flexion, ni cisaillement). Il résout l'équilibre d'un
//! **seul nœud isostatique à deux barres** (déterminant `D = sin(θ2 − θ1)`
//! **non nul** requis : les deux barres ne doivent pas être colinéaires) ; ce
//! n'est **pas** une résolution globale de la structure (matrice de rigidité,
//! méthode des sections, nœuds à plus de deux barres). Le **signe** des
//! efforts indique traction (+) ou compression (−). Les grandeurs `E`, `A`,
//! `I` et les charges appliquées sont **fournies par l'appelant** ; aucune
//! propriété matérielle ni géométrique n'est inventée. La charge d'Euler
//! suppose un comportement **élastique linéaire** et des **appuis
//! bi-articulés** (longueur de flambement = `L`) ; l'appelant reste
//! responsable de la comparaison à la résistance de la barre et de la prise en
//! compte des imperfections (courbes de flambement de l'Eurocode 3).

/// Résout l'équilibre d'un **nœud articulé à deux barres** et renvoie le
/// couple d'efforts axiaux `(F1, F2)` (N) tels que
/// `F1·cos(θ1) + F2·cos(θ2) + Px = 0` et
/// `F1·sin(θ1) + F2·sin(θ2) + Py = 0`, avec `applied_load_x` = `Px` et
/// `applied_load_y` = `Py` les composantes de la charge appliquée au nœud (N)
/// et `angle_1` = `θ1`, `angle_2` = `θ2` les angles des barres depuis l'axe
/// `x` (radians). Résolution 2×2 par la règle de Cramer :
/// `D = sin(θ2 − θ1)`, `F1 = (−Px·sin(θ2) + Py·cos(θ2)) / D` et
/// `F2 = (Px·sin(θ1) − Py·cos(θ1)) / D`. Un effort **positif** est une
/// **traction**, un effort **négatif** une **compression**.
///
/// Panique si le déterminant `D = sin(θ2 − θ1)` est nul ou quasi nul
/// (`|D| <= 1e-12`), c'est-à-dire si les deux barres sont **colinéaires** :
/// le système est alors singulier (mécanisme, pas d'équilibre déterminé).
pub fn truss_member_force_two_bars(
    applied_load_x: f64,
    applied_load_y: f64,
    angle_1: f64,
    angle_2: f64,
) -> (f64, f64) {
    let (s1, c1) = angle_1.sin_cos();
    let (s2, c2) = angle_2.sin_cos();
    let det = c1 * s2 - c2 * s1;
    assert!(
        det.abs() > 1e-12,
        "les deux barres sont colinéaires (déterminant sin(θ2 − θ1) ≈ 0) : \
         système singulier, équilibre indéterminé"
    );
    let force_1 = (-applied_load_x * s2 + applied_load_y * c2) / det;
    let force_2 = (applied_load_x * s1 - applied_load_y * c1) / det;
    (force_1, force_2)
}

/// Contrainte axiale `σ = N / A` (Pa) d'une barre, avec `member_force` = `N`
/// l'effort axial (N, traction > 0) et `cross_section_area` = `A` l'aire de la
/// section (m²). Le signe de `σ` suit celui de `N` : positif en traction,
/// négatif en compression.
///
/// Panique si `cross_section_area <= 0` (aire nulle ou négative : division par
/// zéro, géométrie non physique).
pub fn truss_axial_stress(member_force: f64, cross_section_area: f64) -> f64 {
    assert!(
        cross_section_area > 0.0,
        "l'aire de la section A doit être strictement positive (m²)"
    );
    member_force / cross_section_area
}

/// Allongement `ΔL = N·L / (E·A)` (m) d'une barre en comportement élastique
/// linéaire, avec `member_force` = `N` l'effort axial (N, traction > 0),
/// `length` = `L` la longueur (m), `elastic_modulus` = `E` le module de Young
/// (Pa) et `cross_section_area` = `A` l'aire de la section (m²). `ΔL` est
/// positif en traction (barre qui s'allonge) et négatif en compression.
///
/// Panique si `length <= 0`, si `elastic_modulus <= 0` ou si
/// `cross_section_area <= 0` (division par zéro ou grandeurs non physiques).
pub fn truss_elongation(
    member_force: f64,
    length: f64,
    elastic_modulus: f64,
    cross_section_area: f64,
) -> f64 {
    assert!(
        length > 0.0,
        "la longueur L doit être strictement positive (m)"
    );
    assert!(
        elastic_modulus > 0.0,
        "le module de Young E doit être strictement positif (Pa)"
    );
    assert!(
        cross_section_area > 0.0,
        "l'aire de la section A doit être strictement positive (m²)"
    );
    member_force * length / (elastic_modulus * cross_section_area)
}

/// Charge critique de flambement d'Euler `Ncr = π²·E·I / L²` (N) d'une barre
/// comprimée **bi-articulée**, avec `elastic_modulus` = `E` le module de Young
/// (Pa), `inertia` = `I` le moment quadratique de la section (m⁴) et `length`
/// = `L` la longueur de flambement (m). `Ncr` est la charge de compression
/// axiale (valeur positive) au-delà de laquelle la barre flambe.
///
/// Panique si `elastic_modulus <= 0`, si `inertia <= 0` ou si `length <= 0`
/// (grandeurs non physiques ou division par zéro).
pub fn truss_euler_buckling_load(elastic_modulus: f64, inertia: f64, length: f64) -> f64 {
    assert!(
        elastic_modulus > 0.0,
        "le module de Young E doit être strictement positif (Pa)"
    );
    assert!(
        inertia > 0.0,
        "le moment quadratique I doit être strictement positif (m⁴)"
    );
    assert!(
        length > 0.0,
        "la longueur L doit être strictement positive (m)"
    );
    core::f64::consts::PI * core::f64::consts::PI * elastic_modulus * inertia / (length * length)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use core::f64::consts::{FRAC_PI_2, FRAC_PI_4, PI};

    #[test]
    fn two_bars_satisfy_equilibrium_by_construction() {
        // Réciprocité : les efforts renvoyés doivent annuler les deux équations
        // d'équilibre, quelles que soient la charge et les angles (non colinéaires).
        let px = 1500.0_f64;
        let py = -800.0_f64;
        let a1 = FRAC_PI_4; // 45°
        let a2 = 2.0 * FRAC_PI_4 + FRAC_PI_4; // 135°
        let (f1, f2) = truss_member_force_two_bars(px, py, a1, a2);
        assert_relative_eq!(f1 * a1.cos() + f2 * a2.cos() + px, 0.0, epsilon = 1e-6);
        assert_relative_eq!(f1 * a1.sin() + f2 * a2.sin() + py, 0.0, epsilon = 1e-6);
    }

    #[test]
    fn two_bars_orthogonal_worked_case() {
        // Barre 1 horizontale (θ1 = 0), barre 2 verticale (θ2 = π/2).
        // D = cos0·sin(π/2) − cos(π/2)·sin0 = 1·1 − 0·0 = 1.
        // F1 = (−Px·sin(π/2) + Py·cos(π/2))/1 = −Px = −2000.
        // F2 = ( Px·sin0    − Py·cos0    )/1 = −Py = −(−500) = 500.
        let px = 2000.0_f64;
        let py = -500.0_f64;
        let (f1, f2) = truss_member_force_two_bars(px, py, 0.0, FRAC_PI_2);
        assert_relative_eq!(f1, -2000.0, epsilon = 1e-6);
        assert_relative_eq!(f2, 500.0, epsilon = 1e-6);
    }

    #[test]
    fn stress_and_elongation_are_proportional_and_signed() {
        // Cas chiffré : N = 100 000 N, A = 2,0e-3 m², L = 3,0 m, E = 210e9 Pa.
        // σ  = 100000 / 2,0e-3 = 5,0e7 Pa (= 50 MPa), traction (> 0).
        // ΔL = 100000·3,0 / (210e9·2,0e-3) = 300000 / 4,2e8
        //    = 7,142857…e-4 m ≈ 0,714 mm.
        let n = 100_000.0_f64;
        let area = 2.0e-3_f64;
        let length = 3.0_f64;
        let e = 210.0e9_f64;
        assert_relative_eq!(truss_axial_stress(n, area), 5.0e7, epsilon = 1.0);
        assert_relative_eq!(
            truss_elongation(n, length, e, area),
            7.142_857_142_857e-4,
            epsilon = 1e-9
        );
        // Compression : effort négatif => contrainte et allongement négatifs,
        // de même amplitude (proportionnalité linéaire à N).
        assert_relative_eq!(truss_axial_stress(-n, area), -5.0e7, epsilon = 1.0);
        assert_relative_eq!(
            truss_elongation(-n, length, e, area),
            -7.142_857_142_857e-4,
            epsilon = 1e-9
        );
    }

    #[test]
    fn euler_load_scales_inverse_square_with_length() {
        // Ncr ∝ 1/L² : doubler la longueur divise la charge critique par 4.
        let e = 210.0e9_f64;
        let i = 8.0e-6_f64;
        let n1 = truss_euler_buckling_load(e, i, 2.0);
        let n2 = truss_euler_buckling_load(e, i, 4.0);
        assert_relative_eq!(n1 / n2, 4.0, epsilon = 1e-12);
    }

    #[test]
    fn euler_load_worked_case() {
        // E = 210e9 Pa, I = 8,0e-6 m⁴, L = 4,0 m.
        // Ncr = π²·210e9·8,0e-6 / 4,0² = π²·1,68e6 / 16 = π²·105000.
        //     = 9,8696044011…·105000 = 1 036 308,462… N ≈ 1036 kN.
        let ncr = truss_euler_buckling_load(210.0e9, 8.0e-6, 4.0);
        let expected = PI * PI * 105_000.0_f64;
        assert_relative_eq!(ncr, expected, epsilon = 1e-3);
        assert_relative_eq!(ncr, 1_036_308.462_f64, epsilon = 1.0);
    }

    #[test]
    #[should_panic(expected = "colinéaires")]
    fn two_bars_reject_collinear_members() {
        // θ1 = 0 et θ2 = π : barres alignées => déterminant nul, système singulier.
        truss_member_force_two_bars(1000.0, 500.0, 0.0, PI);
    }
}
