//! Cartes de contrôle de **Shewhart** X-barre et R (limites de contrôle à partir
//! des constantes tabulées `A2`, `D3`, `D4`).
//!
//! ```text
//! carte X̄  UCL = X̄̄ + A2·R̄        (limite de contrôle supérieure)
//!          CL  = X̄̄                (ligne centrale = grande moyenne)
//!          LCL = X̄̄ − A2·R̄        (limite de contrôle inférieure)
//! carte R  UCL = D4·R̄            (limite supérieure de l'étendue)
//!          CL  = R̄                (ligne centrale = étendue moyenne)
//!          LCL = D3·R̄            (limite inférieure de l'étendue)
//! statut   sous contrôle ⇔ LCL ≤ valeur ≤ UCL
//! ```
//!
//! `X̄̄` (grand_mean) grande moyenne (moyenne des moyennes de sous-groupes),
//! `R̄` (rbar) étendue moyenne des sous-groupes, `A2`, `D3`, `D4` constantes de
//! carte de contrôle dépendant de la taille de sous-groupe `n`. `X̄̄`, `R̄` et la
//! valeur testée partagent l'unité de la grandeur mesurée (m, mm, N, … au choix
//! de l'appelant, cohérente entre elles) ; `A2`, `D3`, `D4` sont sans dimension.
//!
//! **Limite honnête** : les constantes `A2`, `D3`, `D4` sont **tabulées** selon la
//! taille de sous-groupe `n` (tables de Shewhart, p. ex. `n=5` → `A2=0,577`,
//! `D3=0`, `D4=2,114`) et **fournies par l'appelant** — aucune valeur « par
//! défaut » n'est inventée ici. Le procédé est supposé **normal** et les
//! sous-groupes rationnels/indépendants ; l'estimation de `X̄̄` et `R̄` à partir
//! des données brutes n'est pas faite dans ce module.

/// Limite de contrôle **supérieure** de la carte X̄ : `UCL = X̄̄ + A2·R̄`.
///
/// Panique si `a2 < 0` ou `rbar < 0`.
pub fn xbar_upper_control_limit(grand_mean: f64, a2: f64, rbar: f64) -> f64 {
    assert!(a2 >= 0.0, "la constante A2 doit être positive ou nulle");
    assert!(
        rbar >= 0.0,
        "l'étendue moyenne R̄ doit être positive ou nulle"
    );
    grand_mean + a2 * rbar
}

/// Limite de contrôle **inférieure** de la carte X̄ : `LCL = X̄̄ − A2·R̄`.
///
/// Panique si `a2 < 0` ou `rbar < 0`.
pub fn xbar_lower_control_limit(grand_mean: f64, a2: f64, rbar: f64) -> f64 {
    assert!(a2 >= 0.0, "la constante A2 doit être positive ou nulle");
    assert!(
        rbar >= 0.0,
        "l'étendue moyenne R̄ doit être positive ou nulle"
    );
    grand_mean - a2 * rbar
}

/// Ligne centrale de la carte X̄ : `CL = X̄̄` (grande moyenne).
pub fn xbar_center_line(grand_mean: f64) -> f64 {
    grand_mean
}

/// Limite de contrôle **supérieure** de la carte R : `UCL = D4·R̄`.
///
/// Panique si `d4 < 0` ou `rbar < 0`.
pub fn rchart_upper_control_limit(d4: f64, rbar: f64) -> f64 {
    assert!(d4 >= 0.0, "la constante D4 doit être positive ou nulle");
    assert!(
        rbar >= 0.0,
        "l'étendue moyenne R̄ doit être positive ou nulle"
    );
    d4 * rbar
}

/// Limite de contrôle **inférieure** de la carte R : `LCL = D3·R̄`.
///
/// Panique si `d3 < 0` ou `rbar < 0`.
pub fn rchart_lower_control_limit(d3: f64, rbar: f64) -> f64 {
    assert!(d3 >= 0.0, "la constante D3 doit être positive ou nulle");
    assert!(
        rbar >= 0.0,
        "l'étendue moyenne R̄ doit être positive ou nulle"
    );
    d3 * rbar
}

/// Ligne centrale de la carte R : `CL = R̄` (étendue moyenne).
///
/// Panique si `rbar < 0`.
pub fn rchart_center_line(rbar: f64) -> f64 {
    assert!(
        rbar >= 0.0,
        "l'étendue moyenne R̄ doit être positive ou nulle"
    );
    rbar
}

/// Statut de contrôle d'un point : `vrai` si `LCL ≤ valeur ≤ UCL`.
///
/// Panique si `lcl > ucl` (limites incohérentes).
pub fn shewhart_process_in_control(value: f64, lcl: f64, ucl: f64) -> bool {
    assert!(
        lcl <= ucl,
        "la limite inférieure doit être inférieure ou égale à la limite supérieure"
    );
    value >= lcl && value <= ucl
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn xbar_limits_are_symmetric_about_center() {
        // UCL et LCL sont à ±A2·R̄ de la ligne centrale : leur milieu = X̄̄.
        let (xbb, a2, rbar) = (10.0, 0.577, 2.0);
        let ucl = xbar_upper_control_limit(xbb, a2, rbar);
        let lcl = xbar_lower_control_limit(xbb, a2, rbar);
        assert_relative_eq!(0.5 * (ucl + lcl), xbar_center_line(xbb), epsilon = 1e-12);
        assert_relative_eq!(ucl - lcl, 2.0 * a2 * rbar, epsilon = 1e-12);
    }

    #[test]
    fn xbar_realistic_case_n5() {
        // n=5 (A2=0,577), X̄̄=100 mm, R̄=5 mm → UCL=102,885 ; LCL=97,115.
        let (xbb, a2, rbar) = (100.0, 0.577, 5.0);
        assert_relative_eq!(
            xbar_upper_control_limit(xbb, a2, rbar),
            102.885,
            epsilon = 1e-9
        );
        assert_relative_eq!(
            xbar_lower_control_limit(xbb, a2, rbar),
            97.115,
            epsilon = 1e-9
        );
    }

    #[test]
    fn rchart_limits_scale_with_rbar() {
        // Les limites de la carte R sont proportionnelles à R̄ (D3, D4 fixés).
        let (d3, d4) = (0.0, 2.114);
        assert_relative_eq!(
            rchart_upper_control_limit(d4, 10.0),
            2.0 * rchart_upper_control_limit(d4, 5.0),
            epsilon = 1e-12
        );
        // n=5 → D3=0 : la limite inférieure de la carte R est nulle.
        assert_relative_eq!(rchart_lower_control_limit(d3, 5.0), 0.0, epsilon = 1e-12);
    }

    #[test]
    fn rchart_realistic_case_n5() {
        // n=5 (D3=0, D4=2,114), R̄=5 mm → UCL=10,57 ; LCL=0 ; CL=5.
        assert_relative_eq!(
            rchart_upper_control_limit(2.114, 5.0),
            10.57,
            epsilon = 1e-9
        );
        assert_relative_eq!(rchart_lower_control_limit(0.0, 5.0), 0.0, epsilon = 1e-12);
        assert_relative_eq!(rchart_center_line(5.0), 5.0, epsilon = 1e-12);
    }

    #[test]
    fn in_control_detects_inside_and_outside() {
        let (lcl, ucl) = (97.115, 102.885);
        // Points aux bornes et au centre : sous contrôle.
        assert!(shewhart_process_in_control(lcl, lcl, ucl));
        assert!(shewhart_process_in_control(ucl, lcl, ucl));
        assert!(shewhart_process_in_control(100.0, lcl, ucl));
        // Points hors limites : hors contrôle.
        assert!(!shewhart_process_in_control(103.0, lcl, ucl));
        assert!(!shewhart_process_in_control(96.0, lcl, ucl));
    }

    #[test]
    fn zero_rbar_collapses_xbar_limits_to_center() {
        // R̄ = 0 (aucune dispersion) → UCL = LCL = X̄̄.
        let xbb = 42.0;
        assert_relative_eq!(
            xbar_upper_control_limit(xbb, 0.577, 0.0),
            xbb,
            epsilon = 1e-12
        );
        assert_relative_eq!(
            xbar_lower_control_limit(xbb, 0.577, 0.0),
            xbb,
            epsilon = 1e-12
        );
    }

    #[test]
    #[should_panic(expected = "limite inférieure")]
    fn inverted_limits_panics() {
        shewhart_process_in_control(100.0, 105.0, 95.0);
    }
}
