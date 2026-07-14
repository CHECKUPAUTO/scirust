//! Empilage de rondelles **Belleville** (disc springs) — combinaison des lois
//! effort-flèche d'un empilage en **série** et/ou en **parallèle**.
//!
//! ```text
//! série     (n rondelles tête-bêche)   δ_stack = n_s · δ_1      F_stack = F_1
//! parallèle (n rondelles empilées)     F_stack = n_p · F_1      δ_stack = δ_1
//! combiné   (n_p en //, n_s en série)  F_stack = n_p · F_1
//!                                       δ_stack = n_s · δ_1
//! raideur   k_stack = (n_p / n_s) · k_1
//! ```
//!
//! `δ_1`/`F_1` flèche (m) et effort (N) d'**une seule** rondelle, `k_1`
//! raideur d'une rondelle (N/m), `n_s` nombre de rondelles en série (entier
//! ≥ 1), `n_p` nombre de rondelles en parallèle (entier ≥ 1). En **série** les
//! flèches s'additionnent à effort constant ; en **parallèle** les efforts
//! s'additionnent à flèche constante.
//!
//! **Convention** : SI cohérent (m, N, N/m). **Limite honnête** : empilage
//! **idéal** de rondelles **identiques**, sans **frottement** inter-rondelles
//! ni sur le guide, sans jeu ni tassement, dans le domaine élastique. La loi
//! effort-flèche d'une rondelle unique (`F_1`, `δ_1`, `k_1`) est **fournie par
//! l'appelant** (p. ex. via `belleville_washers`) : ce module ne fait que la
//! composer et n'invente aucune constante de matériau.

/// Flèche d'un empilage en **série** : `δ_stack = n_s · δ_1` (m).
///
/// En série (rondelles tête-bêche) les flèches s'additionnent à effort constant.
///
/// Panique si `n_in_series == 0` ou `single_deflection < 0`.
pub fn stack_deflection_series(single_deflection: f64, n_in_series: u32) -> f64 {
    assert!(n_in_series >= 1, "n_in_series doit être ≥ 1");
    assert!(single_deflection >= 0.0, "single_deflection doit être ≥ 0");
    f64::from(n_in_series) * single_deflection
}

/// Effort d'un empilage en **parallèle** : `F_stack = n_p · F_1` (N).
///
/// En parallèle (rondelles empilées dans le même sens) les efforts
/// s'additionnent à flèche constante.
///
/// Panique si `n_in_parallel == 0` ou `single_load < 0`.
pub fn stack_load_parallel(single_load: f64, n_in_parallel: u32) -> f64 {
    assert!(n_in_parallel >= 1, "n_in_parallel doit être ≥ 1");
    assert!(single_load >= 0.0, "single_load doit être ≥ 0");
    f64::from(n_in_parallel) * single_load
}

/// Effort et flèche d'un empilage **combiné** (`n_p` en parallèle, `n_s` en
/// série) : renvoie le couple `(F_stack, δ_stack) = (n_p · F_1, n_s · δ_1)`.
///
/// Panique si `n_series == 0`, `n_parallel == 0`, `single_load < 0` ou
/// `single_deflection < 0`.
pub fn stack_combined(
    single_load: f64,
    single_deflection: f64,
    n_series: u32,
    n_parallel: u32,
) -> (f64, f64) {
    let load = stack_load_parallel(single_load, n_parallel);
    let deflection = stack_deflection_series(single_deflection, n_series);
    (load, deflection)
}

/// Raideur d'un empilage combiné : `k_stack = (n_p / n_s) · k_1` (N/m).
///
/// La série divise la raideur, le parallèle la multiplie.
///
/// Panique si `n_series == 0`, `n_parallel == 0` ou `single_stiffness < 0`.
pub fn stack_stiffness(single_stiffness: f64, n_series: u32, n_parallel: u32) -> f64 {
    assert!(n_series >= 1, "n_series doit être ≥ 1");
    assert!(n_parallel >= 1, "n_parallel doit être ≥ 1");
    assert!(single_stiffness >= 0.0, "single_stiffness doit être ≥ 0");
    f64::from(n_parallel) / f64::from(n_series) * single_stiffness
}

/// Nombre total de rondelles d'un empilage combiné : `N = n_s · n_p`.
///
/// Panique si `n_series == 0` ou `n_parallel == 0`.
pub fn stack_washer_count(n_series: u32, n_parallel: u32) -> u32 {
    assert!(n_series >= 1, "n_series doit être ≥ 1");
    assert!(n_parallel >= 1, "n_parallel doit être ≥ 1");
    n_series * n_parallel
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn series_deflections_add_up() {
        // 5 rondelles en série : la flèche est exactement 5 fois celle d'une seule.
        let delta_1 = 0.0008_f64;
        assert_relative_eq!(
            stack_deflection_series(delta_1, 5),
            5.0 * delta_1,
            max_relative = 1e-12
        );
    }

    #[test]
    fn parallel_loads_add_up() {
        // 3 rondelles en parallèle : l'effort est exactement 3 fois celui d'une seule.
        let f_1 = 1250.0_f64;
        assert_relative_eq!(stack_load_parallel(f_1, 3), 3.0 * f_1, max_relative = 1e-12);
    }

    #[test]
    fn single_washer_is_neutral() {
        // n_s = n_p = 1 : l'empilage se réduit à la rondelle unique.
        let (f_1, d_1) = (900.0_f64, 0.0005_f64);
        let (f, d) = stack_combined(f_1, d_1, 1, 1);
        assert_relative_eq!(f, f_1, max_relative = 1e-12);
        assert_relative_eq!(d, d_1, max_relative = 1e-12);
    }

    #[test]
    fn combined_matches_individual_primitives() {
        // Le combiné doit coïncider avec les primitives série/parallèle appliquées séparément.
        let (f_1, d_1) = (2000.0_f64, 0.0006_f64);
        let (n_s, n_p) = (4, 2);
        let (f, d) = stack_combined(f_1, d_1, n_s, n_p);
        assert_relative_eq!(f, stack_load_parallel(f_1, n_p), max_relative = 1e-12);
        assert_relative_eq!(d, stack_deflection_series(d_1, n_s), max_relative = 1e-12);
    }

    #[test]
    fn stiffness_is_load_over_deflection() {
        // Identité physique : k_stack = F_stack / δ_stack doit valoir (n_p/n_s)·k_1,
        // avec k_1 = F_1/δ_1 pour une rondelle (loi linéarisée localement).
        let (f_1, d_1) = (1600.0_f64, 0.0004_f64);
        let k_1 = f_1 / d_1;
        let (n_s, n_p) = (6, 3);
        let (f, d) = stack_combined(f_1, d_1, n_s, n_p);
        assert_relative_eq!(f / d, stack_stiffness(k_1, n_s, n_p), max_relative = 1e-12);
    }

    #[test]
    fn total_count_multiplies() {
        // Un empilage 4×3 compte 12 rondelles.
        assert_eq!(stack_washer_count(4, 3), 12);
    }

    #[test]
    #[should_panic(expected = "n_in_series doit être ≥ 1")]
    fn zero_series_panics() {
        stack_deflection_series(0.001, 0);
    }
}
