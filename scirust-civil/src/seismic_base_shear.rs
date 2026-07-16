//! Effort tranchant sismique à la base d'un bâtiment selon l'**Eurocode 8**
//! (EN 1998-1), par la **méthode des forces latérales équivalentes** applicable
//! aux bâtiments réguliers : effort tranchant à la base, période fondamentale
//! approchée, répartition des forces d'étage, ordonnée du spectre de calcul sur
//! le plateau et masse effective à partir du poids sismique.
//!
//! ```text
//! effort à la base         Fb     = Sd(T1)·m·λ
//! période approchée        T1     = Ct·H^0,75
//! force d'étage            Fi     = Fb·(mi·zi)/Σ(mj·zj)
//! spectre plateau          Sd     = ag·S·β/q
//! masse effective          m      = W/g
//! ```
//!
//! `Fb` effort tranchant à la base (N), `Sd(T1)` accélération spectrale de
//! calcul au niveau de la période fondamentale (m/s²), `m` masse totale mise en
//! mouvement (kg), `λ` facteur de correction (–, ≈ 0,85 pour les bâtiments de
//! plus de deux niveaux dont `T1 ≤ 2·TC`, sinon 1,0), `T1` période fondamentale
//! (s), `Ct` coefficient dépendant du type de structure (s·m⁻⁰·⁷⁵), `H` hauteur
//! du bâtiment au-dessus des fondations (m), `Fi` force latérale à l'étage `i`
//! (N), `mi` masse de l'étage `i` (kg), `zi` hauteur de l'étage `i` au-dessus de
//! la base (m), `Σ(mj·zj)` somme des produits masse·hauteur de tous les étages
//! (kg·m), `ag` accélération de calcul du sol (m/s²), `S` paramètre de sol (–),
//! `β` amplification spectrale sur le plateau (–, ≈ 2,5), `q` facteur de
//! comportement (–), `W` poids sismique (N), `g` accélération de la pesanteur
//! (m/s²).
//!
//! **Convention** : SI strict et cohérent — newtons (N) pour les forces et
//! poids, kilogrammes (kg) pour les masses, mètres (m) pour les hauteurs,
//! secondes (s) pour les périodes, m/s² pour les accélérations. Types `f64`.
//!
//! **Limite honnête** : ce module applique la méthode des forces latérales
//! équivalentes de l'Eurocode 8, valable pour les **bâtiments réguliers** dont
//! la réponse est dominée par le mode fondamental. Il ne réalise **aucune**
//! analyse modale spectrale ni temporelle. L'accélération spectrale de calcul
//! `Sd(T1)`, le facteur de correction `λ`, le coefficient `Ct`, le paramètre de
//! sol `S`, l'amplification `β`, l'accélération `ag` et le facteur de
//! comportement `q` sont **fournis par l'appelant** d'après l'Eurocode et son
//! Annexe Nationale — jamais inventés ici. L'accélération de la pesanteur `g`
//! est également un **paramètre fourni**. La répartition des forces d'étage
//! utilise l'approximation du **mode fondamental linéaire** (déformée
//! triangulaire) via les hauteurs `zi` ; les distributions modales plus fines
//! restent à la charge de l'appelant. Ce module ne fournit aucune valeur de
//! carte de zonage ni de table de spectre par défaut.

/// Effort tranchant sismique à la base `Fb = Sd(T1)·m·λ` (N), produit de
/// l'accélération spectrale de calcul, de la masse totale mise en mouvement et
/// du facteur de correction.
///
/// Panique si `spectral_acceleration < 0`, `total_mass <= 0` ou
/// `correction_factor <= 0`.
pub fn seis_base_shear(spectral_acceleration: f64, total_mass: f64, correction_factor: f64) -> f64 {
    assert!(
        spectral_acceleration >= 0.0,
        "l'accélération spectrale de calcul Sd(T1) doit être positive ou nulle"
    );
    assert!(
        total_mass > 0.0,
        "la masse totale m doit être strictement positive"
    );
    assert!(
        correction_factor > 0.0,
        "le facteur de correction λ doit être strictement positif"
    );
    spectral_acceleration * total_mass * correction_factor
}

/// Période fondamentale approchée `T1 = Ct·H^0,75` (s), en fonction du
/// coefficient de structure `Ct` (s·m⁻⁰·⁷⁵) et de la hauteur `H` (m) du
/// bâtiment au-dessus des fondations.
///
/// Panique si `coefficient <= 0` ou `height <= 0`.
pub fn seis_fundamental_period_approx(coefficient: f64, height: f64) -> f64 {
    assert!(
        coefficient > 0.0,
        "le coefficient de structure Ct doit être strictement positif"
    );
    assert!(
        height > 0.0,
        "la hauteur H du bâtiment doit être strictement positive"
    );
    coefficient * height.powf(0.75)
}

/// Force latérale à un étage `Fi = Fb·(mi·zi)/Σ(mj·zj)` (N), répartition selon
/// la déformée triangulaire du mode fondamental (les hauteurs `zi` croissantes
/// donnent des forces croissantes vers le sommet).
///
/// Panique si `base_shear < 0`, `floor_mass <= 0`, `floor_height < 0` ou
/// `sum_mass_height <= 0`.
pub fn seis_lateral_force_distribution(
    base_shear: f64,
    floor_mass: f64,
    floor_height: f64,
    sum_mass_height: f64,
) -> f64 {
    assert!(
        base_shear >= 0.0,
        "l'effort tranchant à la base Fb doit être positif ou nul"
    );
    assert!(
        floor_mass > 0.0,
        "la masse d'étage mi doit être strictement positive"
    );
    assert!(
        floor_height >= 0.0,
        "la hauteur d'étage zi doit être positive ou nulle"
    );
    assert!(
        sum_mass_height > 0.0,
        "la somme Σ(mj·zj) doit être strictement positive"
    );
    base_shear * floor_mass * floor_height / sum_mass_height
}

/// Ordonnée du spectre de calcul sur le plateau `Sd = ag·S·β/q` (m/s²), produit
/// de l'accélération de calcul du sol par le paramètre de sol et l'amplification
/// spectrale, divisé par le facteur de comportement.
///
/// Panique si `ground_acceleration < 0`, `soil_factor <= 0`,
/// `spectral_amplification <= 0` ou `behaviour_factor < 1`.
pub fn seis_design_spectrum_plateau(
    ground_acceleration: f64,
    soil_factor: f64,
    spectral_amplification: f64,
    behaviour_factor: f64,
) -> f64 {
    assert!(
        ground_acceleration >= 0.0,
        "l'accélération de calcul du sol ag doit être positive ou nulle"
    );
    assert!(
        soil_factor > 0.0,
        "le paramètre de sol S doit être strictement positif"
    );
    assert!(
        spectral_amplification > 0.0,
        "l'amplification spectrale β doit être strictement positive"
    );
    assert!(
        behaviour_factor >= 1.0,
        "le facteur de comportement q doit être supérieur ou égal à 1"
    );
    ground_acceleration * soil_factor * spectral_amplification / behaviour_factor
}

/// Masse effective mise en mouvement `m = W/g` (kg), à partir du poids sismique
/// `W` (N) et de l'accélération de la pesanteur `g` (m/s²).
///
/// Panique si `seismic_weight < 0` ou `gravity <= 0`.
pub fn seis_effective_mass(seismic_weight: f64, gravity: f64) -> f64 {
    assert!(
        seismic_weight >= 0.0,
        "le poids sismique W doit être positif ou nul"
    );
    assert!(
        gravity > 0.0,
        "l'accélération de la pesanteur g doit être strictement positive"
    );
    seismic_weight / gravity
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn base_shear_is_trilinear_product() {
        // Fb = Sd·m·λ : cas chiffré Sd = 1,5 m/s², m = 200000 kg, λ = 0,85.
        // Fb = 1,5·200000·0,85 = 255000 N.
        let fb = seis_base_shear(1.5, 200_000.0, 0.85);
        assert_relative_eq!(fb, 255_000.0, max_relative = 1e-12);
        // Proportionnalité : doubler la masse double l'effort à la base.
        assert_relative_eq!(
            seis_base_shear(1.5, 400_000.0, 0.85),
            2.0 * fb,
            max_relative = 1e-12
        );
        // Une accélération spectrale nulle donne un effort nul.
        assert_relative_eq!(seis_base_shear(0.0, 200_000.0, 0.85), 0.0, epsilon = 1e-12);
    }

    #[test]
    fn fundamental_period_scaling_law() {
        // T1 = Ct·H^0,75 : le rapport de deux périodes ne dépend que de H^0,75.
        let ct = 0.075;
        let t1 = seis_fundamental_period_approx(ct, 20.0);
        let t2 = seis_fundamental_period_approx(ct, 40.0);
        // T2/T1 = (40/20)^0,75 = 2^0,75.
        assert_relative_eq!(t2 / t1, 2.0_f64.powf(0.75), max_relative = 1e-9);
        // Cas chiffré : Ct = 0,075, H = 20 m.
        // T1 = 0,075·20^0,75 = 0,075·9,457416 = 0,709306 s.
        assert_relative_eq!(t1, 0.709306, max_relative = 1e-3);
    }

    #[test]
    fn force_distribution_sums_to_base_shear() {
        // Deux étages de mêmes masses aux hauteurs 3 m et 6 m.
        // Σ(mj·zj) = 1000·3 + 1000·6 = 9000 kg·m.
        let sum_mz = 1000.0 * 3.0 + 1000.0 * 6.0;
        let fb = 90_000.0;
        let f1 = seis_lateral_force_distribution(fb, 1000.0, 3.0, sum_mz);
        let f2 = seis_lateral_force_distribution(fb, 1000.0, 6.0, sum_mz);
        // La somme des forces d'étage restitue l'effort à la base.
        assert_relative_eq!(f1 + f2, fb, max_relative = 1e-12);
        // Déformée triangulaire : l'étage supérieur reçoit le double de l'étage
        // inférieur (rapport des hauteurs 6/3 = 2).
        assert_relative_eq!(f2, 2.0 * f1, max_relative = 1e-12);
        // Cas chiffré : F1 = 90000·(1000·3)/9000 = 30000 N.
        assert_relative_eq!(f1, 30_000.0, max_relative = 1e-12);
    }

    #[test]
    fn design_spectrum_inversely_proportional_to_q() {
        // Sd = ag·S·β/q : doubler q divise Sd par deux (dissipation accrue).
        let sd1 = seis_design_spectrum_plateau(2.0, 1.2, 2.5, 1.5);
        let sd2 = seis_design_spectrum_plateau(2.0, 1.2, 2.5, 3.0);
        assert_relative_eq!(sd2, 0.5 * sd1, max_relative = 1e-12);
        // Cas chiffré : ag = 2,0 m/s², S = 1,2, β = 2,5, q = 1,5.
        // Sd = 2,0·1,2·2,5/1,5 = 6,0/1,5 = 4,0 m/s².
        assert_relative_eq!(sd1, 4.0, max_relative = 1e-12);
    }

    #[test]
    fn effective_mass_inverts_weight() {
        // m = W/g : réciprocité avec le poids W = m·g.
        // W = 1962000 N, g = 9,81 m/s² -> m = 200000 kg.
        let m = seis_effective_mass(1_962_000.0, 9.81);
        assert_relative_eq!(m, 200_000.0, max_relative = 1e-9);
        // Recomposition du poids à partir de la masse trouvée.
        assert_relative_eq!(m * 9.81, 1_962_000.0, max_relative = 1e-9);
    }

    #[test]
    fn worked_case_regular_building() {
        // Bâtiment régulier : W = 1962000 N, g = 9,81 m/s².
        let m = seis_effective_mass(1_962_000.0, 9.81);
        assert_relative_eq!(m, 200_000.0, max_relative = 1e-9);
        // Spectre de calcul : ag = 1,6 m/s², S = 1,25, β = 2,5, q = 4,0.
        // Sd = 1,6·1,25·2,5/4,0 = 5,0/4,0 = 1,25 m/s².
        let sd = seis_design_spectrum_plateau(1.6, 1.25, 2.5, 4.0);
        assert_relative_eq!(sd, 1.25, max_relative = 1e-12);
        // Effort à la base avec λ = 0,85.
        // Fb = 1,25·200000·0,85 = 212500 N.
        let fb = seis_base_shear(sd, m, 0.85);
        assert_relative_eq!(fb, 212_500.0, max_relative = 1e-9);
    }

    #[test]
    #[should_panic(expected = "le facteur de comportement q doit être supérieur ou égal à 1")]
    fn behaviour_factor_below_one_panics() {
        let _ = seis_design_spectrum_plateau(2.0, 1.2, 2.5, 0.8);
    }
}
