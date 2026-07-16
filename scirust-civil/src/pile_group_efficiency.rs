//! **Géotechnique — efficacité d'un groupe de pieux** : coefficient
//! d'efficacité `η` par la formule empirique de **Converse-Labarre**, capacité
//! du groupe déduite de la capacité du pieu isolé, capacité en **rupture
//! d'ensemble** (bloc) et rapport de tassement groupe / pieu isolé (approché).
//!
//! ```text
//! efficacité (Converse-Labarre)  η = 1 − θ·[(n−1)·m + (m−1)·n] / (90·m·n)
//!   avec                         θ = atan(d/s)   (en DEGRÉS)
//! capacité de groupe             Qg = Qsingle · N · η
//! capacité en bloc               Qbloc = P·L·fs + Ab·qb
//! rapport de tassement           Rs = √(Bg / d)
//! ```
//!
//! `η` efficacité du groupe (sans dimension, `0 < η ≤ 1`) ; `θ` angle
//! `atan(d/s)` exprimé en **degrés** (voir convention ci-dessous), `d` =
//! `pile_diameter` diamètre du pieu (m), `s` = `spacing` entraxe des pieux (m),
//! `n` = `rows` nombre de rangées et `m` = `columns` nombre de colonnes du
//! maillage (sans dimension, `≥ 1`). `Qg` capacité du groupe (N), `Qsingle` =
//! `single_pile_capacity` capacité d'un pieu isolé (N), `N` =
//! `number_of_piles` nombre total de pieux (sans dimension). `Qbloc` capacité
//! en bloc (N), `P` = `perimeter` périmètre du bloc équivalent (m), `L` =
//! `length` longueur du bloc (m), `fs` = `unit_skin_friction` frottement
//! latéral unitaire (Pa), `Ab` = `base_area` aire de la base du bloc (m²), `qb`
//! = `unit_base_resistance` résistance unitaire de base (Pa). `Rs` rapport de
//! tassement (sans dimension), `Bg` = `group_width` largeur du groupe (m).
//!
//! **Convention** : SI strict — **N, m, Pa** (avec `1 Pa = 1 N/m²`). Les
//! frottements et résistances unitaires (`fs`, `qb`) sont en **pascals**, les
//! longueurs/périmètres en **mètres**, les aires en **mètres carrés**, les
//! efforts (`Qsingle`, `Qg`, `Qbloc`) en **newtons**. L'angle `θ` de
//! Converse-Labarre est calculé en **degrés** (`θ = atan(d/s)` converti de
//! radians en degrés) et divisé par la constante `90` de la formule, elle aussi
//! en degrés : la convention est cohérente et **indépendante des radians**.
//!
//! **Limite honnête** : formule d'efficacité **empirique** de Converse-Labarre,
//! surtout indicative en **sols cohérents** ; la capacité du **pieu isolé**
//! `Qsingle` et les frottements/résistances unitaires (`fs`, `qb`) sont
//! **fournis par l'appelant** (voir `pile_capacity`), jamais inventés. La
//! capacité réelle d'un groupe est le **minimum** entre la somme réduite
//! `Qg = Qsingle·N·η` et la rupture en bloc `Qbloc` : ce module fournit les deux
//! termes mais laisse l'appelant en prendre le minimum selon son cas. Les
//! résistances caractéristiques du sol **et** les coefficients partiels de
//! sécurité (Eurocode 7 et son Annexe Nationale) sont **fournis par
//! l'appelant** ; aucune valeur « par défaut » n'est inventée. Résultat
//! **indicatif**.

/// Efficacité d'un groupe de pieux par la formule de **Converse-Labarre**
/// `η = 1 − θ·[(n−1)·m + (m−1)·n] / (90·m·n)`, avec `θ = atan(d/s)` **en
/// degrés**, `n = rows` rangées et `m = columns` colonnes.
///
/// La constante `90` est en degrés, cohérente avec `θ` en degrés.
///
/// Panique si `pile_diameter <= 0`, si `spacing <= 0`, si `rows < 1` ou si
/// `columns < 1`.
pub fn pilegrp_converse_labarre(pile_diameter: f64, spacing: f64, rows: f64, columns: f64) -> f64 {
    assert!(
        pile_diameter > 0.0,
        "le diamètre du pieu d doit être strictement positif"
    );
    assert!(
        spacing > 0.0,
        "l'entraxe des pieux s doit être strictement positif"
    );
    assert!(rows >= 1.0, "le nombre de rangées n doit être ≥ 1");
    assert!(columns >= 1.0, "le nombre de colonnes m doit être ≥ 1");
    let theta_deg = (pile_diameter / spacing).atan().to_degrees();
    let coupling = (rows - 1.0) * columns + (columns - 1.0) * rows;
    1.0 - theta_deg * coupling / (90.0 * rows * columns)
}

/// Capacité portante d'un groupe de pieux
/// `Qg = Qsingle · N · η` (N) : capacité du pieu isolé multipliée par le nombre
/// de pieux et affectée du coefficient d'efficacité.
///
/// Panique si `single_pile_capacity < 0`, si `number_of_piles <= 0` ou si
/// `efficiency < 0`.
pub fn pilegrp_capacity(single_pile_capacity: f64, number_of_piles: f64, efficiency: f64) -> f64 {
    assert!(
        single_pile_capacity >= 0.0,
        "la capacité du pieu isolé Qsingle doit être ≥ 0"
    );
    assert!(
        number_of_piles > 0.0,
        "le nombre de pieux N doit être strictement positif"
    );
    assert!(efficiency >= 0.0, "l'efficacité η doit être ≥ 0");
    single_pile_capacity * number_of_piles * efficiency
}

/// Capacité en **rupture d'ensemble** (bloc équivalent)
/// `Qbloc = P·L·fs + Ab·qb` (N) : frottement latéral sur le pourtour du bloc
/// plus résistance de base, avec `fs` et `qb` en Pa.
///
/// Panique si `perimeter <= 0`, si `length <= 0`, si `unit_skin_friction < 0`,
/// si `base_area <= 0` ou si `unit_base_resistance < 0`.
pub fn pilegrp_block_capacity(
    perimeter: f64,
    length: f64,
    unit_skin_friction: f64,
    base_area: f64,
    unit_base_resistance: f64,
) -> f64 {
    assert!(
        perimeter > 0.0,
        "le périmètre du bloc P doit être strictement positif"
    );
    assert!(
        length > 0.0,
        "la longueur du bloc L doit être strictement positive"
    );
    assert!(
        unit_skin_friction >= 0.0,
        "le frottement unitaire fs doit être ≥ 0"
    );
    assert!(
        base_area > 0.0,
        "l'aire de base Ab doit être strictement positive"
    );
    assert!(
        unit_base_resistance >= 0.0,
        "la résistance unitaire de base qb doit être ≥ 0"
    );
    perimeter * length * unit_skin_friction + base_area * unit_base_resistance
}

/// Rapport de tassement groupe / pieu isolé (approché)
/// `Rs = √(Bg / d)` (sans dimension) : estimation empirique de l'amplification
/// du tassement d'un groupe par rapport au pieu isolé.
///
/// Panique si `group_width <= 0` ou si `pile_diameter <= 0`.
pub fn pilegrp_settlement_ratio(group_width: f64, pile_diameter: f64) -> f64 {
    assert!(
        group_width > 0.0,
        "la largeur du groupe Bg doit être strictement positive"
    );
    assert!(
        pile_diameter > 0.0,
        "le diamètre du pieu d doit être strictement positif"
    );
    (group_width / pile_diameter).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn single_pile_has_unit_efficiency() {
        // Un « groupe » 1×1 : aucun pieu voisin, terme de couplage nul → η = 1.
        let eta = pilegrp_converse_labarre(0.4, 1.2, 1.0, 1.0);
        assert_relative_eq!(eta, 1.0, epsilon = 1e-12);
    }

    #[test]
    fn efficiency_increases_with_spacing() {
        // Plus l'entraxe s augmente, plus θ = atan(d/s) diminue → η croît vers 1.
        let eta_tight = pilegrp_converse_labarre(0.3, 0.9, 3.0, 3.0);
        let eta_wide = pilegrp_converse_labarre(0.3, 1.8, 3.0, 3.0);
        assert!(eta_wide > eta_tight);
        assert!(eta_wide < 1.0);
        assert!(eta_tight < 1.0);
    }

    #[test]
    fn converse_labarre_worked_value() {
        // Maillage n = 3 rangées, m = 4 colonnes ; d = 0,3 m, s = 0,9 m.
        //   θ  = atan(0,3/0,9) = atan(1/3) = 18,434948823 °
        //   couplage = (3−1)·4 + (4−1)·3 = 8 + 9 = 17
        //   η  = 1 − 18,434948823·17 / (90·3·4)
        //      = 1 − 313,394129998 / 1080
        //      = 1 − 0,290179750 = 0,709820250
        // (littéral recalculé : 18,434948823·17 = 313,394129998 ;
        //  313,394129998/1080 = 0,290179750 ; 1 − 0,290179750 = 0,709820250)
        let eta = pilegrp_converse_labarre(0.3, 0.9, 3.0, 4.0);
        assert_relative_eq!(eta, 0.709_820_250, max_relative = 1e-3);
    }

    #[test]
    fn group_capacity_scales_linearly() {
        // Qg = Qsingle·N·η : proportionnel à N et à η.
        let qsingle = 800_000.0_f64;
        let qg = pilegrp_capacity(qsingle, 9.0, 0.7);
        assert_relative_eq!(qg, qsingle * 9.0 * 0.7, epsilon = 1e-6);
        // Doubler l'efficacité double la capacité de groupe.
        let qg_double = pilegrp_capacity(qsingle, 9.0, 1.4);
        assert_relative_eq!(qg_double, 2.0 * qg, epsilon = 1e-6);
    }

    #[test]
    fn block_capacity_is_friction_plus_base() {
        // Qbloc = P·L·fs + Ab·qb : additivité frottement + base.
        // Bloc 2,4 × 2,4 m, L = 15 m → P = 9,6 m, Ab = 5,76 m².
        //   frottement = 9,6·15·40 000 = 5 760 000 N
        //   base       = 5,76·1 500 000 = 8 640 000 N
        //   Qbloc      = 14 400 000 N
        // (littéral recalculé : 9,6·15 = 144 ; 144·40 000 = 5 760 000 ;
        //  5,76·1 500 000 = 8 640 000 ; total 14 400 000)
        let qbloc = pilegrp_block_capacity(9.6, 15.0, 40_000.0, 5.76, 1_500_000.0);
        assert_relative_eq!(qbloc, 14_400_000.0, max_relative = 1e-3);
        // Sans résistance de base, il ne reste que le frottement.
        let only_friction = pilegrp_block_capacity(9.6, 15.0, 40_000.0, 5.76, 0.0);
        assert_relative_eq!(only_friction, 5_760_000.0, max_relative = 1e-3);
    }

    #[test]
    fn settlement_ratio_is_square_root() {
        // Rs = √(Bg/d) : pour Bg = 4·d, Rs = 2 ; identité de réciprocité Rs² = Bg/d.
        let d = 0.5_f64;
        let rs = pilegrp_settlement_ratio(2.0, d);
        assert_relative_eq!(rs, 2.0, max_relative = 1e-3);
        assert_relative_eq!(rs * rs, 2.0 / d, epsilon = 1e-9);
    }

    #[test]
    #[should_panic(expected = "l'entraxe des pieux s doit être strictement positif")]
    fn converse_labarre_rejects_zero_spacing() {
        // s = 0 interdit : division par zéro dans atan(d/s).
        pilegrp_converse_labarre(0.4, 0.0, 3.0, 3.0);
    }
}
