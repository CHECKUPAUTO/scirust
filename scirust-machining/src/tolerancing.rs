//! Systèmes de tolérancement de dessin — tolérances **générales** ISO 2768-1
//! (dimensions linéaires et angulaires) et ISO 2768-2 (géométriques), plus un
//! catalogue documenté des normes **GPS** (Geometrical Product Specifications)
//! de l'industrie.
//!
//! Ce module couvre le versant *déterministe et normatif* du tolérancement —
//! celui qu'on lit sur un plan sans cotation explicite (renvoi « ISO 2768-mK »
//! dans le cartouche). Il est complémentaire de la crate `scirust-tolerance`,
//! qui traite le versant *statistique et inertiel* :
//!
//! - **ISO 286** (ajustements arbre/alésage, grades `IT`, écarts fondamentaux)
//!   → `scirust_tolerance::fits` ;
//! - **ISO 1101** (évaluation numérique planéité/rectitude/position/battement…)
//!   → `scirust_tolerance::geometry` et `scirust_tolerance::position` ;
//! - capabilité, chaînes de cotes, MSA, plans d'échantillonnage → le reste de
//!   `scirust-tolerance`.
//!
//! Ici on fournit les **tables de tolérances générales**, celles qui
//! s'appliquent par défaut à toute cote non tolérancée individuellement.
//!
//! **Limite honnête** : les valeurs reproduisent les tables publiées d'ISO 2768
//! (parties 1 et 2). La norme elle-même reste la référence contractuelle ; ce
//! module en donne un accès calculable, sans se substituer au texte officiel ni
//! aux éventuelles révisions/amendements.

/// Classe de tolérance générale linéaire/angulaire ISO 2768-1.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GeneralClass {
    /// `f` — fine.
    Fine,
    /// `m` — moyenne.
    Medium,
    /// `c` — grossière.
    Coarse,
    /// `v` — très grossière.
    VeryCoarse,
}

impl GeneralClass {
    /// Lettre de désignation de la classe (`f`, `m`, `c`, `v`).
    pub fn letter(self) -> char {
        match self
        {
            GeneralClass::Fine => 'f',
            GeneralClass::Medium => 'm',
            GeneralClass::Coarse => 'c',
            GeneralClass::VeryCoarse => 'v',
        }
    }
    fn index(self) -> usize {
        match self
        {
            GeneralClass::Fine => 0,
            GeneralClass::Medium => 1,
            GeneralClass::Coarse => 2,
            GeneralClass::VeryCoarse => 3,
        }
    }
}

// ISO 2768-1 — écarts admissibles (±mm) des dimensions linéaires.
// Colonnes : [f, m, c, v] ; NaN = valeur non définie par la norme pour la classe.
// Bornes supérieures (mm) des plages : 3, 6, 30, 120, 400, 1000, 2000, 4000.
const LINEAR_UPPER: [f64; 8] = [3.0, 6.0, 30.0, 120.0, 400.0, 1000.0, 2000.0, 4000.0];
const LINEAR_DEV: [[f64; 4]; 8] = [
    // 0,5–3 : v non défini
    [0.05, 0.1, 0.2, f64::NAN],
    // 3–6
    [0.05, 0.1, 0.3, 0.5],
    // 6–30
    [0.1, 0.2, 0.5, 1.0],
    // 30–120
    [0.15, 0.3, 0.8, 1.5],
    // 120–400
    [0.2, 0.5, 1.2, 2.5],
    // 400–1000
    [0.3, 0.8, 2.0, 4.0],
    // 1000–2000
    [0.5, 1.2, 3.0, 6.0],
    // 2000–4000 : f non défini
    [f64::NAN, 2.0, 4.0, 8.0],
];

/// Écart admissible `±` (mm) d'une dimension linéaire `nominal` (mm) sous la
/// tolérance générale `class` (ISO 2768-1).
///
/// Renvoie `None` si la dimension est hors table (`< 0,5` ou `> 4000` mm) ou si
/// la norme ne définit pas de valeur pour cette classe dans cette plage (p. ex.
/// classe `v` sous 3 mm, classe `f` au-delà de 2000 mm).
pub fn general_linear_tolerance(nominal_mm: f64, class: GeneralClass) -> Option<f64> {
    if nominal_mm < 0.5 || nominal_mm > 4000.0
    {
        return None;
    }
    let row = LINEAR_UPPER.iter().position(|&u| nominal_mm <= u)?;
    let v = LINEAR_DEV[row][class.index()];
    if v.is_nan() { None } else { Some(v) }
}

// ISO 2768-1 — écarts admissibles (±degrés décimaux) des dimensions
// angulaires, par longueur du côté le plus court (mm). f et m partagent la
// même colonne. Colonnes : [f/m, c, v].
// Bornes supérieures des plages : 10, 50, 120, 400, +∞.
const ANGULAR_UPPER: [f64; 5] = [10.0, 50.0, 120.0, 400.0, f64::INFINITY];
const ANGULAR_DEV: [[f64; 3]; 5] = [
    // ≤10 mm  : 1°,    1°30',  3°
    [1.0, 1.5, 3.0],
    // 10–50   : 0°30', 1°,     2°
    [0.5, 1.0, 2.0],
    // 50–120  : 0°20', 0°30',  1°
    [1.0 / 3.0, 0.5, 1.0],
    // 120–400 : 0°10', 0°15',  0°30'
    [1.0 / 6.0, 0.25, 0.5],
    // >400    : 0°5',  0°10',  0°20'
    [1.0 / 12.0, 1.0 / 6.0, 1.0 / 3.0],
];

/// Écart angulaire admissible `±` (degrés décimaux) pour un côté le plus court
/// `shorter_side` (mm) sous la tolérance générale `class` (ISO 2768-1).
///
/// Les classes `f` et `m` partagent la même colonne. Renvoie `None` si
/// `shorter_side < 0` (aucune borne supérieure : la dernière plage est ouverte).
pub fn general_angular_tolerance(shorter_side_mm: f64, class: GeneralClass) -> Option<f64> {
    if shorter_side_mm < 0.0
    {
        return None;
    }
    let col = match class
    {
        GeneralClass::Fine | GeneralClass::Medium => 0,
        GeneralClass::Coarse => 1,
        GeneralClass::VeryCoarse => 2,
    };
    let row = ANGULAR_UPPER.iter().position(|&u| shorter_side_mm <= u)?;
    Some(ANGULAR_DEV[row][col])
}

/// Classe de tolérance géométrique générale ISO 2768-2.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GeometricalClass {
    /// `H`.
    H,
    /// `K`.
    K,
    /// `L`.
    L,
}

impl GeometricalClass {
    fn index(self) -> usize {
        match self
        {
            GeometricalClass::H => 0,
            GeometricalClass::K => 1,
            GeometricalClass::L => 2,
        }
    }
}

// ISO 2768-2 — rectitude et planéité (mm), par plage de longueur nominale.
// Bornes supérieures : 10, 30, 100, 300, 1000, 3000. Colonnes : [H, K, L].
const SF_UPPER: [f64; 6] = [10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0];
const SF_DEV: [[f64; 3]; 6] = [
    [0.02, 0.05, 0.1],
    [0.05, 0.1, 0.2],
    [0.1, 0.2, 0.4],
    [0.2, 0.4, 0.8],
    [0.3, 0.6, 1.2],
    [0.4, 0.8, 1.6],
];

/// Tolérance générale de **rectitude / planéité** (mm) pour une longueur
/// nominale `nominal` (mm) et une classe géométrique (ISO 2768-2).
///
/// Renvoie `None` au-delà de 3000 mm ou pour `nominal < 0`.
pub fn general_straightness_flatness(nominal_mm: f64, class: GeometricalClass) -> Option<f64> {
    if nominal_mm < 0.0 || nominal_mm > 3000.0
    {
        return None;
    }
    let row = SF_UPPER.iter().position(|&u| nominal_mm <= u)?;
    Some(SF_DEV[row][class.index()])
}

// ISO 2768-2 — perpendicularité (mm), par longueur du côté le plus court.
// Bornes supérieures : 100, 300, 1000, 3000. Colonnes : [H, K, L].
const PERP_UPPER: [f64; 4] = [100.0, 300.0, 1000.0, 3000.0];
const PERP_DEV: [[f64; 3]; 4] = [
    [0.2, 0.4, 0.6],
    [0.3, 0.6, 1.0],
    [0.4, 0.8, 1.5],
    [0.5, 1.0, 2.0],
];

/// Tolérance générale de **perpendicularité** (mm) pour un côté le plus court
/// `shorter_side` (mm) et une classe géométrique (ISO 2768-2).
pub fn general_perpendicularity(shorter_side_mm: f64, class: GeometricalClass) -> Option<f64> {
    if shorter_side_mm < 0.0 || shorter_side_mm > 3000.0
    {
        return None;
    }
    let row = PERP_UPPER.iter().position(|&u| shorter_side_mm <= u)?;
    Some(PERP_DEV[row][class.index()])
}

// ISO 2768-2 — symétrie (mm), par plage. Bornes : 100, 300, 1000, 3000.
const SYM_UPPER: [f64; 4] = [100.0, 300.0, 1000.0, 3000.0];
const SYM_DEV: [[f64; 3]; 4] = [
    [0.5, 0.6, 0.6],
    [0.5, 0.6, 1.0],
    [0.5, 0.8, 1.5],
    [0.5, 1.0, 2.0],
];

/// Tolérance générale de **symétrie** (mm) pour une longueur nominale
/// `nominal` (mm) et une classe géométrique (ISO 2768-2).
pub fn general_symmetry(nominal_mm: f64, class: GeometricalClass) -> Option<f64> {
    if nominal_mm < 0.0 || nominal_mm > 3000.0
    {
        return None;
    }
    let row = SYM_UPPER.iter().position(|&u| nominal_mm <= u)?;
    Some(SYM_DEV[row][class.index()])
}

/// Tolérance générale de **battement circulaire** (mm) — valeur fixe par classe
/// (ISO 2768-2) : `H = 0,1`, `K = 0,2`, `L = 0,5`.
pub fn general_circular_runout(class: GeometricalClass) -> f64 {
    match class
    {
        GeometricalClass::H => 0.1,
        GeometricalClass::K => 0.2,
        GeometricalClass::L => 0.5,
    }
}

/// Normes du système **GPS** (ISO Geometrical Product Specifications) et du
/// tolérancement industriel — catalogue de référence.
///
/// Chaque variante donne son numéro ([`Self::number`]), son titre
/// ([`Self::title`]) et son objet ([`Self::scope`]). L'énumération n'est pas
/// exhaustive de tout l'écosystème GPS (des dizaines de parties), mais couvre
/// les normes structurantes rencontrées en productique mécanique.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpsStandard {
    /// Principe fondamental d'indépendance.
    Iso8015,
    /// Tolérancement géométrique (formes, orientation, position, battement).
    Iso1101,
    /// Références et systèmes de références (datums).
    Iso5459,
    /// Tolérancement de localisation.
    Iso5458,
    /// Système d'ajustements arbre/alésage (grades IT, écarts).
    Iso286,
    /// Tolérances générales dimensionnelles et angulaires.
    Iso2768_1,
    /// Tolérances géométriques générales.
    Iso2768_2,
    /// Tolérancement dimensionnel (tailles).
    Iso14405,
    /// Exigence du maximum / minimum de matière (MMR/LMR) et de réciprocité.
    Iso2692,
    /// Indication des états de surface.
    Iso1302,
    /// État de surface : méthode du profil, paramètres (Ra, Rz…).
    Iso4287,
    /// État de surface : règles et procédures d'évaluation du profil.
    Iso4288,
}

impl GpsStandard {
    /// Numéro normatif (ex. `"ISO 1101"`).
    pub fn number(self) -> &'static str {
        match self
        {
            GpsStandard::Iso8015 => "ISO 8015",
            GpsStandard::Iso1101 => "ISO 1101",
            GpsStandard::Iso5459 => "ISO 5459",
            GpsStandard::Iso5458 => "ISO 5458",
            GpsStandard::Iso286 => "ISO 286",
            GpsStandard::Iso2768_1 => "ISO 2768-1",
            GpsStandard::Iso2768_2 => "ISO 2768-2",
            GpsStandard::Iso14405 => "ISO 14405",
            GpsStandard::Iso2692 => "ISO 2692",
            GpsStandard::Iso1302 => "ISO 1302",
            GpsStandard::Iso4287 => "ISO 4287",
            GpsStandard::Iso4288 => "ISO 4288",
        }
    }

    /// Titre abrégé de la norme.
    pub fn title(self) -> &'static str {
        match self
        {
            GpsStandard::Iso8015 => "Principe fondamental GPS (indépendance)",
            GpsStandard::Iso1101 => "Tolérancement géométrique",
            GpsStandard::Iso5459 => "Références et systèmes de références",
            GpsStandard::Iso5458 => "Tolérancement de localisation",
            GpsStandard::Iso286 => "Système d'ajustements ISO (tolérances et écarts)",
            GpsStandard::Iso2768_1 => "Tolérances générales — dimensions linéaires et angulaires",
            GpsStandard::Iso2768_2 => "Tolérances générales — géométriques",
            GpsStandard::Iso14405 => "Tolérancement dimensionnel",
            GpsStandard::Iso2692 => "Exigence du maximum/minimum de matière (MMR/LMR)",
            GpsStandard::Iso1302 => "Indication des états de surface",
            GpsStandard::Iso4287 => "État de surface — paramètres du profil",
            GpsStandard::Iso4288 => "État de surface — règles d'évaluation",
        }
    }

    /// Objet de la norme (rôle en productique).
    pub fn scope(self) -> &'static str {
        match self
        {
            GpsStandard::Iso8015 =>
            {
                "Pose le principe d'indépendance : chaque exigence de taille et de \
                 géométrie s'applique indépendamment sauf modificateur explicite."
            },
            GpsStandard::Iso1101 =>
            {
                "Définit les tolérances de forme, orientation, position et battement, \
                 leurs symboles et zones de tolérance."
            },
            GpsStandard::Iso5459 =>
            {
                "Établit les références (datums) simples, communes et systèmes de \
                 références qui orientent les tolérances géométriques."
            },
            GpsStandard::Iso5458 =>
            {
                "Spécifie le tolérancement de localisation de motifs de features \
                 (groupes de trous, etc.)."
            },
            GpsStandard::Iso286 =>
            {
                "Système d'ajustements arbre/alésage : grades de tolérance IT01–IT18 \
                 et écarts fondamentaux (lettres a–zc)."
            },
            GpsStandard::Iso2768_1 =>
            {
                "Tolérances générales par défaut des cotes linéaires et angulaires \
                 non tolérancées (classes f, m, c, v)."
            },
            GpsStandard::Iso2768_2 =>
            {
                "Tolérances géométriques générales par défaut (classes H, K, L) : \
                 rectitude, planéité, perpendicularité, symétrie, battement."
            },
            GpsStandard::Iso14405 =>
            {
                "Définit les tailles linéaires (locale, globale, calculée) et leurs \
                 modificateurs de spécification."
            },
            GpsStandard::Iso2692 =>
            {
                "Modificateurs Ⓜ/Ⓛ : transfert de tolérance selon l'état de matière et \
                 exigence de réciprocité."
            },
            GpsStandard::Iso1302 =>
            {
                "Codifie l'inscription des états de surface sur les dessins techniques."
            },
            GpsStandard::Iso4287 =>
            {
                "Définit les paramètres d'amplitude du profil de rugosité (Ra, Rz, Rt…)."
            },
            GpsStandard::Iso4288 =>
            {
                "Fixe les règles, longueurs de base et procédures d'évaluation du \
                 profil de surface."
            },
        }
    }
}

/// Catalogue complet des normes GPS répertoriées par ce module, dans un ordre
/// stable — pratique pour construire un tableau de synthèse.
pub const GPS_CATALOGUE: [GpsStandard; 12] = [
    GpsStandard::Iso8015,
    GpsStandard::Iso1101,
    GpsStandard::Iso5459,
    GpsStandard::Iso5458,
    GpsStandard::Iso286,
    GpsStandard::Iso2768_1,
    GpsStandard::Iso2768_2,
    GpsStandard::Iso14405,
    GpsStandard::Iso2692,
    GpsStandard::Iso1302,
    GpsStandard::Iso4287,
    GpsStandard::Iso4288,
];

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn linear_tolerance_reads_the_iso2768_table() {
        // 6–30 mm, classe m → ±0,2 mm.
        assert_relative_eq!(
            general_linear_tolerance(20.0, GeneralClass::Medium).unwrap(),
            0.2,
            epsilon = 1e-12
        );
        // borne : exactement 30 mm reste dans la plage 6–30.
        assert_relative_eq!(
            general_linear_tolerance(30.0, GeneralClass::Coarse).unwrap(),
            0.5,
            epsilon = 1e-12
        );
        // juste au-dessus bascule dans 30–120.
        assert_relative_eq!(
            general_linear_tolerance(30.001, GeneralClass::Coarse).unwrap(),
            0.8,
            epsilon = 1e-12
        );
    }

    #[test]
    fn linear_tolerance_undefined_cells_return_none() {
        // classe v sous 3 mm : non définie.
        assert!(general_linear_tolerance(2.0, GeneralClass::VeryCoarse).is_none());
        // classe f entre 2000 et 4000 : non définie.
        assert!(general_linear_tolerance(3000.0, GeneralClass::Fine).is_none());
        // hors table.
        assert!(general_linear_tolerance(0.4, GeneralClass::Medium).is_none());
        assert!(general_linear_tolerance(5000.0, GeneralClass::Coarse).is_none());
    }

    #[test]
    fn angular_tolerance_shares_f_and_m_column() {
        // côté 20 mm (plage 10–50), f et m → ±0,5° ; c → ±1° ; v → ±2°.
        assert_relative_eq!(
            general_angular_tolerance(20.0, GeneralClass::Fine).unwrap(),
            0.5,
            epsilon = 1e-12
        );
        assert_relative_eq!(
            general_angular_tolerance(20.0, GeneralClass::Medium).unwrap(),
            0.5,
            epsilon = 1e-12
        );
        assert_relative_eq!(
            general_angular_tolerance(20.0, GeneralClass::Coarse).unwrap(),
            1.0,
            epsilon = 1e-12
        );
        // plage ouverte au-delà de 400 mm.
        assert_relative_eq!(
            general_angular_tolerance(5000.0, GeneralClass::VeryCoarse).unwrap(),
            1.0 / 3.0,
            epsilon = 1e-12
        );
    }

    #[test]
    fn geometrical_general_tolerances_read_their_tables() {
        // rectitude/planéité, 30–100 mm, classe K → 0,2 mm.
        assert_relative_eq!(
            general_straightness_flatness(50.0, GeometricalClass::K).unwrap(),
            0.2,
            epsilon = 1e-12
        );
        // perpendicularité, côté ≤100 mm, classe H → 0,2 mm.
        assert_relative_eq!(
            general_perpendicularity(80.0, GeometricalClass::H).unwrap(),
            0.2,
            epsilon = 1e-12
        );
        // symétrie classe H : 0,5 mm sur toutes les plages.
        assert_relative_eq!(
            general_symmetry(2500.0, GeometricalClass::H).unwrap(),
            0.5,
            epsilon = 1e-12
        );
        // battement circulaire : valeurs fixes.
        assert_relative_eq!(
            general_circular_runout(GeometricalClass::L),
            0.5,
            epsilon = 1e-12
        );
    }

    #[test]
    fn gps_catalogue_is_self_consistent() {
        // chaque entrée a un numéro non vide commençant par "ISO".
        for s in GPS_CATALOGUE
        {
            assert!(s.number().starts_with("ISO"));
            assert!(!s.title().is_empty());
            assert!(!s.scope().is_empty());
        }
        // repère : ISO 2768-1 bien présente.
        assert!(GPS_CATALOGUE.contains(&GpsStandard::Iso2768_1));
    }
}
