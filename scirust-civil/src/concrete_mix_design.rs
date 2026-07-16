//! Formulation du béton (rapport eau/ciment, E/C) — méthode indicative des
//! **volumes absolus** avec les lois empiriques d'**Abrams/Bolomey** :
//! rapport E/C déduit de la résistance visée, résistance moyenne à viser,
//! dosage en ciment, rapport granulats/ciment et volume rendu du mélange.
//!
//! ```text
//! rapport E/C          w/c = a/(fcm + b)                (Abrams/Bolomey inversée)
//! résistance visée     fcm = fck + Δf                   (marge statistique)
//! dosage ciment        C   = W/(w/c)
//! rapport G/C          g/c = (ρ − C − W)/C
//! volume rendu         V   = C/ρc + W/ρw + G/ρg         (volumes absolus)
//! ```
//!
//! `w/c` rapport massique eau/ciment (–), `a` et `b` constantes empiriques de
//! la loi d'Abrams/Bolomey (`a` en MPa, `b` en MPa), `fcm` résistance moyenne
//! visée à la compression (MPa), `fck` résistance caractéristique (MPa), `Δf`
//! marge statistique (MPa), `C` dosage en ciment (kg/m³), `W` dosage en eau
//! (kg/m³), `g/c` rapport massique granulats/ciment (–), `ρ` masse volumique
//! du béton frais (kg/m³), `G` masse de granulats (kg), `V` volume rendu du
//! mélange (m³ si les masses sont en kg et les masses volumiques en kg/m³).
//!
//! **Convention** : SI cohérent — masses en kg, masses volumiques en kg/m³,
//! dosages en kg/m³, volumes en m³, résistances en MPa (N/mm²). Les rapports
//! E/C et G/C sont sans dimension. Types `f64`.
//!
//! **Limite honnête** : formulation **indicative**. Les constantes empiriques
//! `a` et `b` d'Abrams/Bolomey (calées sur les matériaux et le type de ciment),
//! la marge statistique `Δf` (dépendant de l'écart-type de production et du
//! fractile visé), les dosages en eau, ainsi que les masses volumiques du
//! ciment `ρc`, de l'eau `ρw`, des granulats `ρg` et du béton frais `ρ` sont
//! **fournis par l'appelant** — jamais inventés. Ce module applique la méthode
//! des volumes absolus ; il **n'optimise pas** la granulométrie, ni les
//! adjuvants, ni l'air occlus, ni l'ouvrabilité.

/// Rapport eau/ciment `w/c = a/(fcm + b)` déduit de la loi d'Abrams/Bolomey
/// inversée, à partir des constantes empiriques `a` et `b` (MPa) et de la
/// résistance moyenne visée `fcm` (MPa).
///
/// Panique si `strength_constant_a <= 0`, si `target_strength <= 0`, ou si
/// `target_strength + strength_constant_b <= 0` (dénominateur non strictement
/// positif).
pub fn mix_water_cement_from_strength(
    strength_constant_a: f64,
    strength_constant_b: f64,
    target_strength: f64,
) -> f64 {
    assert!(
        strength_constant_a > 0.0,
        "la constante empirique a doit être strictement positive"
    );
    assert!(
        target_strength > 0.0,
        "la résistance visée fcm doit être strictement positive"
    );
    let denominator = target_strength + strength_constant_b;
    assert!(
        denominator > 0.0,
        "le dénominateur (fcm + b) doit être strictement positif"
    );
    strength_constant_a / denominator
}

/// Résistance moyenne à viser `fcm = fck + Δf` (MPa) : résistance
/// caractéristique majorée de la marge statistique de production.
///
/// Panique si `characteristic_strength <= 0` ou si `margin < 0`.
pub fn mix_target_mean_strength(characteristic_strength: f64, margin: f64) -> f64 {
    assert!(
        characteristic_strength > 0.0,
        "la résistance caractéristique fck doit être strictement positive"
    );
    assert!(
        margin >= 0.0,
        "la marge statistique Δf doit être positive ou nulle"
    );
    characteristic_strength + margin
}

/// Dosage en ciment `C = W/(w/c)` (kg/m³) à partir du dosage en eau `W`
/// (kg/m³) et du rapport eau/ciment `w/c`.
///
/// Panique si `water_content < 0` ou si `water_cement_ratio <= 0`.
pub fn mix_cement_content(water_content: f64, water_cement_ratio: f64) -> f64 {
    assert!(
        water_content >= 0.0,
        "le dosage en eau W doit être positif ou nul"
    );
    assert!(
        water_cement_ratio > 0.0,
        "le rapport eau/ciment w/c doit être strictement positif"
    );
    water_content / water_cement_ratio
}

/// Rapport granulats/ciment `g/c = (ρ − C − W)/C` déduit de la masse volumique
/// du béton frais `ρ` (kg/m³), du dosage en ciment `C` (kg/m³) et du dosage en
/// eau `W` (kg/m³) : la masse de granulats est le complément massique.
///
/// Panique si `cement_content <= 0`, si `water_content < 0`, ou si la masse de
/// granulats déduite `total_mass_density − cement_content − water_content` est
/// négative (masse volumique du béton frais incohérente avec les dosages).
pub fn mix_aggregate_cement_ratio(
    total_mass_density: f64,
    cement_content: f64,
    water_content: f64,
) -> f64 {
    assert!(
        cement_content > 0.0,
        "le dosage en ciment C doit être strictement positif"
    );
    assert!(
        water_content >= 0.0,
        "le dosage en eau W doit être positif ou nul"
    );
    let aggregate_mass = total_mass_density - cement_content - water_content;
    assert!(
        aggregate_mass >= 0.0,
        "la masse de granulats déduite (ρ − C − W) doit être positive ou nulle"
    );
    aggregate_mass / cement_content
}

/// Volume rendu du mélange par la méthode des volumes absolus
/// `V = C/ρc + W/ρw + G/ρg` (m³), somme des volumes absolus du ciment, de
/// l'eau et des granulats.
///
/// Panique si l'une des masses `cement_mass`, `water_mass`, `aggregate_mass`
/// est négative, ou si l'une des masses volumiques `cement_density`,
/// `water_density`, `aggregate_density` n'est pas strictement positive.
pub fn mix_yield_volume(
    cement_mass: f64,
    water_mass: f64,
    aggregate_mass: f64,
    cement_density: f64,
    water_density: f64,
    aggregate_density: f64,
) -> f64 {
    assert!(
        cement_mass >= 0.0,
        "la masse de ciment doit être positive ou nulle"
    );
    assert!(
        water_mass >= 0.0,
        "la masse d'eau doit être positive ou nulle"
    );
    assert!(
        aggregate_mass >= 0.0,
        "la masse de granulats doit être positive ou nulle"
    );
    assert!(
        cement_density > 0.0,
        "la masse volumique du ciment ρc doit être strictement positive"
    );
    assert!(
        water_density > 0.0,
        "la masse volumique de l'eau ρw doit être strictement positive"
    );
    assert!(
        aggregate_density > 0.0,
        "la masse volumique des granulats ρg doit être strictement positive"
    );
    cement_mass / cement_density + water_mass / water_density + aggregate_mass / aggregate_density
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn target_mean_strength_is_additive() {
        // fcm = fck + Δf : additivité et cas dégénéré marge nulle.
        assert_relative_eq!(
            mix_target_mean_strength(25.0, 0.0),
            25.0,
            max_relative = 1e-12
        );
        assert_relative_eq!(
            mix_target_mean_strength(25.0, 8.0),
            33.0,
            max_relative = 1e-12
        );
    }

    #[test]
    fn water_cement_ratio_decreases_with_strength() {
        // w/c = a/(fcm + b) est strictement décroissant avec fcm.
        let a = 27.0;
        let b = 3.0;
        let low = mix_water_cement_from_strength(a, b, 20.0);
        let high = mix_water_cement_from_strength(a, b, 40.0);
        assert!(
            high < low,
            "un béton plus résistant demande un E/C plus faible"
        );
        // Cas chiffré : a = 27, b = 3, fcm = 33.
        // w/c = 27/(33 + 3) = 27/36 = 0,75.
        assert_relative_eq!(
            mix_water_cement_from_strength(27.0, 3.0, 33.0),
            0.75,
            max_relative = 1e-12
        );
    }

    #[test]
    fn cement_content_reciprocity() {
        // C = W/(w/c) puis W = C·(w/c) : réciprocité du dosage.
        let water = 180.0;
        let wc = 0.5;
        let cement = mix_cement_content(water, wc);
        // C = 180/0,5 = 360 kg/m³.
        assert_relative_eq!(cement, 360.0, max_relative = 1e-12);
        assert_relative_eq!(cement * wc, water, max_relative = 1e-12);
    }

    #[test]
    fn aggregate_ratio_complement() {
        // g/c = (ρ − C − W)/C. Avec ρ = 2400, C = 360, W = 180 :
        // granulats = 2400 − 360 − 180 = 1860 kg/m³ ;
        // g/c = 1860/360 = 5,166667.
        let gc = mix_aggregate_cement_ratio(2400.0, 360.0, 180.0);
        assert_relative_eq!(gc, 5.166667, max_relative = 1e-3);
        // Reconstitution : ρ = C·(1 + g/c) + W.
        let rho = 360.0 * (1.0 + gc) + 180.0;
        assert_relative_eq!(rho, 2400.0, max_relative = 1e-12);
    }

    #[test]
    fn yield_volume_worked_case() {
        // Méthode des volumes absolus, 1 m³ de béton.
        // Ciment 360 kg à ρc = 3150 kg/m³ : 360/3150 = 0,114286 m³.
        // Eau    180 kg à ρw = 1000 kg/m³ : 180/1000 = 0,180000 m³.
        // Granulats 1860 kg à ρg = 2650 kg/m³ : 1860/2650 = 0,701887 m³.
        // V = 0,114286 + 0,180000 + 0,701887 = 0,996172 m³.
        let v = mix_yield_volume(360.0, 180.0, 1860.0, 3150.0, 1000.0, 2650.0);
        assert_relative_eq!(v, 0.996172, max_relative = 1e-3);
        // Volume nul si toutes les masses sont nulles.
        assert_relative_eq!(
            mix_yield_volume(0.0, 0.0, 0.0, 3150.0, 1000.0, 2650.0),
            0.0,
            max_relative = 1e-12
        );
    }

    #[test]
    #[should_panic(expected = "le rapport eau/ciment w/c doit être strictement positif")]
    fn zero_water_cement_ratio_panics() {
        let _ = mix_cement_content(180.0, 0.0);
    }
}
