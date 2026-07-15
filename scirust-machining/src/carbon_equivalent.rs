//! Soudabilité — **équivalent carbone** d'un acier au carbone-manganèse par la
//! formule de l'**IIW** (Institut international de la soudure) et classement du
//! risque de fissuration à froid associé.
//!
//! ```text
//! équivalent carbone (IIW)
//!   CE = C + Mn/6 + (Cr + Mo + V)/5 + (Ni + Cu)/15
//!
//! classes de soudabilité
//!   CE < 0,40          → Bonne     (préchauffage rarement nécessaire)
//!   0,40 ≤ CE ≤ 0,60   → Moyenne   (préchauffage souvent requis)
//!   CE > 0,60          → Difficile (préchauffage et précautions marqués)
//! ```
//!
//! `CE` équivalent carbone (adimensionnel, exprimé comme une teneur en % massique
//! équivalente de carbone) ; `C`, `Mn`, `Cr`, `Mo`, `V`, `Ni`, `Cu` teneurs
//! massiques de l'analyse chimique, en **pour-cent massique** (`%`, p. ex. `0,18`
//! pour 0,18 % de carbone).
//!
//! **Convention** : entrées en % massique, sortie adimensionnelle homogène à un %
//! massique de carbone. **Limite honnête** : la formule IIW est **empirique** et
//! s'applique aux aciers au **carbone-manganèse** ; les teneurs proviennent de
//! l'**analyse de coulée ou produit fournie par l'appelant** — aucune valeur
//! « par défaut » n'est inventée. L'équivalent carbone est un **indicateur de
//! risque de fissuration à froid** (hydrogène), pas une garantie de soudabilité :
//! il ne remplace ni le calcul d'énergie de soudage, ni la température de
//! préchauffage, ni les essais de qualification.

/// Borne haute de la classe « Bonne » (et borne basse de « Moyenne »).
pub const CE_THRESHOLD_GOOD: f64 = 0.40;

/// Borne haute de la classe « Moyenne » (et borne basse de « Difficile »).
pub const CE_THRESHOLD_DIFFICULT: f64 = 0.60;

/// Classe de soudabilité déduite de l'équivalent carbone IIW.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WeldabilityClass {
    /// `CE < 0,40` : bonne soudabilité, préchauffage rarement nécessaire.
    Good,
    /// `0,40 ≤ CE ≤ 0,60` : soudabilité moyenne, préchauffage souvent requis.
    Moderate,
    /// `CE > 0,60` : soudabilité difficile, préchauffage et précautions marqués.
    Difficult,
}

/// Équivalent carbone (IIW) `CE = C + Mn/6 + (Cr+Mo+V)/5 + (Ni+Cu)/15`.
///
/// Teneurs `c`, `mn`, `cr`, `mo`, `v`, `ni`, `cu` en **% massique**. Résultat
/// adimensionnel, homogène à une teneur en carbone en % massique.
///
/// Panique si l'une des teneurs est négative.
pub fn carbon_equivalent_iiw(c: f64, mn: f64, cr: f64, mo: f64, v: f64, ni: f64, cu: f64) -> f64 {
    assert!(c >= 0.0, "la teneur en carbone doit être positive");
    assert!(mn >= 0.0, "la teneur en manganèse doit être positive");
    assert!(cr >= 0.0, "la teneur en chrome doit être positive");
    assert!(mo >= 0.0, "la teneur en molybdène doit être positive");
    assert!(v >= 0.0, "la teneur en vanadium doit être positive");
    assert!(ni >= 0.0, "la teneur en nickel doit être positive");
    assert!(cu >= 0.0, "la teneur en cuivre doit être positive");
    c + mn / 6.0 + (cr + mo + v) / 5.0 + (ni + cu) / 15.0
}

/// Classe de soudabilité selon l'équivalent carbone `ce` (voir [`WeldabilityClass`]).
///
/// `CE < 0,40` → `Good` ; `0,40 ≤ CE ≤ 0,60` → `Moderate` ; `CE > 0,60`
/// → `Difficult`.
///
/// Panique si `ce` est négatif ou non fini.
pub fn ce_weldability_class(ce: f64) -> WeldabilityClass {
    assert!(
        ce.is_finite(),
        "l'équivalent carbone doit être un nombre fini"
    );
    assert!(ce >= 0.0, "l'équivalent carbone doit être positif");
    if ce < CE_THRESHOLD_GOOD
    {
        WeldabilityClass::Good
    }
    else if ce <= CE_THRESHOLD_DIFFICULT
    {
        WeldabilityClass::Moderate
    }
    else
    {
        WeldabilityClass::Difficult
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn pure_carbon_equals_itself() {
        // Sans éléments d'alliage, CE se réduit exactement à la teneur en carbone.
        assert_relative_eq!(
            carbon_equivalent_iiw(0.30, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            0.30,
            epsilon = 1e-12
        );
    }

    #[test]
    fn manganese_term_is_one_sixth() {
        // Le manganèse seul contribue Mn/6 : 1,20 % → 0,20.
        assert_relative_eq!(
            carbon_equivalent_iiw(0.0, 1.20, 0.0, 0.0, 0.0, 0.0, 0.0),
            1.20 / 6.0,
            epsilon = 1e-12
        );
    }

    #[test]
    fn grouped_terms_share_their_divisor() {
        // Cr+Mo+V sont divisés par 5 : le triplet équivaut à leur somme groupée.
        let split = carbon_equivalent_iiw(0.0, 0.0, 0.10, 0.20, 0.15, 0.0, 0.0);
        let lumped = carbon_equivalent_iiw(0.0, 0.0, 0.45, 0.0, 0.0, 0.0, 0.0);
        assert_relative_eq!(split, lumped, epsilon = 1e-12);
        // Ni+Cu sont divisés par 15 : même identité.
        let split_15 = carbon_equivalent_iiw(0.0, 0.0, 0.0, 0.0, 0.0, 0.30, 0.20);
        let lumped_15 = carbon_equivalent_iiw(0.0, 0.0, 0.0, 0.0, 0.0, 0.50, 0.0);
        assert_relative_eq!(split_15, lumped_15, epsilon = 1e-12);
    }

    #[test]
    fn realistic_cmn_steel() {
        // Acier C-Mn courant : C 0,18 ; Mn 1,20 ; Cr 0,10 ; Mo 0,02 ; V 0 ;
        // Ni 0,05 ; Cu 0,15.
        // CE = 0,18 + 1,20/6 + (0,10+0,02+0)/5 + (0,05+0,15)/15
        //    = 0,18 + 0,20 + 0,024 + 0,0133333… = 0,4173333…
        let ce = carbon_equivalent_iiw(0.18, 1.20, 0.10, 0.02, 0.0, 0.05, 0.15);
        assert_relative_eq!(ce, 0.417_333_333_333_333, epsilon = 1e-9);
        // Juste au-dessus de 0,40 → soudabilité moyenne.
        assert_eq!(ce_weldability_class(ce), WeldabilityClass::Moderate);
    }

    #[test]
    fn class_boundaries_are_inclusive_for_moderate() {
        // Bornes exactes : 0,40 et 0,60 appartiennent à la classe « Moyenne ».
        assert_eq!(ce_weldability_class(0.39), WeldabilityClass::Good);
        assert_eq!(
            ce_weldability_class(CE_THRESHOLD_GOOD),
            WeldabilityClass::Moderate
        );
        assert_eq!(ce_weldability_class(0.50), WeldabilityClass::Moderate);
        assert_eq!(
            ce_weldability_class(CE_THRESHOLD_DIFFICULT),
            WeldabilityClass::Moderate
        );
        assert_eq!(ce_weldability_class(0.65), WeldabilityClass::Difficult);
    }

    #[test]
    #[should_panic(expected = "teneur en carbone")]
    fn negative_carbon_panics() {
        carbon_equivalent_iiw(-0.05, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    }
}
