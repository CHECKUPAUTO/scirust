//! Filetages métriques ISO — géométrie du profil triangulaire à 60° (ISO 68-1)
//! et grandeurs de calcul de la vis (diamètres primitif/noyau, section
//! résistante ISO 898-1, pas et hauteur d'hélice).
//!
//! Le profil de base a une hauteur de triangle fondamental
//! `H = (√3/2)·P` (P = pas). Les diamètres caractéristiques d'un filetage
//! nominal `d` (diamètre extérieur, mm) en découlent :
//!
//! ```text
//! d2 = d − 0,6495·P     (diamètre primitif = d − 3/4·H)
//! d1 = d − 1,0825·P     (diamètre sur flancs / noyau théorique = d − 5/4·H)
//! d3 = d − 1,2269·P     (diamètre du noyau de la vis, fond arrondi)
//! ```
//!
//! La **section résistante** `As` (ISO 898-1), utilisée pour dimensionner une
//! vis en traction, prend le diamètre moyen entre primitif et noyau :
//!
//! ```text
//! As = (π/4) · ((d2 + d3)/2)²
//! ```
//!
//! **Limite honnête** : ce module donne la **géométrie de base** et la section
//! résistante normalisées. Il ne modélise pas les classes de tolérance de
//! filetage (6H/6g, ISO 965), ni la répartition réelle de charge entre filets,
//! ni le desserrage/serrage (couple de serrage, coefficient de frottement) —
//! calculs distincts que l'appelant mène avec ses propres données.

use core::f64::consts::PI;

/// Filetage métrique ISO, défini par son diamètre nominal et son pas.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MetricThread {
    /// Diamètre nominal `d` (extérieur, mm) — le « 10 » d'un M10.
    pub major_diameter_mm: f64,
    /// Pas `P` (mm) — le « 1,5 » d'un M10×1,5.
    pub pitch_mm: f64,
}

impl MetricThread {
    /// Hauteur du triangle fondamental `H = (√3/2)·P` (mm).
    pub fn fundamental_height(&self) -> f64 {
        3f64.sqrt() / 2.0 * self.pitch_mm
    }

    /// Diamètre primitif `d2 = d − 0,6495·P` (mm).
    pub fn pitch_diameter(&self) -> f64 {
        self.major_diameter_mm - 0.649_519_052_838_329 * self.pitch_mm
    }

    /// Diamètre sur flancs / noyau théorique `d1 = d − 1,0825·P` (mm).
    pub fn minor_diameter(&self) -> f64 {
        self.major_diameter_mm - 1.082_531_754_730_548 * self.pitch_mm
    }

    /// Diamètre du noyau de la vis `d3 = d − 1,2269·P` (mm, fond arrondi).
    pub fn root_diameter(&self) -> f64 {
        self.major_diameter_mm - 1.226_869_322_150_637 * self.pitch_mm
    }

    /// Section résistante `As = (π/4)·((d2 + d3)/2)²` (mm², ISO 898-1).
    pub fn tensile_stress_area(&self) -> f64 {
        let dm = (self.pitch_diameter() + self.root_diameter()) / 2.0;
        PI / 4.0 * dm * dm
    }

    /// Pas de l'hélice `Ph = P·n` (mm) pour `starts` filets (1 = filet simple).
    ///
    /// Panique si `starts == 0`.
    pub fn lead(&self, starts: u32) -> f64 {
        assert!(starts > 0, "un filetage a au moins un filet");
        self.pitch_mm * starts as f64
    }

    /// Angle d'hélice au primitif `ψ` (degrés) pour `starts` filets :
    /// `tan ψ = Ph / (π·d2)`.
    ///
    /// Panique si `starts == 0`.
    pub fn helix_angle_deg(&self, starts: u32) -> f64 {
        let lead = self.lead(starts);
        (lead / (PI * self.pitch_diameter())).atan().to_degrees()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn m10() -> MetricThread {
        // M10×1,5 (pas gros standard).
        MetricThread {
            major_diameter_mm: 10.0,
            pitch_mm: 1.5,
        }
    }

    #[test]
    fn m10_characteristic_diameters_match_the_standard() {
        let t = m10();
        // Valeurs tabulées ISO 261/724.
        assert_relative_eq!(t.pitch_diameter(), 9.026, epsilon = 1e-3);
        assert_relative_eq!(t.minor_diameter(), 8.376, epsilon = 1e-3);
        assert_relative_eq!(t.root_diameter(), 8.160, epsilon = 1e-3);
    }

    #[test]
    fn m10_tensile_stress_area_is_58_mm2() {
        // As normalisée d'un M10 : 58,0 mm².
        assert_relative_eq!(m10().tensile_stress_area(), 58.0, epsilon = 0.1);
    }

    #[test]
    fn fundamental_height_follows_the_60_degree_profile() {
        // H = (√3/2)·P = 0,86603·1,5 ≈ 1,299 mm.
        assert_relative_eq!(
            m10().fundamental_height(),
            3f64.sqrt() / 2.0 * 1.5,
            epsilon = 1e-12
        );
    }

    #[test]
    fn lead_multiplies_pitch_by_starts() {
        // Filet double : Ph = 2·P = 3 mm.
        assert_relative_eq!(m10().lead(2), 3.0, epsilon = 1e-12);
    }

    #[test]
    fn helix_angle_is_small_for_a_single_start() {
        // tan ψ = P/(π·d2) = 1,5/(π·9,026) ≈ 0,0529 → ψ ≈ 3,03°.
        assert_relative_eq!(m10().helix_angle_deg(1), 3.028, epsilon = 1e-2);
    }

    #[test]
    #[should_panic(expected = "au moins un filet")]
    fn zero_starts_panics() {
        m10().lead(0);
    }
}
