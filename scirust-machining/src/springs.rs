//! Ressorts hélicoïdaux de compression à fil rond (EN 13906-1) — raideur,
//! flèche et contrainte de cisaillement corrigée (facteur de Wahl).
//!
//! Un ressort est décrit par son diamètre de fil `d` (mm), son diamètre moyen
//! d'enroulement `D` (mm), son nombre de spires actives `n` et le module de
//! cisaillement `G` (MPa) du matériau. L'indice du ressort `C = D/d` gouverne
//! la concentration de contrainte.
//!
//! ```text
//! raideur         k = G·d⁴ / (8·D³·n)              (N/mm)
//! flèche          s = F / k                         (mm)
//! cisaillement    τ = 8·F·D / (π·d³)                (MPa)
//! facteur de Wahl Kw = (4C−1)/(4C−4) + 0,615/C
//! τ corrigé       τ_c = Kw · τ
//! ```
//!
//! Le facteur de Wahl `Kw` majore la contrainte nominale pour tenir compte de
//! la courbure du fil et du cisaillement direct côté intérieur de la spire, où
//! la rupture s'amorce.
//!
//! **Limite honnête** : modèle statique linéaire du fil rond. `G` est une
//! donnée matériau fournie par l'appelant. Ce module ne couvre ni le flambage
//! du ressort élancé, ni la fréquence propre / résonance en service dynamique,
//! ni la tenue en fatigue (diagramme de Goodman/Haigh) — calculs distincts à
//! mener avec les données du matériau et du chargement.

use core::f64::consts::PI;

/// Ressort hélicoïdal de compression à fil rond.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HelicalSpring {
    /// Diamètre de fil `d` (mm).
    pub wire_diameter_mm: f64,
    /// Diamètre moyen d'enroulement `D` (mm).
    pub mean_diameter_mm: f64,
    /// Nombre de spires actives `n`.
    pub active_coils: f64,
    /// Module de cisaillement `G` (MPa) — ~81 500 pour l'acier à ressort.
    pub shear_modulus_mpa: f64,
}

impl HelicalSpring {
    /// Indice du ressort `C = D/d` (sans dimension).
    ///
    /// Panique si `wire_diameter_mm <= 0`.
    pub fn spring_index(&self) -> f64 {
        assert!(
            self.wire_diameter_mm > 0.0,
            "le diamètre de fil doit être strictement positif"
        );
        self.mean_diameter_mm / self.wire_diameter_mm
    }

    /// Raideur `k = G·d⁴ / (8·D³·n)` (N/mm).
    ///
    /// Panique si `active_coils <= 0` ou `mean_diameter_mm <= 0`.
    pub fn rate_n_per_mm(&self) -> f64 {
        assert!(
            self.active_coils > 0.0 && self.mean_diameter_mm > 0.0,
            "spires actives et diamètre moyen doivent être strictement positifs"
        );
        let d = self.wire_diameter_mm;
        let dm = self.mean_diameter_mm;
        self.shear_modulus_mpa * d.powi(4) / (8.0 * dm.powi(3) * self.active_coils)
    }

    /// Flèche `s = F / k` (mm) sous un effort `force` (N).
    pub fn deflection_mm(&self, force_n: f64) -> f64 {
        force_n / self.rate_n_per_mm()
    }

    /// Contrainte de cisaillement nominale `τ = 8·F·D / (π·d³)` (MPa) sous un
    /// effort `force` (N), sans correction.
    pub fn shear_stress_mpa(&self, force_n: f64) -> f64 {
        8.0 * force_n * self.mean_diameter_mm / (PI * self.wire_diameter_mm.powi(3))
    }

    /// Facteur de Wahl `Kw = (4C−1)/(4C−4) + 0,615/C`.
    pub fn wahl_factor(&self) -> f64 {
        let c = self.spring_index();
        (4.0 * c - 1.0) / (4.0 * c - 4.0) + 0.615 / c
    }

    /// Contrainte de cisaillement corrigée `τ_c = Kw · τ` (MPa) sous un effort
    /// `force` (N) — la valeur dimensionnante.
    pub fn corrected_shear_stress_mpa(&self, force_n: f64) -> f64 {
        self.wahl_factor() * self.shear_stress_mpa(force_n)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn spring() -> HelicalSpring {
        // d=4, D=30, n=6, acier G=81500 MPa.
        HelicalSpring {
            wire_diameter_mm: 4.0,
            mean_diameter_mm: 30.0,
            active_coils: 6.0,
            shear_modulus_mpa: 81_500.0,
        }
    }

    #[test]
    fn spring_index_is_diameter_ratio() {
        // C = 30/4 = 7,5.
        assert_relative_eq!(spring().spring_index(), 7.5, epsilon = 1e-12);
    }

    #[test]
    fn rate_matches_the_en13906_formula() {
        // k = 81500·256/(8·27000·6) ≈ 16,10 N/mm.
        assert_relative_eq!(spring().rate_n_per_mm(), 16.0988, epsilon = 1e-3);
    }

    #[test]
    fn deflection_is_force_over_rate() {
        // F=100 N → s = 100/16,0988 ≈ 6,213 mm.
        let s = spring().deflection_mm(100.0);
        assert_relative_eq!(s, 100.0 / spring().rate_n_per_mm(), epsilon = 1e-12);
        assert_relative_eq!(s, 6.2116, epsilon = 1e-3);
    }

    #[test]
    fn nominal_shear_stress_matches_the_formula() {
        // τ = 8·100·30/(π·64) ≈ 119,37 MPa.
        assert_relative_eq!(spring().shear_stress_mpa(100.0), 119.366, epsilon = 1e-2);
    }

    #[test]
    fn wahl_factor_exceeds_one_and_corrects_upward() {
        // Kw = 29/26 + 0,615/7,5 ≈ 1,1974 > 1.
        let s = spring();
        assert_relative_eq!(s.wahl_factor(), 1.1974, epsilon = 1e-3);
        // La contrainte corrigée est bien supérieure à la nominale.
        assert!(s.corrected_shear_stress_mpa(100.0) > s.shear_stress_mpa(100.0));
        assert_relative_eq!(
            s.corrected_shear_stress_mpa(100.0),
            s.wahl_factor() * s.shear_stress_mpa(100.0),
            epsilon = 1e-9
        );
    }

    #[test]
    #[should_panic(expected = "spires actives")]
    fn zero_active_coils_panics() {
        let mut s = spring();
        s.active_coils = 0.0;
        s.rate_n_per_mm();
    }
}
