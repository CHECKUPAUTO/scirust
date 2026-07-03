//! Protection de distance (relais d'impédance, caractéristique mho) —
//! IEEE C37.113-2015 (*Guide for Protective Relay Applications to
//! Transmission Lines*) §5.2 pour la caractéristique mho, §6 pour la
//! coordination des zones.
//!
//! Le relais mesure l'impédance apparente vue depuis son point
//! d'installation, `Z = V/I` (phaseurs tension/courant à fréquence
//! fondamentale), et déclenche si cette impédance tombe dans une des
//! zones réglées. Chaque zone est un cercle mho passant par l'origine
//! (l'emplacement du relais) dont le diamètre va jusqu'à un point de
//! portée `reach` (Ω, sur l'angle caractéristique de la ligne protégée) —
//! la caractéristique directionnelle standard qui ne déclenche que pour
//! des défauts *devant* le relais (IEEE C37.113 §5.2, éq. du comparateur
//! `Re[(Z_reach - Z)·conj(Z)] ≥ 0`, équivalente à « Z est dans le cercle
//! de diamètre [0, Z_reach] »).
//!
//! ## Zones et temporisation (pratique industrielle usuelle)
//! - **Zone 1** : portée ~80-90% de la ligne, **instantanée** (pas de
//!   retard intentionnel) — ne couvre pas 100% pour ne jamais déclencher
//!   sur un défaut de la ligne suivante (marge d'erreur de mesure/CT/PT).
//! - **Zone 2** : portée ~120-150% (couvre le reste de la ligne + marge),
//!   retardée ~0.2-0.4 s pour laisser Zone 1 de la ligne suivante agir
//!   en premier.
//! - **Zone 3** : portée ~150-250% (secours à distance), retardée
//!   ~0.6-1.0 s.
//!
//! Ces pourcentages/retards sont des **pratiques de réglage courantes**,
//! pas des constantes physiques : ce module ne les code pas en dur, il
//! laisse l'appelant fournir ses propres `RelayZone` (portée complexe +
//! retard) — le rôle du code est la géométrie du comparateur mho et la
//! sélection de la zone la plus rapide qui s'applique, pas le réglage.
//!
//! ## Hypothèse de zones emboîtées
//! [`DistanceRelay::evaluate`] renvoie la **première** zone (dans l'ordre
//! de la liste) dont le cercle contient l'impédance mesurée. Pour un jeu
//! de zones standard (même angle caractéristique, portées croissantes),
//! les cercles sont emboîtés (Zone 1 ⊂ Zone 2 ⊂ Zone 3) et lister les
//! zones de la plus rapide à la plus lente donne bien le temps de
//! déclenchement le plus court applicable. Si cette hypothèse ne tient
//! pas (angles différents entre zones), chaque zone est quand même
//! évaluée indépendamment dans l'ordre de la liste — mettez la zone la
//! plus contraignante en premier.

use scirust_signal::Complex;

/// Division complexe `a / b` (non fournie par [`scirust_signal::Complex`],
/// qui ne l'utilise pas ailleurs — implémentée localement plutôt que
/// d'étendre un type partagé par d'autres crates pour un seul usage).
fn complex_div(a: Complex, b: Complex) -> Complex {
    let denom = b.mag_sq();
    Complex::new(
        (a.re * b.re + a.im * b.im) / denom,
        (a.im * b.re - a.re * b.im) / denom,
    )
}

/// Impédance apparente `Z = V/I`. `None` si le courant mesuré est
/// (quasi) nul — impédance non définie, aucune zone ne peut être évaluée
/// (physiquement : pas de courant de défaut, pas de décision à prendre).
pub fn apparent_impedance(voltage: Complex, current: Complex) -> Option<Complex> {
    if current.mag_sq() < 1e-18
    {
        return None;
    }
    Some(complex_div(voltage, current))
}

/// Le comparateur mho : `true` si `z_measured` est à l'intérieur du
/// cercle de diamètre `[0, reach]` (IEEE C37.113 §5.2).
pub fn mho_operates(z_measured: Complex, reach: Complex) -> bool {
    let diff = reach - z_measured;
    (diff * z_measured.conj()).re >= 0.0
}

/// Une zone de protection de distance : portée (Ω, phaseur complexe sur
/// l'angle caractéristique de la ligne) et son retard de déclenchement
/// intentionnel.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RelayZone {
    pub reach: Complex,
    pub delay_s: f64,
}

/// Décision de déclenchement d'un relais de distance.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TripDecision {
    /// Index de zone (0-based ; "Zone `zone_index+1`" dans la convention usuelle).
    pub zone_index: usize,
    pub delay_s: f64,
}

/// Un relais de distance multi-zones.
#[derive(Debug, Clone, PartialEq)]
pub struct DistanceRelay {
    /// Zones dans l'ordre d'évaluation — la plus rapide (Zone 1) en premier.
    pub zones: Vec<RelayZone>,
}

impl DistanceRelay {
    pub fn new(zones: Vec<RelayZone>) -> Self {
        Self { zones }
    }

    /// Évalue l'impédance apparente `V/I` contre chaque zone dans l'ordre
    /// et renvoie la première qui opère (voir la note d'emboîtement en
    /// tête de module). `None` si aucune zone n'opère, ou si le courant
    /// est nul (pas de défaut mesurable).
    pub fn evaluate(&self, voltage: Complex, current: Complex) -> Option<TripDecision> {
        let z = apparent_impedance(voltage, current)?;
        for (i, zone) in self.zones.iter().enumerate()
        {
            if mho_operates(z, zone.reach)
            {
                return Some(TripDecision {
                    zone_index: i,
                    delay_s: zone.delay_s,
                });
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn apparent_impedance_matches_ohms_law() {
        // V = 100∠0°, I = 10∠-30° -> Z = 10∠30° = 8.660+5.0j.
        let v = Complex::new(100.0, 0.0);
        let i = Complex::cis((-30.0_f64).to_radians()) * 10.0;
        let z = apparent_impedance(v, i).unwrap();
        assert_relative_eq!(z.re, 10.0 * 30.0_f64.to_radians().cos(), epsilon = 1e-9);
        assert_relative_eq!(z.im, 10.0 * 30.0_f64.to_radians().sin(), epsilon = 1e-9);
    }

    #[test]
    fn zero_current_yields_no_impedance() {
        let v = Complex::new(100.0, 0.0);
        let i = Complex::zero();
        assert!(apparent_impedance(v, i).is_none());
    }

    #[test]
    fn mho_circle_membership_matches_geometric_construction() {
        // Reach purely resistive: 10+0j -> circle centered (5,0), radius 5.
        let reach = Complex::new(10.0, 0.0);
        assert!(mho_operates(Complex::new(5.0, 0.0), reach)); // center: inside
        assert!(!mho_operates(Complex::new(15.0, 0.0), reach)); // beyond reach: outside
        assert!(!mho_operates(Complex::new(0.0, 5.0), reach)); // orthogonal: outside
        // On the boundary (5,5): distance from center (5,0) is exactly 5.
        let diff = reach - Complex::new(5.0, 5.0);
        let boundary_test = (diff * Complex::new(5.0, 5.0).conj()).re;
        assert_relative_eq!(boundary_test, 0.0, epsilon = 1e-9);
    }

    #[test]
    fn evaluate_picks_the_fastest_applicable_zone() {
        // Common characteristic angle (80°, typical for a HV line), reach 5/12/20 Ω.
        let angle = 80.0_f64.to_radians();
        let reach = |ohms: f64| Complex::cis(angle) * ohms;
        let relay = DistanceRelay::new(vec![
            RelayZone {
                reach: reach(5.0),
                delay_s: 0.0,
            },
            RelayZone {
                reach: reach(12.0),
                delay_s: 0.3,
            },
            RelayZone {
                reach: reach(20.0),
                delay_s: 0.8,
            },
        ]);

        // A fault at 4 Ω along the line angle: inside Zone 1 -> instantaneous.
        let v_close = reach(4.0) * Complex::new(1.0, 0.0); // V = Z*I with I=1∠0°
        let i_unit = Complex::new(1.0, 0.0);
        let d1 = relay.evaluate(v_close, i_unit).unwrap();
        assert_eq!(d1.zone_index, 0);
        assert_relative_eq!(d1.delay_s, 0.0);

        // A fault at 10 Ω: outside Zone 1, inside Zone 2 -> 0.3s.
        let v_mid = reach(10.0) * Complex::new(1.0, 0.0);
        let d2 = relay.evaluate(v_mid, i_unit).unwrap();
        assert_eq!(d2.zone_index, 1);
        assert_relative_eq!(d2.delay_s, 0.3);

        // A fault at 30 Ω: beyond all zones -> no trip.
        let v_far = reach(30.0) * Complex::new(1.0, 0.0);
        assert!(relay.evaluate(v_far, i_unit).is_none());
    }

    #[test]
    fn zero_current_never_trips_any_zone() {
        let relay = DistanceRelay::new(vec![RelayZone {
            reach: Complex::new(10.0, 10.0),
            delay_s: 0.0,
        }]);
        assert!(
            relay
                .evaluate(Complex::new(100.0, 0.0), Complex::zero())
                .is_none()
        );
    }
}
