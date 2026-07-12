//! Torseurs — outil de la mécanique du solide (statique et cinématique).
//! Un torseur `{T}` réduit en un point `P` est le couple (résultante `R`,
//! moment `M_P`) d'un champ de vecteurs ; il modélise aussi bien une action
//! mécanique (torseur d'action : résultante des forces + moment) qu'un mouvement
//! (torseur cinématique : `Ω` + vitesse `V`).
//!
//! **Transport du moment** (formule fondamentale, en un point `Q`) :
//!
//! ```text
//! M_Q = M_P + QP ∧ R = M_P + (P − Q) ∧ R
//! ```
//!
//! **Invariants** :
//! - la résultante `R` ;
//! - l'**automoment** (invariant scalaire) `I = R · M`, indépendant du point ;
//! - le **pas** `p = (R·M) / ‖R‖²` (le long de l'axe central).
//!
//! **Cas particuliers** : un torseur est un **couple** si `R = 0` (moment
//! uniforme), un **glisseur** (résultante pure sur une droite) si `R ≠ 0` et
//! `R·M = 0`. L'**axe central** est le lieu où le moment est colinéaire à `R`.
//!
//! La **comoment** de deux torseurs `{T1}` et `{T2}` réduits au même point vaut
//! `R1·M2 + R2·M1` ; c'est un invariant, base du calcul de puissance (comoment
//! du torseur d'action et du torseur cinématique).
//!
//! **Convention** : vecteurs 3D `[x, y, z]` en unités cohérentes de l'appelant
//! (par ex. forces en N, moments en N·m, points en m). Le module est purement
//! géométrique et n'impose aucune unité.

/// Produit scalaire `a · b`.
pub fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Produit vectoriel `a ∧ b`.
pub fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn add(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}
fn scale(a: [f64; 3], k: f64) -> [f64; 3] {
    [a[0] * k, a[1] * k, a[2] * k]
}
fn norm2(a: [f64; 3]) -> f64 {
    dot(a, a)
}

/// Torseur réduit en un point : résultante `resultant`, moment `moment` exprimé
/// au point de réduction `point`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Torseur {
    /// Résultante `R`.
    pub resultant: [f64; 3],
    /// Moment `M_P` au point de réduction.
    pub moment: [f64; 3],
    /// Point de réduction `P`.
    pub point: [f64; 3],
}

impl Torseur {
    /// Construit un torseur à partir de ses éléments de réduction.
    pub fn new(resultant: [f64; 3], moment: [f64; 3], point: [f64; 3]) -> Self {
        Torseur {
            resultant,
            moment,
            point,
        }
    }

    /// **Glisseur** : force `force` s'appliquant au point `application_point`
    /// (moment nul en ce point).
    pub fn force(force: [f64; 3], application_point: [f64; 3]) -> Self {
        Torseur {
            resultant: force,
            moment: [0.0; 3],
            point: application_point,
        }
    }

    /// **Couple** pur : résultante nulle, moment `moment` (uniforme partout).
    pub fn couple(moment: [f64; 3]) -> Self {
        Torseur {
            resultant: [0.0; 3],
            moment,
            point: [0.0; 3],
        }
    }

    /// Moment du torseur exprimé en un point `q` : `M_Q = M_P + (P − Q) ∧ R`.
    pub fn moment_at(&self, q: [f64; 3]) -> [f64; 3] {
        add(self.moment, cross(sub(self.point, q), self.resultant))
    }

    /// Même torseur, réduit (transporté) au point `q`.
    pub fn transport(&self, q: [f64; 3]) -> Torseur {
        Torseur {
            resultant: self.resultant,
            moment: self.moment_at(q),
            point: q,
        }
    }

    /// Somme de deux torseurs (torseur résultant), réduite au point de `self`.
    pub fn add_torseur(&self, other: &Torseur) -> Torseur {
        let m_other = other.moment_at(self.point);
        Torseur {
            resultant: add(self.resultant, other.resultant),
            moment: add(self.moment, m_other),
            point: self.point,
        }
    }

    /// Automoment (invariant scalaire) `I = R · M`, indépendant du point.
    pub fn automoment(&self) -> f64 {
        dot(self.resultant, self.moment)
    }

    /// `true` si le torseur est un **couple** (`R = 0` à la tolérance `eps`).
    pub fn is_couple(&self, eps: f64) -> bool {
        norm2(self.resultant).sqrt() <= eps
    }

    /// `true` si le torseur est un **glisseur** (`R ≠ 0` et `R·M = 0`).
    pub fn is_glisseur(&self, eps: f64) -> bool {
        !self.is_couple(eps) && self.automoment().abs() <= eps
    }

    /// Pas du torseur `p = (R·M) / ‖R‖²` (le long de l'axe central).
    ///
    /// Panique si `R = 0` (pas indéfini pour un couple).
    pub fn pitch(&self) -> f64 {
        let r2 = norm2(self.resultant);
        assert!(r2 > 0.0, "le pas n'est pas défini pour un couple (R = 0)");
        self.automoment() / r2
    }

    /// Un point de l'**axe central** : `P0 = P + (R ∧ M_P) / ‖R‖²`.
    /// Sur cet axe, le moment est minimal et colinéaire à `R`.
    ///
    /// Panique si `R = 0`.
    pub fn central_axis_point(&self) -> [f64; 3] {
        let r2 = norm2(self.resultant);
        assert!(
            r2 > 0.0,
            "l'axe central n'est pas défini pour un couple (R = 0)"
        );
        add(
            self.point,
            scale(cross(self.resultant, self.moment), 1.0 / r2),
        )
    }

    /// Comoment avec un autre torseur, réduits au même point :
    /// `R1·M2 + R2·M1`. Invariant ; sert au calcul de puissance.
    pub fn comoment(&self, other: &Torseur) -> f64 {
        let m_other = other.moment_at(self.point);
        dot(self.resultant, m_other) + dot(other.resultant, self.moment)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn cross_and_dot_basics() {
        assert_eq!(cross([1.0, 0.0, 0.0], [0.0, 1.0, 0.0]), [0.0, 0.0, 1.0]);
        assert_relative_eq!(dot([1.0, 2.0, 3.0], [4.0, 5.0, 6.0]), 32.0, epsilon = 1e-12);
    }

    #[test]
    fn moment_transport_of_a_force() {
        // Force 10 selon x appliquée à l'origine ; moment en (0,1,0) = 10 selon z.
        let g = Torseur::force([10.0, 0.0, 0.0], [0.0, 0.0, 0.0]);
        let m = g.moment_at([0.0, 1.0, 0.0]);
        assert_relative_eq!(m[0], 0.0, epsilon = 1e-12);
        assert_relative_eq!(m[1], 0.0, epsilon = 1e-12);
        assert_relative_eq!(m[2], 10.0, epsilon = 1e-12);
    }

    #[test]
    fn couple_has_uniform_moment() {
        let c = Torseur::couple([0.0, 0.0, 5.0]);
        assert!(c.is_couple(1e-9));
        // même moment quel que soit le point.
        assert_eq!(c.moment_at([3.0, -2.0, 7.0]), [0.0, 0.0, 5.0]);
    }

    #[test]
    fn a_pure_force_is_a_glisseur_with_zero_automoment() {
        let g = Torseur::force([10.0, 0.0, 0.0], [0.0, 0.0, 0.0]);
        assert_relative_eq!(g.automoment(), 0.0, epsilon = 1e-12);
        assert!(g.is_glisseur(1e-9));
        // l'automoment reste nul après transport (invariant).
        let t = g.transport([0.0, 1.0, 0.0]);
        assert_relative_eq!(t.automoment(), 0.0, epsilon = 1e-9);
    }

    #[test]
    fn automoment_is_transport_invariant() {
        // Torseur quelconque (ni couple ni glisseur).
        let t = Torseur::new([2.0, 0.0, 0.0], [0.0, 0.0, 3.0], [0.0, 0.0, 0.0]);
        let i0 = t.automoment();
        let i1 = t.transport([1.0, 2.0, 3.0]).automoment();
        assert_relative_eq!(i0, i1, epsilon = 1e-9);
    }

    #[test]
    fn central_axis_moment_is_parallel_to_resultant() {
        // Sur l'axe central, le moment est colinéaire à R (produit vectoriel nul).
        let t = Torseur::new([2.0, 0.0, 0.0], [1.0, 4.0, 3.0], [0.0, 0.0, 0.0]);
        let p0 = t.central_axis_point();
        let m0 = t.moment_at(p0);
        let par = cross(t.resultant, m0);
        assert_relative_eq!(par[0], 0.0, epsilon = 1e-9);
        assert_relative_eq!(par[1], 0.0, epsilon = 1e-9);
        assert_relative_eq!(par[2], 0.0, epsilon = 1e-9);
        // le moment minimal vaut pas × R.
        assert_relative_eq!(m0[0], t.pitch() * t.resultant[0], epsilon = 1e-9);
    }

    #[test]
    fn sum_of_two_forces_reduces_correctly() {
        // Deux forces opposées décalées → couple pur (résultante nulle).
        let f1 = Torseur::force([0.0, 10.0, 0.0], [0.0, 0.0, 0.0]);
        let f2 = Torseur::force([0.0, -10.0, 0.0], [1.0, 0.0, 0.0]);
        let s = f1.add_torseur(&f2);
        assert_relative_eq!(s.resultant[1], 0.0, epsilon = 1e-12);
        assert!(s.is_couple(1e-9));
        // moment du couple = 10 (à x=1) selon z : cross((1,0,0),(0,-10,0)) = (0,0,-10).
        assert_relative_eq!(s.moment[2], -10.0, epsilon = 1e-9);
    }

    #[test]
    fn comoment_is_symmetric_and_point_invariant() {
        let a = Torseur::new([1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 0.0]);
        let b = Torseur::new([0.0, 3.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]);
        let cab = a.comoment(&b);
        let cba = b.comoment(&a);
        assert_relative_eq!(cab, cba, epsilon = 1e-12);
        // invariance : réduire a en un autre point ne change pas la comoment.
        let cab2 = a.transport([2.0, -1.0, 5.0]).comoment(&b);
        assert_relative_eq!(cab, cab2, epsilon = 1e-9);
    }

    #[test]
    #[should_panic(expected = "couple")]
    fn pitch_of_a_couple_panics() {
        Torseur::couple([0.0, 0.0, 1.0]).pitch();
    }
}
