//! Économie d'usinage — modèle de **Gilbert** (1950) donnant les vitesses de
//! coupe optimales dérivées de la loi de Taylor : celle de **production
//! maximale** (temps par pièce minimal) et celle de **coût minimal**.
//!
//! À partir de `Vc·T^n = C`, la durée de vie d'arête optimale s'écrit :
//!
//! ```text
//! Production maximale :  T = (1/n − 1) · t_c
//! Coût minimal :         T = (1/n − 1) · (t_c + C_t / C_o)
//! ```
//!
//! où `t_c` est le temps de changement d'arête (min), `C_t` le coût d'une arête
//! (outil + affûtage/indexation, €), et `C_o` le taux d'exploitation
//! machine + opérateur (€/min). La vitesse correspondante se lit ensuite sur la
//! loi de Taylor, `Vc = C / T^n`.
//!
//! La vitesse de coût minimal est toujours **inférieure** à celle de production
//! maximale : entre les deux s'étend le « domaine de haute efficience » (Gilbert),
//! où l'on choisit son compromis débit/coût.
//!
//! **Limite honnête** : ce modèle optimise le seul régime de coupe vis-à-vis de
//! l'usure de Taylor. Il suppose `n < 1` (toujours vrai en pratique) et des
//! coûts constants ; il ne modélise ni les contraintes de puissance/rigidité
//! machine, ni la qualité de surface visée, ni les temps hors coupe autres que
//! le changement d'outil.

/// Paramètres économiques d'une opération : loi de Taylor (`n`, `C`), temps de
/// changement d'arête et coûts.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MachiningEconomics {
    /// Exposant de Taylor `n` (0 < n < 1).
    pub n: f64,
    /// Constante de Taylor `C` (m/min).
    pub c: f64,
    /// Temps de changement d'arête `t_c` (min).
    pub tool_change_time_min: f64,
    /// Coût d'une arête `C_t` (€ ou toute unité monétaire cohérente).
    pub tool_cost: f64,
    /// Taux d'exploitation machine + opérateur `C_o` (même unité par minute).
    pub operating_rate: f64,
}

impl MachiningEconomics {
    fn check(&self) {
        assert!(
            self.n > 0.0 && self.n < 1.0,
            "l'exposant de Taylor doit vérifier 0 < n < 1"
        );
        assert!(self.c > 0.0, "la constante C de Taylor doit être positive");
    }

    /// Durée de vie d'arête `T` (min) maximisant le débit de pièces :
    /// `T = (1/n − 1)·t_c`.
    pub fn tool_life_max_production(&self) -> f64 {
        self.check();
        (1.0 / self.n - 1.0) * self.tool_change_time_min
    }

    /// Durée de vie d'arête `T` (min) minimisant le coût par pièce :
    /// `T = (1/n − 1)·(t_c + C_t/C_o)`.
    ///
    /// Panique si `operating_rate <= 0`.
    pub fn tool_life_min_cost(&self) -> f64 {
        self.check();
        assert!(
            self.operating_rate > 0.0,
            "le taux d'exploitation doit être strictement positif"
        );
        (1.0 / self.n - 1.0) * (self.tool_change_time_min + self.tool_cost / self.operating_rate)
    }

    /// Vitesse de coupe `Vc` (m/min) de production maximale.
    pub fn speed_max_production(&self) -> f64 {
        self.c / self.tool_life_max_production().powf(self.n)
    }

    /// Vitesse de coupe `Vc` (m/min) de coût minimal.
    pub fn speed_min_cost(&self) -> f64 {
        self.c / self.tool_life_min_cost().powf(self.n)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn eco() -> MachiningEconomics {
        MachiningEconomics {
            n: 0.25,
            c: 300.0,
            tool_change_time_min: 2.0,
            tool_cost: 5.0,
            operating_rate: 1.0,
        }
    }

    #[test]
    fn max_production_tool_life_matches_gilbert() {
        // (1/0,25 − 1)·2 = 3·2 = 6 min.
        assert_relative_eq!(eco().tool_life_max_production(), 6.0, epsilon = 1e-9);
    }

    #[test]
    fn min_cost_tool_life_matches_gilbert() {
        // (1/0,25 − 1)·(2 + 5/1) = 3·7 = 21 min.
        assert_relative_eq!(eco().tool_life_min_cost(), 21.0, epsilon = 1e-9);
    }

    #[test]
    fn min_cost_speed_is_below_max_production_speed() {
        // T plus grand ⇒ Vc plus faible : la coupe économique est plus lente.
        let e = eco();
        assert!(e.speed_min_cost() < e.speed_max_production());
    }

    #[test]
    fn speeds_satisfy_the_taylor_relation() {
        // Vc·T^n doit redonner C aux deux optimums.
        let e = eco();
        assert_relative_eq!(
            e.speed_max_production() * e.tool_life_max_production().powf(e.n),
            e.c,
            epsilon = 1e-9
        );
        assert_relative_eq!(
            e.speed_min_cost() * e.tool_life_min_cost().powf(e.n),
            e.c,
            epsilon = 1e-9
        );
    }

    #[test]
    #[should_panic(expected = "0 < n < 1")]
    fn exponent_out_of_range_panics() {
        let mut e = eco();
        e.n = 1.5;
        e.tool_life_max_production();
    }
}
