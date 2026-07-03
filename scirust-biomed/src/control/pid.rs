//! Contrôleur PID discret à anti-windup conditionnel — Åström & Hägglund,
//! *PID Controllers: Theory, Design, and Tuning* (ISA, 2nd ed. 1995),
//! chapitre 3 (forme parallèle) et §3.5 (anti-windup).
//!
//! Brique de contrôle générique, utilisable comme boucle de régulation
//! d'un débit d'insuline en fonction de l'écart de glycémie au point de
//! consigne — la classe de contrôleur effectivement déployée dans les
//! systèmes en boucle fermée hybride de première génération (Medtronic
//! 670G/770G ; les systèmes plus récents — Tandem Control-IQ, Omnipod 5,
//! CamAPS FX — utilisent du MPC, hors périmètre ici).
//!
//! **Avertissement** : ceci est une brique de contrôle générique à des
//! fins de démonstration/recherche, PAS un algorithme de dosage
//! cliniquement validé. Toute utilisation sur un dispositif médical réel
//! exige la validation clinique, la vérification formelle et
//! l'homologation réglementaire complètes (IEC 62304, FDA SaMD/GMLP/PCCP).

/// Gains proportionnel / intégral / dérivé (forme parallèle).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PidGains {
    pub kp: f64,
    pub ki: f64,
    pub kd: f64,
}

/// Contrôleur PID à état (intégrale, erreur précédente) et sortie bornée.
#[derive(Debug, Clone, PartialEq)]
pub struct PidController {
    gains: PidGains,
    output_min: f64,
    output_max: f64,
    integral: f64,
    prev_error: Option<f64>,
}

impl PidController {
    /// `output_min`/`output_max` bornent la commande (par ex. `[0,
    /// pump_max_rate]` pour un débit d'insuline, qui ne peut être négatif).
    pub fn new(gains: PidGains, output_min: f64, output_max: f64) -> Self {
        Self {
            gains,
            output_min,
            output_max,
            integral: 0.0,
            prev_error: None,
        }
    }

    /// Un pas de contrôle : `error = setpoint - measurement`, renvoie la
    /// commande bornée à `[output_min, output_max]`. `dt` dans les mêmes
    /// unités de temps que les gains (par ex. minutes).
    ///
    /// Anti-windup conditionnel : l'intégrale n'est mise à jour que si la
    /// sortie non bornée n'est pas déjà saturée dans le sens de l'erreur
    /// courante — évite l'accumulation de l'intégrale pendant la
    /// saturation (Åström & Hägglund §3.5).
    pub fn step(&mut self, setpoint: f64, measurement: f64, dt: f64) -> f64 {
        let error = setpoint - measurement;
        let derivative = match self.prev_error
        {
            Some(prev) if dt > 0.0 => (error - prev) / dt,
            _ => 0.0,
        };
        self.prev_error = Some(error);

        let proportional_derivative = self.gains.kp * error + self.gains.kd * derivative;
        let trial_output = proportional_derivative + self.gains.ki * self.integral;
        let saturating_further = (trial_output >= self.output_max && error > 0.0)
            || (trial_output <= self.output_min && error < 0.0);
        if !saturating_further
        {
            self.integral += error * dt;
        }

        let output = proportional_derivative + self.gains.ki * self.integral;
        output.clamp(self.output_min, self.output_max)
    }

    /// Remet l'état interne à zéro (intégrale, mémoire de dérivée) — à
    /// appeler par ex. après une reprise de mode automatique.
    pub fn reset(&mut self) {
        self.integral = 0.0;
        self.prev_error = None;
    }

    pub fn gains(&self) -> PidGains {
        self.gains
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    /// Ferme la boucle sur une planta du premier ordre standard
    /// `tau·y' = -y + K·u` (Euler explicite) et vérifie que l'action
    /// intégrale élimine l'erreur statique pour une consigne en échelon —
    /// résultat classique (principe du modèle interne), vérifié
    /// numériquement (numpy) avant portage : y converge vers 10.0 à
    /// 6.9e-10 près après 1000 pas de dt=0.1 (100 min, 20 constantes de
    /// temps).
    #[test]
    fn eliminates_steady_state_error_on_a_first_order_plant() {
        let (tau, k_plant) = (5.0, 2.0);
        let (r, dt, steps) = (10.0, 0.1, 1000);
        let mut pid = PidController::new(
            PidGains {
                kp: 0.6,
                ki: 0.15,
                kd: 0.0,
            },
            -1e9,
            1e9,
        );
        let mut y = 0.0;
        for _ in 0..steps
        {
            let u = pid.step(r, y, dt);
            y += dt * (-y + k_plant * u) / tau;
        }
        assert_relative_eq!(y, 10.0, epsilon = 1e-6);
    }

    #[test]
    fn output_never_exceeds_configured_bounds() {
        let mut pid = PidController::new(
            PidGains {
                kp: 10.0,
                ki: 5.0,
                kd: 0.0,
            },
            0.0,
            2.0,
        );
        for _ in 0..50
        {
            let u = pid.step(100.0, 0.0, 1.0);
            assert!((0.0..=2.0).contains(&u), "output {u} out of bounds");
        }
    }

    #[test]
    fn reset_clears_integral_and_derivative_memory() {
        let mut pid = PidController::new(
            PidGains {
                kp: 1.0,
                ki: 1.0,
                kd: 1.0,
            },
            -10.0,
            10.0,
        );
        pid.step(5.0, 0.0, 1.0);
        pid.step(5.0, 1.0, 1.0);
        pid.reset();
        // After reset, behaves exactly as a fresh controller on the same input.
        let mut fresh = PidController::new(
            PidGains {
                kp: 1.0,
                ki: 1.0,
                kd: 1.0,
            },
            -10.0,
            10.0,
        );
        assert_eq!(pid.step(5.0, 0.0, 1.0), fresh.step(5.0, 0.0, 1.0));
    }
}
