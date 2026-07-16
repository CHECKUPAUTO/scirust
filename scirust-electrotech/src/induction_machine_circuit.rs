//! Électrotechnique — **machine asynchrone** (moteur à induction) : schéma
//! équivalent par phase en régime permanent, répartition de la puissance de
//! l'entrefer vers la puissance mécanique via le glissement, couple
//! électromagnétique et glissement au couple maximal (schéma approché).
//!
//! ```text
//! glissement            g     = (N_s − N) / N_s              [sans dim.]
//! puissance d'entrefer  P_ag  = P_cu2 / g                    [W]
//! puissance mécanique   P_mec = P_ag · (1 − g)               [W]
//! couple électromagn.   T_em  = P_ag / Ω_s                   [N·m]
//! glissement au couple  g_max = R2 / (X1 + X2)               [sans dim.]
//! maximal (approché)
//! ```
//!
//! `N_s` vitesse de synchronisme [tr/min ou rad/s, même unité que `N`], `N`
//! vitesse mécanique du rotor [même unité que `N_s`], `g` glissement [sans
//! dimension, ×100 pour des %], `P_cu2` pertes Joule au rotor [W], `P_ag`
//! puissance transmise à l'entrefer [W], `P_mec` puissance mécanique
//! développée sur l'arbre (avant pertes mécaniques) [W], `Ω_s` vitesse de
//! synchronisme angulaire [rad/s], `T_em` couple électromagnétique [N·m],
//! `R2` résistance rotorique ramenée au stator [Ω], `X1` réactance de fuite
//! statorique [Ω], `X2` réactance de fuite rotorique ramenée au stator [Ω].
//! La relation exacte `P_ag = P_cu2 + P_mec = P_cu2 / g` découle du fait que
//! les pertes Joule rotor valent `g · P_ag`.
//!
//! **Limite honnête** : machine asynchrone en **régime permanent équilibré**
//! et sinusoïdal, schéma équivalent **par phase**. La répartition puissance
//! d'entrefer → mécanique par le glissement (`P_cu2 = g · P_ag`,
//! `P_mec = (1 − g) · P_ag`) et le couple `T_em = P_ag / Ω_s` sont
//! **exacts** dans ce modèle. Le glissement au couple maximal `g_max` utilise
//! le **schéma équivalent approché** (résistance statorique `R1` **négligée**
//! et branche magnétisante reportée aux bornes) ; en toute rigueur `R1`
//! intervient. Les résistances et réactances (`R2`, `X1`, `X2`), les
//! vitesses et les pertes sont **fournies par l'appelant** (essais à
//! vide/rotor bloqué, plaque signalétique, catalogue) ; aucune valeur « par
//! défaut » n'est inventée. La saturation, les harmoniques d'espace, l'effet
//! pelliculaire rotorique et les régimes transitoires sont **négligés**.

/// Glissement `g = (N_s − N) / N_s` [sans dimension] (multiplier par 100 pour
/// l'exprimer en pourcentage).
///
/// `synchronous_speed` vitesse de synchronisme `N_s` (tr/min ou rad/s,
/// strictement positive), `rotor_speed` vitesse mécanique du rotor `N` dans
/// la **même unité** que `synchronous_speed` ; le résultat est sans
/// dimension. À l'arrêt (`N = 0`) le glissement vaut `1` ; au synchronisme
/// (`N = N_s`) il vaut `0`.
///
/// Panique si `synchronous_speed <= 0` (division).
pub fn imc_slip(synchronous_speed: f64, rotor_speed: f64) -> f64 {
    assert!(
        synchronous_speed > 0.0,
        "la vitesse de synchronisme doit être strictement positive"
    );
    (synchronous_speed - rotor_speed) / synchronous_speed
}

/// Puissance transmise à l'entrefer `P_ag = P_cu2 / g` [W] (les pertes Joule
/// rotor valent `g · P_ag`).
///
/// `rotor_copper_loss` pertes Joule au rotor `P_cu2` en watts (W, positives
/// ou nulles), `slip` glissement `g` (sans dimension, strictement positif) ;
/// le résultat est la puissance d'entrefer en watts (W).
///
/// Panique si `rotor_copper_loss < 0`, ou si `slip <= 0` (division / sens
/// physique en fonctionnement moteur).
pub fn imc_airgap_power(rotor_copper_loss: f64, slip: f64) -> f64 {
    assert!(
        rotor_copper_loss >= 0.0,
        "les pertes Joule rotor doivent être positives ou nulles (W)"
    );
    assert!(
        slip > 0.0,
        "le glissement doit être strictement positif (division)"
    );
    rotor_copper_loss / slip
}

/// Puissance mécanique développée `P_mec = P_ag · (1 − g)` [W] (puissance sur
/// l'arbre avant déduction des pertes mécaniques).
///
/// `airgap_power` puissance transmise à l'entrefer `P_ag` en watts (W,
/// positive ou nulle), `slip` glissement `g` (sans dimension, dans `[0, 1]`) ;
/// le résultat est la puissance mécanique en watts (W). On a
/// `P_mec = P_ag − P_cu2` avec `P_cu2 = g · P_ag`.
///
/// Panique si `airgap_power < 0`, ou si `slip` n'est pas dans `[0, 1]`.
pub fn imc_mechanical_power(airgap_power: f64, slip: f64) -> f64 {
    assert!(
        airgap_power >= 0.0,
        "la puissance d'entrefer doit être positive ou nulle (W)"
    );
    assert!(
        (0.0..=1.0).contains(&slip),
        "le glissement doit être dans [0, 1]"
    );
    airgap_power * (1.0 - slip)
}

/// Couple électromagnétique `T_em = P_ag / Ω_s` [N·m] (couple rapporté à la
/// vitesse de synchronisme).
///
/// `airgap_power` puissance transmise à l'entrefer `P_ag` en watts (W,
/// positive ou nulle), `synchronous_speed_rad` vitesse de synchronisme
/// angulaire `Ω_s` en radians par seconde (rad/s, strictement positive) ; le
/// résultat est le couple électromagnétique en newtons-mètres (N·m). Ce
/// couple est bien défini au synchronisme alors que `P_mec / Ω_mec` y est
/// indéterminé.
///
/// Panique si `airgap_power < 0`, ou si `synchronous_speed_rad <= 0`
/// (division).
pub fn imc_torque(airgap_power: f64, synchronous_speed_rad: f64) -> f64 {
    assert!(
        airgap_power >= 0.0,
        "la puissance d'entrefer doit être positive ou nulle (W)"
    );
    assert!(
        synchronous_speed_rad > 0.0,
        "la vitesse de synchronisme angulaire doit être strictement positive (rad/s)"
    );
    airgap_power / synchronous_speed_rad
}

/// Glissement au couple maximal `g_max = R2 / (X1 + X2)` [sans dimension]
/// (schéma équivalent **approché**, résistance statorique négligée).
///
/// `rotor_resistance` résistance rotorique ramenée au stator `R2` en ohms
/// (Ω, positive ou nulle), `stator_reactance` réactance de fuite statorique
/// `X1` en ohms (Ω), `rotor_reactance` réactance de fuite rotorique ramenée
/// au stator `X2` en ohms (Ω) ; le résultat est le glissement au couple
/// maximal, sans dimension. Le couple est maximal lorsque `R2 / g` égale la
/// réactance de fuite totale `X1 + X2`.
///
/// Panique si `rotor_resistance < 0`, ou si `stator_reactance +
/// rotor_reactance <= 0` (division).
pub fn imc_slip_at_max_torque(
    rotor_resistance: f64,
    stator_reactance: f64,
    rotor_reactance: f64,
) -> f64 {
    assert!(
        rotor_resistance >= 0.0,
        "la résistance rotorique doit être positive ou nulle (Ω)"
    );
    let total_reactance = stator_reactance + rotor_reactance;
    assert!(
        total_reactance > 0.0,
        "la réactance de fuite totale doit être strictement positive (Ω)"
    );
    rotor_resistance / total_reactance
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use core::f64::consts::PI;

    #[test]
    fn slip_limits_and_realistic() {
        // N_s = 1500 tr/min, N = 1425 tr/min ⇒ g = 75/1500 = 0,05.
        assert_relative_eq!(imc_slip(1500.0, 1425.0), 0.05, epsilon = 1e-12);
        // À l'arrêt (N = 0) ⇒ g = 1 ; au synchronisme (N = N_s) ⇒ g = 0.
        assert_relative_eq!(imc_slip(1500.0, 0.0), 1.0, epsilon = 1e-12);
        assert_relative_eq!(imc_slip(1500.0, 1500.0), 0.0, epsilon = 1e-12);
    }

    #[test]
    fn power_split_reciprocity() {
        // P_cu2 = 500 W, g = 0,05 ⇒ P_ag = 500/0,05 = 10 000 W.
        let p_ag = imc_airgap_power(500.0, 0.05);
        assert_relative_eq!(p_ag, 10_000.0, epsilon = 1e-9);
        // P_mec = P_ag·(1−g) = 10 000·0,95 = 9 500 W.
        let p_mec = imc_mechanical_power(p_ag, 0.05);
        assert_relative_eq!(p_mec, 9_500.0, epsilon = 1e-9);
        // Bilan exact : P_ag = P_cu2 + P_mec ⇒ P_cu2 = P_ag − P_mec = 500 W.
        assert_relative_eq!(p_ag - p_mec, 500.0, epsilon = 1e-9);
    }

    #[test]
    fn airgap_power_recovers_copper_loss() {
        // Réciprocité : P_cu2 = g · P_ag pour toute paire (P_cu2, g) valide.
        let (loss, g) = (750.0_f64, 0.03_f64);
        let p_ag = imc_airgap_power(loss, g);
        assert_relative_eq!(g * p_ag, loss, epsilon = 1e-9);
    }

    #[test]
    fn torque_matches_airgap_over_omega() {
        // N_s = 1500 tr/min ⇒ Ω_s = 1500·2π/60 = 50π = 157,07963... rad/s.
        let omega_s = 1500.0_f64 * 2.0 * PI / 60.0;
        assert_relative_eq!(omega_s, 50.0 * PI, epsilon = 1e-12);
        // P_ag = 10 000 W ⇒ T_em = 10 000 / (50π) = 200/π = 63,66197... N·m.
        let t_em = imc_torque(10_000.0, omega_s);
        assert_relative_eq!(t_em, 200.0 / PI, epsilon = 1e-9);
        // Cohérence : T_em · Ω_s = P_ag.
        assert_relative_eq!(t_em * omega_s, 10_000.0, epsilon = 1e-6);
    }

    #[test]
    fn torque_scales_linearly_with_airgap_power() {
        // À Ω_s fixé : T_em ∝ P_ag (doubler P_ag double le couple).
        let omega_s = 157.079_632_679_49_f64;
        let t1 = imc_torque(5_000.0, omega_s);
        let t2 = imc_torque(10_000.0, omega_s);
        assert_relative_eq!(t2, 2.0 * t1, epsilon = 1e-9);
    }

    #[test]
    fn slip_at_max_torque_realistic() {
        // R2 = 0,5 Ω, X1 = 1,5 Ω, X2 = 1,0 Ω ⇒ g_max = 0,5/2,5 = 0,2.
        assert_relative_eq!(imc_slip_at_max_torque(0.5, 1.5, 1.0), 0.2, epsilon = 1e-12);
        // Proportionnalité : g_max ∝ R2 à réactances fixées.
        let g1 = imc_slip_at_max_torque(0.5, 1.5, 1.0);
        let g2 = imc_slip_at_max_torque(1.0, 1.5, 1.0);
        assert_relative_eq!(g2, 2.0 * g1, epsilon = 1e-9);
    }

    #[test]
    #[should_panic(expected = "glissement doit être strictement positif")]
    fn zero_slip_airgap_panics() {
        imc_airgap_power(500.0, 0.0);
    }
}
