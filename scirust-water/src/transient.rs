//! Water-hammer transient physics.
//!
//! When a valve slams shut, the moving water column is stopped and its momentum
//! converts to a pressure spike that travels back up the pipe at the acoustic
//! wave speed. Two governing equations:
//!
//! - **Joukowsky**: the head/pressure rise from an instantaneous velocity change,
//!   `Δp = ρ·c·Δv` (equivalently `ΔH = c·Δv/g`).
//! - **Korteweg**: the pressure-wave speed in a liquid-filled elastic pipe,
//!   `c = √( (K/ρ) / (1 + (K/E)·(D/e)) )`, which is below the free-fluid sound
//!   speed because the pipe wall stretches.

/// Joukowsky pressure surge (Pa) from an instantaneous velocity change.
///
/// `rho` — fluid density (kg/m³); `wave_speed` — acoustic wave speed (m/s);
/// `delta_v` — change in flow velocity (m/s, magnitude). `Δp = ρ·c·Δv`.
pub fn joukowsky_surge(rho: f64, wave_speed: f64, delta_v: f64) -> f64 {
    rho * wave_speed * delta_v
}

/// Joukowsky head rise (m) — the surge expressed as a column height.
/// `ΔH = c·Δv/g` with `g = 9.80665 m/s²`.
pub fn joukowsky_head(wave_speed: f64, delta_v: f64) -> f64 {
    wave_speed * delta_v / 9.806_65
}

/// Korteweg pressure-wave speed (m/s) in a liquid-filled thin-walled elastic pipe.
///
/// `k` — fluid bulk modulus (Pa); `rho` — fluid density (kg/m³); `e_pipe` — pipe
/// Young's modulus (Pa); `diameter` and `wall` — pipe inner diameter and wall
/// thickness (same unit). The wall's elasticity lowers `c` below `√(K/ρ)`.
pub fn korteweg_wave_speed(k: f64, rho: f64, e_pipe: f64, diameter: f64, wall: f64) -> f64 {
    let free = k / rho; // squared free-fluid sound speed
    (free / (1.0 + (k / e_pipe) * (diameter / wall))).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn joukowsky_surge_for_a_textbook_case() {
        // Water (ρ=1000), c=1200 m/s, stopping a 2 m/s flow:
        // Δp = 1000·1200·2 = 2.4 MPa (~24 bar) — a realistic, dangerous surge.
        let dp = joukowsky_surge(1000.0, 1200.0, 2.0);
        assert!((dp - 2.4e6).abs() < 1.0, "got {dp}");
        // Same surge as a head: 1200·2/9.80665 ≈ 244.7 m.
        let h = joukowsky_head(1200.0, 2.0);
        assert!((h - 244.73).abs() < 0.1, "got {h}");
    }

    #[test]
    fn korteweg_matches_a_hand_computed_value() {
        // c² = (K/ρ) / (1 + (K/E)(D/e))
        //    = (2.2e9/1000) / (1 + (2.2e9/200e9)(0.5/0.01))
        //    = 2.2e6 / (1 + 0.011·50) = 2.2e6 / 1.55 = 1_419_354.84
        //  → c = 1191.37 m/s.
        let c = korteweg_wave_speed(2.2e9, 1000.0, 200e9, 0.5, 0.01);
        assert!((c - 1191.37).abs() < 0.1, "got {c}");
    }

    #[test]
    fn korteweg_is_below_the_free_fluid_sound_speed() {
        // Water in a steel pipe: K=2.2 GPa, ρ=1000, E=200 GPa, D=0.5 m, e=0.01 m.
        let k = 2.2e9_f64;
        let rho = 1000.0;
        let free = (k / rho).sqrt(); // ~1483 m/s
        let c = korteweg_wave_speed(k, rho, 200e9, 0.5, 0.01);
        assert!(
            c < free,
            "pipe wall must lower the wave speed: {c} vs {free}"
        );
        // Steel is stiff, so the reduction is modest — still ~1190 m/s.
        assert!((1150.0..1250.0).contains(&c), "got {c}");
    }

    #[test]
    fn a_softer_pipe_slows_the_wave_more() {
        // Same water, a far more compliant pipe (PVC-like, E=3 GPa) drops c hard.
        let stiff = korteweg_wave_speed(2.2e9, 1000.0, 200e9, 0.5, 0.01);
        let soft = korteweg_wave_speed(2.2e9, 1000.0, 3e9, 0.5, 0.01);
        assert!(
            soft < stiff,
            "compliant pipe must be slower: {soft} vs {stiff}"
        );
        assert!(
            soft < 600.0,
            "PVC wave speed should be well under 600 m/s: {soft}"
        );
    }
}
