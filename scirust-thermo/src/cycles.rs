//! Ideal (air-standard) thermodynamic power and refrigeration cycles.
//!
//! All efficiencies are dimensionless in `[0, 1)`; temperatures are
//! absolute \[K\].

use crate::error::{ThermoError, in_range, positive};

fn check_gamma(gamma: f64) -> Result<f64, ThermoError> {
    in_range("gamma", gamma, 1.0 + 1e-12, 2.0)
}

/// Carnot efficiency `η = 1 − T_cold/T_hot` (absolute temperatures,
/// `0 < T_cold < T_hot`).
pub fn carnot_efficiency(t_hot: f64, t_cold: f64) -> Result<f64, ThermoError> {
    positive("t_hot", t_hot)?;
    positive("t_cold", t_cold)?;
    if t_cold >= t_hot
    {
        return Err(ThermoError::OutOfRange {
            name: "t_cold",
            value: t_cold,
            min: 0.0,
            max: t_hot,
        });
    }
    Ok(1.0 - t_cold / t_hot)
}

/// Carnot coefficient of performance of a **refrigerator**,
/// `COP = T_cold/(T_hot − T_cold)`.
pub fn carnot_cop_refrigerator(t_hot: f64, t_cold: f64) -> Result<f64, ThermoError> {
    carnot_efficiency(t_hot, t_cold)?; // same validation
    Ok(t_cold / (t_hot - t_cold))
}

/// Carnot coefficient of performance of a **heat pump**,
/// `COP = T_hot/(T_hot − T_cold)` (= refrigerator COP + 1).
pub fn carnot_cop_heat_pump(t_hot: f64, t_cold: f64) -> Result<f64, ThermoError> {
    carnot_efficiency(t_hot, t_cold)?;
    Ok(t_hot / (t_hot - t_cold))
}

/// Air-standard **Otto** cycle efficiency,
/// `η = 1 − r^{1−γ}` for compression ratio `r > 1`.
pub fn otto_efficiency(compression_ratio: f64, gamma: f64) -> Result<f64, ThermoError> {
    in_range(
        "compression_ratio",
        compression_ratio,
        1.0 + 1e-12,
        f64::MAX,
    )?;
    check_gamma(gamma)?;
    Ok(1.0 - compression_ratio.powf(1.0 - gamma))
}

/// Air-standard **Diesel** cycle efficiency for compression ratio
/// `r > 1` and cut-off ratio `r_c > 1`:
///
/// `η = 1 − (1/r^{γ−1}) · (r_c^γ − 1)/(γ (r_c − 1))`
pub fn diesel_efficiency(
    compression_ratio: f64,
    cutoff_ratio: f64,
    gamma: f64,
) -> Result<f64, ThermoError> {
    in_range(
        "compression_ratio",
        compression_ratio,
        1.0 + 1e-12,
        f64::MAX,
    )?;
    in_range("cutoff_ratio", cutoff_ratio, 1.0 + 1e-12, compression_ratio)?;
    check_gamma(gamma)?;
    let r = compression_ratio;
    let rc = cutoff_ratio;
    Ok(1.0 - (rc.powf(gamma) - 1.0) / (gamma * (rc - 1.0) * r.powf(gamma - 1.0)))
}

/// Air-standard **Brayton** (gas-turbine) cycle efficiency,
/// `η = 1 − r_p^{−(γ−1)/γ}` for pressure ratio `r_p > 1`.
pub fn brayton_efficiency(pressure_ratio: f64, gamma: f64) -> Result<f64, ThermoError> {
    in_range("pressure_ratio", pressure_ratio, 1.0 + 1e-12, f64::MAX)?;
    check_gamma(gamma)?;
    Ok(1.0 - pressure_ratio.powf(-(gamma - 1.0) / gamma))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn carnot_textbook_values() {
        // 600 K / 300 K → 50 %; COPs: fridge 1, heat pump 2.
        assert!((carnot_efficiency(600.0, 300.0).unwrap() - 0.5).abs() < 1e-15);
        assert!((carnot_cop_refrigerator(600.0, 300.0).unwrap() - 1.0).abs() < 1e-15);
        assert!((carnot_cop_heat_pump(600.0, 300.0).unwrap() - 2.0).abs() < 1e-15);
        // Heat-pump COP = fridge COP + 1, always.
        let (th, tc) = (310.0, 275.0);
        let cop_r = carnot_cop_refrigerator(th, tc).unwrap();
        let cop_h = carnot_cop_heat_pump(th, tc).unwrap();
        assert!((cop_h - cop_r - 1.0).abs() < 1e-12);
    }

    #[test]
    fn otto_r8_is_56_5_percent() {
        // Classic result: r = 8, γ = 1.4 → η = 1 − 8^{−0.4} ≈ 0.5647.
        let eta = otto_efficiency(8.0, 1.4).unwrap();
        assert!((eta - 0.5647).abs() < 1e-4, "eta = {eta}");
    }

    #[test]
    fn diesel_r18_rc2_is_63_2_percent() {
        // Cengel & Boles example: r = 18, r_c = 2, γ = 1.4 → η ≈ 0.632.
        let eta = diesel_efficiency(18.0, 2.0, 1.4).unwrap();
        assert!((eta - 0.6316).abs() < 2e-4, "eta = {eta}");
    }

    #[test]
    fn diesel_tends_to_otto_as_cutoff_shrinks() {
        // lim r_c → 1⁺ of the Diesel efficiency is the Otto efficiency.
        let eta_d = diesel_efficiency(12.0, 1.0 + 1e-6, 1.4).unwrap();
        let eta_o = otto_efficiency(12.0, 1.4).unwrap();
        assert!((eta_d - eta_o).abs() < 1e-5);
        // and Diesel is strictly below Otto at the same r for r_c > 1.
        assert!(diesel_efficiency(12.0, 2.0, 1.4).unwrap() < eta_o);
    }

    #[test]
    fn brayton_rp10_is_48_2_percent() {
        // r_p = 10, γ = 1.4 → η = 1 − 10^{−2/7} ≈ 0.4821.
        let eta = brayton_efficiency(10.0, 1.4).unwrap();
        assert!((eta - 0.4821).abs() < 2e-4, "eta = {eta}");
    }

    #[test]
    fn rejects_degenerate_cycles() {
        assert!(carnot_efficiency(300.0, 300.0).is_err());
        assert!(carnot_efficiency(300.0, 400.0).is_err());
        assert!(otto_efficiency(1.0, 1.4).is_err());
        assert!(diesel_efficiency(18.0, 20.0, 1.4).is_err()); // r_c > r
        assert!(brayton_efficiency(0.5, 1.4).is_err());
    }
}
