//! Open-channel (free-surface) hydraulics: Manning's equation, critical
//! flow, specific energy and the hydraulic jump.
//!
//! SI units throughout; Manning's `n` is the metric coefficient.

use crate::error::{FluidsError, in_range, non_negative, positive};

/// Mean velocity from **Manning's equation** `V = (1/n) R^{2/3} √S` \[m/s\].
///
/// * `n` — Manning roughness coefficient \[s/m^{1/3}\], > 0
/// * `hydraulic_radius` — R = A/P \[m\], > 0
/// * `slope` — energy/bed slope S \[−\], ≥ 0
pub fn manning_velocity(n: f64, hydraulic_radius: f64, slope: f64) -> Result<f64, FluidsError> {
    positive("n", n)?;
    positive("hydraulic_radius", hydraulic_radius)?;
    non_negative("slope", slope)?;
    Ok(hydraulic_radius.powf(2.0 / 3.0) * slope.sqrt() / n)
}

/// Discharge from Manning's equation, `Q = A V` \[m³/s\].
pub fn manning_discharge(
    n: f64,
    area: f64,
    hydraulic_radius: f64,
    slope: f64,
) -> Result<f64, FluidsError> {
    positive("area", area)?;
    Ok(area * manning_velocity(n, hydraulic_radius, slope)?)
}

/// Critical depth of a **rectangular** channel, `y_c = (q²/g)^{1/3}` \[m\],
/// where `q` is the discharge per unit width \[m²/s\].
pub fn critical_depth_rectangular(q_per_width: f64, gravity: f64) -> Result<f64, FluidsError> {
    positive("q_per_width", q_per_width)?;
    positive("gravity", gravity)?;
    Ok((q_per_width * q_per_width / gravity).powf(1.0 / 3.0))
}

/// Specific energy of a channel section, `E = y + V²/(2g)` \[m\].
pub fn specific_energy(depth: f64, speed: f64, gravity: f64) -> Result<f64, FluidsError> {
    positive("depth", depth)?;
    non_negative("speed", speed)?;
    positive("gravity", gravity)?;
    Ok(depth + speed * speed / (2.0 * gravity))
}

/// Conjugate (sequent) depth ratio of a **hydraulic jump** in a
/// rectangular channel (Bélanger equation):
///
/// `y₂/y₁ = ( √(1 + 8 Fr₁²) − 1 ) / 2`
///
/// Requires a supercritical approach flow, `Fr₁ ≥ 1`.
pub fn hydraulic_jump_depth_ratio(fr1: f64) -> Result<f64, FluidsError> {
    in_range("fr1", fr1, 1.0, f64::MAX)?;
    Ok(0.5 * ((1.0 + 8.0 * fr1 * fr1).sqrt() - 1.0))
}

/// Normal (uniform-flow) depth of a **rectangular** channel carrying
/// discharge `Q` \[m³/s\] on slope `S` with width `b` \[m\], by solving
/// Manning's equation for the depth.
///
/// Uses deterministic bracketing + 200 bisection steps (the conveyance
/// is strictly increasing in depth), so identical inputs give
/// bit-identical results everywhere.
pub fn normal_depth_rectangular(
    discharge: f64,
    width: f64,
    n: f64,
    slope: f64,
) -> Result<f64, FluidsError> {
    positive("discharge", discharge)?;
    positive("width", width)?;
    positive("n", n)?;
    positive("slope", slope)?;

    let q_of = |y: f64| -> f64 {
        let area = width * y;
        let radius = area / (width + 2.0 * y);
        area * radius.powf(2.0 / 3.0) * slope.sqrt() / n
    };

    let mut lo = 1e-12;
    let mut hi = 1.0;
    let mut bracketed = false;
    for _ in 0..200
    {
        if q_of(hi) > discharge
        {
            bracketed = true;
            break;
        }
        lo = hi;
        hi *= 2.0;
    }
    if !bracketed
    {
        return Err(FluidsError::NoConvergence {
            what: "normal-depth bracket",
        });
    }
    for _ in 0..200
    {
        let mid = 0.5 * (lo + hi);
        if q_of(mid) > discharge
        {
            hi = mid;
        }
        else
        {
            lo = mid;
        }
    }
    Ok(0.5 * (lo + hi))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn manning_smooth_concrete() {
        // n = 0.013, R = 1 m, S = 0.001: V = √0.001/0.013 = 2.4325 m/s.
        let v = manning_velocity(0.013, 1.0, 0.001).unwrap();
        assert!((v - 2.432_52).abs() < 1e-4, "V = {v}");
        let q = manning_discharge(0.013, 2.0, 1.0, 0.001).unwrap();
        assert!((q - 2.0 * v).abs() < 1e-12);
    }

    #[test]
    fn critical_depth_known_value() {
        // q = 2 m²/s, g = 9.81: y_c = (4/9.81)^{1/3} = 0.7416 m.
        let yc = critical_depth_rectangular(2.0, 9.81).unwrap();
        assert!((yc - 0.741_56).abs() < 1e-4, "yc = {yc}");
        // At critical depth the Froude number is exactly 1.
        let v = 2.0 / yc;
        let fr = v / (9.81 * yc).sqrt();
        assert!((fr - 1.0).abs() < 1e-12);
    }

    #[test]
    fn belanger_at_froude_2() {
        // y₂/y₁ = (√33 − 1)/2 = 2.37228.
        let r = hydraulic_jump_depth_ratio(2.0).unwrap();
        assert!((r - 2.372_281).abs() < 1e-5, "ratio = {r}");
        // Fr₁ = 1 is the degenerate (no-jump) case: ratio 1.
        assert!((hydraulic_jump_depth_ratio(1.0).unwrap() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn normal_depth_roundtrip() {
        // Whatever depth we solve for must reproduce the discharge.
        let (b, n, s) = (3.0, 0.015, 0.002);
        let q = 8.5;
        let y = normal_depth_rectangular(q, b, n, s).unwrap();
        let area = b * y;
        let radius = area / (b + 2.0 * y);
        let q_back = manning_discharge(n, area, radius, s).unwrap();
        assert!((q_back - q).abs() / q < 1e-9, "y = {y}, Q = {q_back}");
    }

    #[test]
    fn specific_energy_minimum_at_critical_depth() {
        // For fixed q, E(y) is minimal at y_c: check E(yc) < E(yc ± 20 %).
        let (q, g) = (2.0, 9.81);
        let yc = critical_depth_rectangular(q, g).unwrap();
        let e = |y: f64| specific_energy(y, q / y, g).unwrap();
        assert!(e(yc) < e(yc * 0.8));
        assert!(e(yc) < e(yc * 1.2));
        // And the minimum value is 1.5 y_c for a rectangular channel.
        assert!((e(yc) - 1.5 * yc).abs() < 1e-12);
    }

    #[test]
    fn rejects_subcritical_jump() {
        assert!(hydraulic_jump_depth_ratio(0.7).is_err());
        assert!(manning_velocity(0.0, 1.0, 0.001).is_err());
    }
}
