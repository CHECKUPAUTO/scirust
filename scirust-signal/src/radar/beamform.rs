//! Array processing: direction-of-arrival (DOA) estimation with a uniform
//! linear array (ULA).
//!
//! A plane wave from angle `θ` (measured from broadside) reaches element `m` of
//! a ULA with element spacing `d` wavelengths at phase `2π·d·m·sin θ` — the
//! **steering vector** `a(θ)`. The conventional (delay-and-sum / Bartlett)
//! beamformer scans a steering angle across the field of view and reports the
//! output power `aᴴ(θ)·R·a(θ)`; its spatial spectrum peaks at the source
//! directions. This is the angle stage that complements the range-Doppler
//! chain; higher-resolution estimators (MVDR, MUSIC) build on the same steering
//! vectors.

use crate::complex::Complex;
use std::f64::consts::PI;

/// The ULA **steering vector** `a(θ)`: `num_sensors` elements at spacing
/// `spacing` (wavelengths), for a plane wave from `angle_rad` (radians from
/// broadside). Element `m` is `exp(j·2π·spacing·m·sin θ)` — unit magnitude, and
/// all ones at broadside (`θ = 0`).
pub fn steering_vector(num_sensors: usize, spacing: f64, angle_rad: f64) -> Vec<Complex> {
    let phase_step = 2.0 * PI * spacing * angle_rad.sin();
    (0..num_sensors)
        .map(|m| Complex::cis(phase_step * m as f64))
        .collect()
}

/// The conventional (delay-and-sum / Bartlett) beamformer spatial spectrum:
/// for each steering angle, the average output power `mean_t |aᴴ(θ)·x[t]|²`
/// over the array `snapshots` (`snapshots[t]` is one length-`M` snapshot). The
/// spectrum peaks at the source directions. Snapshots of the wrong length are
/// skipped; an empty snapshot set yields an all-zero spectrum.
pub fn beamform_spectrum(snapshots: &[Vec<Complex>], spacing: f64, angles: &[f64]) -> Vec<f64> {
    if snapshots.is_empty()
    {
        return vec![0.0; angles.len()];
    }
    let m = snapshots[0].len();
    angles
        .iter()
        .map(|&theta| {
            let a = steering_vector(m, spacing, theta);
            let mut power = 0.0;
            let mut used = 0usize;
            for x in snapshots
            {
                if x.len() != m
                {
                    continue;
                }
                // y = aᴴ x = Σ conj(a[i])·x[i]
                let y = a
                    .iter()
                    .zip(x)
                    .fold(Complex::zero(), |acc, (ai, xi)| acc + ai.conj() * *xi);
                power += y.mag_sq();
                used += 1;
            }
            if used == 0 { 0.0 } else { power / used as f64 }
        })
        .collect()
}

/// The steering angle at which a spatial `spectrum` peaks — a single-source DOA
/// estimate. `None` for an empty spectrum or mismatched lengths.
pub fn estimate_doa(spectrum: &[f64], angles: &[f64]) -> Option<f64> {
    if spectrum.is_empty() || spectrum.len() != angles.len()
    {
        return None;
    }
    let (idx, _) = spectrum
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))?;
    angles.get(idx).copied()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn source_snapshots(
        m: usize,
        spacing: f64,
        dirs: &[(f64, f64)],
        t: usize,
    ) -> Vec<Vec<Complex>> {
        // dirs: (angle, signal-phase-rate). Noiseless sum of plane waves.
        let steer: Vec<Vec<Complex>> = dirs
            .iter()
            .map(|&(theta, _)| steering_vector(m, spacing, theta))
            .collect();
        (0..t)
            .map(|k| {
                (0..m)
                    .map(|i| {
                        dirs.iter()
                            .enumerate()
                            .fold(Complex::zero(), |acc, (s, &(_, rate))| {
                                let sig = Complex::cis(rate * k as f64 + s as f64);
                                acc + sig * steer[s][i]
                            })
                    })
                    .collect()
            })
            .collect()
    }

    #[test]
    fn steering_vector_is_all_ones_at_broadside_and_unit_magnitude() {
        for c in &steering_vector(6, 0.5, 0.0)
        {
            assert!((c.re - 1.0).abs() < 1e-12 && c.im.abs() < 1e-12);
        }
        for c in &steering_vector(6, 0.5, 0.5)
        {
            assert!((c.mag() - 1.0).abs() < 1e-12);
        }
    }

    #[test]
    fn beamformer_peaks_at_the_source_direction() {
        let (m, spacing, theta0) = (8usize, 0.5, 0.35_f64);
        let snaps = source_snapshots(m, spacing, &[(theta0, 0.3)], 16);
        let angles: Vec<f64> = (-90..=90).map(|d| (d as f64).to_radians()).collect();
        let spec = beamform_spectrum(&snaps, spacing, &angles);
        let est = estimate_doa(&spec, &angles).unwrap();
        assert!(
            (est - theta0).abs() < 2.0_f64.to_radians(),
            "DOA {est} vs {theta0}"
        );
    }

    #[test]
    fn beamformer_sees_two_separated_sources_above_an_empty_direction() {
        let (m, spacing) = (12usize, 0.5);
        let (t1, t2) = (-0.4_f64, 0.5_f64);
        let snaps = source_snapshots(m, spacing, &[(t1, 0.3), (t2, 0.7)], 64);
        // Probe the two source angles and one empty direction.
        let probe = [t1, t2, 1.4_f64];
        let p = beamform_spectrum(&snaps, spacing, &probe);
        assert!(p[0] > 3.0 * p[2], "source 1 not above the floor");
        assert!(p[1] > 3.0 * p[2], "source 2 not above the floor");
    }

    #[test]
    fn beamform_edge_cases() {
        assert!(
            beamform_spectrum(&[], 0.5, &[0.0, 0.1])
                .iter()
                .all(|&p| p == 0.0)
        );
        assert!(estimate_doa(&[], &[]).is_none());
        assert!(estimate_doa(&[1.0, 2.0], &[0.0]).is_none()); // length mismatch
    }
}
