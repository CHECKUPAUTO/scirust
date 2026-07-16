//! The canonical velocity scenarios, curvature weighting and reference
//! configuration used by the ITD simulator, ported verbatim.
//!
//! With `indexing="xy"` meshes, the field at row `i`, column `j` samples the
//! velocity at `(x = xc[j], y = yc[i])`. The scenario functions therefore take
//! the 1-D axis coordinates and build the 2-D fields directly.

use std::f64::consts::PI;

use crate::field::Field2;

/// The reference simulation configuration (grid, domain, time horizon and the
/// curvature characteristic length).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Config {
    /// Number of grid points per axis.
    pub grid_size: usize,
    /// Lower bound of the (square) domain.
    pub domain_min: f64,
    /// Upper bound of the (square) domain.
    pub domain_max: f64,
    /// Total simulated duration.
    pub duration: f64,
    /// Number of time samples.
    pub time_steps: usize,
    /// Characteristic length `L` in the curvature weight `exp(L²·κ)`.
    pub characteristic_length: f64,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            grid_size: 161,
            domain_min: -2.0,
            domain_max: 2.0,
            duration: 10.0,
            time_steps: 401,
            characteristic_length: 0.5,
        }
    }
}

impl Config {
    /// The per-axis grid coordinates (`grid_size` points from `domain_min` to
    /// `domain_max`).
    pub fn coordinates(&self) -> Vec<f64> {
        linspace(self.domain_min, self.domain_max, self.grid_size)
    }

    /// The time samples (`time_steps` points from `0` to `duration`).
    pub fn times(&self) -> Vec<f64> {
        linspace(0.0, self.duration, self.time_steps)
    }

    /// The uniform grid spacing implied by [`Config::coordinates`].
    pub fn spacing(&self) -> f64 {
        let c = self.coordinates();
        c[1] - c[0]
    }
}

/// `num` evenly spaced samples over `[start, stop]`, reproducing NumPy's
/// `linspace` (the final sample is set exactly to `stop`).
pub fn linspace(start: f64, stop: f64, num: usize) -> Vec<f64> {
    if num == 0
    {
        return Vec::new();
    }
    if num == 1
    {
        return vec![start];
    }
    let delta = (stop - start) / (num - 1) as f64;
    let mut out: Vec<f64> = (0..num).map(|k| start + k as f64 * delta).collect();
    out[num - 1] = stop;
    out
}

/// The three canonical velocity scenarios.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Scenario {
    /// Pure expansion `(a·x, a·y)` — analytically irrotational.
    Calm,
    /// Coherent rigid rotation `(−a·y, a·x)`.
    Coherent,
    /// A deterministic superposition of four moving Gaussian vortices plus a
    /// weak background.
    Multi,
}

impl Scenario {
    /// The reference name of the scenario.
    pub fn name(&self) -> &'static str {
        match self
        {
            Scenario::Calm => "calm",
            Scenario::Coherent => "coherent",
            Scenario::Multi => "multi",
        }
    }

    /// Evaluates the velocity field `(vx, vy)` at time `t` on the grid given by
    /// column coordinates `xc` and row coordinates `yc`.
    pub fn velocity(&self, xc: &[f64], yc: &[f64], t: f64) -> (Field2, Field2) {
        match self
        {
            Scenario::Calm => calm_field(xc, yc, t),
            Scenario::Coherent => coherent_vortex(xc, yc, t),
            Scenario::Multi => multi_vortex_field(xc, yc, t),
        }
    }
}

/// Pure expansion `vx = a(t)·x`, `vy = a(t)·y`. Its analytic curl is zero.
pub fn calm_field(xc: &[f64], yc: &[f64], t: f64) -> (Field2, Field2) {
    let amplitude = 0.35 + 0.05 * (0.4 * t).sin();
    let ny = yc.len();
    let nx = xc.len();
    let vx = Field2::from_fn(ny, nx, |_, j| amplitude * xc[j]);
    let vy = Field2::from_fn(ny, nx, |i, _| amplitude * yc[i]);
    (vx, vy)
}

/// Coherent rigid rotation `vx = −a(t)·y`, `vy = a(t)·x`.
pub fn coherent_vortex(xc: &[f64], yc: &[f64], t: f64) -> (Field2, Field2) {
    let amplitude = 1.0 + 0.35 * (0.6 * t).sin();
    let ny = yc.len();
    let nx = xc.len();
    let vx = Field2::from_fn(ny, nx, |i, _| -amplitude * yc[i]);
    let vy = Field2::from_fn(ny, nx, |_, j| amplitude * xc[j]);
    (vx, vy)
}

/// A deterministic superposition of four moving Gaussian vortices plus a weak
/// sinusoidal background.
pub fn multi_vortex_field(xc: &[f64], yc: &[f64], t: f64) -> (Field2, Field2) {
    let ny = yc.len();
    let nx = xc.len();

    let vortex_data = [
        (
            -0.85 + 0.15 * (0.37 * t).cos(),
            -0.65 + 0.12 * (0.43 * t).sin(),
            1.30,
            0.55,
        ),
        (
            0.80 + 0.13 * (0.31 * t).sin(),
            -0.55 + 0.14 * (0.47 * t).cos(),
            -1.05,
            0.48,
        ),
        (
            -0.45 + 0.12 * (0.53 * t).sin(),
            0.85 + 0.10 * (0.29 * t).cos(),
            0.90,
            0.62,
        ),
        (
            0.75 + 0.11 * (0.41 * t).cos(),
            0.70 + 0.13 * (0.35 * t).sin(),
            -1.20,
            0.52,
        ),
    ];

    let background = 0.08 * (0.5 * t).sin();

    let mut vx = Field2::zeros(ny, nx);
    let mut vy = Field2::zeros(ny, nx);

    for i in 0..ny
    {
        for j in 0..nx
        {
            let x = xc[j];
            let y = yc[i];
            let mut ux = 0.0;
            let mut uy = 0.0;
            for &(center_x, center_y, strength, width) in &vortex_data
            {
                let dx = x - center_x;
                let dy = y - center_y;
                let radius_squared = dx * dx + dy * dy;
                let envelope = (-radius_squared / (2.0 * width * width)).exp();
                let temporal_strength = strength * (1.0 + 0.18 * (0.7 * t + strength).sin());
                ux += -temporal_strength * dy * envelope;
                uy += temporal_strength * dx * envelope;
            }
            ux += background * (PI * y).sin();
            uy += background * (PI * x).sin();
            *vx.get_mut(i, j) = ux;
            *vy.get_mut(i, j) = uy;
        }
    }

    (vx, vy)
}

/// The synthetic curvature weighting shared by all scenarios. It is a
/// deterministic function of position and time, not a physical metric.
pub fn curvature_field(xc: &[f64], yc: &[f64], t: f64) -> Field2 {
    let ny = yc.len();
    let nx = xc.len();
    let mx = 0.65 * (0.22 * t).cos();
    let my = 0.65 * (0.22 * t).sin();
    Field2::from_fn(ny, nx, |i, j| {
        let x = xc[j];
        let y = yc[i];
        let central = 0.42 * (-(x * x + y * y) / (0.9 * 0.9)).exp();
        let moving = 0.20 * (-(((x - mx) * (x - mx)) + (y - my) * (y - my)) / (0.45 * 0.45)).exp();
        let negative_region =
            -0.12 * (-(((x + 1.0) * (x + 1.0)) + (y - 0.75) * (y - 0.75)) / (0.6 * 0.6)).exp();
        central + moving + negative_region
    })
}
