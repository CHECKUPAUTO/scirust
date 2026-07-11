//! A single-zone building thermal plant: the **2R2C** model (air node + wall
//! thermal mass) that HVAC control and fault-detection (the `scirust-hvac`
//! vertical) reason about. Here it is a simulator: fix the outside temperature
//! and the HVAC heat input and integrate.
//!
//! State `y = [t_air, t_wall]`:
//!
//! - `t_air'  = ((t_wall − t_air)/R_aw + Q) / C_air`
//! - `t_wall' = ((t_air − t_wall)/R_aw + (t_out − t_wall)/R_wo) / C_wall`
//!
//! `R_aw` couples the air to the wall mass, `R_wo` the wall to the outside,
//! `Q` is the HVAC power delivered to the air. The steady state is exact and
//! linear in `Q` — `t_wall = t_out + Q·R_wo`, `t_air = t_out + Q·(R_aw+R_wo)`
//! — the oracle the tests use; with `Q = 0` the zone relaxes to the outside
//! temperature (biexponentially, two thermal masses).

use crate::engine::{SimError, System};

fn check_positive(name: &str, value: f64) -> Result<(), SimError> {
    if value.is_finite() && value > 0.0
    {
        Ok(())
    }
    else
    {
        Err(SimError::BadInput(format!(
            "{name} = {value} must be finite and positive"
        )))
    }
}

fn check_finite(name: &str, value: f64) -> Result<(), SimError> {
    if value.is_finite()
    {
        Ok(())
    }
    else
    {
        Err(SimError::BadInput(format!(
            "{name} = {value} must be finite"
        )))
    }
}

/// A 2R2C single-zone thermal model driven by a constant outside temperature
/// and HVAC heat input.
#[derive(Debug, Clone, PartialEq)]
pub struct ZoneThermal2R2C {
    c_air: f64,
    c_wall: f64,
    r_aw: f64,
    r_wo: f64,
    t_outside: f64,
    q_hvac: f64,
}

impl ZoneThermal2R2C {
    /// Create the model. `c_air`, `c_wall`, `r_aw`, `r_wo` must be finite and
    /// positive; `t_outside` and `q_hvac` finite.
    pub fn new(
        c_air: f64,
        c_wall: f64,
        r_aw: f64,
        r_wo: f64,
        t_outside: f64,
        q_hvac: f64,
    ) -> Result<Self, SimError> {
        check_positive("c_air", c_air)?;
        check_positive("c_wall", c_wall)?;
        check_positive("r_aw", r_aw)?;
        check_positive("r_wo", r_wo)?;
        check_finite("t_outside", t_outside)?;
        check_finite("q_hvac", q_hvac)?;
        Ok(ZoneThermal2R2C {
            c_air,
            c_wall,
            r_aw,
            r_wo,
            t_outside,
            q_hvac,
        })
    }

    /// The steady-state temperatures `(t_air, t_wall)`:
    /// `t_air = t_out + Q·(R_aw + R_wo)`, `t_wall = t_out + Q·R_wo`.
    pub fn steady_state(&self) -> (f64, f64) {
        let t_wall = self.t_outside + self.q_hvac * self.r_wo;
        let t_air = self.t_outside + self.q_hvac * (self.r_aw + self.r_wo);
        (t_air, t_wall)
    }

    /// The static heat-loss coefficient of the zone (W/K), `1/(R_aw + R_wo)`:
    /// the HVAC power needed per kelvin of steady-state air-temperature lift.
    pub fn conductance(&self) -> f64 {
        1.0 / (self.r_aw + self.r_wo)
    }
}

impl System for ZoneThermal2R2C {
    fn dim(&self) -> usize {
        2
    }

    fn derivatives(&self, _t: f64, y: &[f64], dydt: &mut [f64]) {
        let (t_air, t_wall) = (y[0], y[1]);
        dydt[0] = ((t_wall - t_air) / self.r_aw + self.q_hvac) / self.c_air;
        dydt[1] =
            ((t_air - t_wall) / self.r_aw + (self.t_outside - t_wall) / self.r_wo) / self.c_wall;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::simulate;

    fn heated() -> ZoneThermal2R2C {
        // Air: 1.2 kJ/K, wall mass: 20 kJ/K; R_aw = 0.05 K/W, R_wo = 0.2 K/W;
        // 5 °C outside, 500 W of heating.
        ZoneThermal2R2C::new(1_200.0, 20_000.0, 0.05, 0.2, 5.0, 500.0).unwrap()
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn heated_zone_reaches_the_linear_steady_state() {
        let zone = heated();
        let (air_ss, wall_ss) = zone.steady_state();
        // t_air = 5 + 500*0.25 = 130; t_wall = 5 + 500*0.2 = 105.
        assert!((air_ss - 130.0).abs() < 1e-9 && (wall_ss - 105.0).abs() < 1e-9);
        // The wall mass dominates the time constant; integrate several hours.
        let traj = simulate(&zone, &[5.0, 5.0], 0.0, 40_000.0, 1.0).unwrap();
        let last = traj.last_state().unwrap();
        assert!((last[0] - air_ss).abs() < 1e-2, "t_air = {}", last[0]);
        assert!((last[1] - wall_ss).abs() < 1e-2, "t_wall = {}", last[1]);
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn unheated_zone_relaxes_to_the_outside_temperature() {
        let zone = ZoneThermal2R2C::new(1_200.0, 20_000.0, 0.05, 0.2, 5.0, 0.0).unwrap();
        let traj = simulate(&zone, &[22.0, 20.0], 0.0, 60_000.0, 1.0).unwrap();
        let last = traj.last_state().unwrap();
        assert!((last[0] - 5.0).abs() < 1e-2 && (last[1] - 5.0).abs() < 1e-2);
        // Every temperature stays within the initial-to-outside band (no
        // overshoot for a passive two-mass system starting monotone).
        for row in &traj.y
        {
            assert!(
                (5.0 - 1e-6..=22.0 + 1e-6).contains(&row[0]),
                "t_air escaped: {}",
                row[0]
            );
        }
    }

    #[test]
    fn conductance_and_constructor_validation() {
        let zone = heated();
        assert!((zone.conductance() - 1.0 / 0.25).abs() < 1e-12);
        assert!(ZoneThermal2R2C::new(0.0, 20_000.0, 0.05, 0.2, 5.0, 500.0).is_err());
        assert!(ZoneThermal2R2C::new(1_200.0, 20_000.0, -0.05, 0.2, 5.0, 500.0).is_err());
        assert!(ZoneThermal2R2C::new(1_200.0, 20_000.0, 0.05, 0.2, f64::NAN, 500.0).is_err());
        assert!(ZoneThermal2R2C::new(1_200.0, 20_000.0, 0.05, 0.2, 5.0, f64::INFINITY).is_err());
    }
}
