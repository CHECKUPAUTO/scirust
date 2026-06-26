//! PID controller with anti-windup, and relay (Åström–Hägglund) auto-tuning.

use serde::{Deserialize, Serialize};

/// Discrete PID controller with output clamping and clamping anti-windup.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Pid {
    kp: f64,
    ki: f64,
    kd: f64,
    dt: f64,
    out_min: f64,
    out_max: f64,
    integral: f64,
    prev_error: f64,
    has_prev: bool,
}

impl Pid {
    /// New PID with gains and sample time `dt`, output unbounded.
    pub fn new(kp: f64, ki: f64, kd: f64, dt: f64) -> Self {
        Self {
            kp,
            ki,
            kd,
            dt,
            out_min: f64::NEG_INFINITY,
            out_max: f64::INFINITY,
            integral: 0.0,
            prev_error: 0.0,
            has_prev: false,
        }
    }

    /// Set actuator saturation limits.
    pub fn with_limits(mut self, out_min: f64, out_max: f64) -> Self {
        self.out_min = out_min;
        self.out_max = out_max;
        self
    }

    /// Reset internal state.
    pub fn reset(&mut self) {
        self.integral = 0.0;
        self.prev_error = 0.0;
        self.has_prev = false;
    }

    /// Compute the control output for a setpoint and measurement.
    pub fn update(&mut self, setpoint: f64, measurement: f64) -> f64 {
        let error = setpoint - measurement;
        let trial_integral = self.integral + error * self.dt;
        let deriv = if self.has_prev
        {
            (error - self.prev_error) / self.dt
        }
        else
        {
            0.0
        };
        let unclamped = self.kp * error + self.ki * trial_integral + self.kd * deriv;
        let out = unclamped.clamp(self.out_min, self.out_max);
        // Clamping anti-windup: only integrate when not saturated (or when
        // integrating would move the output back into range).
        if (out - unclamped).abs() < 1e-12
            || (unclamped > self.out_max && error < 0.0)
            || (unclamped < self.out_min && error > 0.0)
        {
            self.integral = trial_integral;
        }
        self.prev_error = error;
        self.has_prev = true;
        out
    }
}

/// Result of relay (Åström–Hägglund) auto-tuning: the ultimate gain `ku` and
/// period `tu`, plus Ziegler–Nichols PID gains.
#[derive(Debug, Clone, Copy)]
pub struct RelayTuning {
    pub ku: f64,
    pub tu: f64,
    pub kp: f64,
    pub ki: f64,
    pub kd: f64,
}

/// Relay-feedback auto-tune around a first-order-plus-deadtime plant simulated
/// by `plant(u) -> y` (a closure advancing the plant one `dt` step). Drives a
/// relay of amplitude `d` and reads the sustained oscillation amplitude/period.
pub fn relay_autotune(
    mut plant: impl FnMut(f64) -> f64,
    setpoint: f64,
    d: f64,
    dt: f64,
    steps: usize,
) -> Option<RelayTuning> {
    let mut last_sign = 1.0;
    let mut crossings = Vec::new();
    let mut amax = f64::MIN;
    let mut amin = f64::MAX;
    // Relay feedback: exactly one plant step per control period. The relay value
    // is chosen from the previous output, applied once, and the returned value
    // is the new measurement. (An earlier version called `plant(0.0)` first to
    // "measure" — but the closure *advances* the plant, so that injected a
    // spurious zero-input step every period and double-advanced the clock,
    // distorting the limit cycle and halving the identified Tu.)
    let mut y = 0.0; // plant assumed to start at rest
    for k in 0..steps
    {
        let e = setpoint - y;
        let sign = if e >= 0.0 { 1.0 } else { -1.0 };
        y = plant(d * sign); // apply relay, read the resulting output
        if k > steps / 3
        {
            // settle, then record
            amax = amax.max(y);
            amin = amin.min(y);
            if sign != last_sign
            {
                crossings.push(k as f64 * dt);
            }
        }
        last_sign = sign;
    }
    if crossings.len() < 3
    {
        return None;
    }
    // Period from successive same-direction crossings (two half-periods).
    let periods: Vec<f64> = crossings.windows(2).map(|w| 2.0 * (w[1] - w[0])).collect();
    let tu = periods.iter().sum::<f64>() / periods.len() as f64;
    let a = 0.5 * (amax - amin); // oscillation amplitude
    if a <= 0.0
    {
        return None;
    }
    let ku = 4.0 * d / (core::f64::consts::PI * a);
    // Ziegler–Nichols classic PID.
    Some(RelayTuning {
        ku,
        tu,
        kp: 0.6 * ku,
        ki: 1.2 * ku / tu,
        kd: 0.075 * ku * tu,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// First-order plant y' = (-y + g·u)/tau, Euler-discretized.
    struct Plant {
        y: f64,
        tau: f64,
        g: f64,
        dt: f64,
    }
    impl Plant {
        fn step(&mut self, u: f64) -> f64 {
            self.y += self.dt * (-self.y + self.g * u) / self.tau;
            self.y
        }
    }

    #[test]
    fn pid_reaches_setpoint_with_no_steady_state_error() {
        let dt = 0.1;
        let mut pid = Pid::new(2.0, 1.0, 0.05, dt).with_limits(-10.0, 10.0);
        let mut plant = Plant {
            y: 0.0,
            tau: 1.0,
            g: 1.0,
            dt,
        };
        let sp = 5.0;
        let mut y = 0.0;
        for _ in 0..2000
        {
            let u = pid.update(sp, y);
            y = plant.step(u);
        }
        assert!((y - sp).abs() < 0.02, "y {y} did not reach setpoint {sp}");
    }

    #[test]
    fn anti_windup_keeps_integral_bounded() {
        // Saturating actuator + unreachable setpoint: integral must not blow up.
        let dt = 0.1;
        let mut pid = Pid::new(1.0, 2.0, 0.0, dt).with_limits(-1.0, 1.0);
        let mut plant = Plant {
            y: 0.0,
            tau: 1.0,
            g: 1.0,
            dt,
        };
        let sp = 100.0; // far beyond what u∈[-1,1] can reach (steady ~1.0)
        let mut y = 0.0;
        for _ in 0..500
        {
            let u = pid.update(sp, y);
            assert!((-1.0..=1.0).contains(&u));
            y = plant.step(u);
        }
        assert!(
            pid.integral.abs() < 1e3,
            "integral wound up: {}",
            pid.integral
        );
    }

    #[test]
    fn relay_autotune_recovers_the_ultimate_point_of_a_cubic_plant() {
        // Plant G(s) = 1/(s+1)³ — three unity first-order lags in series. Its
        // ultimate (−180° phase) point is ANALYTIC:
        //   ∠G(jω) = −3·atan(ω) = −π  ⇒  ω_u = tan(π/3) = √3,
        //   T_u = 2π/ω_u = 2π/√3 ≈ 3.6276 s,
        //   |G(jω_u)| = 1/(1+ω_u²)^{3/2} = 1/8  ⇒  K_u = 1/|G| = 8.
        // Relay feedback must rediscover this point: the limit-cycle period is
        // T_u, and the describing-function gain 4d/(πa) recovers K_u to within
        // the method's first-harmonic accuracy.
        let dt = 0.01;
        let (mut x1, mut x2, mut x3) = (0.0_f64, 0.0_f64, 0.0_f64);
        let plant = move |u: f64| -> f64 {
            x1 += dt * (-x1 + u);
            x2 += dt * (-x2 + x1);
            x3 += dt * (-x3 + x2);
            x3
        };
        let t = relay_autotune(plant, 0.0, 1.0, dt, 6000)
            .expect("a cubic plant sustains a relay limit cycle");
        let tu_exact = 2.0 * core::f64::consts::PI / 3.0_f64.sqrt();
        // In practice this recovers Tu to ≈0.3% and Ku to ≈1%; the bands below
        // leave generous margin for the describing-function approximation while
        // still pinning the result to the analytic ultimate point.
        assert!(
            (t.tu - tu_exact).abs() / tu_exact < 0.03,
            "Tu {} not within 3% of analytic {tu_exact}",
            t.tu
        );
        assert!(
            (t.ku - 8.0).abs() / 8.0 < 0.08,
            "Ku {} not within 8% of analytic 8",
            t.ku
        );
        // Ziegler–Nichols gains are the documented functions of (Ku, Tu).
        assert!((t.kp - 0.6 * t.ku).abs() < 1e-12);
        assert!((t.ki - 1.2 * t.ku / t.tu).abs() < 1e-12);
        assert!((t.kd - 0.075 * t.ku * t.tu).abs() < 1e-12);
    }

    #[test]
    fn relay_autotune_returns_none_without_enough_oscillation() {
        // Too few steps to observe ≥3 relay switches ⇒ no usable estimate.
        let mut plant = Plant {
            y: 0.0,
            tau: 1.0,
            g: 1.0,
            dt: 0.05,
        };
        let step = move |u: f64| plant.step(u);
        assert!(relay_autotune(step, 1.0, 1.0, 0.05, 6).is_none());
    }
}
