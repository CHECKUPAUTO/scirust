//! Planar inertial dead-reckoning (a simplified strapdown mechanization).
//!
//! Given acceleration already expressed in the navigation (tangent) frame, the
//! navigation equations integrate it to velocity and position. This is the part
//! of inertial navigation that needs **no external signal** — and exactly why
//! it drifts: any acceleration bias integrates twice into an unbounded position
//! error. That drift is what GNSS fusion ([`crate::fusion`]) corrects.
//!
//! Honesty note: this is a *planar* mechanization with acceleration supplied in
//! the nav frame. It does not integrate attitude, Earth rotation, or Coriolis
//! terms — it is the kinematic core, not a full inertial navigation system.

/// A 2-D dead-reckoning integrator: position and velocity in a tangent frame.
#[derive(Debug, Clone, Copy)]
pub struct Ins2d {
    /// Position `[x, y]` (metres).
    pub pos: [f64; 2],
    /// Velocity `[vx, vy]` (metres/second).
    pub vel: [f64; 2],
}

impl Ins2d {
    /// New mechanization from an initial position and velocity.
    pub fn new(pos: [f64; 2], vel: [f64; 2]) -> Self {
        Self { pos, vel }
    }

    /// Advance by `dt` seconds under nav-frame acceleration `accel` `[ax, ay]`.
    /// Exact for constant acceleration over the step (`p += v·dt + ½·a·dt²`).
    #[allow(clippy::needless_range_loop)]
    pub fn propagate(&mut self, accel: [f64; 2], dt: f64) {
        for i in 0..2
        {
            self.pos[i] += self.vel[i] * dt + 0.5 * accel[i] * dt * dt;
            self.vel[i] += accel[i] * dt;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[allow(clippy::needless_range_loop)]
    fn constant_acceleration_matches_the_kinematic_closed_form() {
        // a = (0.5, -0.2), start at rest at origin; after t the closed form is
        // p = ½ a t², v = a t. Integrate in small steps and compare.
        let a = [0.5, -0.2];
        let dt = 1e-3;
        let mut ins = Ins2d::new([0.0, 0.0], [0.0, 0.0]);
        let steps = 2000;
        for _ in 0..steps
        {
            ins.propagate(a, dt);
        }
        let t = dt * steps as f64;
        for i in 0..2
        {
            assert!((ins.pos[i] - 0.5 * a[i] * t * t).abs() < 1e-9, "pos drift");
            assert!((ins.vel[i] - a[i] * t).abs() < 1e-9, "vel drift");
        }
    }

    #[test]
    #[allow(clippy::needless_range_loop)]
    fn nonzero_initial_velocity_obeys_the_full_kinematic_law() {
        // The earlier test starts from rest, so it never exercises the v₀·dt term.
        // Here both the initial velocity and the acceleration are nonzero. Closed
        // form: p(t) = p₀ + v₀·t + ½·a·t², v(t) = v₀ + a·t. With p₀=(1,−2),
        // v₀=(3,1), a=(0.5,−0.4) at t=4 s:
        //   pₓ = 1 + 3·4 + ½·0.5·16 = 17       p_y = −2 + 1·4 + ½·(−0.4)·16 = −1.2
        //   vₓ = 3 + 0.5·4 = 5                 v_y = 1 + (−0.4)·4 = −0.6
        let p0 = [1.0, -2.0];
        let v0 = [3.0, 1.0];
        let a = [0.5, -0.4];
        let dt = 1e-3;
        let steps = 4000;
        let mut ins = Ins2d::new(p0, v0);
        for _ in 0..steps
        {
            ins.propagate(a, dt);
        }
        let expect_p = [17.0, -1.2];
        let expect_v = [5.0, -0.6];
        for i in 0..2
        {
            assert!(
                (ins.pos[i] - expect_p[i]).abs() < 1e-9,
                "pos[{i}]={}",
                ins.pos[i]
            );
            assert!(
                (ins.vel[i] - expect_v[i]).abs() < 1e-9,
                "vel[{i}]={}",
                ins.vel[i]
            );
        }
    }

    #[test]
    fn a_constant_bias_drifts_quadratically() {
        // The signature failure mode: a small unmodelled acceleration bias
        // integrates into a position error that grows like t².
        let bias = [0.01, 0.0];
        let dt = 0.01;
        let mut ins = Ins2d::new([0.0, 0.0], [0.0, 0.0]);
        for _ in 0..1000
        {
            ins.propagate(bias, dt); // 10 s of pure bias
        }
        let t = 10.0;
        assert!((ins.pos[0] - 0.5 * bias[0] * t * t).abs() < 1e-6);
        assert!(ins.pos[0] > 0.4, "bias should accumulate a real error");
    }
}
