//! Control algorithms for autonomous systems and trajectory tracking.

/// A simple PID (Proportional-Integral-Derivative) controller.
pub struct PidController {
    pub kp: f64,
    pub ki: f64,
    pub kd: f64,
    integral: f64,
    prev_error: f64,
    prev_time: Option<f64>,
}

impl PidController {
    pub fn new(kp: f64, ki: f64, kd: f64) -> Self {
        Self {
            kp,
            ki,
            kd,
            integral: 0.0,
            prev_error: 0.0,
            prev_time: None,
        }
    }

    /// Update the PID controller with the current error and current time.
    /// Returns the control output.
    pub fn update(&mut self, error: f64, current_time: f64) -> f64 {
        let dt = match self.prev_time {
            Some(t) => current_time - t,
            None => 0.0,
        };

        if dt > 0.0 {
            self.integral += error * dt;
            let derivative = (error - self.prev_error) / dt;
            let output = self.kp * error + self.ki * self.integral + self.kd * derivative;
            self.prev_error = error;
            self.prev_time = Some(current_time);
            output
        } else {
            self.prev_time = Some(current_time);
            self.prev_error = error;
            self.kp * error
        }
    }

    pub fn reset(&mut self) {
        self.integral = 0.0;
        self.prev_error = 0.0;
        self.prev_time = None;
    }
}

/// A basic 1D Kalman Filter.
pub struct KalmanFilter1D {
    /// State estimate
    x: f64,
    /// Estimation error covariance
    p: f64,
    /// Process noise covariance
    q: f64,
    /// Measurement noise covariance
    r: f64,
}

impl KalmanFilter1D {
    pub fn new(initial_x: f64, initial_p: f64, q: f64, r: f64) -> Self {
        Self { x: initial_x, p: initial_p, q, r }
    }

    /// Prediction step
    pub fn predict(&mut self) {
        // x = x (constant model for 1D)
        self.p = self.p + self.q;
    }

    /// Update step with a new measurement
    pub fn update(&mut self, z: f64) {
        let k = self.p / (self.p + self.r);
        self.x = self.x + k * (z - self.x);
        self.p = (1.0 - k) * self.p;
    }

    pub fn state(&self) -> f64 {
        self.x
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pid_controller() {
        let mut pid = PidController::new(1.0, 0.1, 0.05);
        let error = 10.0;
        let out1 = pid.update(error, 0.0);
        assert_eq!(out1, 10.0); // Kp * error = 1 * 10

        let out2 = pid.update(8.0, 1.0);
        // dt = 1.0
        // integral = 8.0 * 1.0 = 8.0
        // derivative = (8.0 - 10.0) / 1.0 = -2.0
        // output = 1.0 * 8.0 + 0.1 * 8.0 + 0.05 * (-2.0) = 8.0 + 0.8 - 0.1 = 8.7
        assert!((out2 - 8.7).abs() < 1e-10);
    }

    #[test]
    fn test_kalman_filter() {
        let mut kf = KalmanFilter1D::new(0.0, 1.0, 0.1, 0.1);

        // Measurement is 10.0
        kf.predict();
        kf.update(10.0);
        let state1 = kf.state();
        assert!(state1 > 0.0 && state1 < 10.0);

        // After many measurements, state should approach 10.0
        for _ in 0..100 {
            kf.predict();
            kf.update(10.0);
        }
        assert!((kf.state() - 10.0).abs() < 0.1);
    }
}
