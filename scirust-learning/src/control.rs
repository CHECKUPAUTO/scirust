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
        let dt = match self.prev_time
        {
            Some(t) => current_time - t,
            None => 0.0,
        };

        if dt > 0.0
        {
            self.integral += error * dt;
            let derivative = (error - self.prev_error) / dt;
            let output = self.kp * error + self.ki * self.integral + self.kd * derivative;
            self.prev_error = error;
            self.prev_time = Some(current_time);
            output
        }
        else
        {
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
        Self {
            x: initial_x,
            p: initial_p,
            q,
            r,
        }
    }

    /// Prediction step
    pub fn predict(&mut self) {
        // x = x (constant model for 1D)
        self.p += self.q;
    }

    /// Update step with a new measurement
    pub fn update(&mut self, z: f64) {
        let k = self.p / (self.p + self.r);
        self.x = self.x + k * (z - self.x);
        self.p *= 1.0 - k;
    }

    pub fn state(&self) -> f64 {
        self.x
    }
}

/// A Matrix-based Kalman Filter for multi-dimensional state estimation.
/// x_{k} = F * x_{k-1} + B * u_{k} + w_{k}
/// z_{k} = H * x_{k} + v_{k}
pub struct KalmanFilter {
    /// State estimate vector (n x 1)
    pub x: Vec<f64>,
    /// State covariance matrix (n x n)
    pub p: Vec<Vec<f64>>,
    /// State transition matrix (n x n)
    pub f: Vec<Vec<f64>>,
    /// Observation matrix (m x n)
    pub h: Vec<Vec<f64>>,
    /// Process noise covariance (n x n)
    pub q: Vec<Vec<f64>>,
    /// Measurement noise covariance (m x m)
    pub r: Vec<Vec<f64>>,
}

impl KalmanFilter {
    pub fn new(
        x: Vec<f64>,
        p: Vec<Vec<f64>>,
        f: Vec<Vec<f64>>,
        h: Vec<Vec<f64>>,
        q: Vec<Vec<f64>>,
        r: Vec<Vec<f64>>,
    ) -> Self {
        Self { x, p, f, h, q, r }
    }

    /// Predict the next state
    pub fn predict(&mut self) {
        // x = F * x
        let n = self.x.len();
        let mut new_x = vec![0.0; n];
        #[allow(clippy::needless_range_loop)]
        for i in 0..n
        {
            #[allow(clippy::needless_range_loop)]
            for j in 0..n
            {
                new_x[i] += self.f[i][j] * self.x[j];
            }
        }
        self.x = new_x;

        // P = F * P * F^T + Q
        let mut fp = vec![vec![0.0; n]; n];
        #[allow(clippy::needless_range_loop)]
        for i in 0..n
        {
            #[allow(clippy::needless_range_loop)]
            for j in 0..n
            {
                for k in 0..n
                {
                    fp[i][j] += self.f[i][k] * self.p[k][j];
                }
            }
        }
        let mut fpf_t = vec![vec![0.0; n]; n];
        #[allow(clippy::needless_range_loop)]
        for i in 0..n
        {
            #[allow(clippy::needless_range_loop)]
            for j in 0..n
            {
                for k in 0..n
                {
                    fpf_t[i][j] += fp[i][k] * self.f[j][k]; // F^T means f[j][k] instead of f[k][j]
                }
            }
        }
        #[allow(clippy::needless_range_loop)]
        for i in 0..n
        {
            #[allow(clippy::needless_range_loop)]
            for j in 0..n
            {
                self.p[i][j] = fpf_t[i][j] + self.q[i][j];
            }
        }
    }

    /// Update the state with a measurement z
    #[allow(clippy::needless_range_loop)]
    pub fn update(&mut self, z: &[f64]) {
        let n = self.x.len();
        let m = z.len();

        // y = z - H * x (innovation)
        let mut y = vec![0.0; m];
        #[allow(clippy::needless_range_loop)]
        for i in 0..m
        {
            let mut hx = 0.0;
            #[allow(clippy::needless_range_loop)]
            for j in 0..n
            {
                hx += self.h[i][j] * self.x[j];
            }
            y[i] = z[i] - hx;
        }

        // S = H * P * H^T + R
        let mut hp = vec![vec![0.0; n]; m];
        #[allow(clippy::needless_range_loop)]
        for i in 0..m
        {
            #[allow(clippy::needless_range_loop)]
            for j in 0..n
            {
                for k in 0..n
                {
                    hp[i][j] += self.h[i][k] * self.p[k][j];
                }
            }
        }
        let mut s = vec![vec![0.0; m]; m];
        #[allow(clippy::needless_range_loop)]
        for i in 0..m
        {
            for j in 0..m
            {
                for k in 0..n
                {
                    s[i][j] += hp[i][k] * self.h[j][k];
                }
                s[i][j] += self.r[i][j];
            }
        }

        // K = P * H^T * S^-1  (general m-dimensional measurement update).
        let s_inv = invert_matrix_gj(&s);

        // P * H^T  (n x m)
        let mut pht = vec![vec![0.0; m]; n];
        for i in 0..n
        {
            for j in 0..m
            {
                let mut acc = 0.0;
                for k in 0..n
                {
                    acc += self.p[i][k] * self.h[j][k];
                }
                pht[i][j] = acc;
            }
        }

        // K = (P H^T) * S^-1  (n x m)
        let mut k_gain = vec![vec![0.0; m]; n];
        for i in 0..n
        {
            for j in 0..m
            {
                let mut acc = 0.0;
                for l in 0..m
                {
                    acc += pht[i][l] * s_inv[l][j];
                }
                k_gain[i][j] = acc;
            }
        }

        // x = x + K * y
        for i in 0..n
        {
            let mut acc = 0.0;
            for j in 0..m
            {
                acc += k_gain[i][j] * y[j];
            }
            self.x[i] += acc;
        }

        // P = (I - K * H) * P
        let mut kh = vec![vec![0.0; n]; n];
        for i in 0..n
        {
            for j in 0..n
            {
                let mut acc = 0.0;
                for l in 0..m
                {
                    acc += k_gain[i][l] * self.h[l][j];
                }
                kh[i][j] = acc;
            }
        }
        let mut new_p = vec![vec![0.0; n]; n];
        for i in 0..n
        {
            for j in 0..n
            {
                let mut khp = 0.0;
                for k_idx in 0..n
                {
                    khp += kh[i][k_idx] * self.p[k_idx][j];
                }
                new_p[i][j] = self.p[i][j] - khp;
            }
        }
        self.p = new_p;
    }
}

/// Invert a square matrix by Gauss-Jordan elimination with partial pivoting.
///
/// Used for the Kalman innovation covariance `S` (m×m), which is positive
/// definite for a valid filter (`S = H P Hᵀ + R`). A (near-)singular pivot is
/// left as identity for that column rather than panicking.
#[allow(clippy::needless_range_loop)]
fn invert_matrix_gj(a: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let m = a.len();
    // Augmented [A | I].
    let mut aug: Vec<Vec<f64>> = (0..m)
        .map(|i| {
            let mut row = a[i].clone();
            row.extend((0..m).map(|j| if i == j { 1.0 } else { 0.0 }));
            row
        })
        .collect();

    for col in 0..m
    {
        // Partial pivot: largest magnitude in this column at/below the diagonal.
        let mut piv = col;
        for r in (col + 1)..m
        {
            if aug[r][col].abs() > aug[piv][col].abs()
            {
                piv = r;
            }
        }
        if aug[piv][col].abs() < 1e-12
        {
            continue; // singular column; skip (leaves identity there)
        }
        aug.swap(col, piv);

        let d = aug[col][col];
        for j in 0..(2 * m)
        {
            aug[col][j] /= d;
        }
        for r in 0..m
        {
            if r != col
            {
                let f = aug[r][col];
                if f != 0.0
                {
                    for j in 0..(2 * m)
                    {
                        aug[r][j] -= f * aug[col][j];
                    }
                }
            }
        }
    }

    aug.iter().map(|row| row[m..(2 * m)].to_vec()).collect()
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
    fn kalman_multidim_update_moves_state_and_shrinks_covariance() {
        // 2D state, 2D measurement, H = I. Prior x=[0,0], P=I, R=0.1 I, z=[1,1].
        // The m>1 update path was previously a silent no-op; it must now move the
        // state toward the measurement and shrink the covariance.
        let mut kf = KalmanFilter::new(
            vec![0.0, 0.0],
            vec![vec![1.0, 0.0], vec![0.0, 1.0]], // P
            vec![vec![1.0, 0.0], vec![0.0, 1.0]], // F
            vec![vec![1.0, 0.0], vec![0.0, 1.0]], // H
            vec![vec![0.0, 0.0], vec![0.0, 0.0]], // Q
            vec![vec![0.1, 0.0], vec![0.0, 0.1]], // R
        );
        kf.update(&[1.0, 1.0]);
        // K = P (P+R)^-1 = 1/1.1 I; x = K z = [1/1.1, 1/1.1].
        assert!((kf.x[0] - 1.0 / 1.1).abs() < 1e-9, "x0={}", kf.x[0]);
        assert!((kf.x[1] - 1.0 / 1.1).abs() < 1e-9, "x1={}", kf.x[1]);
        // Posterior variance (I-K)P = 1/11 < 1 (prior); dims stay independent.
        assert!(kf.p[0][0] > 0.0 && kf.p[0][0] < 0.5, "p00={}", kf.p[0][0]);
        assert!(kf.p[0][1].abs() < 1e-9, "p01={}", kf.p[0][1]);
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
        for _ in 0..100
        {
            kf.predict();
            kf.update(10.0);
        }
        assert!((kf.state() - 10.0).abs() < 0.1);
    }

    #[test]
    fn test_kalman_filter_matrix() {
        // 1D motion: x = [pos, vel]^T
        let x = vec![0.0, 1.0];
        let p = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let f = vec![vec![1.0, 1.0], vec![0.0, 1.0]]; // dt = 1
        let h = vec![vec![1.0, 0.0]]; // measure position
        let q = vec![vec![0.1, 0.0], vec![0.0, 0.1]];
        let r = vec![vec![0.1]];

        let mut kf = KalmanFilter::new(x, p, f, h, q, r);

        kf.predict();
        // x should be [1, 1]
        assert!((kf.x[0] - 1.0).abs() < 1e-10);

        kf.update(&[1.1]);
        // pos should be close to 1.1
        assert!((kf.x[0] - 1.1).abs() < 0.1);
    }
}
