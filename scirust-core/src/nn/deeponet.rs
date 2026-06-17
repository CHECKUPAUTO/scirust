//! **DeepONet** — operator learning (Lu, Jin, Pang, Zhang & Karniadakis, *Nature
//! Machine Intelligence* 2021).
//!
//! A DeepONet learns a whole **operator** `G : u ↦ G(u)` (a function-to-function
//! map) rather than a single function. It factors the output as a **branch × trunk**
//! inner product: `G(u)(y) ≈ Σ_k b_k(u)·t_k(y)`, where the **branch** encodes the
//! input function `u` (sampled at fixed sensors) and the **trunk** encodes the
//! query location `y`. Once trained it generalises to **unseen input functions**.
//!
//! This implementation is the **fixed-trunk / linear-branch** variant
//! (POD-DeepONet, Lu et al. 2022): the trunk is a fixed Fourier-cosine basis
//! `t_k(y) = cos(kπy)` and the branch is a learned **linear** map of the sensor
//! values. That makes the fit a **convex** least-squares problem and is *exact*
//! for linear operators such as the **antiderivative** `G(u)(y) = ∫₀^y u`. Pure
//! `f32` in a fixed order ⇒ **deterministic**.

use std::f32::consts::PI;

/// A DeepONet with a fixed cosine trunk and a learned linear branch.
pub struct DeepONet {
    branch: Vec<f32>, // `trunk_dim × sensors`, row-major: `B[k][i] = branch[k*m+i]`
    p: usize,         // trunk dimension (number of basis functions)
    m: usize,         // number of sensors
}

impl DeepONet {
    /// New DeepONet for inputs sampled at `sensors` points and a trunk of
    /// `trunk_dim` cosine basis functions (`cos(kπy)`, `k = 0..trunk_dim`).
    pub fn new(sensors: usize, trunk_dim: usize) -> Self {
        assert!(
            sensors > 0 && trunk_dim > 0,
            "DeepONet: sensors, trunk_dim > 0"
        );
        Self {
            branch: vec![0.0f32; trunk_dim * sensors],
            p: trunk_dim,
            m: sensors,
        }
    }

    /// The fixed trunk basis at `y`: `[cos(0), cos(πy), …, cos((p−1)πy)]`.
    fn trunk(&self, y: f32) -> Vec<f32> {
        (0..self.p).map(|k| (k as f32 * PI * y).cos()).collect()
    }

    /// Evaluate `G(u)(y) ≈ Σ_k (Σ_i B[k][i]·u[i])·cos(kπy)`.
    pub fn eval(&self, u: &[f32], y: f32) -> f32 {
        assert_eq!(u.len(), self.m, "DeepONet: sensor count mismatch");
        let phi = self.trunk(y);
        (0..self.p)
            .map(|k| {
                let row = &self.branch[k * self.m..(k + 1) * self.m];
                let bk: f32 = row.iter().zip(u).map(|(&b, &ui)| b * ui).sum();
                bk * phi[k]
            })
            .sum()
    }

    /// Convex gradient-descent fit on samples `(us[s], ys[s], targets[s])`
    /// (`us[s]` are sensor readings, `ys[s]` a query location, `targets[s] =
    /// G(u)(y)`). Returns the final mean squared error.
    pub fn fit(
        &mut self,
        us: &[Vec<f32>],
        ys: &[f32],
        targets: &[f32],
        steps: usize,
        lr: f32,
    ) -> f32 {
        let s = us.len();
        assert!(
            s > 0 && ys.len() == s && targets.len() == s,
            "DeepONet: dataset length mismatch"
        );
        let phis: Vec<Vec<f32>> = ys.iter().map(|&y| self.trunk(y)).collect();
        let inv_s = 1.0 / s as f32;
        let mut mse = 0.0f32;
        for _ in 0..steps
        {
            let mut grad = vec![0.0f32; self.branch.len()];
            mse = 0.0;
            for ((u, phi), &t) in us.iter().zip(&phis).zip(targets)
            {
                // pred = Σ_{k,i} B[k][i]·u[i]·phi[k].
                let mut pred = 0.0f32;
                for (k, &pk) in phi.iter().enumerate()
                {
                    let row = &self.branch[k * self.m..(k + 1) * self.m];
                    let bk: f32 = row.iter().zip(u).map(|(&b, &ui)| b * ui).sum();
                    pred += bk * pk;
                }
                let e = pred - t;
                mse += e * e;
                let ge = 2.0 * e * inv_s;
                for (k, &pk) in phi.iter().enumerate()
                {
                    let gphi = ge * pk;
                    let grow = &mut grad[k * self.m..(k + 1) * self.m];
                    for (g, &ui) in grow.iter_mut().zip(u)
                    {
                        *g += gphi * ui;
                    }
                }
            }
            mse *= inv_s;
            for (b, g) in self.branch.iter_mut().zip(&grad)
            {
                *b -= lr * g;
            }
        }
        mse
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::PcgEngine;

    /// Build a dataset for the **antiderivative** operator `G(u)(y) = ∫₀^y u(s)ds`,
    /// with input functions `u(x) = Σ_{k=1}^{4} a_k sin(kπx)` (random `a_k`). The
    /// exact antiderivative is `Σ_k a_k (1 − cos(kπy))/(kπ)`.
    fn antideriv_dataset(
        rng: &mut PcgEngine,
        n_funcs: usize,
        m: usize,
        ys: &[f32],
    ) -> (Vec<Vec<f32>>, Vec<f32>, Vec<f32>) {
        let kmax = 4usize;
        let (mut us, mut yy, mut tt) = (Vec::new(), Vec::new(), Vec::new());
        for _ in 0..n_funcs
        {
            let a: Vec<f32> = (0..kmax).map(|_| rng.float_signed()).collect();
            let u: Vec<f32> = (0..m)
                .map(|i| {
                    let x = i as f32 / (m as f32 - 1.0);
                    a.iter()
                        .enumerate()
                        .map(|(k, &ak)| ak * ((k + 1) as f32 * PI * x).sin())
                        .sum()
                })
                .collect();
            for &y in ys
            {
                let g: f32 = a
                    .iter()
                    .enumerate()
                    .map(|(k, &ak)| {
                        let kk = (k + 1) as f32;
                        ak * (1.0 - (kk * PI * y).cos()) / (kk * PI)
                    })
                    .sum();
                us.push(u.clone());
                yy.push(y);
                tt.push(g);
            }
        }
        (us, yy, tt)
    }

    /// **DeepONet operator learning, tested.** Trained on some input functions, it
    /// approximates the antiderivative operator on **unseen** input functions to
    /// low error — far below a constant (mean) predictor.
    #[test]
    fn deeponet_learns_antiderivative_and_generalizes() {
        let mut rng = PcgEngine::new(5);
        let m = 20usize;
        let ys: Vec<f32> = (0..=10).map(|i| i as f32 / 10.0).collect(); // y ∈ [0,1]
        // Trunk dim 5 = {1, cos(πy), …, cos(4πy)} spans the antiderivatives.
        let (utr, ytr, ttr) = antideriv_dataset(&mut rng, 40, m, &ys);
        let (ute, yte, tte) = antideriv_dataset(&mut rng, 20, m, &ys);

        let mut net = DeepONet::new(m, 5);
        net.fit(&utr, &ytr, &ttr, 3000, 0.1);

        let test_mse: f32 = (0..ute.len())
            .map(|s| (net.eval(&ute[s], yte[s]) - tte[s]).powi(2))
            .sum::<f32>()
            / ute.len() as f32;
        let mean: f32 = tte.iter().sum::<f32>() / tte.len() as f32;
        let base_mse: f32 = tte.iter().map(|&t| (t - mean).powi(2)).sum::<f32>() / tte.len() as f32;
        assert!(test_mse < 0.01, "DeepONet test MSE {test_mse} too high");
        assert!(
            test_mse < 0.1 * base_mse,
            "DeepONet did not learn the operator: {test_mse} vs baseline {base_mse}"
        );
    }

    /// The fit is bit-for-bit deterministic.
    #[test]
    fn deeponet_is_deterministic() {
        let mut rng = PcgEngine::new(2);
        let m = 12usize;
        let ys: Vec<f32> = (0..=5).map(|i| i as f32 / 5.0).collect();
        let (u, y, t) = antideriv_dataset(&mut rng, 10, m, &ys);
        let fit = || {
            let mut net = DeepONet::new(m, 4);
            let e = net.fit(&u, &y, &t, 200, 0.1);
            (e.to_bits(), net.eval(&u[0], 0.5).to_bits())
        };
        assert_eq!(fit(), fit());
    }
}
