// scirust-core/src/nn/batch_norm_2d.rs
//
// BatchNorm2d — normalisation par batch pour convolutions.
//
// Input attendu : (N, C*H*W) — la sortie flatten de Conv2d, en layout NCHW
// (channel-major : dans chaque ligne, le canal c occupe le bloc contigu
// [c*H*W .. (c+1)*H*W)).  Pour normaliser PAR CANAL sur (N, H, W), on
// réorganise en une matrice channel-major (C, N*H*W) où la ligne c contient
// les N*H*W valeurs du canal c ; les statistiques par canal sont alors une
// réduction par ligne.  (L'ancien reshape (N,C*H*W)->(N*H*W,C) mélangeait les
// canaux : il n'est correct que pour un input NHWC.)

use crate::autodiff::reverse::{Tape, Tensor, Var};
use crate::nn::module::Module;
use std::collections::HashMap;

pub struct BatchNorm2d {
    pub num_channels: usize,
    pub gamma: Tensor, // (1, C)
    pub beta: Tensor,  // (1, C)
    pub eps: f32,
    pub momentum: f32,
    pub running_mean: Tensor, // (1, C)
    pub running_var: Tensor,  // (1, C)
    pub training: bool,
    last_g_idx: Option<usize>,
    last_b_idx: Option<usize>,
    pub name: String,
}

impl BatchNorm2d {
    pub fn new(num_channels: usize) -> Self {
        Self {
            num_channels,
            gamma: Tensor::from_vec(vec![1.0; num_channels], 1, num_channels),
            beta: Tensor::zeros(1, num_channels),
            eps: 1e-5,
            momentum: 0.1,
            running_mean: Tensor::zeros(1, num_channels),
            running_var: Tensor::from_vec(vec![1.0; num_channels], 1, num_channels),
            training: true,
            last_g_idx: None,
            last_b_idx: None,
            name: format!("bn2d_{num_channels}"),
        }
    }

    #[must_use]
    pub fn with_name(mut self, name: &str) -> Self {
        self.name = name.into();
        self
    }

    pub fn set_training(&mut self, mode: bool) {
        self.training = mode;
    }

    fn update_running_stats(&mut self, batch_mean: &[f32], batch_var: &[f32]) {
        let alpha = self.momentum;
        for j in 0..self.num_channels
        {
            self.running_mean.data[j] =
                (1.0 - alpha) * self.running_mean.data[j] + alpha * batch_mean[j];
            self.running_var.data[j] =
                (1.0 - alpha) * self.running_var.data[j] + alpha * batch_var[j];
        }
    }
}

impl Module for BatchNorm2d {
    fn forward<'t>(&mut self, tape: &'t Tape, input: Var<'t>) -> Var<'t> {
        let (n, total_features) = input.shape();
        let c = self.num_channels;
        assert_eq!(
            total_features % c,
            0,
            "BatchNorm2d: total_features {total_features} not divisible by num_channels {c}"
        );
        let ncol = total_features / c * n; // N * H * W  (per-channel element count)

        // NCHW (N, C*HW) -> channel-major (C, N*HW): each row = one channel.
        //   transpose : (N, C*HW) -> (C*HW, N)
        //   reshape   : (C*HW, N) -> (C, HW*N)     [both are pure re-layouts]
        // xc[c, s*N + k] == input[k, c, s], so a per-row reduction over the
        // (N*HW) columns is exactly the per-channel statistic over (N, H, W).
        let xc = input.transpose().reshape(&[c, ncol]);

        // Params are stored (1, C); reshape to (C, 1) so they broadcast per-row.
        // Keep the (1, C) input node ids for parameter_indices/sync.
        let gamma_in = tape.input(self.gamma.clone());
        let beta_in = tape.input(self.beta.clone());
        self.last_g_idx = Some(gamma_in.idx());
        self.last_b_idx = Some(beta_in.idx());
        let gamma_col = gamma_in.reshape(&[c, 1]);
        let beta_col = beta_in.reshape(&[c, 1]);

        let eps_col = tape.input(Tensor::from_vec(vec![self.eps; c], c, 1));

        let normed = if self.training
        {
            let mean = xc.mean_axis(1); // (C,1) per-channel mean over N*HW
            let centered = xc.sub(mean.broadcast(c, ncol));
            let var = centered.hadamard(centered).mean_axis(1); // (C,1) biased var

            // Running-stat update from the detached batch statistics.
            let mean_v = tape.value(mean.idx());
            let var_v = tape.value(var.idx());
            self.update_running_stats(&mean_v.data, &var_v.data);

            let inv_std = var.add(eps_col).sqrt().reciprocal(); // (C,1)
            let x_hat = centered.hadamard(inv_std.broadcast(c, ncol));
            let scaled = x_hat.hadamard(gamma_col.broadcast(c, ncol));
            scaled.add(beta_col.broadcast(c, ncol))
        }
        else
        {
            let rmean = tape.input(self.running_mean.clone()).reshape(&[c, 1]);
            let rvar = tape.input(self.running_var.clone()).reshape(&[c, 1]);
            let centered = xc.sub(rmean.broadcast(c, ncol));
            let inv_std = rvar.add(eps_col).sqrt().reciprocal();
            let x_hat = centered.hadamard(inv_std.broadcast(c, ncol));
            let scaled = x_hat.hadamard(gamma_col.broadcast(c, ncol));
            scaled.add(beta_col.broadcast(c, ncol))
        };

        // Channel-major (C, N*HW) -> back to NCHW (N, C*HW): inverse re-layout.
        normed.reshape(&[total_features, n]).transpose()
    }

    fn parameter_indices(&self) -> Vec<usize> {
        let mut v = Vec::new();
        if let Some(i) = self.last_g_idx
        {
            v.push(i);
        }
        if let Some(i) = self.last_b_idx
        {
            v.push(i);
        }
        v
    }

    fn sync(&mut self, tape: &Tape) {
        if let Some(i) = self.last_g_idx
        {
            self.gamma = tape.value(i);
        }
        if let Some(i) = self.last_b_idx
        {
            self.beta = tape.value(i);
        }
    }

    fn state_dict(&self) -> HashMap<String, Tensor> {
        let mut map = HashMap::new();
        map.insert(format!("{}.gamma", self.name), self.gamma.clone());
        map.insert(format!("{}.beta", self.name), self.beta.clone());
        map.insert(
            format!("{}.running_mean", self.name),
            self.running_mean.clone(),
        );
        map.insert(
            format!("{}.running_var", self.name),
            self.running_var.clone(),
        );
        map
    }

    fn load_state_dict(&mut self, sd: &HashMap<String, Tensor>) -> crate::error::Result<()> {
        let g = sd
            .get(&format!("{}.gamma", self.name))
            .ok_or_else(|| format!("missing key: {}.gamma", self.name))?;
        let b = sd
            .get(&format!("{}.beta", self.name))
            .ok_or_else(|| format!("missing key: {}.beta", self.name))?;
        let rm = sd
            .get(&format!("{}.running_mean", self.name))
            .ok_or_else(|| format!("missing key: {}.running_mean", self.name))?;
        let rv = sd
            .get(&format!("{}.running_var", self.name))
            .ok_or_else(|| format!("missing key: {}.running_var", self.name))?;

        self.gamma = g.clone();
        self.beta = b.clone();
        self.running_mean = rm.clone();
        self.running_var = rv.clone();
        Ok(())
    }
}

impl Clone for BatchNorm2d {
    fn clone(&self) -> Self {
        Self {
            num_channels: self.num_channels,
            gamma: self.gamma.clone(),
            beta: self.beta.clone(),
            eps: self.eps,
            momentum: self.momentum,
            running_mean: self.running_mean.clone(),
            running_var: self.running_var.clone(),
            training: self.training,
            last_g_idx: None,
            last_b_idx: None,
            name: self.name.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Extract the N*H*W values of channel `ci` from an NCHW (N, C*HW) output.
    fn channel_values(out: &Tensor, n: usize, num_c: usize, ci: usize) -> Vec<f32> {
        let total = out.cols; // C*HW
        let hw = total / num_c;
        let mut v = Vec::new();
        for row in 0..n
        {
            for s in 0..hw
            {
                v.push(out.data[row * total + ci * hw + s]);
            }
        }
        v
    }

    fn mean_var(v: &[f32]) -> (f32, f32) {
        let n = v.len() as f32;
        let m = v.iter().sum::<f32>() / n;
        let var = v.iter().map(|x| (x - m) * (x - m)).sum::<f32>() / n;
        (m, var)
    }

    // The critical regression test. Feed GENUINE NCHW data (channel-major within
    // each row, exactly what Conv2d emits) with two channels on very different
    // scales, and assert each channel is independently normalized to mean 0,
    // var 1. The old (N,C*HW)->(N*HW,C) reshape mixed the channels together, so
    // this fails on the buggy code and passes on the fix.
    #[test]
    fn bn2d_normalizes_each_channel_to_zero_mean_unit_var() {
        let mut bn = BatchNorm2d::new(2);
        let tape = Tape::new();
        // N=2, C=2, HW=2. channel 0 = [1,2,3,4], channel 1 = [10,20,30,40].
        // NCHW row layout per sample: [c0s0, c0s1, c1s0, c1s1].
        //   n0: [1, 2, 10, 20]     n1: [3, 4, 30, 40]
        let x = tape.input(Tensor::from_vec(
            vec![1.0, 2.0, 10.0, 20.0, 3.0, 4.0, 30.0, 40.0],
            2,
            4,
        ));
        let y = bn.forward(&tape, x);
        let yt = tape.value(y.idx());
        assert_eq!(yt.shape(), (2, 4));

        for ci in 0..2
        {
            let (m, v) = mean_var(&channel_values(&yt, 2, 2, ci));
            assert!(m.abs() < 1e-3, "channel {ci} mean {m} should be ~0");
            assert!((v - 1.0).abs() < 1e-2, "channel {ci} var {v} should be ~1");
        }
    }

    #[test]
    fn bn2d_running_stats_are_per_channel() {
        let mut bn = BatchNorm2d::new(2);
        let tape = Tape::new();
        // N=1, C=2, HW=2. NCHW row [1,2,3,4] => channel0=[1,2], channel1=[3,4].
        let x = tape.input(Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], 1, 4));
        let _ = bn.forward(&tape, x);
        // batch_mean = [1.5, 3.5]; running = 0.9*0 + 0.1*batch_mean.
        assert!(
            (bn.running_mean.data[0] - 0.15).abs() < 1e-5,
            "running_mean[0] = {}",
            bn.running_mean.data[0]
        );
        assert!(
            (bn.running_mean.data[1] - 0.35).abs() < 1e-5,
            "running_mean[1] = {}",
            bn.running_mean.data[1]
        );
    }

    #[test]
    fn bn2d_eval_uses_running_stats() {
        let mut bn = BatchNorm2d::new(2);
        bn.running_mean = Tensor::from_vec(vec![2.0, 4.0], 1, 2);
        bn.running_var = Tensor::from_vec(vec![4.0, 9.0], 1, 2);
        bn.set_training(false);

        let tape = Tape::new();
        // N=1, C=2, HW=2. NCHW [2,2,4,6] => channel0=[2,2], channel1=[4,6].
        let x = tape.input(Tensor::from_vec(vec![2.0, 2.0, 4.0, 6.0], 1, 4));
        let y = bn.forward(&tape, x);
        let yt = tape.value(y.idx());
        // channel0: mean=2,var=4,std=2 -> (2-2)/2 = 0 for both spatials.
        assert!(yt.data[0].abs() < 1e-3, "c0s0 = {}", yt.data[0]);
        assert!(yt.data[1].abs() < 1e-3, "c0s1 = {}", yt.data[1]);
        // channel1: mean=4,var=9,std=3 -> (4-4)/3=0, (6-4)/3=0.667.
        assert!(yt.data[2].abs() < 1e-3, "c1s0 = {}", yt.data[2]);
        assert!(
            (yt.data[3] - 2.0 / 3.0).abs() < 1e-3,
            "c1s1 = {}",
            yt.data[3]
        );
    }

    #[test]
    fn bn2d_gradient_flows() {
        let mut bn = BatchNorm2d::new(2);
        let tape = Tape::new();
        let x = tape.input(Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], 1, 4));
        let x_idx = x.idx();
        let y = bn.forward(&tape, x);
        let weights = tape.input(Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], 1, 4));
        let loss = y.hadamard(weights).sum();
        loss.backward();
        let g = tape.grad(x_idx);
        assert!(g.data.iter().any(|&v| v.abs() > 1e-6), "gradient is zero");
    }

    #[test]
    fn bn2d_state_dict_round_trip() {
        let mut bn1 = BatchNorm2d::new(3);
        bn1.gamma = Tensor::from_vec(vec![2.0; 3], 1, 3);
        bn1.beta = Tensor::from_vec(vec![1.0; 3], 1, 3);
        let sd = bn1.state_dict();

        let mut bn2 = BatchNorm2d::new(3);
        bn2.load_state_dict(&sd).unwrap();
        assert_eq!(bn2.gamma.data, bn1.gamma.data);
        assert_eq!(bn2.beta.data, bn1.beta.data);
        assert_eq!(bn2.running_mean.data, bn1.running_mean.data);
        assert_eq!(bn2.running_var.data, bn1.running_var.data);
    }
}
