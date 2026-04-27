// scirust-core/src/nn/batch_norm.rs
//
// BatchNorm1d — normalisation par batch sur les dimensions de features.
//
// Forme :
//   Input  : (N, F)    N = batch_size, F = features
//   Output : (N, F)
//
// Forward (training) :
//   μ_j  = (1/N) Σ_i x[i,j]                ← (1, F) via sum_axis(0) / N
//   σ²_j = (1/N) Σ_i (x[i,j] - μ_j)²       ← variance
//   x̂[i,j] = (x[i,j] - μ_j) / √(σ²_j + ε)
//   y[i,j] = γ_j · x̂[i,j] + β_j
//
// γ et β sont des paramètres apprenables, shape (1, F).
//
// Forward (eval) : utilise les running_mean / running_var accumulés
//                  pendant le training. (Non implémenté en v6 — v6.1)

use std::collections::HashMap;
use crate::autodiff::reverse::{Tape, Tensor, Var};
use crate::nn::module::Module;

pub struct BatchNorm1d {
    pub gamma:        Tensor,             // (1, F) scale, init à 1
    pub beta:         Tensor,             // (1, F) shift, init à 0
    pub eps:          f32,
    pub momentum:     f32,
    pub running_mean: Tensor,             // (1, F) — utilisé en eval mode
    pub running_var:  Tensor,             // (1, F)
    pub training:     bool,
    last_g_idx:       Option<usize>,
    last_b_idx:       Option<usize>,
    pub name:         String,
}

impl BatchNorm1d {
    pub fn new(num_features: usize) -> Self {
        Self {
            gamma:        Tensor::from_vec(vec![1.0; num_features], 1, num_features),
            beta:         Tensor::from_vec(vec![0.0; num_features], 1, num_features),
            eps:          1e-5,
            momentum:     0.1,
            running_mean: Tensor::zeros(1, num_features),
            running_var:  Tensor::from_vec(vec![1.0; num_features], 1, num_features),
            training:     true,
            last_g_idx:   None,
            last_b_idx:   None,
            name:         format!("bn_{num_features}"),
        }
    }

    pub fn with_name(mut self, name: &str) -> Self {
        self.name = name.into();
        self
    }
}

impl Module for BatchNorm1d {
    fn forward<'t>(&mut self, tape: &'t Tape, input: Var<'t>) -> Var<'t> {
        let (n, _f) = input.shape();
        let inv_n = 1.0 / n as f32;

        // Paramètres apprenables sur le tape
        let gamma_v = tape.input(self.gamma.clone());
        let beta_v  = tape.input(self.beta.clone());
        self.last_g_idx = Some(gamma_v.idx());
        self.last_b_idx = Some(beta_v.idx());

        if self.training {
            // ----- Statistiques du batch ----- //
            // μ : (1, F) = sum_axis(input, 0) * (1/N)
            let mu = input.clone().sum_axis(0).scale(inv_n);

            // centré : (N, F) = input - μ (broadcast)
            let mu_neg = mu.clone().neg();
            let centered = input.add_broadcast(mu_neg);

            // variance : (1, F) = sum_axis(centered², 0) * (1/N)
            let centered_sq = centered.clone().hadamard(centered.clone());
            let var = centered_sq.sum_axis(0).scale(inv_n);

            // std = sqrt(var + ε) : (1, F)
            // var + ε via add d'un tenseur constant
            let eps_t = tape.input(
                Tensor::from_vec(vec![self.eps; var.shape().1], 1, var.shape().1));
            let var_eps = var.add(eps_t);
            let std = var_eps.sqrt();

            // x_hat = centered * (1/std), broadcast sur axis 0
            let inv_std = std.reciprocal();              // (1, F)
            let x_hat = centered.mul_broadcast(inv_std);

            // y = γ · x_hat + β  (γ et β broadcast)
            let scaled = x_hat.mul_broadcast(gamma_v);
            scaled.add_broadcast(beta_v)

            // NOTE : en v6 on ne met PAS à jour running_mean/running_var.
            // Ces stats sont nécessaires pour le mode eval. À ajouter en v6.1.
        } else {
            // Mode eval : utilise les running stats
            // x_hat = (x - running_mean) / sqrt(running_var + ε)
            let rmean_v = tape.input(self.running_mean.clone());
            let rvar_v  = tape.input(self.running_var.clone());
            let centered = input.add_broadcast(rmean_v.neg());

            let eps_t = tape.input(
                Tensor::from_vec(vec![self.eps; self.running_var.cols],
                                 1, self.running_var.cols));
            let std = rvar_v.add(eps_t).sqrt();
            let x_hat = centered.mul_broadcast(std.reciprocal());
            let scaled = x_hat.mul_broadcast(gamma_v);
            scaled.add_broadcast(beta_v)
        }
    }

    fn parameter_indices(&self) -> Vec<usize> {
        let mut v = Vec::new();
        if let Some(i) = self.last_g_idx { v.push(i); }
        if let Some(i) = self.last_b_idx { v.push(i); }
        v
    }

    fn sync(&mut self, tape: &Tape) {
        if let Some(i) = self.last_g_idx { self.gamma = tape.value(i); }
        if let Some(i) = self.last_b_idx { self.beta  = tape.value(i); }
    }

    fn state_dict(&self) -> Vec<(String, Tensor)> {
        vec![
            (format!("{}.gamma",        self.name), self.gamma.clone()),
            (format!("{}.beta",         self.name), self.beta.clone()),
            (format!("{}.running_mean", self.name), self.running_mean.clone()),
            (format!("{}.running_var",  self.name), self.running_var.clone()),
        ]
    }

    fn load_state_dict(&mut self, dict: &HashMap<String, Tensor>) -> usize {
        let mut loaded = 0;
        if let Some(t) = dict.get(&format!("{}.gamma", self.name))
            { self.gamma = t.clone(); loaded += 1; }
        if let Some(t) = dict.get(&format!("{}.beta", self.name))
            { self.beta = t.clone(); loaded += 1; }
        if let Some(t) = dict.get(&format!("{}.running_mean", self.name))
            { self.running_mean = t.clone(); loaded += 1; }
        if let Some(t) = dict.get(&format!("{}.running_var", self.name))
            { self.running_var = t.clone(); loaded += 1; }
        loaded
    }

    fn train(&mut self, mode: bool) { self.training = mode; }
}

// ================================================================== //
//  Tests                                                              //
// ================================================================== //
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bn_normalizes_to_zero_mean_unit_var() {
        // Avec γ=1, β=0, le BN doit normaliser les colonnes à
        // moyenne ≈ 0 et variance ≈ 1.
        let mut bn = BatchNorm1d::new(3);
        let tape = Tape::new();
        // Batch de 4 échantillons, 3 features
        let x = tape.input(Tensor::from_vec(
            vec![1.0, 10.0, -5.0,
                 3.0, 12.0, -3.0,
                 5.0, 14.0, -1.0,
                 7.0, 16.0,  1.0],
            4, 3));
        let y = bn.forward(&tape, x);
        let yt = tape.value(y.idx());

        // Vérif : pour chaque colonne, moyenne ≈ 0
        for j in 0..3 {
            let mean: f32 = (0..4).map(|i| yt.data[i*3 + j]).sum::<f32>() / 4.0;
            assert!(mean.abs() < 1e-4,
                    "col {j} mean = {mean}, attendu ≈ 0");
        }

        // Variance ≈ 1 (dans la limite de eps)
        for j in 0..3 {
            let mean: f32 = (0..4).map(|i| yt.data[i*3 + j]).sum::<f32>() / 4.0;
            let var: f32 = (0..4).map(|i| {
                let d = yt.data[i*3 + j] - mean;
                d * d
            }).sum::<f32>() / 4.0;
            // var = N/(N) ≈ 1, mais avec eps on a var = 1 - eps/var_originale ≈ 1
            assert!((var - 1.0).abs() < 0.01, "col {j} var = {var}");
        }
    }

    #[test]
    fn bn_eval_mode_uses_running_stats() {
        let mut bn = BatchNorm1d::new(2);
        // Forcer running_mean / running_var à des valeurs connues
        bn.running_mean = Tensor::from_vec(vec![5.0, 10.0], 1, 2);
        bn.running_var  = Tensor::from_vec(vec![4.0, 9.0],  1, 2);
        bn.train(false);

        let tape = Tape::new();
        let x = tape.input(Tensor::from_vec(vec![5.0, 10.0,  7.0, 13.0], 2, 2));
        let y = bn.forward(&tape, x);
        let yt = tape.value(y.idx());

        // Ligne 0 : (5-5)/√(4+ε) = 0, (10-10)/√(9+ε) = 0
        assert!(yt.data[0].abs() < 1e-3);
        assert!(yt.data[1].abs() < 1e-3);
        // Ligne 1 : (7-5)/√4 = 1, (13-10)/√9 = 1
        assert!((yt.data[2] - 1.0).abs() < 1e-3);
        assert!((yt.data[3] - 1.0).abs() < 1e-3);
    }

    #[test]
    fn bn_state_dict_round_trip() {
        let bn = BatchNorm1d::new(8).with_name("test_bn");
        let dict = bn.state_dict();
        assert_eq!(dict.len(), 4);
        assert!(dict.iter().any(|(k, _)| k.contains("gamma")));
        assert!(dict.iter().any(|(k, _)| k.contains("running_mean")));
    }

    #[test]
    fn bn_parameter_indices_after_forward() {
        let mut bn = BatchNorm1d::new(4);
        let tape = Tape::new();
        let x = tape.input(Tensor::zeros(3, 4));
        let _ = bn.forward(&tape, x);
        // γ + β = 2 paramètres apprenables (running_* ne sont PAS apprenables)
        assert_eq!(bn.parameter_indices().len(), 2);
    }
}
