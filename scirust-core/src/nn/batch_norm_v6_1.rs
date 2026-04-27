// scirust-core/src/nn/batch_norm_v6_1.rs
//
// BatchNorm1d v6.1 — clôture du chapitre training en ajoutant la
// mise à jour des running_mean / running_var pendant le forward training.
//
// CHANGEMENTS PAR RAPPORT À v6 :
//   - calcul ET stockage des stats du batch en CPU pendant le forward
//     (en plus de la version graphe pour le gradient)
//   - mise à jour de running_mean / running_var avec moving average :
//
//     running_mean ← (1 - momentum) · running_mean + momentum · batch_mean
//     running_var  ← (1 - momentum) · running_var  + momentum · batch_var
//
//   - le mode eval utilise correctement les running stats (déjà v6)
//
// IMPORTANT : la mise à jour est un effet de bord SUR LE MODULE, pas
// sur le tape. Le graphe AD n'a pas besoin de la connaître. C'est la
// raison pour laquelle on duplique le calcul de mean/var en CPU pur.

use std::collections::HashMap;
use crate::autodiff::reverse::{Tape, Tensor, Var};
use crate::nn::module::Module;

pub struct BatchNorm1d {
    pub gamma:        Tensor,
    pub beta:         Tensor,
    pub eps:          f32,
    pub momentum:     f32,
    pub running_mean: Tensor,
    pub running_var:  Tensor,
    pub training:     bool,
    last_g_idx:       Option<usize>,
    last_b_idx:       Option<usize>,
    pub name:         String,
}

impl BatchNorm1d {
    pub fn new(num_features: usize) -> Self {
        Self {
            gamma:        Tensor::from_vec(vec![1.0; num_features], 1, num_features),
            beta:         Tensor::zeros(1, num_features),
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

    /// Calcule batch_mean et batch_var en CPU pur (pour la mise à jour
    /// des running_stats — pas pour le graphe AD qui les recalcule via
    /// sum_axis).
    fn compute_batch_stats(&self, input_data: &[f32], n: usize, f: usize)
        -> (Vec<f32>, Vec<f32>)
    {
        let inv_n = 1.0 / n as f32;
        let mut mean = vec![0.0f32; f];
        for i in 0..n {
            for j in 0..f {
                mean[j] += input_data[i * f + j];
            }
        }
        for v in mean.iter_mut() { *v *= inv_n; }

        let mut var = vec![0.0f32; f];
        for i in 0..n {
            for j in 0..f {
                let d = input_data[i * f + j] - mean[j];
                var[j] += d * d;
            }
        }
        for v in var.iter_mut() { *v *= inv_n; }

        (mean, var)
    }

    fn update_running_stats(&mut self, batch_mean: &[f32], batch_var: &[f32]) {
        let alpha = self.momentum;
        for j in 0..self.running_mean.cols {
            self.running_mean.data[j] =
                (1.0 - alpha) * self.running_mean.data[j] + alpha * batch_mean[j];
            self.running_var.data[j] =
                (1.0 - alpha) * self.running_var.data[j] + alpha * batch_var[j];
        }
    }
}

impl Module for BatchNorm1d {
    fn forward<'t>(&mut self, tape: &'t Tape, input: Var<'t>) -> Var<'t> {
        let (n, f) = input.shape();
        let inv_n = 1.0 / n as f32;

        let gamma_v = tape.input(self.gamma.clone());
        let beta_v  = tape.input(self.beta.clone());
        self.last_g_idx = Some(gamma_v.idx());
        self.last_b_idx = Some(beta_v.idx());

        if self.training {
            // Effet de bord : mise à jour des running stats. Lit l'input depuis
            // le tape, calcule batch_mean / batch_var, met à jour les buffers
            // du module. Le graphe AD continue avec sum_axis comme avant.
            let input_t = tape.value(input.idx());
            let (batch_mean, batch_var) =
                self.compute_batch_stats(&input_t.data, n, f);
            self.update_running_stats(&batch_mean, &batch_var);

            // Graphe AD identique à v6
            let mu = input.clone().sum_axis(0).scale(inv_n);
            let mu_neg = mu.neg();
            let centered = input.add_broadcast(mu_neg);
            let centered_sq = centered.clone().hadamard(centered.clone());
            let var = centered_sq.sum_axis(0).scale(inv_n);
            let eps_t = tape.input(
                Tensor::from_vec(vec![self.eps; f], 1, f));
            let std = var.add(eps_t).sqrt();
            let inv_std = std.reciprocal();
            let x_hat = centered.mul_broadcast(inv_std);
            let scaled = x_hat.mul_broadcast(gamma_v);
            scaled.add_broadcast(beta_v)
        } else {
            // Eval : running stats
            let rmean_v = tape.input(self.running_mean.clone());
            let rvar_v  = tape.input(self.running_var.clone());
            let centered = input.add_broadcast(rmean_v.neg());
            let eps_t = tape.input(
                Tensor::from_vec(vec![self.eps; f], 1, f));
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn running_stats_update_on_training_forward() {
        let mut bn = BatchNorm1d::new(2);
        let initial_mean = bn.running_mean.data.clone();
        // Forward avec un batch dont la moyenne est non-nulle
        let tape = Tape::new();
        let x = tape.input(Tensor::from_vec(vec![10.0, 20.0,  10.0, 20.0], 2, 2));
        let _ = bn.forward(&tape, x);

        // running_mean doit avoir bougé : (1-0.1)*0 + 0.1*10 = 1.0
        assert!((bn.running_mean.data[0] - 1.0).abs() < 1e-5,
                "got {}", bn.running_mean.data[0]);
        assert!((bn.running_mean.data[1] - 2.0).abs() < 1e-5);
        // Différent de l'initial
        assert_ne!(bn.running_mean.data, initial_mean);
    }

    #[test]
    fn eval_mode_does_not_update_stats() {
        let mut bn = BatchNorm1d::new(2);
        bn.train(false);
        let snapshot = bn.running_mean.data.clone();

        let tape = Tape::new();
        let x = tape.input(Tensor::from_vec(vec![100.0, 100.0,  100.0, 100.0], 2, 2));
        let _ = bn.forward(&tape, x);

        // running_mean inchangée en eval
        assert_eq!(bn.running_mean.data, snapshot);
    }

    #[test]
    fn multiple_forward_passes_converge_running_to_batch_mean() {
        // Si on fait beaucoup de forwards avec le même batch, running_mean
        // doit converger vers batch_mean
        let mut bn = BatchNorm1d::new(1);
        for _ in 0..200 {
            let tape = Tape::new();
            let x = tape.input(Tensor::from_vec(vec![5.0, 5.0, 5.0, 5.0], 4, 1));
            let _ = bn.forward(&tape, x);
        }
        // Après 200 itérations avec momentum 0.1, running_mean ≈ 5.0
        assert!((bn.running_mean.data[0] - 5.0).abs() < 0.01,
                "got {}", bn.running_mean.data[0]);
    }
}
