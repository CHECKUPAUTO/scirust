use crate::autodiff::reverse::Tensor;

/// Entraînement en précision mixte FP16/FP32 avec loss scaling dynamique.
/// Conserve une copie master FP32 des poids et effectue les forward/backward
/// en FP16 avec scale pour éviter l'underflow des gradients.
pub struct MixedPrecisionTrainer {
    pub master_weights: Vec<Tensor>,
    pub fp16_weights: Vec<Tensor>,
    pub loss_scale: f32,
    scale_growth_factor: f32,
    scale_backoff_factor: f32,
    growth_interval: usize,
    step_counter: usize,
    max_scale: f32,
}

impl MixedPrecisionTrainer {
    pub fn new(model_params: &[Tensor], initial_scale: f32) -> Self {
        let master_weights = model_params.to_vec();
        let fp16_weights = master_weights.iter().map(cast_to_fp16).collect();
        Self {
            master_weights,
            fp16_weights,
            loss_scale: initial_scale,
            scale_growth_factor: 2.0,
            scale_backoff_factor: 0.5,
            growth_interval: 2000,
            step_counter: 0,
            max_scale: 65536.0,
        }
    }

    /// Rafraîchit les poids FP16 (consommés par le forward) à partir de la
    /// copie master FP32, sans jamais écraser cette copie master. Le forward
    /// doit lire `fp16_weights`; `master_weights` reste la source de vérité
    /// FP32 mise à jour par l'optimiseur.
    pub fn before_forward(&mut self) {
        self.update_fp16_from_master();
    }

    /// Après le backward: rescale les gradients, vérifie overflow, met à jour
    pub fn after_backward(&mut self, grads: &[Tensor]) -> Result<f32, String> {
        self.step_counter += 1;
        let mut any_overflow = false;
        let mut scaled_grads = Vec::with_capacity(grads.len());

        for grad in grads
        {
            let unscaled = unscale_tensor(grad, 1.0 / self.loss_scale);
            if has_nan_or_inf(&unscaled)
            {
                any_overflow = true;
                break;
            }
            scaled_grads.push(unscaled);
        }

        if any_overflow
        {
            self.loss_scale *= self.scale_backoff_factor;
            return Err(format!(
                "Overflow detected, loss scale reduced to {}",
                self.loss_scale
            ));
        }

        // Appliquer les gradients (à faire par l'optimiseur externe)
        Ok(self.loss_scale)
    }

    /// Croissance périodique du loss scale
    pub fn maybe_grow_scale(&mut self) {
        if self.step_counter.is_multiple_of(self.growth_interval)
        {
            self.loss_scale = (self.loss_scale * self.scale_growth_factor).min(self.max_scale);
        }
    }

    /// Cast master FP32 → FP16 pour le prochain forward
    pub fn update_fp16_from_master(&mut self) {
        for (fp16, master) in self.fp16_weights.iter_mut().zip(&self.master_weights)
        {
            *fp16 = cast_to_fp16(master);
        }
    }
}

fn cast_to_fp16(t: &Tensor) -> Tensor {
    let mut out = Tensor::zeros(t.rows, t.cols);
    for (dst, &src) in out.data.iter_mut().zip(&t.data)
    {
        let half = half::f16::from_f32(src);
        *dst = half.to_f32();
    }
    out
}

fn unscale_tensor(t: &Tensor, scale: f32) -> Tensor {
    let mut out = Tensor::zeros(t.rows, t.cols);
    for (dst, &src) in out.data.iter_mut().zip(&t.data)
    {
        let half = half::f16::from_f32(src);
        *dst = half.to_f32() * scale;
    }
    out
}

fn has_nan_or_inf(t: &Tensor) -> bool {
    t.data.iter().any(|&x| x.is_nan() || x.is_infinite())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_creation() {
        let params = vec![Tensor::zeros(10, 20), Tensor::zeros(5, 5)];
        let trainer = MixedPrecisionTrainer::new(&params, 1.0);
        assert_eq!(trainer.master_weights.len(), 2);
        assert_eq!(trainer.fp16_weights.len(), 2);
    }

    #[test]
    fn test_loss_scale_backoff() {
        let params = vec![Tensor::zeros(10, 20)];
        let mut trainer = MixedPrecisionTrainer::new(&params, 1.0);
        let grads = vec![Tensor {
            data: vec![f32::NAN; 200],
            rows: 10,
            cols: 20,
        }];
        let result = trainer.after_backward(&grads);
        assert!(result.is_err(), "Should fail on NaN");
        assert!(trainer.loss_scale < 1.0, "Loss scale should decrease");
    }

    #[test]
    fn test_before_forward_preserves_master_precision() {
        // 0.1 n'est pas représentable exactement en FP16: un aller-retour FP16
        // perd de la précision. before_forward() ne doit jamais écraser la copie
        // master FP32 avec cette valeur arrondie.
        let master = Tensor::from_vec(vec![0.1, 0.2, 0.3, 0.4], 2, 2);
        let params = vec![master.clone()];
        let mut trainer = MixedPrecisionTrainer::new(&params, 1.0);

        trainer.before_forward();

        // La copie master conserve exactement les valeurs FP32 d'origine.
        assert_eq!(
            trainer.master_weights[0].data, master.data,
            "before_forward() ne doit pas dégrader les poids master FP32"
        );

        // Les poids FP16 destinés au forward sont bien la version arrondie.
        let expected_fp16: Vec<f32> = master
            .data
            .iter()
            .map(|&x| half::f16::from_f32(x).to_f32())
            .collect();
        assert_eq!(trainer.fp16_weights[0].data, expected_fp16);

        // La version FP16 diffère effectivement du master (sinon le test est vide).
        assert_ne!(
            trainer.fp16_weights[0].data, master.data,
            "l'arrondi FP16 devrait modifier 0.1/0.2/0.3"
        );
    }

    #[test]
    fn test_before_forward_tracks_master_updates() {
        // Après une mise à jour de la copie master par l'optimiseur, un nouvel
        // appel à before_forward() doit refléter les nouvelles valeurs en FP16.
        let params = vec![Tensor::from_vec(vec![0.1, 0.2], 1, 2)];
        let mut trainer = MixedPrecisionTrainer::new(&params, 1.0);
        trainer.before_forward();

        trainer.master_weights[0].data = vec![1.5, 2.5];
        trainer.before_forward();

        let expected: Vec<f32> = [1.5_f32, 2.5]
            .iter()
            .map(|&x| half::f16::from_f32(x).to_f32())
            .collect();
        assert_eq!(trainer.fp16_weights[0].data, expected);
        // Le master reste inchangé par before_forward().
        assert_eq!(trainer.master_weights[0].data, vec![1.5, 2.5]);
    }

    #[test]
    fn test_scale_growth() {
        let params = vec![Tensor::zeros(10, 20)];
        let mut trainer = MixedPrecisionTrainer::new(&params, 1.0);
        trainer.growth_interval = 5;
        for _ in 0..5
        {
            let grads = vec![Tensor::zeros(10, 20)];
            let _ = trainer.after_backward(&grads);
        }
        trainer.maybe_grow_scale();
        assert!(trainer.loss_scale > 1.0, "Loss scale should grow");
    }
}
