// scirust-core/src/nn/loss/cross_entropy.rs
//
// Cross-Entropy Loss avec softmax integre et stable numeriquement.
// MULTI-BATCH : supporte batch > 1 via sum_axis (depuis reverse.rs).

use crate::autodiff::reverse::{Tape, Tensor, Var};
use crate::nn::loss::Loss;

pub struct CrossEntropyLoss;

impl CrossEntropyLoss {
    pub fn new() -> Self {
        CrossEntropyLoss
    }
}

impl Default for CrossEntropyLoss {
    fn default() -> Self {
        CrossEntropyLoss
    }
}

impl CrossEntropyLoss {
    /// pred : (batch, n_classes) — logits bruts
    /// target_indices : (batch, 1) — indices entiers des classes (0..n_classes)
    /// Renvoie : Var scalaire = mean(CE par sample)
    pub fn forward_with_indices<'t>(
        &self,
        tape: &'t Tape,
        pred: Var<'t>,
        target_indices: &Tensor,
    ) -> Var<'t> {
        let (batch, n_classes) = pred.shape();
        assert_eq!(
            target_indices.shape(),
            (batch, 1),
            "CrossEntropy: target_indices shape {:?} != ({}, 1)",
            target_indices.shape(),
            batch
        );

        let mut mask_data = vec![0.0f32; batch * n_classes];
        for b in 0..batch
        {
            let label = target_indices.data[b] as usize;
            assert!(
                label < n_classes,
                "label {} out of range [0, {})",
                label,
                n_classes
            );
            mask_data[b * n_classes + label] = 1.0;
        }
        let mask = tape.input(Tensor::from_vec(mask_data, batch, n_classes));
        self.forward(tape, pred, mask)
    }
}

impl Loss for CrossEntropyLoss {
    /// pred : (batch, n_classes) — logits bruts (avant softmax)
    /// target : (batch, n_classes) — encodage one-hot
    /// renvoie : Var scalaire = mean(CE par sample)
    fn forward<'t>(&self, tape: &'t Tape, pred: Var<'t>, target: Var<'t>) -> Var<'t> {
        let (batch, n_classes) = pred.shape();
        assert_eq!(
            target.shape(),
            pred.shape(),
            "CrossEntropy: target shape {:?} != pred shape {:?}",
            target.shape(),
            pred.shape()
        );

        // 1) Calcul du max par row (en CPU, traite comme constante)
        let pred_t = tape.value(pred.idx());
        let mut max_per_row = vec![0.0f32; batch];
        for (r, row) in pred_t.data.chunks_exact(n_classes).enumerate()
        {
            max_per_row[r] = row.iter().copied().fold(row[0], f32::max);
        }

        // 2) Construction d'un tenseur "max broadcaste" (batch, n_classes)
        let mut max_broadcast_data = vec![0.0f32; batch * n_classes];
        for (r, chunk) in max_broadcast_data.chunks_exact_mut(n_classes).enumerate()
        {
            chunk.fill(max_per_row[r]);
        }
        let max_var = tape.input(Tensor::from_vec(max_broadcast_data, batch, n_classes));

        // 3) shifted = pred - max (numeriquement stable)
        let shifted = pred.try_sub(max_var).unwrap();

        // 4) exp_shifted = exp(shifted)
        let exp_shifted = shifted.exp();

        // 5) Z_per_row = sum sur axis 1 (cols) -> shape (batch, 1)
        let z_per_row = exp_shifted.sum_axis(1); // (batch, 1)

        // 6) log(Z_per_row) — shape (batch, 1)
        let log_z = z_per_row.log(); // (batch, 1)

        // 7) Broadcast log_z sur (batch, n_classes) via broadcast natif
        let log_z_broadcast = log_z.broadcast(batch, n_classes); // (batch, n_classes)

        // 8) log_softmax = shifted - log_z_broadcast
        let log_softmax = shifted.try_sub(log_z_broadcast).unwrap();

        // 9) ce = -sum(target ⊙ log_softmax) / batch  (mean par sample)
        let prod = target.try_hadamard(log_softmax).unwrap();

        prod.sum().scale(-1.0 / batch as f32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::autodiff::reverse::Tensor;

    #[test]
    fn ce_low_when_confident_correct() {
        let tape = Tape::new();
        let pred = tape.input(Tensor::from_vec(vec![10.0, 0.0, 0.0], 1, 3));
        let target = tape.input(Tensor::from_vec(vec![1.0, 0.0, 0.0], 1, 3));
        let loss = CrossEntropyLoss::new().forward(&tape, pred, target);
        let v = tape.value(loss.idx());
        assert!(v.data[0] < 0.01, "CE should be near 0, got {}", v.data[0]);
    }

    #[test]
    fn ce_high_when_confident_wrong() {
        let tape = Tape::new();
        let pred = tape.input(Tensor::from_vec(vec![10.0, 0.0, 0.0], 1, 3));
        let target = tape.input(Tensor::from_vec(vec![0.0, 1.0, 0.0], 1, 3));
        let loss = CrossEntropyLoss::new().forward(&tape, pred, target);
        let v = tape.value(loss.idx());
        assert!(
            v.data[0] > 5.0,
            "CE should be > 5 for very wrong pred, got {}",
            v.data[0]
        );
    }

    #[test]
    fn ce_equals_log_n_for_uniform_logits() {
        let tape = Tape::new();
        let pred = tape.input(Tensor::from_vec(vec![0.0, 0.0, 0.0, 0.0], 1, 4));
        let target = tape.input(Tensor::from_vec(vec![1.0, 0.0, 0.0, 0.0], 1, 4));
        let loss = CrossEntropyLoss::new().forward(&tape, pred, target);
        let v = tape.value(loss.idx());
        let expected = 4.0_f32.ln();
        assert!(
            (v.data[0] - expected).abs() < 1e-4,
            "CE = {}, expected log(4) = {}",
            v.data[0],
            expected
        );
    }

    #[test]
    fn ce_stable_with_large_logits() {
        let tape = Tape::new();
        let pred = tape.input(Tensor::from_vec(vec![1000.0, 999.0, 998.0], 1, 3));
        let target = tape.input(Tensor::from_vec(vec![1.0, 0.0, 0.0], 1, 3));
        let loss = CrossEntropyLoss::new().forward(&tape, pred, target);
        let v = tape.value(loss.idx());
        assert!(
            v.data[0].is_finite(),
            "CE produced non-finite value: {}",
            v.data[0]
        );
        assert!(
            v.data[0] < 1.0,
            "CE = {}, should be small for confident correct pred",
            v.data[0]
        );
    }

    #[test]
    fn ce_gradient_flows_to_pred() {
        let tape = Tape::new();
        let pred = tape.input(Tensor::from_vec(vec![1.0, 2.0, 3.0], 1, 3));
        let target = tape.input(Tensor::from_vec(vec![0.0, 1.0, 0.0], 1, 3));
        let loss = CrossEntropyLoss::new().forward(&tape, pred, target);
        tape.backward(loss.idx());

        let g = tape.grad(pred.idx());
        let max_abs: f32 = g.data.iter().map(|v| v.abs()).fold(0.0, f32::max);
        assert!(max_abs > 1e-6, "Gradient on pred is zero, autograd broken");

        // Propriete mathematique : partialCE/partialpred = softmax(pred) - target
        let e1 = 1.0_f32.exp();
        let e2 = 2.0_f32.exp();
        let e3 = 3.0_f32.exp();
        let z = e1 + e2 + e3;
        let s = [e1 / z, e2 / z, e3 / z];
        let expected_grad = [s[0] - 0.0, s[1] - 1.0, s[2] - 0.0];
        for (i, &expected) in expected_grad.iter().enumerate()
        {
            assert!(
                (g.data[i] - expected).abs() < 1e-3,
                "grad[{}] = {}, expected {} (softmax - target)",
                i,
                g.data[i],
                expected
            );
        }
    }

    #[test]
    fn ce_multi_batch_works() {
        // batch=2, n_classes=3
        // Sample 0: [10, 0, 0] target classe 0 -> CE ~ 0
        // Sample 1: [0, 10, 0] target classe 1 -> CE ~ 0
        // mean CE ~= 0
        let tape = Tape::new();
        let pred = tape.input(Tensor::from_vec(vec![10.0, 0.0, 0.0, 0.0, 10.0, 0.0], 2, 3));
        let target = tape.input(Tensor::from_vec(vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0], 2, 3));
        let loss = CrossEntropyLoss::new().forward(&tape, pred, target);
        let v = tape.value(loss.idx());
        assert!(
            v.data[0] < 0.01,
            "Multi-batch CE should be near 0, got {}",
            v.data[0]
        );

        // Verifier que le gradient remonte bien pour les 2 samples
        tape.backward(loss.idx());
        let g = tape.grad(pred.idx());
        assert!(
            g.data.iter().all(|v| v.abs() > 1e-6),
            "Multi-batch gradient zero on some entries"
        );
    }

    #[test]
    fn ce_multi_batch_mixed_confidence() {
        // batch=2, n_classes=2
        // Sample 0: confident correct -> CE ~ 0
        // Sample 1: confident wrong -> CE ~ 10
        // mean CE ~= 5
        let tape = Tape::new();
        let pred = tape.input(Tensor::from_vec(vec![10.0, 0.0, 0.0, 10.0], 2, 2));
        let target = tape.input(Tensor::from_vec(vec![1.0, 0.0, 1.0, 0.0], 2, 2));
        let loss = CrossEntropyLoss::new().forward(&tape, pred, target);
        let v = tape.value(loss.idx());
        // sample0 CE ~ 0, sample1 CE ~ 10 -> mean ~ 5
        assert!(
            v.data[0] > 4.0 && v.data[0] < 6.0,
            "Mixed batch mean CE = {}, expected ~5",
            v.data[0]
        );
    }

    #[test]
    fn ce_int_labels_matches_one_hot() {
        let tape = Tape::new();
        let pred = tape.input(Tensor::from_vec(vec![10.0, 0.0, 0.0, 0.0, 10.0, 0.0], 2, 3));
        let target_one_hot = tape.input(Tensor::from_vec(vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0], 2, 3));
        let target_int = Tensor::from_vec(vec![0.0, 1.0], 2, 1);

        let loss_oh = CrossEntropyLoss::new().forward(&tape, pred, target_one_hot);
        let loss_int = CrossEntropyLoss::new().forward_with_indices(&tape, pred, &target_int);

        let v_oh = tape.value(loss_oh.idx()).data[0];
        let v_int = tape.value(loss_int.idx()).data[0];
        assert!(
            (v_oh - v_int).abs() < 1e-5,
            "CE int={} != CE one-hot={}",
            v_int,
            v_oh
        );
    }

    #[test]
    fn ce_int_labels_gradient_flows() {
        let tape = Tape::new();
        let pred = tape.input(Tensor::from_vec(vec![1.0, 2.0, 3.0], 1, 3));
        let target_int = Tensor::from_vec(vec![1.0], 1, 1);
        let loss = CrossEntropyLoss::new().forward_with_indices(&tape, pred, &target_int);
        tape.backward(loss.idx());
        let g = tape.grad(pred.idx());
        let max_abs: f32 = g.data.iter().map(|v| v.abs()).fold(0.0, f32::max);
        assert!(max_abs > 1e-6, "Gradient on pred is zero with int labels");
    }
}
