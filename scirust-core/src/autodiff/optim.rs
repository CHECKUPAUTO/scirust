// scirust-core/src/autodiff/optim.rs
//
// Optimiseurs pour entraînement de modèles construits sur la Tape AD.
//
// Convention :
//   - Les paramètres sont des Var créés via tape.input(...).
//   - Après backward(), tape.grad(idx) contient le gradient.
//   - L'optimiseur lit le gradient et met à jour la valeur stockée
//     dans le tape via Tape::set_value(idx, ...).
//
// Pour itérer correctement (multi-époque), recréer un Tape neuf à
// chaque pas et ré-injecter les paramètres mis à jour. La Tape
// accumule en effet les noeuds — voir adam_converges_on_quadratic
// dans les tests pour le pattern complet.
//
// PLACE DANS L'ARCHITECTURE (les trois familles d'optimiseurs) :
//   - Ce module : la famille *tape* (`Sgd`, `Adam`, `AdamW`), trait
//     `Optimizer` — entraînement 2-D sur la tape reverse-mode.
//   - `crate::nn::nd_optim` : la famille *N-D* (`NdAdam`, `NdLion`, …),
//     trait `NdOptimizer` — entraînement des couches N-D.
//   - `crate::optim` : la famille *raw-slice* (`RMSprop`, `AdamW`,
//     `LAMB`) — boucles d'entraînement manuelles sur des `&[f32]`.
// Les trois familles implémentent `crate::optim::HasLearningRate`
// (les optimiseurs tape via un blanket impl sur `Optimizer`), ce qui
// permet à n'importe quel `LrSchedule` de piloter n'importe quel
// optimiseur via `LrSchedule::drive`.

use crate::autodiff::reverse::{Tape, Tensor};
use std::collections::HashMap;

// ================================================================== //
//  Trait Optimizer                                                    //
// ================================================================== //

pub trait Optimizer {
    /// Applique un pas de descente sur les paramètres listés.
    /// Lit les gradients via `tape.grad(idx)`, écrit les valeurs via
    /// `tape.set_value(idx, ...)`.
    fn step(&mut self, params: &[usize], tape: &Tape);

    /// Lit le learning rate courant.
    fn lr(&self) -> f32;

    /// Modifie le learning rate (utilisé par les schedulers).
    fn set_lr(&mut self, lr: f32);
}

// ================================================================== //
//  SGD avec momentum optionnel                                        //
// ================================================================== //

pub struct Sgd {
    pub lr: f32,
    pub momentum: f32,
    pub weight_decay: f32,
    velocities: HashMap<usize, Tensor>,
}

impl Sgd {
    pub fn new(lr: f32) -> Self {
        Self {
            lr,
            momentum: 0.0,
            weight_decay: 0.0,
            velocities: HashMap::new(),
        }
    }

    pub fn with_momentum(mut self, m: f32) -> Self {
        self.momentum = m;
        self
    }

    pub fn with_weight_decay(mut self, wd: f32) -> Self {
        self.weight_decay = wd;
        self
    }
}

impl Optimizer for Sgd {
    fn step(&mut self, params: &[usize], tape: &Tape) {
        for &idx in params
        {
            let mut value = tape.value(idx);
            let grad = tape.grad(idx);
            assert_eq!(
                value.shape(),
                grad.shape(),
                "Sgd::step: shape mismatch param/grad (idx={})",
                idx
            );

            let v = self
                .velocities
                .entry(idx)
                .or_insert_with(|| Tensor::zeros(value.rows, value.cols));

            for i in 0..value.data.len()
            {
                let mut g = grad.data[i];
                // Weight decay (L2 regularization)
                if self.weight_decay > 0.0
                {
                    g += self.weight_decay * value.data[i];
                }
                // Momentum : v = momentum * v + g
                v.data[i] = self.momentum * v.data[i] + g;
                // Update : θ -= lr * v
                value.data[i] -= self.lr * v.data[i];
            }
            tape.set_value(idx, value);
        }
    }

    fn lr(&self) -> f32 {
        self.lr
    }
    fn set_lr(&mut self, lr: f32) {
        self.lr = lr;
    }
}

// ================================================================== //
//  Adam — Adaptive Moment Estimation (Kingma & Ba, 2014)              //
// ================================================================== //

pub struct Adam {
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub epsilon: f32,
    pub weight_decay: f32,
    pub amsgrad: bool,
    t: usize,                      // step counter (pour bias correction)
    m: HashMap<usize, Tensor>,     // 1er moment (moyenne mobile gradient)
    v: HashMap<usize, Tensor>,     // 2e moment (moyenne mobile gradient²)
    v_max: HashMap<usize, Tensor>, // max historique du 2e moment (AMSGrad)
}

impl Adam {
    pub fn new(lr: f32) -> Self {
        Self {
            lr,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            weight_decay: 0.0,
            amsgrad: false,
            t: 0,
            m: HashMap::new(),
            v: HashMap::new(),
            v_max: HashMap::new(),
        }
    }

    pub fn with_betas(mut self, b1: f32, b2: f32) -> Self {
        self.beta1 = b1;
        self.beta2 = b2;
        self
    }

    pub fn with_weight_decay(mut self, wd: f32) -> Self {
        self.weight_decay = wd;
        self
    }

    pub fn with_epsilon(mut self, eps: f32) -> Self {
        self.epsilon = eps;
        self
    }

    /// Active la variante **AMSGrad** (Reddi, Kale & Kumar, *On the
    /// Convergence of Adam and Beyond*, ICLR 2018) : le dénominateur du pas
    /// utilise le **maximum historique** du second moment au lieu de sa
    /// moyenne mobile. Il ne peut donc jamais décroître, ce qui évite les
    /// pas géants après un pic de gradient et restaure la garantie de
    /// convergence qu'Adam perd sur certains problèmes convexes.
    pub fn with_amsgrad(mut self) -> Self {
        self.amsgrad = true;
        self
    }
}

impl Optimizer for Adam {
    fn step(&mut self, params: &[usize], tape: &Tape) {
        self.t += 1;
        let bc1 = 1.0 - self.beta1.powi(self.t as i32); // bias correction 1
        let bc2 = 1.0 - self.beta2.powi(self.t as i32); // bias correction 2

        for &idx in params
        {
            let mut value = tape.value(idx);
            let grad = tape.grad(idx);
            assert_eq!(
                value.shape(),
                grad.shape(),
                "Adam::step: shape mismatch param/grad (idx={})",
                idx
            );

            let m = self
                .m
                .entry(idx)
                .or_insert_with(|| Tensor::zeros(value.rows, value.cols));
            let v = self
                .v
                .entry(idx)
                .or_insert_with(|| Tensor::zeros(value.rows, value.cols));
            let mut v_max = self.amsgrad.then(|| {
                self.v_max
                    .entry(idx)
                    .or_insert_with(|| Tensor::zeros(value.rows, value.cols))
            });

            for i in 0..value.data.len()
            {
                let mut g = grad.data[i];
                if self.weight_decay > 0.0
                {
                    g += self.weight_decay * value.data[i];
                }

                // m = β1 m + (1−β1) g
                m.data[i] = self.beta1 * m.data[i] + (1.0 - self.beta1) * g;
                // v = β2 v + (1−β2) g²
                v.data[i] = self.beta2 * v.data[i] + (1.0 - self.beta2) * g * g;

                // bias correction
                let m_hat = m.data[i] / bc1;
                let mut v_hat = v.data[i] / bc2;

                // AMSGrad : normalisation par le max historique du 2e moment
                // (jamais décroissant), bias-corrigé comme v.
                if let Some(vm) = &mut v_max
                {
                    vm.data[i] = vm.data[i].max(v.data[i]);
                    v_hat = vm.data[i] / bc2;
                }

                // θ -= lr * m̂ / (√v̂ + ε)
                value.data[i] -= self.lr * m_hat / (v_hat.sqrt() + self.epsilon);
            }
            tape.set_value(idx, value);
        }
    }

    fn lr(&self) -> f32 {
        self.lr
    }
    fn set_lr(&mut self, lr: f32) {
        self.lr = lr;
    }
}

// ================================================================== //
//  AdamW — Adam à weight decay découplé (Loshchilov & Hutter 2019)    //
// ================================================================== //

/// **AdamW** (Loshchilov & Hutter, *Decoupled Weight Decay
/// Regularization*, ICLR 2019) sur la Tape : Adam dont le weight decay
/// est **découplé** du gradient.
///
/// Là où `Adam::with_weight_decay` ajoute `wd·θ` au gradient (L2
/// classique — le terme passe par les moments et se fait renormaliser
/// par `√v̂`), AdamW retranche `lr·wd·θ` directement au paramètre :
///
/// ```text
/// θ ← θ − lr·m̂/(√v̂ + ε) − lr·wd·θ
/// ```
///
/// La structure est le miroir exact d'[`Adam`] (moments indexés par
/// nœud de tape, bias correction, AMSGrad optionnel) : avec
/// `weight_decay = 0`, la trajectoire est **bit-à-bit identique** à
/// Adam, donc le swap `Adam → AdamW` est drop-in sur le chemin tape.
/// Le défaut est `weight_decay = 0.01` (le défaut canonique d'AdamW).
pub struct AdamW {
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub epsilon: f32,
    /// Weight decay **découplé** (retranché au paramètre, pas au gradient).
    pub weight_decay: f32,
    pub amsgrad: bool,
    t: usize,                      // step counter (pour bias correction)
    m: HashMap<usize, Tensor>,     // 1er moment (moyenne mobile gradient)
    v: HashMap<usize, Tensor>,     // 2e moment (moyenne mobile gradient²)
    v_max: HashMap<usize, Tensor>, // max historique du 2e moment (AMSGrad)
}

impl AdamW {
    pub fn new(lr: f32) -> Self {
        Self {
            lr,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            weight_decay: 0.01,
            amsgrad: false,
            t: 0,
            m: HashMap::new(),
            v: HashMap::new(),
            v_max: HashMap::new(),
        }
    }

    pub fn with_betas(mut self, b1: f32, b2: f32) -> Self {
        self.beta1 = b1;
        self.beta2 = b2;
        self
    }

    /// Fixe le weight decay découplé (`0.0` = Adam exact).
    pub fn with_weight_decay(mut self, wd: f32) -> Self {
        self.weight_decay = wd;
        self
    }

    pub fn with_epsilon(mut self, eps: f32) -> Self {
        self.epsilon = eps;
        self
    }

    /// Active la variante **AMSGrad** — même sémantique que
    /// [`Adam::with_amsgrad`] : le dénominateur utilise le maximum
    /// historique du 2e moment.
    pub fn with_amsgrad(mut self) -> Self {
        self.amsgrad = true;
        self
    }
}

impl Optimizer for AdamW {
    fn step(&mut self, params: &[usize], tape: &Tape) {
        self.t += 1;
        let bc1 = 1.0 - self.beta1.powi(self.t as i32); // bias correction 1
        let bc2 = 1.0 - self.beta2.powi(self.t as i32); // bias correction 2

        for &idx in params
        {
            let mut value = tape.value(idx);
            let grad = tape.grad(idx);
            assert_eq!(
                value.shape(),
                grad.shape(),
                "AdamW::step: shape mismatch param/grad (idx={})",
                idx
            );

            let m = self
                .m
                .entry(idx)
                .or_insert_with(|| Tensor::zeros(value.rows, value.cols));
            let v = self
                .v
                .entry(idx)
                .or_insert_with(|| Tensor::zeros(value.rows, value.cols));
            let mut v_max = self.amsgrad.then(|| {
                self.v_max
                    .entry(idx)
                    .or_insert_with(|| Tensor::zeros(value.rows, value.cols))
            });

            for i in 0..value.data.len()
            {
                // Différence avec Adam : le gradient reste brut — le weight
                // decay ne passe PAS par les moments.
                let g = grad.data[i];

                // m = β1 m + (1−β1) g
                m.data[i] = self.beta1 * m.data[i] + (1.0 - self.beta1) * g;
                // v = β2 v + (1−β2) g²
                v.data[i] = self.beta2 * v.data[i] + (1.0 - self.beta2) * g * g;

                // bias correction
                let m_hat = m.data[i] / bc1;
                let mut v_hat = v.data[i] / bc2;

                // AMSGrad : normalisation par le max historique du 2e moment.
                if let Some(vm) = &mut v_max
                {
                    vm.data[i] = vm.data[i].max(v.data[i]);
                    v_hat = vm.data[i] / bc2;
                }

                // θ ← θ − lr·m̂/(√v̂ + ε) − lr·wd·θ
                // (decay découplé, calculé sur le θ pré-update ; à wd = 0 le
                // terme est nul et l'update est bit-à-bit celui d'Adam)
                let theta = value.data[i];
                value.data[i] -= self.lr * m_hat / (v_hat.sqrt() + self.epsilon)
                    + self.lr * self.weight_decay * theta;
            }
            tape.set_value(idx, value);
        }
    }

    fn lr(&self) -> f32 {
        self.lr
    }
    fn set_lr(&mut self, lr: f32) {
        self.lr = lr;
    }
}

// ================================================================== //
//  apply_schedule — applique un scheduler à n'importe quel optimizer //
// ================================================================== //

/// Met à jour le learning rate de l'optimizer en fonction du scheduler
/// et du step courant.
///
/// Générique sur le type d'optimizer (Sgd, Adam, AdamW, ou tout autre
/// qui implémente le trait Optimizer). Pour piloter aussi les familles
/// N-D (`crate::nn::nd_optim`) et raw-slice (`crate::optim`), voir le
/// pendant cross-famille
/// [`LrSchedule::drive`](crate::autodiff::scheduler::LrSchedule::drive).
pub fn apply_schedule(
    scheduler: &impl crate::autodiff::scheduler::LrSchedule,
    opt: &mut dyn Optimizer,
    step: usize,
) {
    opt.set_lr(scheduler.lr_at(step));
}

// ================================================================== //
//  Tests                                                              //
// ================================================================== //

#[cfg(test)]
mod tests {
    use super::*;

    // ---------- SGD ---------- //

    #[test]
    fn sgd_step_decreases_param() {
        // f(x) = x²/2, gradient = x. SGD avec lr=0.1 : x ← x - 0.1*x = 9.0
        let tape = Tape::new();
        let x = tape.input(Tensor::from_vec(vec![10.0], 1, 1));
        let half = tape.input(Tensor::from_vec(vec![0.5], 1, 1));
        let sq = x.hadamard(x);
        let loss = sq.hadamard(half).sum();
        loss.tape.backward(loss.idx());

        let mut opt = Sgd::new(0.1);
        opt.step(&[x.idx()], &tape);

        let new_x = tape.value(x.idx()).data[0];
        assert!((new_x - 9.0).abs() < 1e-5, "got {new_x}");
    }

    #[test]
    fn sgd_set_lr_updates() {
        let mut opt = Sgd::new(0.001);
        assert_eq!(opt.lr(), 0.001);
        opt.set_lr(0.01);
        assert_eq!(opt.lr(), 0.01);
    }

    #[test]
    fn sgd_with_momentum_accelerates() {
        // Avec momentum=0.9, le 2ème step doit aller plus loin que sans momentum
        let lr = 0.1;
        let mut x_no_mom = 10.0_f32;
        let mut x_with_mom = 10.0_f32;
        let mut opt_no_mom = Sgd::new(lr);
        let mut opt_with_mom = Sgd::new(lr).with_momentum(0.9);

        for _ in 0..3
        {
            // Sans momentum
            let tape = Tape::new();
            let xv = tape.input(Tensor::from_vec(vec![x_no_mom], 1, 1));
            let loss = xv.hadamard(xv).sum();
            tape.backward(loss.idx());
            opt_no_mom.step(&[xv.idx()], &tape);
            x_no_mom = tape.value(xv.idx()).data[0];

            // Avec momentum
            let tape = Tape::new();
            let xv = tape.input(Tensor::from_vec(vec![x_with_mom], 1, 1));
            let loss = xv.hadamard(xv).sum();
            tape.backward(loss.idx());
            opt_with_mom.step(&[xv.idx()], &tape);
            x_with_mom = tape.value(xv.idx()).data[0];
        }

        // Le momentum doit amener x plus près de 0
        assert!(
            x_with_mom.abs() < x_no_mom.abs(),
            "Avec momentum: |x|={}, sans: |x|={}",
            x_with_mom.abs(),
            x_no_mom.abs()
        );
    }

    // ---------- Adam ---------- //

    #[test]
    fn adam_step_runs() {
        let tape = Tape::new();
        let x = tape.input(Tensor::from_vec(vec![5.0, -3.0], 1, 2));
        let loss = x.hadamard(x).sum();
        tape.backward(loss.idx());

        let mut opt = Adam::new(0.1);
        opt.step(&[x.idx()], &tape);

        let new_x = tape.value(x.idx());
        // Le gradient est 2x = (10, -6). Adam premier pas avec bias correction
        // ≈ ±lr (parce que m_hat / sqrt(v_hat) ≈ sign(g) au premier step).
        assert!(
            new_x.data[0] < 5.0,
            "x[0] = {} should decrease",
            new_x.data[0]
        );
        assert!(
            new_x.data[1] > -3.0,
            "x[1] = {} should increase",
            new_x.data[1]
        );
    }

    #[test]
    fn adam_set_lr_updates() {
        let mut opt = Adam::new(0.001);
        assert_eq!(opt.lr(), 0.001);
        opt.set_lr(0.01);
        assert_eq!(opt.lr(), 0.01);
    }

    // ---------- ORACLE : convergence réelle d'Adam ---------- //

    #[test]
    fn adam_converges_on_quadratic() {
        // ORACLE : minimise f(x) = (x - 3)² partant de x=0.
        // Après N steps Adam, x doit converger vers 3.
        //
        // Si ce test passe, Adam :
        //   - Lit correctement les gradients accumulés sur la tape
        //   - Met à jour les paramètres dans la bonne direction
        //   - Converge réellement vers le minimum (pas juste "fait descendre")

        let lr = 0.1;
        let n_steps = 200;
        let target = 3.0_f32;

        let mut x_value = 0.0_f32;
        let mut opt = Adam::new(lr);

        for _step in 0..n_steps
        {
            // Tape éphémère par step (la tape accumule les nodes sinon).
            // Le paramètre vit dans une variable Rust normale et est ré-injecté
            // à chaque itération.
            let tape = Tape::new();
            let x = tape.input(Tensor::from_vec(vec![x_value], 1, 1));
            let target_t = tape.input(Tensor::from_vec(vec![target], 1, 1));

            // loss = (x - target)²
            let diff = x.sub(target_t);
            let loss = diff.hadamard(diff).sum();
            tape.backward(loss.idx());

            opt.step(&[x.idx()], &tape);
            x_value = tape.value(x.idx()).data[0];
        }

        assert!(
            (x_value - target).abs() < 0.05,
            "Adam n'a pas convergé: x final = {}, target = {}",
            x_value,
            target
        );
    }

    #[test]
    fn adam_converges_on_2d_quadratic() {
        // Vérifie qu'Adam optimise indépendamment chaque dimension.
        // f(x, y) = (x - 1)² + (y + 2)²  → minimum à (1, -2).

        let lr = 0.1;
        let n_steps = 300;
        let target = [1.0_f32, -2.0_f32];

        let mut params = [0.0_f32, 0.0_f32];
        let mut opt = Adam::new(lr);

        for _ in 0..n_steps
        {
            let tape = Tape::new();
            let x = tape.input(Tensor::from_vec(vec![params[0], params[1]], 1, 2));
            let t = tape.input(Tensor::from_vec(vec![target[0], target[1]], 1, 2));

            let diff = x.sub(t);
            let loss = diff.hadamard(diff).sum();
            tape.backward(loss.idx());

            opt.step(&[x.idx()], &tape);
            let v = tape.value(x.idx());
            params[0] = v.data[0];
            params[1] = v.data[1];
        }

        assert!(
            (params[0] - target[0]).abs() < 0.05,
            "x[0] = {}, target = {}",
            params[0],
            target[0]
        );
        assert!(
            (params[1] - target[1]).abs() < 0.05,
            "x[1] = {}, target = {}",
            params[1],
            target[1]
        );
    }

    // ---------- AMSGrad ---------- //

    #[test]
    fn amsgrad_converges_on_quadratic() {
        // Même oracle que adam_converges_on_quadratic, variante AMSGrad :
        // minimise f(x) = (x - 3)² partant de x = 0.
        let lr = 0.1;
        let n_steps = 200;
        let target = 3.0_f32;

        let mut x_value = 0.0_f32;
        let mut opt = Adam::new(lr).with_amsgrad();

        for _ in 0..n_steps
        {
            let tape = Tape::new();
            let x = tape.input(Tensor::from_vec(vec![x_value], 1, 1));
            let target_t = tape.input(Tensor::from_vec(vec![target], 1, 1));
            let diff = x.sub(target_t);
            let loss = diff.hadamard(diff).sum();
            tape.backward(loss.idx());

            opt.step(&[x.idx()], &tape);
            x_value = tape.value(x.idx()).data[0];
        }

        assert!(
            (x_value - target).abs() < 0.05,
            "AMSGrad n'a pas convergé: x final = {}, target = {}",
            x_value,
            target
        );
    }

    #[test]
    fn amsgrad_keeps_denominator_after_gradient_spike() {
        // Propriété définitoire d'AMSGrad (Reddi et al. 2018) : après un pic
        // de gradient, Adam « oublie » le grand 2e moment (moyenne mobile) et
        // reprend de grands pas ; AMSGrad garde le max historique et ses pas
        // restent petits. β2 volontairement bas pour que l'oubli d'Adam soit
        // rapide et le contraste net.
        fn tail_displacement(amsgrad: bool) -> f32 {
            let mut opt = Adam::new(0.1).with_betas(0.9, 0.1);
            if amsgrad
            {
                opt = opt.with_amsgrad();
            }
            let mut x = 0.0_f32;
            let mut tail = 0.0_f32;
            for step in 0..20
            {
                // loss = c·x → gradient constant c : 100 au 1er pas, puis 0,01.
                let c = if step == 0 { 100.0 } else { 0.01 };
                let tape = Tape::new();
                let xv = tape.input(Tensor::from_vec(vec![x], 1, 1));
                let cv = tape.input(Tensor::from_vec(vec![c], 1, 1));
                let loss = xv.hadamard(cv).sum();
                tape.backward(loss.idx());
                opt.step(&[xv.idx()], &tape);
                let new_x = tape.value(xv.idx()).data[0];
                if step > 0
                {
                    tail += (new_x - x).abs();
                }
                x = new_x;
            }
            tail
        }

        let adam = tail_displacement(false);
        let ams = tail_displacement(true);
        assert!(
            ams < adam * 0.1,
            "AMSGrad devrait faire des pas bien plus petits après le pic: \
             adam={adam}, amsgrad={ams}"
        );
    }

    // ---------- AdamW ---------- //

    /// Un pas d'entraînement `f(x) = (x − target)²` sur une tape éphémère :
    /// renvoie la nouvelle valeur du paramètre. Factorise les trajectoires
    /// Adam-vs-AdamW des tests ci-dessous.
    fn quadratic_step(opt: &mut impl Optimizer, x_value: f32, target: f32) -> f32 {
        let tape = Tape::new();
        let x = tape.input(Tensor::from_vec(vec![x_value], 1, 1));
        let target_t = tape.input(Tensor::from_vec(vec![target], 1, 1));
        let diff = x.sub(target_t);
        let loss = diff.hadamard(diff).sum();
        tape.backward(loss.idx());
        opt.step(&[x.idx()], &tape);
        tape.value(x.idx()).data[0]
    }

    #[test]
    fn adamw_set_lr_updates() {
        let mut opt = AdamW::new(0.001);
        assert_eq!(opt.lr(), 0.001);
        opt.set_lr(0.01);
        assert_eq!(opt.lr(), 0.01);
    }

    /// ORACLE drop-in : avec `weight_decay = 0`, AdamW doit produire une
    /// trajectoire **bit-à-bit identique** à Adam (même arithmétique f32,
    /// dans le même ordre) — c'est ce qui rend le swap Adam → AdamW sûr.
    #[test]
    fn adamw_zero_decay_is_bitwise_identical_to_adam() {
        let lr = 0.1;
        let target = 3.0_f32;

        let mut adam = Adam::new(lr);
        let mut adamw = AdamW::new(lr).with_weight_decay(0.0);
        let mut x_adam = 0.0_f32;
        let mut x_adamw = 0.0_f32;

        for step in 0..50
        {
            x_adam = quadratic_step(&mut adam, x_adam, target);
            x_adamw = quadratic_step(&mut adamw, x_adamw, target);
            assert_eq!(
                x_adam.to_bits(),
                x_adamw.to_bits(),
                "step {step}: adam={x_adam}, adamw={x_adamw}"
            );
        }
    }

    /// Avec `weight_decay > 0`, le terme découplé `−lr·wd·θ` tire les
    /// paramètres vers 0 : le point de convergence d'AdamW doit être
    /// strictement plus petit que celui d'Adam sur le même problème.
    #[test]
    fn adamw_weight_decay_shrinks_params_relative_to_adam() {
        let lr = 0.1;
        let target = 3.0_f32;
        let n_steps = 300;

        let mut adam = Adam::new(lr);
        let mut adamw = AdamW::new(lr).with_weight_decay(0.5);
        let mut x_adam = 0.0_f32;
        let mut x_adamw = 0.0_f32;

        for _ in 0..n_steps
        {
            x_adam = quadratic_step(&mut adam, x_adam, target);
            x_adamw = quadratic_step(&mut adamw, x_adamw, target);
        }

        assert!(
            (x_adam - target).abs() < 0.05,
            "Adam n'a pas convergé: {x_adam}"
        );
        assert!(
            x_adamw > 0.0 && x_adamw < x_adam - 0.1,
            "le weight decay devrait tirer AdamW ({x_adamw}) nettement \
             sous Adam ({x_adam})"
        );
    }

    /// ORACLE : même sanity de convergence que `adam_converges_on_quadratic`,
    /// avec le weight decay par défaut (0.01) — le décalage induit sur le
    /// minimum de `(x − 3)²` est négligeable devant la tolérance.
    #[test]
    fn adamw_converges_on_quadratic() {
        let lr = 0.1;
        let n_steps = 200;
        let target = 3.0_f32;

        let mut x_value = 0.0_f32;
        let mut opt = AdamW::new(lr);

        for _ in 0..n_steps
        {
            x_value = quadratic_step(&mut opt, x_value, target);
        }

        assert!(
            (x_value - target).abs() < 0.05,
            "AdamW n'a pas convergé: x final = {}, target = {}",
            x_value,
            target
        );
    }

    #[test]
    fn adam_does_nothing_without_step() {
        // Sanity check : sans appeler step(), les paramètres ne bougent pas.
        // Catch le bug "Adam = stub" qu'on a connu.

        let tape = Tape::new();
        let x = tape.input(Tensor::from_vec(vec![5.0], 1, 1));
        let loss = x.hadamard(x).sum();
        tape.backward(loss.idx());

        // PAS d'opt.step() ici
        let new_x = tape.value(x.idx()).data[0];
        assert_eq!(new_x, 5.0, "x a changé sans step() !");
    }

    // ---------- apply_schedule ---------- //

    #[test]
    fn apply_schedule_works_with_adam() {
        use crate::autodiff::scheduler::LrSchedule;

        struct ConstantHalf;
        impl LrSchedule for ConstantHalf {
            fn lr_at(&self, _step: usize) -> f32 {
                0.5
            }
        }

        let mut opt = Adam::new(0.001);
        apply_schedule(&ConstantHalf, &mut opt, 0);
        assert_eq!(opt.lr(), 0.5);
    }

    #[test]
    fn apply_schedule_works_with_sgd() {
        use crate::autodiff::scheduler::LrSchedule;

        struct StepDecay;
        impl LrSchedule for StepDecay {
            fn lr_at(&self, step: usize) -> f32 {
                if step < 10 { 0.1 } else { 0.01 }
            }
        }

        let mut opt = Sgd::new(0.0);
        apply_schedule(&StepDecay, &mut opt, 5);
        assert_eq!(opt.lr(), 0.1);
        apply_schedule(&StepDecay, &mut opt, 15);
        assert_eq!(opt.lr(), 0.01);
    }
}
