// scirust-core/src/autodiff/scheduler.rs
//
// Learning rate schedulers — outils pour faire varier le LR pendant
// l'entraînement, ce qui est crucial pour atteindre les meilleures
// performances sur les vrais datasets.
//
// USAGE TYPIQUE :
//
//   let mut opt = Adam::new(0.01);
//   let scheduler = WarmupCosine::new(0.01, 0.0001, 1000, 50_000);
//
//   for step in 0..n_steps {
//       opt.set_lr(scheduler.lr_at(step));
//       // ... forward / backward / opt.step ...
//   }
//
// Tous les schedulers sont stateless et déterministes : `lr_at(step)`
// dépend uniquement de `step`. Ça permet de reproduire un entraînement
// exact en redémarrant à n'importe quel step (utile pour checkpointing).

// ================================================================== //
//  Trait                                                              //
// ================================================================== //

pub trait LrSchedule: Send + Sync {
    /// Renvoie le learning rate pour le step donné.
    /// Les implémenteurs doivent garantir que cette fonction est
    /// déterministe : même `step` → même `lr`.
    fn lr_at(&self, step: usize) -> f32;
}

// Les Box<dyn LrSchedule> peuvent eux-mêmes être schedulers
impl<T: LrSchedule + ?Sized> LrSchedule for Box<T> {
    fn lr_at(&self, step: usize) -> f32 {
        (**self).lr_at(step)
    }
}

// ================================================================== //
//  Schedulers concrets                                                //
// ================================================================== //

/// Learning rate constant. Utile comme baseline ou pour des ablations.
#[derive(Clone, Debug)]
pub struct ConstantLr {
    pub lr: f32,
}

impl ConstantLr {
    pub fn new(lr: f32) -> Self {
        Self { lr }
    }
}

impl LrSchedule for ConstantLr {
    fn lr_at(&self, _step: usize) -> f32 {
        self.lr
    }
}

// ------------------------------------------------------------------ //

/// Réduction par paliers : `lr = initial · decay^(step / step_size)`.
/// Exemple typique : initial=0.1, decay=0.1, step_size=30 epochs
/// → /10 toutes les 30 epochs.
#[derive(Clone, Debug)]
pub struct StepLr {
    pub initial: f32,
    pub decay: f32,
    pub step_size: usize,
}

impl StepLr {
    pub fn new(initial: f32, decay: f32, step_size: usize) -> Self {
        assert!(step_size > 0, "StepLr: step_size > 0 requis");
        Self {
            initial,
            decay,
            step_size,
        }
    }
}

impl LrSchedule for StepLr {
    fn lr_at(&self, step: usize) -> f32 {
        let n_decays = (step / self.step_size) as i32;
        self.initial * self.decay.powi(n_decays)
    }
}

// ------------------------------------------------------------------ //

/// Décroissance exponentielle continue : `lr = initial · gamma^step`.
/// Plus lisse que StepLr, sans paliers visibles.
#[derive(Clone, Debug)]
pub struct ExponentialLr {
    pub initial: f32,
    pub gamma: f32,
}

impl ExponentialLr {
    pub fn new(initial: f32, gamma: f32) -> Self {
        assert!(gamma > 0.0 && gamma <= 1.0, "ExponentialLr: gamma ∈ (0, 1]");
        Self { initial, gamma }
    }
}

impl LrSchedule for ExponentialLr {
    fn lr_at(&self, step: usize) -> f32 {
        self.initial * self.gamma.powi(step as i32)
    }
}

// ------------------------------------------------------------------ //

/// Cosine annealing : descend doucement de `initial` à `min_lr` sur
/// `period` steps, en suivant une courbe en cosinus :
///
///   lr(t) = min + (initial - min) · (1 + cos(π·t/period)) / 2
///
/// Au-delà de `period`, reste à `min_lr` (pas de redémarrage cyclique
/// dans cette implémentation simple — voir CosineAnnealingWarmRestarts
/// pour le mode cyclique).
#[derive(Clone, Debug)]
pub struct CosineAnnealing {
    pub initial: f32,
    pub min_lr: f32,
    pub period: usize,
}

impl CosineAnnealing {
    pub fn new(initial: f32, min_lr: f32, period: usize) -> Self {
        assert!(period > 0, "CosineAnnealing: period > 0 requis");
        assert!(
            min_lr <= initial,
            "CosineAnnealing: min_lr <= initial requis"
        );
        Self {
            initial,
            min_lr,
            period,
        }
    }
}

impl LrSchedule for CosineAnnealing {
    fn lr_at(&self, step: usize) -> f32 {
        if step >= self.period
        {
            return self.min_lr;
        }
        let progress = step as f32 / self.period as f32; // [0, 1]
        let cos_factor = 0.5 * (1.0 + (std::f32::consts::PI * progress).cos());
        self.min_lr + (self.initial - self.min_lr) * cos_factor
    }
}

// ------------------------------------------------------------------ //

/// Warmup linéaire suivi d'un cosine annealing :
///   - de 0 à warmup_steps : LR croît linéairement de 0 à `base`
///   - de warmup_steps à total_steps : cosine de `base` vers `min_lr`
///
/// C'est le scheduler par défaut de quasi tous les Transformers
/// modernes. Critique pour les modèles qui ont des gradients instables
/// au démarrage.
#[derive(Clone, Debug)]
pub struct WarmupCosine {
    pub base: f32,
    pub min_lr: f32,
    pub warmup_steps: usize,
    pub total_steps: usize,
}

impl WarmupCosine {
    pub fn new(base: f32, min_lr: f32, warmup_steps: usize, total_steps: usize) -> Self {
        assert!(
            warmup_steps < total_steps,
            "WarmupCosine: warmup_steps < total_steps requis"
        );
        assert!(min_lr <= base);
        Self {
            base,
            min_lr,
            warmup_steps,
            total_steps,
        }
    }
}

impl LrSchedule for WarmupCosine {
    fn lr_at(&self, step: usize) -> f32 {
        if step < self.warmup_steps
        {
            // Linéaire de 0 à base
            self.base * (step as f32 / self.warmup_steps as f32)
        }
        else
        {
            // Cosine de base à min_lr sur le reste
            let post = step - self.warmup_steps;
            let period = self.total_steps - self.warmup_steps;
            if post >= period
            {
                return self.min_lr;
            }
            let progress = post as f32 / period as f32;
            let cos_factor = 0.5 * (1.0 + (std::f32::consts::PI * progress).cos());
            self.min_lr + (self.base - self.min_lr) * cos_factor
        }
    }
}

// ------------------------------------------------------------------ //

/// Réduit le LR quand la métrique de validation stagne.
/// Stateful : nécessite d'être appelé à chaque epoch avec la valeur de loss.
///
///   factor   : facteur de réduction (e.g. 0.5 pour /2 quand patience dépassée)
///   patience : nombre d'epochs sans amélioration avant de réduire
///   min_lr   : plancher absolu
///
/// USAGE :
///   let mut sched = ReduceOnPlateau::new(0.01, 0.5, 5, 1e-6);
///   for epoch {
///       opt.set_lr(sched.current_lr());
///       train_one_epoch(...);
///       let val_loss = evaluate(...);
///       sched.step(val_loss);   // → met à jour le LR si plateau
///   }
pub struct ReduceOnPlateau {
    initial: f32,
    pub factor: f32,
    pub patience: usize,
    pub min_lr: f32,
    current: f32,
    best_loss: f32,
    n_bad_epochs: usize,
}

impl ReduceOnPlateau {
    pub fn new(initial: f32, factor: f32, patience: usize, min_lr: f32) -> Self {
        assert!(factor > 0.0 && factor < 1.0, "factor ∈ (0, 1)");
        Self {
            initial,
            factor,
            patience,
            min_lr,
            current: initial,
            best_loss: f32::INFINITY,
            n_bad_epochs: 0,
        }
    }

    pub fn current_lr(&self) -> f32 {
        self.current
    }

    /// Appelle ça à chaque fin d'epoch avec la val_loss observée.
    /// Renvoie le nouveau LR.
    pub fn step(&mut self, val_loss: f32) -> f32 {
        if val_loss < self.best_loss
        {
            self.best_loss = val_loss;
            self.n_bad_epochs = 0;
        }
        else
        {
            self.n_bad_epochs += 1;
            if self.n_bad_epochs >= self.patience
            {
                self.current = (self.current * self.factor).max(self.min_lr);
                self.n_bad_epochs = 0;
            }
        }
        self.current
    }

    pub fn reset(&mut self) {
        self.current = self.initial;
        self.best_loss = f32::INFINITY;
        self.n_bad_epochs = 0;
    }
}

// Note : ReduceOnPlateau n'implémente pas LrSchedule parce que son
// comportement dépend d'un état externe (val_loss), pas seulement du step.

// ================================================================== //
//  Tests                                                              //
// ================================================================== //
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constant_lr() {
        let s = ConstantLr::new(0.05);
        for step in [0, 1, 100, 10000]
        {
            assert_eq!(s.lr_at(step), 0.05);
        }
    }

    #[test]
    fn step_lr_decays_at_boundaries() {
        let s = StepLr::new(0.1, 0.1, 30);
        assert!((s.lr_at(0) - 0.1).abs() < 1e-7);
        assert!((s.lr_at(29) - 0.1).abs() < 1e-7);
        assert!((s.lr_at(30) - 0.01).abs() < 1e-7);
        assert!((s.lr_at(60) - 0.001).abs() < 1e-7);
        assert!((s.lr_at(90) - 0.0001).abs() < 1e-7);
    }

    #[test]
    fn exponential_decays_smoothly() {
        let s = ExponentialLr::new(1.0, 0.9);
        assert!((s.lr_at(0) - 1.0).abs() < 1e-6);
        assert!((s.lr_at(1) - 0.9).abs() < 1e-6);
        assert!((s.lr_at(10) - 0.9_f32.powi(10)).abs() < 1e-6);
    }

    #[test]
    fn cosine_starts_at_initial_ends_at_min() {
        let s = CosineAnnealing::new(0.1, 0.001, 100);
        assert!((s.lr_at(0) - 0.1).abs() < 1e-5);
        assert!((s.lr_at(100) - 0.001).abs() < 1e-5);
        assert!((s.lr_at(200) - 0.001).abs() < 1e-5); // post-period clamp
        // Au milieu, on est à peu près à mi-chemin
        let mid = s.lr_at(50);
        assert!(mid > 0.04 && mid < 0.06, "mid lr = {mid}");
    }

    #[test]
    fn cosine_monotonic_decreasing_in_period() {
        let s = CosineAnnealing::new(1.0, 0.0, 100);
        let mut prev = f32::INFINITY;
        for step in 0..=100
        {
            let lr = s.lr_at(step);
            assert!(
                lr <= prev + 1e-7,
                "non-monotonic at step {step}: {prev} → {lr}"
            );
            prev = lr;
        }
    }

    #[test]
    fn warmup_cosine_phases() {
        let s = WarmupCosine::new(0.01, 0.0001, 100, 1000);
        // Au step 0 : warmup débute à 0
        assert!(s.lr_at(0) < 1e-6);
        // Au milieu du warmup : 0.005
        assert!((s.lr_at(50) - 0.005).abs() < 1e-5);
        // Fin du warmup : base
        assert!((s.lr_at(100) - 0.01).abs() < 1e-5);
        // Bien après, on tend vers min
        assert!(s.lr_at(999) < 0.001);
    }

    #[test]
    fn warmup_cosine_continuous_at_transition() {
        // Le LR juste avant et juste après warmup doivent être proches
        let s = WarmupCosine::new(0.1, 0.001, 100, 1000);
        let just_before = s.lr_at(99);
        let just_after = s.lr_at(100);
        assert!(
            (just_after - just_before).abs() < 0.01,
            "discontinuité au transition warmup→cosine : {just_before} vs {just_after}"
        );
    }

    #[test]
    fn reduce_on_plateau_reduces_after_patience() {
        let mut s = ReduceOnPlateau::new(0.1, 0.5, 3, 0.0);
        // Loss qui augmente / stagne pendant > patience
        s.step(1.0); // best = 1.0
        assert_eq!(s.current_lr(), 0.1);
        s.step(1.1); // pire → 1 bad
        s.step(1.05); // pire → 2 bad
        s.step(1.05); // pire → 3 bad → réduit
        assert!((s.current_lr() - 0.05).abs() < 1e-7);
    }

    #[test]
    fn reduce_on_plateau_resets_on_improvement() {
        let mut s = ReduceOnPlateau::new(0.1, 0.5, 2, 0.0);
        s.step(1.0);
        s.step(1.5); // 1 bad
        s.step(0.5); // amélioration → reset
        s.step(0.6); // 1 bad
        // Pas de réduction encore car la patience a été reset
        assert_eq!(s.current_lr(), 0.1);
    }

    #[test]
    fn boxed_scheduler_works() {
        let s: Box<dyn LrSchedule> = Box::new(StepLr::new(0.1, 0.5, 10));
        assert_eq!(s.lr_at(0), 0.1);
        assert_eq!(s.lr_at(10), 0.05);
    }
}
