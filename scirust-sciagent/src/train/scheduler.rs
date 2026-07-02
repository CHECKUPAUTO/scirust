use scirust_core::autodiff::scheduler::LrSchedule;

pub struct WarmupCosineSchedule {
    pub base: f32,
    pub min_lr: f32,
    pub warmup_steps: usize,
    pub total_steps: usize,
}

impl WarmupCosineSchedule {
    pub fn new(base: f32, min_lr: f32, warmup_steps: usize, total_steps: usize) -> Self {
        Self {
            base,
            min_lr,
            warmup_steps,
            total_steps,
        }
    }
}

impl LrSchedule for WarmupCosineSchedule {
    fn lr_at(&self, step: usize) -> f32 {
        if step < self.warmup_steps {
            self.base * (step as f32 / self.warmup_steps as f32)
        } else {
            let post = step - self.warmup_steps;
            let period = self.total_steps - self.warmup_steps;
            if post >= period {
                return self.min_lr;
            }
            let progress = post as f32 / period as f32;
            let cos_factor = 0.5 * (1.0 + (std::f32::consts::PI * progress).cos());
            self.min_lr + (self.base - self.min_lr) * cos_factor
        }
    }
}

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_warmup_cosine_phases() {
        let s = WarmupCosineSchedule::new(0.01, 0.0001, 100, 1000);
        assert!(s.lr_at(0) < 1e-6);
        assert!((s.lr_at(50) - 0.005).abs() < 1e-5);
        assert!((s.lr_at(100) - 0.01).abs() < 1e-5);
        assert!(s.lr_at(999) < 0.001);
    }

    #[test]
    fn test_warmup_cosine_continuous() {
        let s = WarmupCosineSchedule::new(0.1, 0.001, 100, 1000);
        let before = s.lr_at(99);
        let after = s.lr_at(100);
        assert!((after - before).abs() < 0.01);
    }

    #[test]
    fn test_constant_lr() {
        let s = ConstantLr::new(0.05);
        assert!((s.lr_at(0) - 0.05).abs() < 1e-7);
        assert!((s.lr_at(100) - 0.05).abs() < 1e-7);
    }
}
