// scirust-core/src/nn/rng.rs
//
// Générateur pseudo-aléatoire PCG (Permuted Congruential Generator).
// Petit, rapide, statistiquement bon. Suffisant pour les inits de poids.
//
// Référence : https://www.pcg-random.org/

pub struct PcgEngine {
    state: u64,
    inc:   u64,
}

impl PcgEngine {
    pub fn new(seed: u64) -> Self {
        let mut rng = Self { state: 0, inc: (seed << 1) | 1 };
        rng.next_u32();
        rng.state = rng.state.wrapping_add(seed);
        rng.next_u32();
        rng
    }

    /// Tire un u32 pseudo-aléatoire.
    pub fn next_u32(&mut self) -> u32 {
        let oldstate = self.state;
        self.state = oldstate.wrapping_mul(6364136223846793005).wrapping_add(self.inc);
        let xorshifted = (((oldstate >> 18) ^ oldstate) >> 27) as u32;
        let rot = (oldstate >> 59) as u32;
        (xorshifted >> rot) | (xorshifted << ((rot.wrapping_neg()) & 31))
    }

    /// Tire un f32 dans [0, 1).
    pub fn float(&mut self) -> f32 {
        (self.next_u32() as f32) / (u32::MAX as f32 + 1.0)
    }

    /// Tire un f32 dans [-1, 1).
    pub fn float_signed(&mut self) -> f32 {
        2.0 * self.float() - 1.0
    }

    /// Échantillonne depuis N(mean, stddev) via Box-Muller transform.
    pub fn normal(&mut self, mean: f32, stddev: f32) -> f32 {
        let u1 = self.float().max(1e-12);
        let u2 = self.float();
        let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
        mean + stddev * z
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_with_same_seed() {
        let mut a = PcgEngine::new(42);
        let mut b = PcgEngine::new(42);
        for _ in 0..100 {
            assert_eq!(a.next_u32(), b.next_u32());
        }
    }

    #[test]
    fn float_in_unit_range() {
        let mut rng = PcgEngine::new(123);
        for _ in 0..1000 {
            let x = rng.float();
            assert!(x >= 0.0 && x < 1.0, "float out of range: {x}");
        }
    }

    #[test]
    fn normal_has_reasonable_stats() {
        let mut rng = PcgEngine::new(7);
        let n = 10000;
        let mut sum = 0.0_f32;
        let mut sq = 0.0_f32;
        for _ in 0..n {
            let x = rng.normal(0.0, 1.0);
            sum += x;
            sq += x * x;
        }
        let mean = sum / n as f32;
        let var = sq / n as f32 - mean * mean;
        // Mean should be close to 0, var close to 1
        assert!(mean.abs() < 0.05, "mean = {mean}");
        assert!((var - 1.0).abs() < 0.1, "var = {var}");
    }
}
