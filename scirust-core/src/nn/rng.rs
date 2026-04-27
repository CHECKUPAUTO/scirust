// scirust-core/src/nn/rng.rs
//
// PcgEngine — RNG portable, seedable, déterministe.
// Implémentation PCG-XSH-RR 64/32 (Melissa O'Neill 2014).
//
// Garanties :
//   - 100 % pure Rust, pas de syscalls, donc reproductible cross-platform
//   - même seed → même séquence sur x86_64, aarch64, wasm32
//   - période 2^64, distribution uniforme cryptographiquement décente
//     pour de l'init de poids (NB : pas pour de la cryptographie)

pub struct PcgEngine {
    state: u64,
    inc:   u64,
}

impl PcgEngine {
    /// Crée un RNG depuis une graine 64 bits.
    pub fn new(seed: u64) -> Self {
        // Initialisation à la PCG : on amorce l'état puis on consomme
        // deux valeurs pour bien diffuser le seed.
        let mut rng = Self { state: 0, inc: (seed << 1) | 1 };
        rng.next_u32();
        rng.state = rng.state.wrapping_add(seed);
        rng.next_u32();
        rng
    }

    /// Tire un u32 uniformément distribué.
    pub fn next_u32(&mut self) -> u32 {
        let oldstate = self.state;
        self.state = oldstate
            .wrapping_mul(6364136223846793005)
            .wrapping_add(self.inc);
        let xorshifted = (((oldstate >> 18) ^ oldstate) >> 27) as u32;
        let rot = (oldstate >> 59) as u32;
        xorshifted.rotate_right(rot)
    }

    /// f32 uniforme dans [0, 1).
    /// Utilise les 24 bits de poids fort pour matcher la mantisse f32.
    pub fn float(&mut self) -> f32 {
        (self.next_u32() >> 8) as f32 / (1u32 << 24) as f32
    }

    /// f32 uniforme dans [low, high).
    pub fn uniform(&mut self, low: f32, high: f32) -> f32 {
        low + (high - low) * self.float()
    }

    /// f32 normal centré-réduit via Box-Muller.
    pub fn normal(&mut self) -> f32 {
        let u1 = self.float().max(1e-7);  // évite log(0)
        let u2 = self.float();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_with_same_seed() {
        let mut a = PcgEngine::new(42);
        let mut b = PcgEngine::new(42);
        for _ in 0..1000 { assert_eq!(a.next_u32(), b.next_u32()); }
    }

    #[test]
    fn float_in_range() {
        let mut rng = PcgEngine::new(7);
        for _ in 0..10000 {
            let f = rng.float();
            assert!(f >= 0.0 && f < 1.0, "got {f}");
        }
    }

    #[test]
    fn normal_roughly_centered() {
        let mut rng = PcgEngine::new(123);
        let n = 10_000;
        let sum: f32 = (0..n).map(|_| rng.normal()).sum();
        let mean = sum / n as f32;
        // Écart-type de la moyenne d'un échantillon N(0,1) de taille n est 1/√n
        // donc à 4σ on tolère ±0.04 pour n=10000
        assert!(mean.abs() < 0.05, "mean = {mean}");
    }
}
