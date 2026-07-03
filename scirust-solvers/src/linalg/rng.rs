//! Générateur pseudo-aléatoire déterministe minimal (SplitMix64, domaine
//! public, S. Vigna) — pour germer les matrices de test de la SVD
//! aléatoire ([`super::randomized_svd`]) sans dépendre de la crate `rand`
//! ni d'un moteur PRNG d'une autre crate du workspace (évite la dépendance
//! circulaire que `scirust-solvers` évite déjà avec `scirust-core`, voir
//! le commentaire de tête de `linalg/mod.rs`). Même graine ⇒ sortie
//! bit-identique, sur toute plateforme.

pub struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }

    /// Uniforme dans `[0, 1)`, 53 bits de mantisse (construction standard :
    /// voir la documentation de xoshiro256).
    fn next_f64_uniform(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64) * (1.0 / (1u64 << 53) as f64)
    }

    /// Loi normale centrée réduite via la transformée de Box-Muller.
    pub fn next_gaussian(&mut self) -> f64 {
        let u1 = self.next_f64_uniform().max(f64::MIN_POSITIVE);
        let u2 = self.next_f64_uniform();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_seed_is_bit_identical() {
        let mut a = SplitMix64::new(42);
        let mut b = SplitMix64::new(42);
        for _ in 0..100
        {
            assert_eq!(a.next_gaussian(), b.next_gaussian());
        }
    }

    #[test]
    fn different_seeds_diverge() {
        let mut a = SplitMix64::new(1);
        let mut b = SplitMix64::new(2);
        let seq_a: Vec<f64> = (0..10).map(|_| a.next_gaussian()).collect();
        let seq_b: Vec<f64> = (0..10).map(|_| b.next_gaussian()).collect();
        assert_ne!(seq_a, seq_b);
    }

    #[test]
    fn gaussian_samples_are_finite_and_roughly_standard() {
        let mut rng = SplitMix64::new(7);
        let samples: Vec<f64> = (0..10_000).map(|_| rng.next_gaussian()).collect();
        assert!(samples.iter().all(|x| x.is_finite()));
        let mean: f64 = samples.iter().sum::<f64>() / samples.len() as f64;
        let var: f64 =
            samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / samples.len() as f64;
        assert!(mean.abs() < 0.1, "mean {mean} should be near 0");
        assert!((var - 1.0).abs() < 0.1, "variance {var} should be near 1");
    }
}
