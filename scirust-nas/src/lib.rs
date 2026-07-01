//! Neural Architecture Search (NAS) for SciRust.
//!
//! Evolutionary architecture search over layer configurations. The objective is
//! a single scalarized fitness — a weighted sum of a zero-cost shape proxy and
//! (negative) parameter/FLOP penalties (see [`NasSearch::evaluate`]) — optimized
//! by truncation selection plus mutation.
//!
//! # Search Space
//!
//! The search space includes:
//! - Number of layers
//! - Layer types (Linear, Conv2d, TransformerBlock)
//! - Hidden dimensions
//! - Activation functions
//! - Dropout rates
//!
//! # Example
//!
//! ```ignore
//! use scirust_nas::{NasConfig, NasSearch, Architecture};
//!
//! let config = NasConfig::default();
//! let mut search = NasSearch::new(config);
//! let population = search.evolve(10, 50).unwrap();
//! let best = &population[0];
//! println!("Best arch: {:?} (fitness={:.4})", best, best.fitness);
//! ```

use serde::{Deserialize, Serialize};
use std::fmt;

/// An architecture candidate in the search space.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Architecture {
    /// Layer specifications.
    pub layers: Vec<LayerSpec>,
    /// Fitness score (higher = better).
    pub fitness: f64,
    /// Number of parameters (millions).
    pub params_m: f64,
    /// Estimated FLOPs.
    pub flops: f64,
    /// Validation accuracy (if evaluated).
    pub accuracy: Option<f64>,
}

impl fmt::Display for Architecture {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[")?;
        for (i, layer) in self.layers.iter().enumerate()
        {
            if i > 0
            {
                write!(f, " → ")?;
            }
            write!(f, "{}", layer)?;
        }
        write!(f, "] (fit={:.4}, {}M params)", self.fitness, self.params_m)
    }
}

/// A single layer specification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LayerSpec {
    Linear {
        out_features: usize,
        activation: ActivationChoice,
    },
    Conv2d {
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
    },
    TransformerBlock {
        embed_dim: usize,
        num_heads: usize,
        ff_dim: usize,
    },
    Dropout {
        rate: f32,
    },
    Pool {
        kind: PoolKind,
    },
}

impl fmt::Display for LayerSpec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self
        {
            LayerSpec::Linear {
                out_features,
                activation,
            } =>
            {
                write!(f, "Linear({},{})", out_features, activation)
            },
            LayerSpec::Conv2d {
                out_channels,
                kernel_size,
                stride,
            } =>
            {
                write!(f, "Conv2d({},{},{})", out_channels, kernel_size, stride)
            },
            LayerSpec::TransformerBlock {
                embed_dim,
                num_heads,
                ff_dim,
            } =>
            {
                write!(f, "Transformer({},{},{})", embed_dim, num_heads, ff_dim)
            },
            LayerSpec::Dropout { rate } => write!(f, "Dropout({})", rate),
            LayerSpec::Pool { kind } => write!(f, "Pool({:?})", kind),
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum ActivationChoice {
    ReLU,
    GELU,
    SiLU,
    Tanh,
}

impl fmt::Display for ActivationChoice {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum PoolKind {
    Max,
    Avg,
    AdaptiveAvg,
}

/// NAS search configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NasConfig {
    /// Search space constraints.
    pub min_layers: usize,
    pub max_layers: usize,
    pub min_hidden: usize,
    pub max_hidden: usize,
    /// Objective weights.
    pub accuracy_weight: f64,
    pub params_weight: f64,
    pub flops_weight: f64,
    /// Optimization budget.
    pub max_models: usize,
    /// Random seed.
    pub seed: u64,
}

impl Default for NasConfig {
    fn default() -> Self {
        Self {
            min_layers: 2,
            max_layers: 8,
            min_hidden: 32,
            max_hidden: 1024,
            accuracy_weight: 1.0,
            params_weight: -0.1,
            flops_weight: -0.05,
            max_models: 500,
            seed: 42,
        }
    }
}

/// Neural Architecture Search engine.
pub struct NasSearch {
    config: NasConfig,
    population: Vec<Architecture>,
    evaluations: usize,
}

impl NasSearch {
    pub fn new(config: NasConfig) -> Self {
        Self {
            config,
            population: Vec::new(),
            evaluations: 0,
        }
    }

    /// Generate a random architecture.
    pub fn random_architecture(&self, seed: u64) -> Architecture {
        let n_layers = self.config.min_layers
            + ((seed as usize) % (self.config.max_layers - self.config.min_layers + 1));

        let mut layers = Vec::new();
        // xorshift is stuck at zero if the state is ever 0, which would make
        // every layer identical (all `Linear(min_hidden, ReLU)`). Seed a
        // nonzero constant for the degenerate input so the PRNG still varies.
        let mut rng = if seed == 0 { 0x9E3779B97F4A7C15 } else { seed };

        for _i in 0..n_layers
        {
            // Simple PRNG: xorshift
            rng ^= rng << 13;
            rng ^= rng >> 17;
            rng ^= rng << 5;

            let out_dim = self.config.min_hidden
                + (rng as usize % (self.config.max_hidden - self.config.min_hidden + 1));

            let act_idx = (rng >> 8) as usize % 4;
            let activation = match act_idx
            {
                0 => ActivationChoice::ReLU,
                1 => ActivationChoice::GELU,
                2 => ActivationChoice::SiLU,
                _ => ActivationChoice::Tanh,
            };

            layers.push(LayerSpec::Linear {
                out_features: out_dim,
                activation,
            });
        }

        let (params_m, flops) = Self::layer_costs(&layers);
        Architecture {
            layers,
            fitness: 0.0,
            params_m,
            flops,
            accuracy: None,
        }
    }

    /// Estimate `(params_m, flops)` for a chain of layers, mirroring the cost
    /// model used when architectures are first sampled. Only `Linear` layers
    /// contribute to the estimate; other layer kinds are treated as free but
    /// still carry the running dimension forward where applicable.
    ///
    /// The input dimension is fixed at 784 (flattened MNIST), matching
    /// [`NasSearch::random_architecture`].
    fn layer_costs(layers: &[LayerSpec]) -> (f64, f64) {
        let mut total_params: f64 = 0.0;
        let mut total_flops: f64 = 0.0;
        let mut prev_dim: usize = 784; // Default input (MNIST)

        for layer in layers
        {
            if let LayerSpec::Linear { out_features, .. } = layer
            {
                let out_dim = *out_features;
                total_params += (prev_dim * out_dim + out_dim) as f64;
                total_flops += 2.0 * prev_dim as f64 * out_dim as f64;
                prev_dim = out_dim;
            }
        }

        (total_params / 1_000_000.0, total_flops)
    }

    /// Evaluate an architecture with a zero-cost proxy objective: reward a
    /// moderate shape — about 4 layers and ~0.5M parameters (two Gaussian bumps
    /// in `[0, 1]`) — then apply the configured (negative) linear penalties on
    /// parameter count and FLOPs.
    pub fn evaluate(&self, arch: &Architecture) -> f64 {
        let capacity = arch.layers.len() as f64;
        let complexity = arch.params_m;

        let capacity_score = (-(capacity - 4.0).powi(2) / 8.0).exp(); // peak at 4 layers
        let complexity_score = (-(complexity - 0.5).powi(2) / 0.5).exp(); // peak at 0.5M params

        self.config.accuracy_weight * capacity_score * complexity_score
            + self.config.params_weight * complexity
            + self.config.flops_weight * (arch.flops / 1e9)
    }

    /// Generate initial population.
    pub fn initialize(&mut self, pop_size: usize) {
        self.population.clear();
        for i in 0..pop_size
        {
            let mut arch = self.random_architecture(
                self.config.seed.wrapping_add(i as u64).wrapping_mul(0x9E3779B9),
            );
            arch.fitness = self.evaluate(&arch);
            self.population.push(arch);
            self.evaluations += 1;
        }
        self.population.sort_by(|a, b| {
            b.fitness
                .partial_cmp(&a.fitness)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    }

    /// Run evolution for `generations` with `pop_size` individuals.
    pub fn evolve(
        &mut self,
        generations: usize,
        pop_size: usize,
    ) -> Result<Vec<Architecture>, Box<dyn std::error::Error>> {
        self.initialize(pop_size);

        for gen in 0..generations
        {
            // Tournament selection: top half survives
            let survive = pop_size / 2;
            self.population.truncate(survive);

            // Generate offspring via mutation
            for i in 0..survive
            {
                let parent = &self.population[i % survive];
                let mut child = parent.clone();

                // Mutate: change one random layer
                if !child.layers.is_empty()
                {
                    let idx = (gen * pop_size + i) % child.layers.len();
                    let seed = self
                        .config
                        .seed
                        .wrapping_add((gen * pop_size + i) as u64)
                        .wrapping_mul(0x9E3779B9);
                    let rng = seed ^ (seed << 13);
                    let new_dim = self.config.min_hidden
                        + ((rng as usize) % (self.config.max_hidden - self.config.min_hidden + 1));

                    child.layers[idx] = LayerSpec::Linear {
                        out_features: new_dim,
                        activation: ActivationChoice::GELU,
                    };
                    // Recompute cost fields so the mutated child is scored with
                    // its own parameter/FLOP counts, not the parent's.
                    let (params_m, flops) = Self::layer_costs(&child.layers);
                    child.params_m = params_m;
                    child.flops = flops;
                    child.fitness = self.evaluate(&child);
                }

                self.population.push(child);
                self.evaluations += 1;

                if self.evaluations >= self.config.max_models
                {
                    break;
                }
            }

            self.population.sort_by(|a, b| {
                b.fitness
                    .partial_cmp(&a.fitness)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        }

        Ok(self.population.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_random_architecture() {
        let config = NasConfig::default();
        let search = NasSearch::new(config);
        let arch = search.random_architecture(42);

        assert!(!arch.layers.is_empty());
        assert!(arch.layers.len() >= 2);
        assert!(arch.params_m > 0.0);
    }

    #[test]
    fn test_architecture_display() {
        let arch = Architecture {
            layers: vec![
                LayerSpec::Linear {
                    out_features: 128,
                    activation: ActivationChoice::ReLU,
                },
                LayerSpec::Dropout { rate: 0.5 },
            ],
            fitness: 0.85,
            params_m: 0.1,
            flops: 100_000.0,
            accuracy: Some(0.95),
        };

        let s = format!("{}", arch);
        assert!(s.contains("Linear(128,ReLU)"));
        assert!(s.contains("Dropout"));
    }

    #[test]
    fn test_evolution_small() {
        let config = NasConfig {
            max_models: 20,
            ..NasConfig::default()
        };
        let mut search = NasSearch::new(config);
        let population = search.evolve(3, 8).unwrap();

        assert!(!population.is_empty());
        // Best fitness should be non-trivial
        assert!(population[0].fitness > 0.0);
    }

    fn arch(n_layers: usize, params_m: f64, flops: f64) -> Architecture {
        Architecture {
            layers: (0..n_layers)
                .map(|_| LayerSpec::Linear {
                    out_features: 128,
                    activation: ActivationChoice::ReLU,
                })
                .collect(),
            fitness: 0.0,
            params_m,
            flops,
            accuracy: None,
        }
    }

    #[test]
    fn evaluate_rewards_moderate_shape_over_bloated() {
        let search = NasSearch::new(NasConfig::default());
        // ~4 layers and ~0.5M params sits at both Gaussian peaks; the deep, huge
        // model is far from both and carries large linear penalties.
        let moderate = arch(4, 0.5, 1.0e8);
        let bloated = arch(8, 8.0, 5.0e9);
        let sm = search.evaluate(&moderate);
        let sb = search.evaluate(&bloated);
        assert!(sm > sb, "moderate {sm} should beat bloated {sb}");
        // The moderate-shape reward term alone is ~accuracy_weight (both bumps ≈1).
        assert!(
            sm > 0.9,
            "moderate score {sm} should be near the reward peak"
        );
    }

    /// Independent reference cost model (input dim 784, Linear-only), used to
    /// check that every architecture in the evolved population carries cost
    /// fields consistent with its own layers rather than a stale ancestor's.
    fn reference_costs(layers: &[LayerSpec]) -> (f64, f64) {
        let mut params: f64 = 0.0;
        let mut flops: f64 = 0.0;
        let mut prev = 784usize;
        for layer in layers
        {
            if let LayerSpec::Linear { out_features, .. } = layer
            {
                params += (prev * out_features + out_features) as f64;
                flops += 2.0 * prev as f64 * *out_features as f64;
                prev = *out_features;
            }
        }
        (params / 1_000_000.0, flops)
    }

    #[test]
    fn mutation_recomputes_params_and_flops() {
        // With enough generations, mutation is guaranteed to fire, so if cost
        // fields were left stale this population would contain an architecture
        // whose params_m/flops disagree with its own layers.
        let cfg = NasConfig {
            max_models: 200,
            ..NasConfig::default()
        };
        let mut search = NasSearch::new(cfg);
        let population = search.evolve(6, 8).unwrap();

        for a in &population
        {
            let (params_m, flops) = reference_costs(&a.layers);
            assert!(
                (a.params_m - params_m).abs() < 1e-9,
                "stale params_m: stored {} vs recomputed {} for layers {:?}",
                a.params_m,
                params_m,
                a.layers
            );
            assert!(
                (a.flops - flops).abs() < 1e-3,
                "stale flops: stored {} vs recomputed {} for layers {:?}",
                a.flops,
                flops,
                a.layers
            );
            // Fitness must equal a fresh evaluation of the stored fields.
            let refit = search.evaluate(a);
            assert!(
                (a.fitness - refit).abs() < 1e-12,
                "fitness {} inconsistent with re-evaluation {}",
                a.fitness,
                refit
            );
        }
    }

    #[test]
    fn random_architecture_zero_seed_is_not_degenerate() {
        // Before the fix, a zero seed left the xorshift state stuck at 0, so
        // every layer collapsed to the same `Linear(min_hidden, ReLU)`.
        let search = NasSearch::new(NasConfig::default());
        let arch = search.random_architecture(0);
        assert!(!arch.layers.is_empty());
        // At least two distinct layer specs (dim or activation must vary).
        let distinct = arch.layers.iter().any(|l| {
            !matches!(
                l,
                LayerSpec::Linear {
                    out_features,
                    activation: ActivationChoice::ReLU,
                } if *out_features == search.config.min_hidden
            )
        });
        assert!(
            distinct,
            "zero-seed architecture is degenerate (all identical): {:?}",
            arch.layers
        );
    }

    #[test]
    fn seed_arithmetic_does_not_overflow_on_extreme_seed() {
        // `seed` is a public u64 field; a large value used to overflow the
        // `+`/`*` seed derivation and panic in debug builds. It must now run.
        let cfg = NasConfig {
            seed: u64::MAX,
            max_models: 40,
            ..NasConfig::default()
        };
        let mut search = NasSearch::new(cfg);
        let population = search.evolve(3, 8).unwrap();
        assert!(!population.is_empty());
    }

    #[test]
    fn evolution_is_deterministic_and_sorted_best_first() {
        let cfg = || NasConfig {
            max_models: 40,
            ..NasConfig::default()
        };
        let pa = NasSearch::new(cfg()).evolve(4, 8).unwrap();
        let pb = NasSearch::new(cfg()).evolve(4, 8).unwrap();
        // Same seed + config → identical search (deterministic).
        assert_eq!(pa[0].fitness, pb[0].fitness);
        // Population is returned sorted best-first.
        for w in pa.windows(2)
        {
            assert!(w[0].fitness >= w[1].fitness, "population not sorted desc");
        }
    }
}
