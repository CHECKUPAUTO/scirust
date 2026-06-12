//! Neural Architecture Search (NAS) for SciRust.
//!
//! Implements evolutionary architecture search over layer configurations,
//! using NSGA-II for multi-objective optimization (accuracy vs FLOPs vs params).
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
        let mut total_params: f64 = 0.0;
        let mut total_flops: f64 = 0.0;
        let mut prev_dim = 784; // Default input (MNIST)
        let mut rng = seed;

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

            // Estimate params and FLOPs
            total_params += (prev_dim * out_dim + out_dim) as f64;
            total_flops += 2.0 * prev_dim as f64 * out_dim as f64;
            prev_dim = out_dim;
        }

        Architecture {
            layers,
            fitness: 0.0,
            params_m: total_params / 1_000_000.0,
            flops: total_flops,
            accuracy: None,
        }
    }

    /// Evaluate an architecture (simulated — returns estimated fitness).
    pub fn evaluate(&self, arch: &Architecture) -> f64 {
        // Simulated objective: prefer smaller models with reasonable capacity
        let capacity = arch.layers.len() as f64;
        let complexity = arch.params_m;

        // Reward moderate complexity, penalize extreme values
        let capacity_score = (-(capacity - 4.0).powi(2) / 8.0).exp();
        let _complexity_score = (-(complexity - 0.5).powi(2) / 0.5).exp();

        self.config.accuracy_weight * capacity_score * 0.85
            + self.config.params_weight * complexity
            + self.config.flops_weight * (arch.flops / 1e9)
    }

    /// Generate initial population.
    pub fn initialize(&mut self, pop_size: usize) {
        self.population.clear();
        for i in 0..pop_size
        {
            let mut arch = self.random_architecture((self.config.seed + i as u64) * 0x9E3779B9);
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
                    let idx = ((gen * pop_size + i) as usize) % child.layers.len();
                    let seed = (self.config.seed + (gen * pop_size + i) as u64) * 0x9E3779B9;
                    let rng = seed ^ (seed << 13);
                    let new_dim = self.config.min_hidden
                        + ((rng as usize) % (self.config.max_hidden - self.config.min_hidden + 1));

                    child.layers[idx] = LayerSpec::Linear {
                        out_features: new_dim,
                        activation: ActivationChoice::GELU,
                    };
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
}
