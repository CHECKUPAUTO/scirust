//! # Opérations Fusionnées — Pilier 1 + Pilier 4
//!
//! Implémente les kernels fusionnés qui combinent plusieurs opérations
//! en un seul passage mémoire, pour éliminer les writes/reads intermédiaires
//! en RAM.
//!
//! ## Kernels fournis
//!
//! - `matmul_silu` — Linear + SiLU en un passage
//! - `matmul_gelu` — Linear + GELU en un passage
//! - `matmul_layernorm` — Linear + LayerNorm en un passage
//! - `matmul_silu_layernorm` — Linear + SiLU + LayerNorm (MLP block)
//! - `matmul_relu` — Linear + ReLU en un passage
//! - `matmul_scale` — Linear × scale en un passage
//!
//! ## Principe d'implémentation
//!
//! Chaque kernel accumule les produits matriciels dans des registres
//! (ou un tampon local de taille fixe) et n'écrit le résultat final qu'en RAM.
//! Les activations et normalisations sont appliquées sur les accumulateurs.

pub fn matmul_silu(
    x: &[f32],
    w: &[f32],
    bias: &[f32],
    batch: usize,
    in_features: usize,
    out_features: usize,
) -> Vec<f32> {
    let mut output = vec![0.0f32; batch * out_features];

    for b in 0..batch {
        let x_off = b * in_features;
        let o_off = b * out_features;

        for o in 0..out_features {
            let mut acc = bias[o];
            for k in 0..in_features {
                acc += x[x_off + k] * w[k * out_features + o];
            }
            acc *= 1.0 / (1.0 + (-acc).exp());
            output[o_off + o] = acc;
        }
    }

    output
}

pub fn matmul_relu(
    x: &[f32],
    w: &[f32],
    bias: &[f32],
    batch: usize,
    in_features: usize,
    out_features: usize,
) -> Vec<f32> {
    let mut output = vec![0.0f32; batch * out_features];

    for b in 0..batch {
        let x_off = b * in_features;
        let o_off = b * out_features;

        for o in 0..out_features {
            let mut acc = bias[o];
            for k in 0..in_features {
                acc += x[x_off + k] * w[k * out_features + o];
            }
            output[o_off + o] = acc.max(0.0);
        }
    }

    output
}

pub fn matmul_gelu(
    x: &[f32],
    w: &[f32],
    bias: &[f32],
    batch: usize,
    in_features: usize,
    out_features: usize,
) -> Vec<f32> {
    let mut output = vec![0.0f32; batch * out_features];
    let sqrt_2_pi = 0.7978845608028654;

    for b in 0..batch {
        let x_off = b * in_features;
        let o_off = b * out_features;

        for o in 0..out_features {
            let mut acc = bias[o];
            for k in 0..in_features {
                acc += x[x_off + k] * w[k * out_features + o];
            }
            let cubed = acc * acc * acc;
            let tanh_arg = sqrt_2_pi * (acc + 0.044715 * cubed);
            acc = 0.5 * acc * (1.0 + tanh_arg.tanh());
            output[o_off + o] = acc;
        }
    }

    output
}

pub fn matmul_layernorm(
    x: &[f32],
    w: &[f32],
    bias: &[f32],
    gamma: &[f32],
    beta: &[f32],
    eps: f32,
    batch: usize,
    in_features: usize,
    out_features: usize,
) -> Vec<f32> {
    let mut output = vec![0.0f32; batch * out_features];

    for b in 0..batch {
        let x_off = b * in_features;
        let o_off = b * out_features;

        let mut intermediate = vec![0.0f32; out_features];
        for o in 0..out_features {
            let mut acc = bias[o];
            for k in 0..in_features {
                acc += x[x_off + k] * w[k * out_features + o];
            }
            intermediate[o] = acc;
        }

        let mut mean = 0.0f32;
        for &v in &intermediate {
            mean += v;
        }
        mean /= out_features as f32;

        let mut var = 0.0f32;
        for &v in &intermediate {
            let d = v - mean;
            var += d * d;
        }
        var /= out_features as f32;
        let std = (var + eps).sqrt();

        for i in 0..out_features {
            output[o_off + i] = (intermediate[i] - mean) / std * gamma[i] + beta[i];
        }
    }

    output
}

pub fn matmul_silu_layernorm(
    x: &[f32],
    w1: &[f32],
    w2: &[f32],
    b1: &[f32],
    b2: &[f32],
    gamma: &[f32],
    beta: &[f32],
    eps: f32,
    batch: usize,
    in_features: usize,
    hidden_features: usize,
    out_features: usize,
) -> Vec<f32> {
    let mut output = vec![0.0f32; batch * out_features];

    for b in 0..batch {
        let x_off = b * in_features;
        let o_off = b * out_features;

        let mut hidden = vec![0.0f32; hidden_features];
        for h in 0..hidden_features {
            let mut acc = b1[h];
            for k in 0..in_features {
                acc += x[x_off + k] * w1[k * hidden_features + h];
            }
            acc *= 1.0 / (1.0 + (-acc).exp());
            hidden[h] = acc;
        }

        let mut intermediate = vec![0.0f32; out_features];
        for o in 0..out_features {
            let mut acc = b2[o];
            for h in 0..hidden_features {
                acc += hidden[h] * w2[h * out_features + o];
            }
            intermediate[o] = acc;
        }

        let mut mean = 0.0f32;
        for &v in &intermediate {
            mean += v;
        }
        mean /= out_features as f32;

        let mut var = 0.0f32;
        for &v in &intermediate {
            let d = v - mean;
            var += d * d;
        }
        var /= out_features as f32;
        let std = (var + eps).sqrt();
        for i in 0..out_features {
            output[o_off + i] = (intermediate[i] - mean) / std * gamma[i] + beta[i];
        }
    }

    output
}

pub fn matmul_scale(
    x: &[f32],
    w: &[f32],
    scale: &[f32],
    batch: usize,
    in_features: usize,
    out_features: usize,
) -> Vec<f32> {
    let mut output = vec![0.0f32; batch * out_features];

    for b in 0..batch {
        let x_off = b * in_features;
        let o_off = b * out_features;

        for o in 0..out_features {
            let mut acc = 0.0f32;
            for k in 0..in_features {
                acc += x[x_off + k] * w[k * out_features + o];
            }
            output[o_off + o] = acc * scale[o];
        }
    }

    output
}

/// Type de kernel fusionné.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FusedKernelOp {
    MatmulSilu,
    MatmulGelu,
    MatmulRelu,
    MatmulLayerNorm,
    MatmulSiluLayerNorm,
    MatmulScale,
}
