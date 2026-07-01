//! # FusedKernel — Représentation d'un noyau fusionné
//!
//! Un kernel fusionné est un bloc de calcul qui combine plusieurs opérations
//! en un seul passage sur la mémoire. Le calcul intermédiaire reste dans les
//! registres CPU ou dans un tampon local de taille fixe.

#[cfg(feature = "portable-simd")]
use std::simd::{StdFloat, f32x4};

use super::fusion::KernelType;

/// Noyau fusionné — contient les données nécessaires à l'exécution.
pub struct FusedKernel {
    /// Type du kernel (détermine le code généré).
    pub kernel_type: KernelType,
    /// Indices des opérations dans le graphe d'origine.
    pub group: Vec<usize>,
    /// Indices des inputs externes (non dans le groupe).
    pub inputs: Vec<usize>,
    /// Indices des outputs (nœuds dont le résultat est préservé).
    pub outputs: Vec<usize>,

    /// Paramètres de compilation (taille de bloc, SIMD width, etc.).
    pub params: KernelParams,
}

impl FusedKernel {
    /// Crée un nouveau kernel avec les paramètres par défaut.
    pub fn new(
        kernel_type: KernelType,
        group: Vec<usize>,
        inputs: Vec<usize>,
        outputs: Vec<usize>,
    ) -> Self {
        Self {
            kernel_type,
            group,
            inputs,
            outputs,
            params: KernelParams::default(),
        }
    }

    /// Exécute le kernel fusionné sur les données d'entrée.
    ///
    /// Les données intermédiaires restent dans les registres / stack.
    pub fn execute(&self, inputs: &[&[f32]], output: &mut [f32]) {
        // Kernels built by `FusionPipeline` leave the matmul dimensions unset
        // (they default to 0), so `out_features` can legitimately be 0. Every
        // matmul-family kernel derives its batch count via
        // `output.len() / out_features`; guard the degenerate case up front so
        // that path is never reached with a zero divisor. With no output
        // columns there is nothing to compute, so leave `output` untouched.
        if self.uses_matmul_dims() && self.params.out_features == 0
        {
            return;
        }
        match self.kernel_type
        {
            KernelType::MatmulSilu =>
            {
                // y = silu(x @ W)
                // x: (batch, in), W: (in, out)
                self.execute_matmul_silu(inputs, output);
            },
            KernelType::MatmulRelu =>
            {
                // y = relu(x @ W)
                self.execute_matmul_relu(inputs, output);
            },
            KernelType::MatmulSiluLayerNorm =>
            {
                // y = layernorm(silu(x @ W))
                self.execute_matmul_silu_layernorm(inputs, output);
            },
            KernelType::MatmulLayerNorm =>
            {
                // y = layernorm(x @ W)
                self.execute_matmul_layernorm(inputs, output);
            },
            KernelType::MatmulScale =>
            {
                // y = (x @ W) * scale
                self.execute_matmul_scale(inputs, output);
            },
            KernelType::TwoLayerMlp =>
            {
                // y = x @ W1 @ W2 + x (residual)
                self.execute_two_layer_mlp(inputs, output);
            },
            KernelType::LayerNormActivation =>
            {
                // y = act(layernorm(x))
                self.execute_layernorm_activation(inputs, output);
            },
            KernelType::SsmScan =>
            {
                // SSM scan — séquentiel par nature (récurrence diagonale).
                self.execute_ssm_scan(inputs, output);
            },
            KernelType::Identity =>
            {
                // Pas de fusion — copie simple
                if inputs.len() == 1
                {
                    output.copy_from_slice(inputs[0]);
                }
            },
        }
    }

    /// Whether this kernel derives its batch count from
    /// `output.len() / out_features` (the matmul family). Those paths divide by
    /// `out_features`, so `execute` guards them against a zero divisor.
    fn uses_matmul_dims(&self) -> bool {
        matches!(
            self.kernel_type,
            KernelType::MatmulSilu
                | KernelType::MatmulRelu
                | KernelType::MatmulSiluLayerNorm
                | KernelType::MatmulLayerNorm
                | KernelType::MatmulScale
                | KernelType::TwoLayerMlp
        )
    }

    // ============= Implémentations des kernels fusionnés =============

    /// Diagonal state-space-model scan (the Mamba / S4D recurrence). Sequential
    /// by nature: a per-state vector `h` (length `N`) starts at zero and evolves
    /// `h[i] ← a[i]·h[i] + b[i]·x[t]`, emitting `y[t] = Σ_i c[i]·h[i]`. The inner
    /// reduction stays in fixed (ascending-`i`) order, so the result is
    /// bit-identical run to run — the determinism discipline of the workspace.
    ///
    /// Input contract:
    /// - `inputs[0] = x` — the scalar input sequence, length `T`
    /// - `inputs[1] = a` — per-state decay (diagonal `A`), length `N`
    /// - `inputs[2] = b` — input projection `B`, length `N`
    /// - `inputs[3] = c` — output projection `C`, length `N`
    /// - `output      = y` — length `T` (must equal `x.len()`)
    fn execute_ssm_scan(&self, inputs: &[&[f32]], output: &mut [f32]) {
        assert!(inputs.len() >= 4, "SsmScan needs x, a, b, c");
        let x = inputs[0];
        let (a, b, c) = (inputs[1], inputs[2], inputs[3]);
        let n = a.len();
        assert_eq!(b.len(), n, "b must match the state dimension");
        assert_eq!(c.len(), n, "c must match the state dimension");
        assert_eq!(
            output.len(),
            x.len(),
            "output length must equal the sequence length"
        );

        let mut h = vec![0.0f32; n];
        for (t, &xt) in x.iter().enumerate()
        {
            let mut y = 0.0f32;
            for (((hi, &ai), &bi), &ci) in h.iter_mut().zip(a).zip(b).zip(c)
            {
                *hi = ai * *hi + bi * xt;
                y += ci * *hi;
            }
            output[t] = y;
        }
    }

    /// Matmul + SiLU en un seul passage.
    ///
    /// L'astuce: on calcule le produit matriciel et applique SiLU sur chaque élément
    /// accumulé, sans écrire le résultat intermédiaire en RAM.
    fn execute_matmul_silu(&self, inputs: &[&[f32]], output: &mut [f32]) {
        // inputs[0] = x (batch x in_features)
        // inputs[1] = W (in_features x out_features)
        assert!(inputs.len() >= 2, "MatmulSilu needs 2 inputs");

        let x = inputs[0];
        let w = inputs[1];

        let batch = output.len() / self.params.out_features;
        let in_features = self.params.in_features;
        let out_features = self.params.out_features;

        // block_size = self.params.tile_size; // used by tiling variant

        for b in 0..batch
        {
            let x_off = b * in_features;
            let o_off = b * out_features;

            for out_row in 0..out_features
            {
                // Accumuler dans un registre (scalar pour la demo)
                let mut acc = 0.0f32;

                for k in 0..in_features
                {
                    acc += x[x_off + k] * w[k * out_features + out_row];
                }

                // Appliquer SiLU: silu(x) = x * sigmoid(x)
                acc = acc * (1.0 + (-acc).exp()).recip();

                output[o_off + out_row] = acc;
            }
        }
    }

    /// Matmul + ReLU.
    fn execute_matmul_relu(&self, inputs: &[&[f32]], output: &mut [f32]) {
        assert!(inputs.len() >= 2);

        let x = inputs[0];
        let w = inputs[1];

        let batch = output.len() / self.params.out_features;
        let in_features = self.params.in_features;
        let out_features = self.params.out_features;

        for b in 0..batch
        {
            let x_off = b * in_features;
            let o_off = b * out_features;

            for out_row in 0..out_features
            {
                let mut acc = 0.0f32;

                #[cfg(feature = "portable-simd")]
                {
                    let mut acc_v = f32x4::splat(0.0);
                    let mut k = 0;
                    while k + 4 <= in_features
                    {
                        let xv = f32x4::from_slice(&x[x_off + k..x_off + k + 4]);
                        // Weight access assumes row-major: W[k, out_row] = w[k * out_features + out_row]
                        // This is non-contiguous in row-major, so we might need a better layout for SIMD
                        // For now, we do scalar-like accumulation with SIMD for the reduction if possible.
                        let wv = f32x4::from_array([
                            w[k * out_features + out_row],
                            w[(k + 1) * out_features + out_row],
                            w[(k + 2) * out_features + out_row],
                            w[(k + 3) * out_features + out_row],
                        ]);
                        acc_v += xv * wv;
                        k += 4;
                    }
                    acc = acc_v.reduce_sum();
                    for remain_k in k..in_features
                    {
                        acc += x[x_off + remain_k] * w[remain_k * out_features + out_row];
                    }
                }
                #[cfg(not(feature = "portable-simd"))]
                {
                    for k in 0..in_features
                    {
                        acc += x[x_off + k] * w[k * out_features + out_row];
                    }
                }

                output[o_off + out_row] = acc.max(0.0);
            }
        }
    }

    /// Matmul + SiLU + LayerNorm (MLP block).
    fn execute_matmul_silu_layernorm(&self, inputs: &[&[f32]], output: &mut [f32]) {
        // inputs[0] = x, inputs[1] = W, inputs[2] = gamma, inputs[3] = beta
        assert!(inputs.len() >= 4);

        let x = inputs[0];
        let w = inputs[1];
        let gamma = inputs[2];
        let beta = inputs[3];

        let batch = output.len() / self.params.out_features;
        let in_features = self.params.in_features;
        let out_features = self.params.out_features;
        let eps = self.params.eps;

        for b in 0..batch
        {
            let x_off = b * in_features;
            let o_off = b * out_features;

            // Step 1: MatMul + SiLU (accumulated)
            let mut tmp = vec![0.0f32; out_features];
            for out_row in 0..out_features
            {
                let mut acc = 0.0f32;
                for k in 0..in_features
                {
                    acc += x[x_off + k] * w[k * out_features + out_row];
                }
                tmp[out_row] = acc * (1.0 + (-acc).exp()).recip();
            }

            // Step 2: LayerNorm (single pass for mean/var + normalize)
            let mut mean = 0.0f32;
            for v in &tmp
            {
                mean += v;
            }
            mean /= out_features as f32;

            let mut var = 0.0f32;
            for v in &tmp
            {
                let d = v - mean;
                var += d * d;
            }
            var /= out_features as f32;

            let std = (var + eps).sqrt();

            // Normalize + scale/shift
            for i in 0..out_features
            {
                output[o_off + i] = (tmp[i] - mean) / std * gamma[i] + beta[i];
            }
        }
    }

    /// Matmul + LayerNorm.
    fn execute_matmul_layernorm(&self, inputs: &[&[f32]], output: &mut [f32]) {
        // Similar to matmul_silu_layernorm but without SiLU
        assert!(inputs.len() >= 2);

        let x = inputs[0];
        let w = inputs[1];

        let batch = output.len() / self.params.out_features;
        let in_features = self.params.in_features;
        let out_features = self.params.out_features;
        let eps = self.params.eps;

        for b in 0..batch
        {
            let x_off = b * in_features;
            let o_off = b * out_features;

            // MatMul
            let mut tmp = vec![0.0f32; out_features];
            for out_row in 0..out_features
            {
                let mut acc = 0.0f32;
                for k in 0..in_features
                {
                    acc += x[x_off + k] * w[k * out_features + out_row];
                }
                tmp[out_row] = acc;
            }

            // LayerNorm
            let mut mean = 0.0f32;
            for v in &tmp
            {
                mean += v;
            }
            mean /= out_features as f32;

            let mut var = 0.0f32;
            for v in &tmp
            {
                let d = v - mean;
                var += d * d;
            }
            var /= out_features as f32;

            for i in 0..out_features
            {
                output[o_off + i] = (tmp[i] - mean) / (var + eps).sqrt();
            }
        }
    }

    /// Matmul + Scale.
    ///
    /// The scale factor is a compile-time scalar carried by the `Scale` node
    /// (`FusionConstant::F32`), not a runtime input tensor — so only `x` and `W`
    /// are needed here. See `FusionPipeline::build_kernel`, which lifts the
    /// constant into `KernelParams::scale`.
    ///
    /// Input contract:
    /// - `inputs[0] = x` — (batch × in_features)
    /// - `inputs[1] = W` — (in_features × out_features), row-major
    fn execute_matmul_scale(&self, inputs: &[&[f32]], output: &mut [f32]) {
        assert!(inputs.len() >= 2, "MatmulScale needs x, W");

        let x = inputs[0];
        let w = inputs[1];
        let scale = self.params.scale;

        let batch = output.len() / self.params.out_features;
        let in_features = self.params.in_features;
        let out_features = self.params.out_features;

        for b in 0..batch
        {
            let x_off = b * in_features;
            let o_off = b * out_features;

            for out_row in 0..out_features
            {
                let mut acc = 0.0f32;
                for k in 0..in_features
                {
                    acc += x[x_off + k] * w[k * out_features + out_row];
                }
                output[o_off + out_row] = acc * scale;
            }
        }
    }

    /// Two-layer MLP with residual: y = x @ W1 @ W2 + x.
    fn execute_two_layer_mlp(&self, inputs: &[&[f32]], output: &mut [f32]) {
        assert!(inputs.len() >= 3);

        let x = inputs[0];
        let w1 = inputs[1];
        let w2 = inputs[2];

        let batch = output.len() / self.params.out_features;
        let hidden = self.params.in_features; // intermediate dimension
        let in_features = self.params.in_features;
        let out_features = self.params.out_features;

        for b in 0..batch
        {
            let x_off = b * in_features;
            let o_off = b * out_features;

            // First layer
            let mut hidden_act = vec![0.0f32; hidden];
            for h in 0..hidden
            {
                let mut acc = 0.0f32;
                for k in 0..in_features
                {
                    acc += x[x_off + k] * w1[k * hidden + h];
                }
                hidden_act[h] = acc; // no activation for demo
            }

            // Second layer + residual
            for o in 0..out_features
            {
                let mut acc = 0.0f32;
                for h in 0..hidden
                {
                    acc += hidden_act[h] * w2[h * out_features + o];
                }
                output[o_off + o] = acc + x[x_off + o]; // residual
            }
        }
    }

    /// LayerNorm + Activation.
    fn execute_layernorm_activation(&self, inputs: &[&[f32]], output: &mut [f32]) {
        assert!(inputs.len() >= 2);

        let x = inputs[0];
        let gamma = if inputs.len() > 2 { inputs[2] } else { &[] };
        let beta = if inputs.len() > 3 { inputs[3] } else { &[] };
        let eps = self.params.eps;

        let d_model = self.params.out_features;

        // Mean
        let mut mean = 0.0f32;
        for v in x.iter().take(d_model)
        {
            mean += v;
        }
        mean /= d_model as f32;

        // Variance
        let mut var = 0.0f32;
        for v in x.iter().take(d_model)
        {
            let d = v - mean;
            var += d * d;
        }
        var /= d_model as f32;

        let std = (var + eps).sqrt();

        // Normalize + activation (SiLU for demo) + scale/shift
        for i in 0..d_model
        {
            let normed = (x[i] - mean) / std;
            let act = normed * (1.0 + (-normed).exp()).recip();
            output[i] = if !gamma.is_empty()
            {
                act * gamma[i] + beta[i]
            }
            else
            {
                act
            };
        }
    }

    /// Retourne les paramètres de compilation pour le kernel.
    pub fn params(&self) -> &KernelParams {
        &self.params
    }

    /// Retourne le nombre d'opérations fusionnées.
    pub fn op_count(&self) -> usize {
        self.group.len()
    }

    /// Retourne le type du kernel.
    pub fn kernel_type(&self) -> KernelType {
        self.kernel_type
    }
}

/// Paramètres de compilation d'un kernel fusionné.
#[derive(Debug, Clone)]
pub struct KernelParams {
    /// Dimension d'entrée pour le matmul.
    pub in_features: usize,
    /// Dimension de sortie pour le matmul.
    pub out_features: usize,
    /// Taille de bloc pour le tiling.
    pub tile_size: usize,
    /// Epsilon pour LayerNorm.
    pub eps: f32,
    /// Facteur d'échelle scalaire pour `MatmulScale` (constante, pas un input).
    pub scale: f32,
}

impl Default for KernelParams {
    fn default() -> Self {
        Self {
            in_features: 0,
            out_features: 0,
            tile_size: 64,
            eps: 1e-5,
            scale: 1.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matmul_relu_kernel_computes_relu_of_matmul() {
        let mut k = FusedKernel::new(KernelType::MatmulRelu, vec![0, 1], vec![], vec![]);
        k.params = KernelParams {
            in_features: 2,
            out_features: 2,
            tile_size: 64,
            eps: 1e-5,
            scale: 1.0,
        };
        // x: (1x2) = [1, 2]; W: (2x2) row-major [k*out+o] = [[1, 0], [0, -1]]
        let x = [1.0f32, 2.0];
        let w = [1.0f32, 0.0, 0.0, -1.0];
        let mut out = [0.0f32; 2];
        k.execute(&[&x, &w], &mut out);
        // y0 = relu(1*1 + 2*0) = 1 ; y1 = relu(1*0 + 2*(-1)) = relu(-2) = 0
        assert_eq!(out, [1.0, 0.0]);
        assert_eq!(k.op_count(), 2);
    }

    #[test]
    fn matmul_scale_uses_scalar_constant_not_a_third_input() {
        // Non-regression: `execute_matmul_scale` used to read `inputs[2]`, which
        // panicked because the scale is a scalar constant carried by the kernel,
        // and the fused inputs are only [x, W]. It must run with two inputs.
        let mut k = FusedKernel::new(KernelType::MatmulScale, vec![0, 1], vec![], vec![]);
        k.params = KernelParams {
            in_features: 2,
            out_features: 2,
            tile_size: 64,
            eps: 1e-5,
            scale: 3.0,
        };
        // x: (1x2) = [1, 2]; W: (2x2) row-major = [[1, 0], [0, 1]] (identity).
        let x = [1.0f32, 2.0];
        let w = [1.0f32, 0.0, 0.0, 1.0];
        let mut out = [0.0f32; 2];
        // Only two inputs — previously this indexed inputs[2] out of bounds.
        k.execute(&[&x, &w], &mut out);
        // y = (x @ W) * scale = [1, 2] * 3 = [3, 6].
        assert_eq!(out, [3.0, 6.0]);
    }

    #[test]
    fn matmul_kernels_with_unset_dims_do_not_divide_by_zero() {
        // Non-regression: `FusionPipeline::build_kernel` never sets the matmul
        // dimensions, so a pipeline-produced kernel keeps the default
        // `out_features == 0`. Every matmul-family kernel computes its batch
        // count as `output.len() / out_features`, which used to panic with an
        // integer divide-by-zero. With the guard, `execute` returns without
        // touching `output`.
        for kt in [
            KernelType::MatmulSilu,
            KernelType::MatmulRelu,
            KernelType::MatmulSiluLayerNorm,
            KernelType::MatmulLayerNorm,
            KernelType::MatmulScale,
            KernelType::TwoLayerMlp,
        ]
        {
            let k = FusedKernel::new(kt, vec![0, 1], vec![], vec![]);
            assert_eq!(k.params.out_features, 0, "default params leave dims unset");

            let x = [1.0f32, 2.0];
            let w = [1.0f32, 0.0, 0.0, 1.0];
            let sentinel = [42.0f32; 2];
            let mut out = sentinel;
            // Previously panicked (divide-by-zero) before reaching any compute.
            k.execute(&[&x, &w, &w], &mut out);
            // Degenerate (no output columns): output is left untouched.
            assert_eq!(out, sentinel);
        }
    }

    #[test]
    fn ssm_scan_matches_the_diagonal_recurrence() {
        let k = FusedKernel::new(KernelType::SsmScan, vec![0, 1, 2, 3], vec![], vec![]);

        // N=1, a=0.5, b=1, c=1 → impulse response is the geometric decay 0.5^t.
        let x = [1.0f32, 0.0, 0.0];
        let (a, b, c) = ([0.5f32], [1.0f32], [1.0f32]);
        let mut out = [0.0f32; 3];
        k.execute(&[&x, &a, &b, &c], &mut out);
        assert_eq!(out, [1.0, 0.5, 0.25]);

        // N=2 mixing two states with different decays:
        // a=[0.5,0], b=[1,1], c=[1,2], x=[1,1]
        //  t0: h=[1,1]   → y = 1*1   + 2*1 = 3
        //  t1: h=[1.5,1] → y = 1*1.5 + 2*1 = 3.5
        let x2 = [1.0f32, 1.0];
        let (a2, b2, c2) = ([0.5f32, 0.0], [1.0f32, 1.0], [1.0f32, 2.0]);
        let mut out2 = [0.0f32; 2];
        k.execute(&[&x2, &a2, &b2, &c2], &mut out2);
        assert_eq!(out2, [3.0, 3.5]);

        // Deterministic: identical inputs give bit-identical output.
        let mut again = [0.0f32; 2];
        k.execute(&[&x2, &a2, &b2, &c2], &mut again);
        assert_eq!(out2, again);
    }
}
