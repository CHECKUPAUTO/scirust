//! # FusedKernel — Représentation d'un noyau fusionné
//!
//! Un kernel fusionné est un bloc de calcul qui combine plusieurs opérations
//! en un seul passage sur la mémoire. Le calcul intermédiaire reste dans les
//! registres CPU ou dans un tampon local de taille fixe.

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
    pub fn new(kernel_type: KernelType, group: Vec<usize>, inputs: Vec<usize>, outputs: Vec<usize>) -> Self {
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
        match self.kernel_type {
            KernelType::MatmulSilu => {
                // y = silu(x @ W)
                // x: (batch, in), W: (in, out)
                self.execute_matmul_silu(inputs, output);
            }
            KernelType::MatmulRelu => {
                // y = relu(x @ W)
                self.execute_matmul_relu(inputs, output);
            }
            KernelType::MatmulSiluLayerNorm => {
                // y = layernorm(silu(x @ W))
                self.execute_matmul_silu_layernorm(inputs, output);
            }
            KernelType::MatmulLayerNorm => {
                // y = layernorm(x @ W)
                self.execute_matmul_layernorm(inputs, output);
            }
            KernelType::MatmulScale => {
                // y = (x @ W) * scale
                self.execute_matmul_scale(inputs, output);
            }
            KernelType::TwoLayerMlp => {
                // y = x @ W1 @ W2 + x (residual)
                self.execute_two_layer_mlp(inputs, output);
            }
            KernelType::LayerNormActivation => {
                // y = act(layernorm(x))
                self.execute_layernorm_activation(inputs, output);
            }
            KernelType::SsmScan => {
                // SSM scan — séquentiel, pas de fusion possible
                unimplemented!("SsmScan kernel not yet implemented")
            }
            KernelType::Identity => {
                // Pas de fusion — copie simple
                if inputs.len() == 1 {
                    output.copy_from_slice(inputs[0]);
                }
            }
        }
    }

    // ============= Implémentations des kernels fusionnés =============

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

        for b in 0..batch {
            let x_off = b * in_features;
            let o_off = b * out_features;

            for out_row in 0..out_features {
                // Accumuler dans un registre (scalar pour la demo)
                let mut acc = 0.0f32;

                for k in 0..in_features {
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

        for b in 0..batch {
            let x_off = b * in_features;
            let o_off = b * out_features;

            for out_row in 0..out_features {
                let mut acc = 0.0f32;
                for k in 0..in_features {
                    acc += x[x_off + k] * w[k * out_features + out_row];
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

        for b in 0..batch {
            let x_off = b * in_features;
            let o_off = b * out_features;

            // Step 1: MatMul + SiLU (accumulated)
            let mut tmp = vec![0.0f32; out_features];
            for out_row in 0..out_features {
                let mut acc = 0.0f32;
                for k in 0..in_features {
                    acc += x[x_off + k] * w[k * out_features + out_row];
                }
                tmp[out_row] = acc * (1.0 + (-acc).exp()).recip();
            }

            // Step 2: LayerNorm (single pass for mean/var + normalize)
            let mut mean = 0.0f32;
            for v in &tmp {
                mean += v;
            }
            mean /= out_features as f32;

            let mut var = 0.0f32;
            for v in &tmp {
                let d = v - mean;
                var += d * d;
            }
            var /= out_features as f32;

            let std = (var + eps).sqrt();

            // Normalize + scale/shift
            for i in 0..out_features {
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

        for b in 0..batch {
            let x_off = b * in_features;
            let o_off = b * out_features;

            // MatMul
            let mut tmp = vec![0.0f32; out_features];
            for out_row in 0..out_features {
                let mut acc = 0.0f32;
                for k in 0..in_features {
                    acc += x[x_off + k] * w[k * out_features + out_row];
                }
                tmp[out_row] = acc;
            }

            // LayerNorm
            let mut mean = 0.0f32;
            for v in &tmp {
                mean += v;
            }
            mean /= out_features as f32;

            let mut var = 0.0f32;
            for v in &tmp {
                let d = v - mean;
                var += d * d;
            }
            var /= out_features as f32;

            for i in 0..out_features {
                output[o_off + i] = (tmp[i] - mean) / (var + eps).sqrt();
            }
        }
    }

    /// Matmul + Scale.
    fn execute_matmul_scale(&self, inputs: &[&[f32]], output: &mut [f32]) {
        assert!(inputs.len() >= 2);

        let x = inputs[0];
        let w = inputs[1];
        let scale = inputs[2];

        let batch = output.len() / self.params.out_features;
        let in_features = self.params.in_features;
        let out_features = self.params.out_features;

        for b in 0..batch {
            let x_off = b * in_features;
            let o_off = b * out_features;

            for out_row in 0..out_features {
                let mut acc = 0.0f32;
                for k in 0..in_features {
                    acc += x[x_off + k] * w[k * out_features + out_row];
                }
                output[o_off + out_row] = acc * scale[out_row];
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

        for b in 0..batch {
            let x_off = b * in_features;
            let o_off = b * out_features;

            // First layer
            let mut hidden_act = vec![0.0f32; hidden];
            for h in 0..hidden {
                let mut acc = 0.0f32;
                for k in 0..in_features {
                    acc += x[x_off + k] * w1[k * hidden + h];
                }
                hidden_act[h] = acc; // no activation for demo
            }

            // Second layer + residual
            for o in 0..out_features {
                let mut acc = 0.0f32;
                for h in 0..hidden {
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
        for v in x.iter().take(d_model) {
            mean += v;
        }
        mean /= d_model as f32;

        // Variance
        let mut var = 0.0f32;
        for v in x.iter().take(d_model) {
            let d = v - mean;
            var += d * d;
        }
        var /= d_model as f32;

        let std = (var + eps).sqrt();

        // Normalize + activation (SiLU for demo) + scale/shift
        for i in 0..d_model {
            let normed = (x[i] - mean) / std;
            let act = normed * (1.0 + (-normed).exp()).recip();
            output[i] = if !gamma.is_empty() {
                act * gamma[i] + beta[i]
            } else {
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
}

impl Default for KernelParams {
    fn default() -> Self {
        Self {
            in_features: 0,
            out_features: 0,
            tile_size: 64,
            eps: 1e-5,
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
}
