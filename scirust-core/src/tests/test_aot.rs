// scirust-core/src/tests/test_aot.rs
#[cfg(test)]
mod tests {
    use crate::aot::{LayerSpec, generate_static_pipeline};

    /// Build the little-endian weight+bias byte stream the AOT compiler expects:
    /// for each Linear layer, `in*out` weight floats followed by `out` bias
    /// floats.
    fn to_bytes(vals: &[f32]) -> Vec<u8> {
        let mut bytes = Vec::new();
        for &v in vals
        {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        bytes
    }

    #[test]
    fn test_aot_generation_basic() {
        let layers = vec![
            LayerSpec::Linear {
                in_features: 2,
                out_features: 3,
            },
            LayerSpec::ReLU,
        ];
        // 2×3 weights then 3 bias values.
        let bytes = to_bytes(&[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]);

        let generated_code = generate_static_pipeline(&layers, &bytes);
        println!("{}", generated_code);
        assert!(generated_code.contains("pub struct StaticModel"));
        assert!(generated_code.contains("weight_0: [[f32; 3]; 2]"));
        assert!(generated_code.contains("0.10000000"));
        assert!(generated_code.contains("0.60000002") || generated_code.contains("0.60000000"));
    }

    /// Regression: a Linear layer is `y = x·W + b`. The generated code must
    /// declare a bias array, initialise it from the byte stream, and seed the
    /// dot-product accumulator with it — previously the bias was dropped
    /// entirely, so every compiled Linear layer computed `x·W` and was wrong.
    #[test]
    fn test_aot_emits_and_applies_bias() {
        let layers = vec![LayerSpec::Linear {
            in_features: 2,
            out_features: 3,
        }];
        // 2×3 weights, then biases 1.5, 2.5, 3.5.
        let bytes = to_bytes(&[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 1.5, 2.5, 3.5]);
        let code = generate_static_pipeline(&layers, &bytes);

        // The bias is a field, is initialised with the supplied values, and the
        // forward accumulator starts from it (not from 0.0).
        assert!(
            code.contains("bias_0: [f32; 3]"),
            "missing bias field:\n{code}"
        );
        assert!(
            code.contains("1.50000000"),
            "bias value not emitted:\n{code}"
        );
        assert!(
            code.contains("3.50000000"),
            "bias value not emitted:\n{code}"
        );
        assert!(
            code.contains("let mut acc = self.bias_0[o];"),
            "forward does not seed accumulator with bias:\n{code}"
        );
        // And it must no longer start the accumulator at zero (the old bug).
        assert!(
            !code.contains("let mut acc = 0.0f32;"),
            "forward still drops the bias (acc starts at 0):\n{code}"
        );
    }
}
