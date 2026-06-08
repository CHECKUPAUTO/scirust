// scirust-core/src/tests/test_aot.rs
#[cfg(test)]
mod tests {
    use crate::aot::{generate_static_pipeline, LayerSpec};

    #[test]
    fn test_aot_generation_basic() {
        let layers = vec![
            LayerSpec::Linear { in_features: 2, out_features: 3 },
            LayerSpec::ReLU,
        ];
        let weights = vec![0.1f32, 0.2, 0.3, 0.4, 0.5, 0.6];
        let mut bytes = Vec::new();
        for w in weights {
            bytes.extend_from_slice(&w.to_le_bytes());
        }

        let generated_code = generate_static_pipeline(&layers, &bytes);
        println!("{}", generated_code);
        assert!(generated_code.contains("pub struct StaticModel"));
        assert!(generated_code.contains("weight_0: [[f32; 3]; 2]"));
        assert!(generated_code.contains("0.10000000"));
        assert!(generated_code.contains("0.60000002") || generated_code.contains("0.60000000"));
    }
}
