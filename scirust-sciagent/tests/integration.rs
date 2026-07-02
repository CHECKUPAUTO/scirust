#[cfg(test)]
mod tests {
use scirust_core::autodiff::reverse::Tape;

use scirust_sciagent::config::SciAgentConfig;
use scirust_sciagent::model::SciAgentModel;
use scirust_sciagent::tokenizer::SciAgentTokenizer;

    #[test]
    fn test_model_creation() {
        let cfg = SciAgentConfig::debug();
        let _model = SciAgentModel::new(&cfg);
        let total = cfg.total_parameters();
        assert!(total > 0, "Model should have parameters");
        println!("Debug model parameters: {total}");
    }

    #[test]
    fn test_forward_shape() {
        let cfg = SciAgentConfig::debug();
        let mut model = SciAgentModel::new(&cfg);
        let tape = Tape::new();
        let input_ids = vec![4usize, 5, 6, 7, 8, 9, 10, 11];
        let seq_len = 8;
        let logits = model.forward(&tape, &input_ids, seq_len);
        let shape = logits.shape();
        assert_eq!(shape.0, seq_len);
        assert_eq!(shape.1, cfg.vocab_size);
    }

    #[test]
    fn test_generates_nonempty() {
        let cfg = SciAgentConfig::debug();
        let mut model = SciAgentModel::new(&cfg);
        let tok = SciAgentTokenizer::new_char_level(&["hello test world"]);
        model.set_tokenizer(tok);
        let prompt = vec![4usize, 5, 6];
        let out = model.generate(&prompt, 10);
        assert!(out.len() > prompt.len(), "Generation should produce tokens");
    }

    #[test]
    fn test_deterministic_forward() {
        let cfg = SciAgentConfig::debug();
        let mut model1 = SciAgentModel::new(&cfg);
        let tape1 = Tape::new();
        let input_ids = vec![4usize, 5, 6, 7, 8, 9, 10, 11];

        let logits1 = model1.forward(&tape1, &input_ids, 8);

        let mut model2 = SciAgentModel::new(&cfg);
        let tape2 = Tape::new();
        let logits2 = model2.forward(&tape2, &input_ids, 8);

        let v1 = tape1.value(logits1.idx());
        let v2 = tape2.value(logits2.idx());
        assert_eq!(v1.data, v2.data, "Deterministic forward failed");
    }

    #[test]
    fn test_deterministic_generation() {
        let cfg = SciAgentConfig::debug();
        let prompt = vec![4usize, 5, 6];

        let mut m1 = SciAgentModel::new(&cfg);
        let out1 = m1.generate(&prompt, 10);

        let mut m2 = SciAgentModel::new(&cfg);
        let out2 = m2.generate(&prompt, 10);

        assert_eq!(out1, out2, "Generation should be deterministic");
    }

    #[test]
    fn test_gradient_flow() {
        let cfg = SciAgentConfig::debug();
        let mut model = SciAgentModel::new(&cfg);
        let tape = Tape::new();
        let input_ids = vec![4usize, 5, 6, 7, 8, 9, 10, 11];

        let logits = model.forward(&tape, &input_ids, 8);
        let loss = logits.sum();
        loss.backward();

        let params = model.parameter_indices();
        assert!(!params.is_empty(), "Should have trainable parameters");
        // Au moins un parametre recoit un gradient non-zero
        let has_some = params.iter().any(|&p| {
            let g = tape.grad(p);
            g.data.iter().map(|x| x.abs()).fold(0.0, f32::max) > 1e-10
        });
        assert!(has_some, "At least some parameters should receive non-zero gradient");
    }
}
