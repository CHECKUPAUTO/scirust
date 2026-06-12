//! SOM inference and oracle-checked evaluation.
//!
//! Every prediction path here is validated against the deterministic
//! ownership oracle: [`evaluate`] measures per-token agreement on labelled
//! samples, and [`predict_program`] runs the model *and* the oracle on the
//! same program and reports their agreement — inference never ships
//! without its ground truth attached.

use scirust_core::autodiff::reverse::Tape;
use scirust_som_dataset::TrainingSample;
use scirust_som_model::SomModel;
use scirust_som_pcg::ast::SomAst;
use scirust_som_symbolic::{Analysis, OwnershipOracle};
use scirust_som_tokenizer::{SomToken, SomVocab};

/// Aggregated per-token agreement between model and ground truth.
#[derive(Debug, Clone, PartialEq)]
pub struct EvalReport {
    pub ownership_accuracy: f32,
    pub borrow_accuracy: f32,
    pub invalid_accuracy: f32,
    pub n_tokens: usize,
}

fn argmax_rows(data: &[f32], rows: usize, cols: usize) -> Vec<usize> {
    (0..rows)
        .map(|r| {
            let row = &data[r * cols..(r + 1) * cols];
            row.iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).expect("finite logits"))
                .map(|(i, _)| i)
                .expect("non-empty row")
        })
        .collect()
}

/// Predicted class ids for one token sequence.
fn predict_ids(model: &mut SomModel, token_ids: &[usize]) -> (Vec<usize>, Vec<usize>, Vec<f32>) {
    let tape = Tape::new();
    let logits = model.forward(&tape, token_ids);
    let seq = token_ids.len();

    let own = tape.value(logits.ownership.idx());
    let bor = tape.value(logits.borrow.idx());
    let inv = tape.value(logits.invalid.sigmoid().idx());

    (
        argmax_rows(&own.data, seq, own.cols),
        argmax_rows(&bor.data, seq, bor.cols),
        inv.data[..seq].to_vec(),
    )
}

/// Measure model agreement with oracle labels over `samples`.
pub fn evaluate(model: &mut SomModel, samples: &[TrainingSample]) -> EvalReport {
    let mut own_hits = 0usize;
    let mut bor_hits = 0usize;
    let mut inv_hits = 0usize;
    let mut n = 0usize;
    for sample in samples
    {
        let (own, bor, inv) = predict_ids(model, &sample.token_ids);
        for i in 0..sample.token_ids.len()
        {
            if own[i] == sample.ownership[i]
            {
                own_hits += 1;
            }
            if bor[i] == sample.borrow[i]
            {
                bor_hits += 1;
            }
            let predicted_fault = inv[i] > 0.5;
            let actual_fault = sample.invalid[i] > 0.5;
            if predicted_fault == actual_fault
            {
                inv_hits += 1;
            }
            n += 1;
        }
    }
    assert!(n > 0, "no tokens to evaluate");
    EvalReport {
        ownership_accuracy: own_hits as f32 / n as f32,
        borrow_accuracy: bor_hits as f32 / n as f32,
        invalid_accuracy: inv_hits as f32 / n as f32,
        n_tokens: n,
    }
}

/// Majority-class baseline for ownership over `samples` (what a constant
/// predictor would score). The model must beat this to be useful.
pub fn ownership_majority_baseline(samples: &[TrainingSample]) -> f32 {
    let mut counts = std::collections::BTreeMap::new();
    let mut n = 0usize;
    for sample in samples
    {
        for &c in &sample.ownership
        {
            *counts.entry(c).or_insert(0usize) += 1;
            n += 1;
        }
    }
    let max = counts.values().copied().max().unwrap_or(0);
    if n == 0 { 0.0 } else { max as f32 / n as f32 }
}

/// One token of an oracle-checked prediction.
#[derive(Debug, Clone)]
pub struct TokenPrediction {
    pub token: SomToken,
    pub ownership: usize,
    pub borrow: usize,
    pub invalid_prob: f32,
}

/// Inference output for a whole program, with the oracle's analysis and
/// the measured agreement attached.
#[derive(Debug, Clone)]
pub struct InferenceReport {
    pub predictions: Vec<TokenPrediction>,
    pub oracle: Analysis,
    pub ownership_agreement: f32,
}

/// Run the model on a program and check it against the oracle.
pub fn predict_program(model: &mut SomModel, ast: &SomAst) -> InferenceReport {
    let oracle = OwnershipOracle::new().analyze(ast);
    let token_ids = SomVocab::encode(&oracle.tokens);
    let (own, bor, inv) = predict_ids(model, &token_ids);

    let hits = oracle
        .labels
        .iter()
        .zip(&own)
        .filter(|(label, &p)| label.ownership == p)
        .count();
    let agreement = hits as f32 / oracle.labels.len().max(1) as f32;

    let predictions = oracle
        .tokens
        .iter()
        .cloned()
        .zip(own.iter().zip(bor.iter().zip(inv.iter())))
        .map(
            |(token, (&ownership, (&borrow, &invalid_prob)))| TokenPrediction {
                token,
                ownership,
                borrow,
                invalid_prob,
            },
        )
        .collect();

    InferenceReport {
        predictions,
        oracle,
        ownership_agreement: agreement,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirust_som_dataset::build_training_set;
    use scirust_som_model::SomModelConfig;
    use scirust_som_trainer::{TrainerConfig, train};

    fn tiny_model(seed: u64) -> SomModel {
        SomModel::new(SomModelConfig {
            vocab_size: SomVocab::vocab_size(),
            d_model: 16,
            n_heads: 2,
            n_layers: 1,
            d_ff: 32,
            max_seq_len: 64,
            seed,
            ..SomModelConfig::default()
        })
    }

    #[test]
    fn evaluation_is_deterministic() {
        let samples = build_training_set(11, 8, 64);
        let r1 = evaluate(&mut tiny_model(5), &samples);
        let r2 = evaluate(&mut tiny_model(5), &samples);
        assert_eq!(r1, r2);
    }

    #[test]
    fn trained_model_beats_ownership_majority_baseline() {
        let train_set = build_training_set(42, 48, 64);
        let eval_set = build_training_set(1042, 16, 64);
        let baseline = ownership_majority_baseline(&eval_set);

        let mut model = tiny_model(42);
        train(
            &mut model,
            &train_set,
            &TrainerConfig {
                epochs: 4,
                learning_rate: 0.01,
            },
        );
        let report = evaluate(&mut model, &eval_set);

        assert!(
            report.ownership_accuracy > baseline,
            "trained accuracy {} must beat majority baseline {}",
            report.ownership_accuracy,
            baseline
        );
        assert!(report.n_tokens > 0);
    }

    /// Measurement probe, not a gate: prints honest metrics for the
    /// README. Run with `cargo test -p scirust-som-inference -- --ignored
    /// --nocapture`.
    #[test]
    #[ignore = "metrics probe — run explicitly to measure"]
    fn metrics_probe() {
        let train_set = build_training_set(42, 200, 64);
        let eval_set = build_training_set(9042, 50, 64);
        let baseline = ownership_majority_baseline(&eval_set);
        let mut model = SomModel::new(SomModelConfig {
            vocab_size: SomVocab::vocab_size(),
            d_model: 32,
            n_heads: 2,
            n_layers: 2,
            d_ff: 64,
            max_seq_len: 64,
            seed: 42,
            ..SomModelConfig::default()
        });
        let report = train(
            &mut model,
            &train_set,
            &TrainerConfig {
                epochs: 8,
                learning_rate: 0.005,
            },
        );
        let eval = evaluate(&mut model, &eval_set);
        println!("losses: {:?}", report.epoch_losses);
        println!("ownership baseline (majority): {baseline:.4}");
        println!("ownership accuracy: {:.4}", eval.ownership_accuracy);
        println!("borrow accuracy:    {:.4}", eval.borrow_accuracy);
        println!("invalid accuracy:   {:.4}", eval.invalid_accuracy);
        println!("tokens evaluated:   {}", eval.n_tokens);
    }

    #[test]
    fn predict_program_reports_oracle_agreement() {
        use scirust_som_pcg::ast::{Expression, Function, Literal, Statement, Type};
        let ast = SomAst::Program(vec![Function {
            name: "main".to_string(),
            params: vec![],
            body: vec![
                Statement::VarDecl {
                    name: "x".to_string(),
                    ty: Type::Int,
                    init: Some(Expression::Literal(Literal::Int(1))),
                },
                Statement::VarDecl {
                    name: "y".to_string(),
                    ty: Type::Int,
                    init: Some(Expression::Variable("x".to_string())),
                },
            ],
        }]);
        let mut model = tiny_model(3);
        let report = predict_program(&mut model, &ast);
        assert_eq!(report.predictions.len(), report.oracle.tokens.len());
        assert!((0.0..=1.0).contains(&report.ownership_agreement));
    }
}
