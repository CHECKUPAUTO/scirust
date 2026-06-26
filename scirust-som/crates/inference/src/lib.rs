//! SOM inference and oracle-checked evaluation.
//!
//! Every prediction path here is validated against the deterministic
//! ownership oracle: [`evaluate`] measures per-token agreement on labelled
//! samples, and [`predict_program`] runs the model *and* the oracle on the
//! same program and reports their agreement — inference never ships
//! without its ground truth attached.

use scirust_core::autodiff::reverse::Tape;
use scirust_som_dataset::TrainingSample;
use scirust_som_model::{SomLabels, SomModel};
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

/// One token sequence run through the model and decoded.
///
/// `labels` is exactly what the model publishes via [`SomLogits::decode`];
/// `invalid_prob[i]` is `sigmoid(invalid_logit_i)`, the fault *probability*
/// that the boolean `labels.invalid[i]` thresholds at 0.5. Both are produced
/// from a single forward pass and have length `token_ids.len()`.
struct Prediction {
    labels: SomLabels,
    invalid_prob: Vec<f32>,
}

/// Run the model on one token sequence and decode it into the *exact* labels
/// the model publishes via [`SomLogits::decode`], plus the per-token fault
/// probability.
///
/// Decoding is delegated to the model so inference can never disagree with the
/// model's own argmax/threshold contract: same first-maximum tie-break on the
/// class heads, same `logit > 0` fault decision (equivalently `prob > 0.5`).
/// Re-deriving the argmax here independently is what previously let the two
/// drift apart on ties (`max_by` keeps the *last* maximum, the model keeps the
/// *first*).
///
/// An empty `token_ids` yields empty vectors *without* invoking the forward
/// pass (which requires `seq_len >= 1`): a sequence with no tokens has no
/// predictions, so callers handle zero-token programs uniformly instead of
/// panicking.
fn predict_labels(model: &mut SomModel, token_ids: &[usize]) -> Prediction {
    if token_ids.is_empty()
    {
        return Prediction {
            labels: SomLabels {
                ownership: Vec::new(),
                borrow: Vec::new(),
                invalid: Vec::new(),
            },
            invalid_prob: Vec::new(),
        };
    }
    let tape = Tape::new();
    let logits = model.forward(&tape, token_ids);
    let labels = logits.decode();
    // Fault probability for the report: sigmoid of the single invalid logit
    // per token. `decode` already thresholds this at 0 (logit) / 0.5 (prob),
    // so `labels.invalid[i] == (invalid_prob[i] > 0.5)` by construction.
    let invalid_prob = tape.value(logits.invalid.idx()).sigmoid().data;
    debug_assert_eq!(labels.ownership.len(), token_ids.len());
    debug_assert_eq!(labels.borrow.len(), token_ids.len());
    debug_assert_eq!(labels.invalid.len(), token_ids.len());
    debug_assert_eq!(invalid_prob.len(), token_ids.len());
    Prediction {
        labels,
        invalid_prob,
    }
}

/// Measure model agreement with oracle labels over `samples`.
///
/// Agreement is per token over the *whole* oracle-labelled stream — every
/// token the oracle emits, special tokens (`FnDecl`, `Drop`, scope markers)
/// included, since the oracle assigns each a real ground-truth label and the
/// model is trained on them. The denominator is the exact count of those
/// tokens; the samples produced by `build_training_set` are unpadded, so no
/// pad position is ever counted. Each accuracy is therefore
/// `matching_tokens / total_tokens`.
///
/// # Panics
/// Panics if `samples` contributes zero tokens in total (nothing to measure).
/// Individual empty samples contribute nothing and are skipped, not faulted.
pub fn evaluate(model: &mut SomModel, samples: &[TrainingSample]) -> EvalReport {
    let mut own_hits = 0usize;
    let mut bor_hits = 0usize;
    let mut inv_hits = 0usize;
    let mut n = 0usize;
    for sample in samples
    {
        debug_assert_eq!(sample.token_ids.len(), sample.ownership.len());
        debug_assert_eq!(sample.token_ids.len(), sample.borrow.len());
        debug_assert_eq!(sample.token_ids.len(), sample.invalid.len());
        let pred = predict_labels(model, &sample.token_ids).labels;
        for i in 0..sample.token_ids.len()
        {
            if pred.ownership[i] == sample.ownership[i]
            {
                own_hits += 1;
            }
            if pred.borrow[i] == sample.borrow[i]
            {
                bor_hits += 1;
            }
            // The oracle's `invalid` channel is 1.0 at a fault, else 0.0; the
            // model's decoded `invalid[i]` is the same fault decision as a
            // bool. Compare the two fault *decisions*, not raw floats.
            let actual_fault = sample.invalid[i] > 0.5;
            if pred.invalid[i] == actual_fault
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
    /// Per-token ownership agreement between the model and the oracle:
    /// `matching_tokens / oracle_tokens`. A program with no tokens has no
    /// disagreements, so its agreement is `1.0` (vacuously total).
    pub ownership_agreement: f32,
}

/// Run the model on a program and check it against the oracle.
///
/// The oracle produces `tokens` and aligned `labels` of equal length; the
/// model is decoded over the same token ids, so every prediction lines up with
/// its ground-truth label index for index. Ownership agreement is the fraction
/// of tokens whose decoded ownership class equals the oracle's — `1.0` for the
/// empty program (no tokens, nothing to disagree on).
pub fn predict_program(model: &mut SomModel, ast: &SomAst) -> InferenceReport {
    let oracle = OwnershipOracle::new().analyze(ast);
    let token_ids = SomVocab::encode(&oracle.tokens);
    debug_assert_eq!(token_ids.len(), oracle.tokens.len());
    debug_assert_eq!(oracle.tokens.len(), oracle.labels.len());

    let Prediction {
        labels,
        invalid_prob,
    } = predict_labels(model, &token_ids);

    let hits = oracle
        .labels
        .iter()
        .zip(&labels.ownership)
        .filter(|(oracle_label, &predicted)| oracle_label.ownership == predicted)
        .count();
    // Empty program: 0 tokens, 0 disagreements ⇒ total (1.0) agreement.
    let agreement = if oracle.labels.is_empty()
    {
        1.0
    }
    else
    {
        hits as f32 / oracle.labels.len() as f32
    };

    let predictions = oracle
        .tokens
        .iter()
        .cloned()
        .zip(
            labels
                .ownership
                .iter()
                .zip(labels.borrow.iter().zip(invalid_prob.iter())),
        )
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

/// Parse real Rust source with the `syn` frontend, lower it, and run the
/// model against the oracle on it.
///
/// Returns `None` only if the input is not valid Rust. Valid Rust that lowers
/// to *no* tokens (empty/whitespace source, or a file with no functions) is
/// handled, not a panic: it yields a report with no predictions and `1.0`
/// agreement.
pub fn predict_rust_source(model: &mut SomModel, src: &str) -> Option<InferenceReport> {
    let lowered = scirust_som_frontend::lower_str(src).ok()?;
    Some(predict_program(model, &lowered.ast))
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
    fn predicts_on_real_rust_source() {
        // The model is trained on the synthetic distribution, then asked to
        // predict on a real Rust file parsed by the syn frontend. We assert
        // the path runs end to end and that a trained model agrees with the
        // oracle better than chance on the real input.
        let train_set = build_training_set(42, 64, 64);
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
        train(
            &mut model,
            &train_set,
            &TrainerConfig {
                epochs: 6,
                learning_rate: 0.005,
            },
        );

        let src = r#"
            fn process(input: String) {
                let owned = input;
                let moved = owned;
                let oops = owned;
                drop(oops);
                drop(moved);
            }
        "#;
        let report = predict_rust_source(&mut model, src).expect("valid rust");
        assert_eq!(report.predictions.len(), report.oracle.tokens.len());
        // Oracle ground truth on real Rust contains the use-after-move fault.
        assert!(
            report
                .oracle
                .diagnostics
                .iter()
                .any(|d| matches!(d.kind, scirust_som_symbolic::FaultKind::UseAfterMove))
        );
        // 1/5 ownership classes ≈ 0.2 by chance; a trained model clears it.
        assert!(
            report.ownership_agreement > 0.4,
            "agreement {} too low on real Rust",
            report.ownership_agreement
        );
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

    // ----- Oracle-checked accuracy tests ----------------------------------
    //
    // These pin the *arithmetic* of the accuracy metric with hand-derived
    // expectations, and the decode path against the model's own
    // `SomLogits::decode`. They are seeded and deterministic.

    use scirust_som_pcg::ast::{Expression, Function, Literal, Statement, Type};
    use scirust_som_symbolic::{OWNERSHIP_DROPPED, OWNERSHIP_NA, OWNERSHIP_OWNED, OwnershipOracle};

    /// `fn main() { let x: i64 = 1; let y = x; }`.
    ///
    /// `x: i64` is a Copy type, so `let y = x;` *copies* — `x` stays `Owned`
    /// (never `Moved`) — and both bindings drop cleanly at scope end. The
    /// oracle stream and labels are therefore fully determined; see
    /// [`tiny_copy_program_oracle_labels_are_hand_derived`] for the locked
    /// token-by-token table this whole section relies on.
    fn tiny_copy_program() -> SomAst {
        SomAst::Program(vec![Function {
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
        }])
    }

    /// Oracle-label one program into a `TrainingSample` (the exact path
    /// `build_training_set` uses, minus the length filter).
    fn sample_of(ast: &SomAst) -> TrainingSample {
        let a = OwnershipOracle::new().analyze(ast);
        TrainingSample {
            token_ids: SomVocab::encode(&a.tokens),
            ownership: a.ownership_ids(),
            borrow: a.borrow_ids(),
            invalid: a.invalid_flags(),
        }
    }

    /// The label stream this section depends on, derived by hand from Rust's
    /// Copy/move semantics and asserted against the oracle. If the oracle ever
    /// changes this program's labels, this test fails first and the dependent
    /// accuracy numbers below are revisited rather than silently drifting.
    #[test]
    fn tiny_copy_program_oracle_labels_are_hand_derived() {
        let a = OwnershipOracle::new().analyze(&tiny_copy_program());
        // Token  | ownership    | borrow | invalid
        // FnDecl | NA           | NA     | false
        // x decl | Owned        | None   | false
        // use x  | Owned (copy) | None   | false
        // y decl | Owned        | None   | false
        // drop y | Dropped      | None   | false
        // drop x | Dropped      | None   | false   (x was copied, not moved)
        use scirust_som_symbolic::{BORROW_NA, BORROW_NONE};
        let expected_own = [
            OWNERSHIP_NA,
            OWNERSHIP_OWNED,
            OWNERSHIP_OWNED,
            OWNERSHIP_OWNED,
            OWNERSHIP_DROPPED,
            OWNERSHIP_DROPPED,
        ];
        let expected_bor = [
            BORROW_NA,
            BORROW_NONE,
            BORROW_NONE,
            BORROW_NONE,
            BORROW_NONE,
            BORROW_NONE,
        ];
        assert_eq!(a.tokens.len(), 6, "stream length is fixed at 6 tokens");
        assert_eq!(a.ownership_ids(), expected_own);
        assert_eq!(a.borrow_ids(), expected_bor);
        assert!(
            a.invalid_flags().iter().all(|&f| f == 0.0),
            "the Copy program is fault-free"
        );
        assert!(a.diagnostics.is_empty());
    }

    /// A model trained to memorise the tiny Copy program reproduces *every*
    /// one of its oracle labels: ownership, borrow and fault accuracy are all
    /// exactly 1.0. Seed 42 / 200 epochs / lr 0.05 sits on a flat 1.0 plateau
    /// (verified stable across the whole 180–300 epoch window, and reproducible
    /// because training is single-threaded and bit-deterministic), so the
    /// expected accuracy is the exact constant 1.0 — not a fragile threshold.
    #[test]
    fn memorized_model_predicts_its_labels_with_full_accuracy() {
        let set = vec![sample_of(&tiny_copy_program())];
        let mut model = tiny_model(42);
        train(
            &mut model,
            &set,
            &TrainerConfig {
                epochs: 200,
                learning_rate: 0.05,
            },
        );
        let report = evaluate(&mut model, &set);
        assert_eq!(report.n_tokens, 6, "all six tokens are scored");
        assert_eq!(
            report.ownership_accuracy, 1.0,
            "memorised ownership must be perfect"
        );
        assert_eq!(
            report.borrow_accuracy, 1.0,
            "memorised borrow must be perfect"
        );
        assert_eq!(
            report.invalid_accuracy, 1.0,
            "memorised fault head must be perfect"
        );
    }

    /// Build a `TrainingSample` whose labels are exactly the (deterministic)
    /// predictions of `model` on `token_ids`. By construction the model agrees
    /// with itself on every token, so evaluating it must score exactly 1.0 on
    /// all three channels — the definition of a "memorised" stream, obtained
    /// without depending on any training trajectory.
    fn self_labelled_sample(model: &mut SomModel, token_ids: Vec<usize>) -> TrainingSample {
        let pred = predict_labels(model, &token_ids);
        let ownership = pred.labels.ownership.clone();
        let borrow = pred.labels.borrow.clone();
        let invalid = pred
            .labels
            .invalid
            .iter()
            .map(|&f| if f { 1.0 } else { 0.0 })
            .collect();
        TrainingSample {
            token_ids,
            ownership,
            borrow,
            invalid,
        }
    }

    #[test]
    fn accuracy_of_identical_label_streams_is_exactly_one() {
        // Use a real oracle token stream so the ids are in-vocab and varied.
        let toks = sample_of(&tiny_copy_program()).token_ids;
        let mut model = tiny_model(8);
        let sample = self_labelled_sample(&mut model, toks);
        let report = evaluate(&mut model, &[sample]);
        assert_eq!(report.n_tokens, 6);
        assert_eq!(report.ownership_accuracy, 1.0);
        assert_eq!(report.borrow_accuracy, 1.0);
        assert_eq!(report.invalid_accuracy, 1.0);
    }

    #[test]
    fn known_mismatch_count_gives_exact_fraction() {
        // Start from a stream the model predicts perfectly (accuracy 1.0), then
        // corrupt exactly k ownership labels to a class the model did *not*
        // predict. Accuracy must drop to exactly (n - k) / n.
        let toks = sample_of(&tiny_copy_program()).token_ids;
        let n = toks.len();
        assert_eq!(n, 6);
        let mut model = tiny_model(8);
        let base = self_labelled_sample(&mut model, toks);

        // Flipping to a class different from the prediction guarantees a miss:
        // for each position pick (predicted + 1) % OWNERSHIP_CLASSES.
        use scirust_som_symbolic::OWNERSHIP_CLASSES;
        for k in 0..=n
        {
            let mut s = base.clone();
            for slot in s.ownership.iter_mut().take(k)
            {
                *slot = (*slot + 1) % OWNERSHIP_CLASSES;
            }
            let report = evaluate(&mut model, &[s]);
            let expected = (n - k) as f32 / n as f32;
            assert_eq!(
                report.ownership_accuracy, expected,
                "k={k} mismatches over n={n} must give exactly {expected}"
            );
            // Borrow/fault channels were left untouched ⇒ still perfect.
            assert_eq!(report.borrow_accuracy, 1.0);
            assert_eq!(report.invalid_accuracy, 1.0);
        }
    }

    #[test]
    fn empty_rust_source_is_handled() {
        let mut model = tiny_model(2);
        // Whitespace-only valid Rust lowers to zero tokens: handled, not a panic.
        let report = predict_rust_source(&mut model, "   \n\t  ").expect("valid rust");
        assert!(report.predictions.is_empty());
        assert!(report.oracle.tokens.is_empty());
        assert_eq!(
            report.ownership_agreement, 1.0,
            "no tokens ⇒ no disagreements ⇒ total agreement"
        );

        // Valid Rust with no functions also yields an empty stream.
        let no_fn = predict_rust_source(&mut model, "struct S; use std::mem;").expect("valid rust");
        assert!(no_fn.predictions.is_empty());
        assert_eq!(no_fn.ownership_agreement, 1.0);

        // Genuinely invalid Rust is still rejected (None), not silently empty.
        assert!(predict_rust_source(&mut model, "fn broken( {").is_none());
    }

    #[test]
    fn empty_program_predict_program_is_handled() {
        let mut model = tiny_model(2);
        let report = predict_program(&mut model, &SomAst::Program(vec![]));
        assert!(report.predictions.is_empty());
        assert!(report.oracle.tokens.is_empty());
        assert_eq!(report.ownership_agreement, 1.0);
    }

    #[test]
    fn predicted_label_streams_match_token_stream_length() {
        // A program with a move, a borrow in a nested scope, and drops, so the
        // stream is non-trivial. Every predicted channel must be exactly as
        // long as the token stream (and the oracle's labels).
        let ast = SomAst::Program(vec![Function {
            name: "main".to_string(),
            params: vec![],
            body: vec![
                Statement::VarDecl {
                    name: "x".to_string(),
                    ty: Type::Str,
                    init: Some(Expression::Literal(Literal::Str("s".to_string()))),
                },
                Statement::VarDecl {
                    name: "y".to_string(),
                    ty: Type::Str,
                    init: Some(Expression::Variable("x".to_string())),
                },
                Statement::Scope(vec![Statement::VarDecl {
                    name: "r".to_string(),
                    ty: Type::Ref(Box::new(Type::Str), false),
                    init: Some(Expression::Reference {
                        name: "y".to_string(),
                        mutable: false,
                    }),
                }]),
            ],
        }]);
        let oracle = OwnershipOracle::new().analyze(&ast);
        let token_ids = SomVocab::encode(&oracle.tokens);
        let mut model = tiny_model(4);

        let pred = predict_labels(&mut model, &token_ids);
        assert_eq!(pred.labels.ownership.len(), token_ids.len());
        assert_eq!(pred.labels.borrow.len(), token_ids.len());
        assert_eq!(pred.labels.invalid.len(), token_ids.len());
        assert_eq!(pred.invalid_prob.len(), token_ids.len());

        let report = predict_program(&mut model, &ast);
        assert_eq!(report.predictions.len(), token_ids.len());
        assert_eq!(report.predictions.len(), report.oracle.labels.len());
    }

    #[test]
    fn inference_decode_matches_models_published_decode() {
        // The decode used by inference must be *identical* to the model's own
        // `SomLogits::decode` — same argmax (first-maximum tie-break) and same
        // fault threshold. We compute both on the same token stream and assert
        // equality element by element. This guards the tie-break fix: the old
        // `max_by`-based argmax kept the *last* maximum and could disagree.
        let token_ids = sample_of(&tiny_copy_program()).token_ids;
        let mut model = tiny_model(13);

        // Inference path.
        let inf = predict_labels(&mut model, &token_ids).labels;

        // Model's own published decode, on a fresh tape.
        let tape = Tape::new();
        let direct = model.forward(&tape, &token_ids).decode();

        assert_eq!(
            inf, direct,
            "inference decode must equal the model's decode"
        );
    }

    #[test]
    fn evaluate_skips_empty_samples_without_panicking() {
        // A real sample plus a degenerate empty one: the empty sample
        // contributes no tokens (and must not hit the forward pass's
        // seq_len >= 1 assert), so n_tokens counts only the real sample.
        let real = sample_of(&tiny_copy_program());
        let empty = TrainingSample {
            token_ids: Vec::new(),
            ownership: Vec::new(),
            borrow: Vec::new(),
            invalid: Vec::new(),
        };
        let mut model = tiny_model(6);
        let report = evaluate(&mut model, &[real.clone(), empty]);
        assert_eq!(report.n_tokens, real.token_ids.len());
        assert_eq!(report.n_tokens, 6);
        assert!((0.0..=1.0).contains(&report.ownership_accuracy));
    }
}
