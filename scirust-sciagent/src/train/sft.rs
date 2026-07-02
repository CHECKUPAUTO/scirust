use std::fs;

use crate::bpe::BpeTokenizer;
use crate::config::SciAgentConfig;
use crate::model::SciAgentModel;
use crate::train::optimizer::TrainOptimizer;
use crate::train::scheduler::WarmupCosineSchedule;

use scirust_core::autodiff::reverse::Tape;
use scirust_core::autodiff::scheduler::LrSchedule;

pub struct SftExample {
    pub instruction: String,
    pub output: String,
    pub tool_calls: Option<Vec<ToolCallExample>>,
}

pub struct ToolCallExample {
    pub name: String,
    pub params: Vec<(String, String)>,
}

pub struct SftDataset {
    examples: Vec<SftExample>,
    #[allow(dead_code)]
    position: usize,
}

impl SftDataset {
    pub fn from_jsonl(path: &str) -> std::io::Result<Self> {
        let file = fs::File::open(path)?;
        let reader = std::io::BufReader::new(file);
        let mut examples = Vec::new();

        for line in std::io::BufRead::lines(reader)
        {
            let line = line?;
            if line.trim().is_empty()
            {
                continue;
            }
            let json: serde_json::Value = serde_json::from_str(&line)
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;

            let instruction = json["instruction"].as_str().unwrap_or("").to_string();
            let output = json["output"].as_str().unwrap_or("").to_string();
            let tool_calls = json["tool_calls"].as_array().map(|arr| {
                arr.iter()
                    .filter_map(|tc| {
                        let name = tc["name"].as_str()?;
                        let params = tc["params"]
                            .as_object()?
                            .iter()
                            .map(|(k, v)| (k.clone(), v.as_str().unwrap_or("").to_string()))
                            .collect();
                        Some(ToolCallExample {
                            name: name.to_string(),
                            params,
                        })
                    })
                    .collect()
            });

            examples.push(SftExample {
                instruction,
                output,
                tool_calls,
            });
        }

        Ok(Self {
            examples,
            position: 0,
        })
    }

    pub fn len(&self) -> usize {
        self.examples.len()
    }

    pub fn is_empty(&self) -> bool {
        self.examples.is_empty()
    }
}

pub fn format_sft_prompt(example: &SftExample, tokenizer: &BpeTokenizer) -> Vec<usize> {
    let bos = tokenizer.special_id("<bos>");
    let eos = tokenizer.special_id("<eos>");

    let mut tokens = vec![bos];
    tokens.extend(tokenizer.encode("user: "));
    tokens.extend(tokenizer.encode(&example.instruction));
    tokens.push(eos);

    tokens.push(bos);
    tokens.extend(tokenizer.encode("assistant: "));
    tokens.extend(tokenizer.encode(&example.output));
    tokens.push(eos);

    tokens
}

#[allow(clippy::too_many_arguments)] // training entry point mirrors the CLI knobs
pub fn sft_train(
    model: &mut SciAgentModel,
    dataset: &SftDataset,
    _config: &SciAgentConfig,
    tokenizer: &BpeTokenizer,
    lr: f32,
    epochs: usize,
    batch_size: usize,
    max_seq_len: usize,
) {
    let total_steps = dataset.len() * epochs / batch_size;
    let mut opt = TrainOptimizer::new_muon(lr);
    let scheduler = WarmupCosineSchedule::new(lr, lr * 0.1, total_steps / 20, total_steps);

    let mut step = 0usize;
    for epoch in 0..epochs
    {
        let mut epoch_loss = 0.0f64;

        for chunk in dataset.examples.chunks(batch_size)
        {
            let tape = Tape::new();
            let mut all_inputs = Vec::new();
            let mut all_targets = Vec::new();

            for ex in chunk
            {
                let tokens = format_sft_prompt(ex, tokenizer);
                if tokens.len() < 2
                {
                    continue;
                }
                let seq = &tokens[..tokens.len().min(max_seq_len)];
                let inputs: Vec<usize> = seq[..seq.len() - 1].to_vec();
                let targets: Vec<usize> = seq[1..].to_vec();
                all_inputs.extend(inputs);
                all_targets.extend(targets);
            }

            if all_inputs.is_empty()
            {
                continue;
            }

            let seq_len = all_inputs.len() / chunk.len();
            let logits = model.forward(&tape, &all_inputs, seq_len);
            let loss = crate::train::cross_entropy_loss(&tape, logits, &all_targets);
            tape.backward(loss.idx());
            let loss_val = tape.value(loss.idx()).data[0] as f64;
            epoch_loss += loss_val;

            let lr = scheduler.lr_at(step);
            opt.set_lr(lr);
            opt.clip_grad_norm(&tape, 1.0);
            let params = model.parameter_indices();
            opt.step(&params, &tape);
            model.sync(&tape);

            step += 1;

            if step % 10 == 0
            {
                println!("[SFT Epoch {epoch} Step {step}] loss: {loss_val:.4} | lr: {lr:.8}");
            }
        }

        println!(
            "[SFT Epoch {epoch}] avg loss: {:.4}",
            epoch_loss / dataset.len() as f64
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bpe::BpeTrainer;

    #[test]
    fn test_format_prompt() {
        let trainer = BpeTrainer::new(50).min_frequency(1);
        let texts = vec!["user: hello world assistant: hi".to_string()];
        let tok = trainer.train(&texts);

        let ex = SftExample {
            instruction: "hello".to_string(),
            output: "world".to_string(),
            tool_calls: None,
        };
        let tokens = format_sft_prompt(&ex, &tok);
        assert!(tokens.len() > 2);
        assert_eq!(tokens[0], 1); // <bos>
    }
}
