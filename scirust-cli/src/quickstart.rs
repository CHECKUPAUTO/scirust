//! The 60-second quickstart, as a deterministic library function.
//!
//! Trains an MLP `2 → 8 → 2` on the XOR-classification task (the same
//! oracle as `examples/quickstart_v2`): not linearly separable, so solving
//! it 4/4 proves the tape, Adam, Linear/ReLU/Sequential and the stable
//! cross-entropy all work together. Seeded (`PcgEngine::new(42)`) and
//! single-threaded ⇒ the loss trajectory is bit-identical across runs.

use scirust_core::autodiff::optim::{Adam, Optimizer};
use scirust_core::autodiff::reverse::{Tape, Tensor};
use scirust_core::nn::{
    CrossEntropyLoss, KaimingNormal, Linear, Loss, Module, PcgEngine, ReLU, Sequential, Zeros,
};

const INPUTS: [[f32; 2]; 4] = [[0.0, 0.0], [1.0, 1.0], [0.0, 1.0], [1.0, 0.0]];
const LABELS: [usize; 4] = [0, 0, 1, 1];

/// Result of a quickstart run.
#[derive(Debug, Clone)]
pub struct Report {
    /// Mean loss recorded every `record_every` epochs, in order.
    pub losses: Vec<f32>,
    /// Correctly classified points out of 4 after training.
    pub correct: usize,
}

impl Report {
    pub fn final_loss(&self) -> f32 {
        *self.losses.last().expect("at least one recorded loss")
    }
}

/// Train the demo for `epochs` epochs, recording the mean loss every
/// `record_every` epochs. Deterministic in the fixed seed.
pub fn train(epochs: usize, record_every: usize) -> Report {
    let mut rng = PcgEngine::new(42);
    let mut model = Sequential::new()
        .add(Linear::new(2, 8, &KaimingNormal, &Zeros, &mut rng))
        .add(ReLU::new())
        .add(Linear::new(8, 2, &KaimingNormal, &Zeros, &mut rng));

    let loss_fn = CrossEntropyLoss::new();
    let mut opt = Adam::new(0.05);
    let mut losses = Vec::new();

    for epoch in 0..epochs
    {
        let mut epoch_loss = 0.0;
        for (x_arr, &label) in INPUTS.iter().zip(LABELS.iter())
        {
            let tape = Tape::new();
            let x = tape.input(Tensor::from_vec(x_arr.to_vec(), 1, 2));
            let mut oh = vec![0.0; 2];
            oh[label] = 1.0;
            let target = tape.input(Tensor::from_vec(oh, 1, 2));

            let logits = model.forward(&tape, x);
            let loss = loss_fn.forward(&tape, logits, target);
            tape.backward(loss.idx());
            opt.step(&model.parameter_indices(), &tape);
            model.sync(&tape);

            epoch_loss += tape.value(loss.idx()).data[0];
        }
        if record_every > 0 && (epoch == 0 || (epoch + 1) % record_every == 0)
        {
            losses.push(epoch_loss / 4.0);
        }
    }

    let mut correct = 0;
    for (x_arr, &label) in INPUTS.iter().zip(LABELS.iter())
    {
        let tape = Tape::new();
        let x = tape.input(Tensor::from_vec(x_arr.to_vec(), 1, 2));
        let logits = model.forward(&tape, x);
        let scores = tape.value(logits.idx());
        let pred = if scores.data[0] > scores.data[1]
        {
            0
        }
        else
        {
            1
        };
        if pred == label
        {
            correct += 1;
        }
    }

    Report { losses, correct }
}

/// Run the quickstart and print a short, human-readable report. Returns the
/// process exit code: 0 if the model solves all 4 points, 1 otherwise.
pub fn run() -> u8 {
    println!("SciRust quickstart — MLP 2→8→2 on XOR classification (seed 42)\n");
    let report = train(1000, 250);
    for (i, l) in report.losses.iter().enumerate()
    {
        println!("  checkpoint {:>2}: loss = {:.6}", i + 1, l);
    }
    println!("\nfinal accuracy: {}/4", report.correct);
    if report.correct == 4
    {
        println!("OK — non-linear model trained end to end, deterministically.");
        0
    }
    else
    {
        println!("UNEXPECTED — the demo should reach 4/4; please report this.");
        1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quickstart_solves_xor_and_loss_decreases() {
        let r = train(1000, 250);
        assert_eq!(r.correct, 4, "XOR demo must reach 4/4");
        assert!(
            r.final_loss() < r.losses[0],
            "loss must decrease: {:?}",
            r.losses
        );
        assert!(r.final_loss().is_finite());
    }

    #[test]
    fn quickstart_is_bit_deterministic() {
        let bits = |r: &Report| r.losses.iter().map(|f| f.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(&train(300, 50)), bits(&train(300, 50)));
    }
}
