// examples/transformer_demo/src/main.rs
//
// SciRust v12.0 — Transformer Demo (sequence classification)
//
// Tache synthetique : vote majoritaire d'une sequence binaire.
//   Input  : sequence de longueur L avec elements {0, 1}
//   Label  : 0 si plus de 0s, 1 si plus de 1s (egalite = classe 0)
//
// Architecture :
//   Embedding (2 -> d_model) + Positional Encoding
//   -> TransformerEncoder (N layers)
//   -> Mean pooling sur la sequence
//   -> Linear (d_model -> 2) -> CrossEntropyLoss

use scirust_core::autodiff::optim::{Adam, Optimizer};
use scirust_core::autodiff::reverse::{Tape, Tensor};
use scirust_core::nn::transformer::TransformerEncoder;
use scirust_core::nn::{CrossEntropyLoss, KaimingNormal, Linear, Loss, Module, PcgEngine, Zeros};
use scirust_core::tensor::tensor3d::{Tensor3D, Var3D};

const SEQ_LEN: usize = 16;
const D_MODEL: usize = 64;
const N_HEADS: usize = 4;
const N_LAYERS: usize = 2;
const D_FF: usize = 128;
const BATCH_SIZE: usize = 64;
const N_EPOCHS: usize = 30;

fn generate_batch(batch_size: usize, seq_len: usize, rng: &mut PcgEngine) -> (Tensor3D, Tensor) {
    let mut x_data = vec![0.0f32; batch_size * seq_len * 2];
    let mut y_data = vec![0.0f32; batch_size * 2];
    for b in 0..batch_size
    {
        let mut sum = 0usize;
        for t in 0..seq_len
        {
            let val = if rng.float() > 0.5 { 1.0 } else { 0.0 };
            if val > 0.5
            {
                sum += 1;
            }
            let idx = b * seq_len * 2 + t * 2;
            x_data[idx] = 1.0 - val; // one-hot : [P(0), P(1)]
            x_data[idx + 1] = val;
        }
        let label = if sum > seq_len / 2 { 1 } else { 0 };
        y_data[b * 2 + label] = 1.0;
    }
    let x = Tensor3D::new(
        Tensor::from_vec(x_data, batch_size * seq_len, 2),
        batch_size,
        seq_len,
        2,
    );
    let y = Tensor::from_vec(y_data, batch_size, 2);
    (x, y)
}

/// Positional encoding sinusoidal fixe (Vaswani et al.)
fn make_positional_encoding(seq_len: usize, d_model: usize) -> Tensor {
    let mut pe = vec![0.0f32; seq_len * d_model];
    for pos in 0..seq_len
    {
        for i in 0..d_model
        {
            let angle = pos as f32 / (10000.0f32).powf(2.0 * (i / 2) as f32 / d_model as f32);
            pe[pos * d_model + i] = if i % 2 == 0 { angle.sin() } else { angle.cos() };
        }
    }
    Tensor::from_vec(pe, seq_len, d_model)
}

/// Repete le PE pour chaque element du batch : (seq_len, d_model) -> (batch*seq_len, d_model)
fn tile_pe(pe: &Tensor, batch: usize) -> Tensor {
    let seq_len = pe.rows;
    let d_model = pe.cols;
    let mut out = vec![0.0f32; batch * seq_len * d_model];
    for b in 0..batch
    {
        for t in 0..seq_len
        {
            for d in 0..d_model
            {
                out[(b * seq_len + t) * d_model + d] = pe.data[t * d_model + d];
            }
        }
    }
    Tensor::from_vec(out, batch * seq_len, d_model)
}

fn main() {
    println!("=== SciRust v12.0 — Transformer Demo (Majority Vote) ===\n");
    println!("Tache : classifier le vote majoritaire de sequences binaires de longueur {SEQ_LEN}");
    println!(
        "Archi : Embedding({D_MODEL}) + PE + Transformer({N_LAYERS}L/{N_HEADS}H) + MeanPool + Linear(2)\n"
    );

    let mut rng = PcgEngine::new(42);

    // Modules
    let mut embed = Linear::new(2, D_MODEL, &KaimingNormal, &Zeros, &mut rng);
    let mut encoder = TransformerEncoder::new(
        N_LAYERS,
        D_MODEL,
        N_HEADS,
        D_FF,
        false,
        &KaimingNormal,
        &Zeros,
        &mut rng,
    );
    let mut classifier = Linear::new(D_MODEL, 2, &KaimingNormal, &Zeros, &mut rng);

    let pe_base = make_positional_encoding(SEQ_LEN, D_MODEL);
    let loss_fn = CrossEntropyLoss::new();
    let mut opt = Adam::new(0.001);

    // Entrainement : 10 batches par epoch
    const BATCHES_PER_EPOCH: usize = 10;
    for epoch in 0..N_EPOCHS
    {
        let mut epoch_loss = 0.0f32;
        for _ in 0..BATCHES_PER_EPOCH
        {
            let (x3d, y2d) = generate_batch(BATCH_SIZE, SEQ_LEN, &mut rng);
            let tape = Tape::new();

            let x_var = Var3D::input_3d(&tape, x3d);
            let flat_in = x_var.as_var();
            let embedded = embed.forward(&tape, flat_in);
            let pe_var = tape.input(tile_pe(&pe_base, BATCH_SIZE));
            let with_pe = embedded.add_broadcast(pe_var);
            let x_encoded = Var3D::from_var(with_pe, BATCH_SIZE, SEQ_LEN, D_MODEL);

            let enc_out = encoder.forward_3d(&tape, x_encoded);
            let enc_flat = enc_out.as_var();
            let pooled = mean_pool(&tape, enc_flat, BATCH_SIZE, SEQ_LEN, D_MODEL);

            let logits = classifier.forward(&tape, pooled);
            let target = tape.input(y2d);
            let loss = loss_fn.forward(&tape, logits, target);

            tape.backward(loss.idx());

            let all_params: Vec<usize> = {
                let mut v = embed.parameter_indices();
                v.extend(encoder.parameter_indices());
                v.extend(classifier.parameter_indices());
                v
            };
            opt.step(&all_params, &tape);
            embed.sync(&tape);
            encoder.sync(&tape);
            classifier.sync(&tape);

            epoch_loss += tape.value(loss.idx()).data[0];
        }
        if (epoch + 1) % 5 == 0 || epoch == 0
        {
            println!(
                "  Epoch {:>2} : loss = {:.4}",
                epoch + 1,
                epoch_loss / BATCHES_PER_EPOCH as f32
            );
        }
    }

    // Evaluation
    println!("\n--- Evaluation ---");
    let mut correct = 0;
    let mut total = 0;
    for _ in 0..10
    {
        let (x3d, y2d) = generate_batch(BATCH_SIZE, SEQ_LEN, &mut rng);
        let tape = Tape::new();
        let x_var = Var3D::input_3d(&tape, x3d);
        let embedded = embed.forward(&tape, x_var.as_var());
        let pe_var = tape.input(tile_pe(&pe_base, BATCH_SIZE));
        let with_pe = embedded.add_broadcast(pe_var);
        let x_encoded = Var3D::from_var(with_pe, BATCH_SIZE, SEQ_LEN, D_MODEL);
        let enc_out = encoder.forward_3d(&tape, x_encoded);
        let pooled = mean_pool(&tape, enc_out.as_var(), BATCH_SIZE, SEQ_LEN, D_MODEL);
        let logits = classifier.forward(&tape, pooled);
        let scores = tape.value(logits.idx());

        for b in 0..BATCH_SIZE
        {
            let p0 = scores.data[b * 2];
            let p1 = scores.data[b * 2 + 1];
            let pred = if p1 > p0 { 1 } else { 0 };
            let true_class = if y2d.data[b * 2 + 1] > y2d.data[b * 2]
            {
                1
            }
            else
            {
                0
            };
            if pred == true_class
            {
                correct += 1;
            }
            total += 1;
        }
    }

    let acc = correct as f32 / total as f32 * 100.0;
    println!("  Accuracy : {:.1}% ({}/{})\n", acc, correct, total);

    if acc >= 80.0
    {
        println!("✅ SUCCES — Le Transformer classifie correctement le vote majoritaire.");
    }
    else if acc >= 60.0
    {
        println!("⚠️  PARTIEL — {:.1}% est acceptable mais < 80%.", acc);
    }
    else
    {
        println!("❌ ECHEC — Convergence insuffisante.");
        std::process::exit(1);
    }
}

/// Mean pooling differentiable : pool_mat (batch, batch*seq_len) @ flat (batch*seq_len, d_model)
fn mean_pool<'a>(
    tape: &'a Tape,
    flat: scirust_core::autodiff::reverse::Var<'a>,
    batch: usize,
    seq_len: usize,
    _d_model: usize,
) -> scirust_core::autodiff::reverse::Var<'a> {
    let inv = 1.0 / seq_len as f32;
    let mut data = vec![0.0f32; batch * (batch * seq_len)];
    for b in 0..batch
    {
        for t in 0..seq_len
        {
            data[b * (batch * seq_len) + b * seq_len + t] = inv;
        }
    }
    let pool_mat = tape.input(Tensor::from_vec(data, batch, batch * seq_len));
    pool_mat.matmul(flat)
}
