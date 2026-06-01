// AUDIT QUANTIFICATION int8 : f32 (oracle) vs int8 sur MNIST test.
use scirust_core::autodiff::reverse::Tape;
use scirust_core::data::{DataLoader, MnistDataset};
use scirust_core::nn::{KaimingNormal, Linear, Module, PcgEngine, ReLU, Sequential, Zeros};
use scirust_core::quantization::{quantize_per_channel, quantized_linear_forward};
use scirust_runtime::{fnv_fold_f32, fnv_init, load_weights};

fn argmax(row: &[f32]) -> usize {
    let mut best = row[0];
    let mut bi = 0usize;
    for (i, &v) in row.iter().enumerate().skip(1) {
        if v > best {
            best = v;
            bi = i;
        }
    }
    bi
}

fn main() {
    let data_dir =
        std::env::var("MNIST_DIR").unwrap_or_else(|_| "/root/scirust/data/mnist".to_string());
    let test = MnistDataset::load_idx(
        format!("{}/t10k-images-idx3-ubyte", data_dir),
        format!("{}/t10k-labels-idx1-ubyte", data_dir),
    )
    .expect("chargement test MNIST");

    let mut rng = PcgEngine::new(123);
    let mut model = Sequential::new()
        .add(Linear::new(784, 256, &KaimingNormal, &Zeros, &mut rng))
        .add(ReLU::new())
        .add(Linear::new(256, 10, &KaimingNormal, &Zeros, &mut rng));
    let sd = load_weights("mnist_mlp.srt").expect("load_weights (lance train_artifact d'abord)");
    model.load_state_dict(&sd).expect("load_state_dict");

    println!("Cles du state_dict :");
    let mut keys: Vec<_> = sd.keys().cloned().collect();
    keys.sort();
    for k in &keys {
        let (r, c) = sd[k].shape();
        println!("  {:<14} ({}, {})", k, r, c);
    }
    let (mut w1, mut b1, mut w2, mut b2) = (None, None, None, None);
    for (_k, t) in &sd {
        match t.shape() {
            (784, 256) => w1 = Some(t.clone()),
            (1, 256) => b1 = Some(t.clone()),
            (256, 10) => w2 = Some(t.clone()),
            (1, 10) => b2 = Some(t.clone()),
            _ => {}
        }
    }
    let w1 = w1.expect("poids Linear1 (784,256)");
    let b1 = b1.expect("biais Linear1 (1,256)");
    let w2 = w2.expect("poids Linear2 (256,10)");
    let b2 = b2.expect("biais Linear2 (1,10)");

    let (w1q, w1s) = quantize_per_channel(&w1.data, 784, 256);
    let (w2q, w2s) = quantize_per_channel(&w2.data, 256, 10);

    let mut loader = DataLoader::new(test.subsample(test.n), 64, false, 42);
    let (mut correct_f32, mut correct_int8, mut total) = (0usize, 0usize, 0usize);
    let mut fp_f32 = fnv_init();
    let mut fp_int8 = fnv_init();

    for (xb, yb) in loader.iter() {
        let (bs, _) = xb.shape();

        let tape = Tape::new();
        let v = tape.input(xb.clone());
        let logits = model.forward(&tape, v);
        let scores = tape.value(logits.idx());
        fp_f32 = fnv_fold_f32(fp_f32, &scores.data);

        let h = quantized_linear_forward(&xb.data, bs, 784, &w1q, &w1s, &b1.data, 256);
        let h_relu: Vec<f32> = h.iter().map(|&x| x.max(0.0)).collect();
        let logits_q = quantized_linear_forward(&h_relu, bs, 256, &w2q, &w2s, &b2.data, 10);
        fp_int8 = fnv_fold_f32(fp_int8, &logits_q);

        for i in 0..bs {
            let tc = argmax(&yb.data[i * 10..(i + 1) * 10]);
            if argmax(&scores.data[i * 10..(i + 1) * 10]) == tc {
                correct_f32 += 1;
            }
            if argmax(&logits_q[i * 10..(i + 1) * 10]) == tc {
                correct_int8 += 1;
            }
            total += 1;
        }
    }

    let acc_f32 = correct_f32 as f32 / total as f32 * 100.0;
    let acc_int8 = correct_int8 as f32 / total as f32 * 100.0;
    let n_params = 784 * 256 + 256 * 10;
    let bytes_f32 = n_params * 4;
    let bytes_int8 = n_params + (256 + 10) * 4;
    let ratio = bytes_f32 as f64 / bytes_int8 as f64;

    println!();
    println!("=== AUDIT int8 (MNIST test, {} echantillons) ===", total);
    println!("Accuracy f32 (oracle) : {:.2}% ({}/{})", acc_f32, correct_f32, total);
    println!("Accuracy int8         : {:.2}% ({}/{})", acc_int8, correct_int8, total);
    println!("Delta                 : {:+.2} points", acc_int8 - acc_f32);
    println!();
    println!("Poids f32  : {} octets", bytes_f32);
    println!("Poids int8 : {} octets (poids + scales per-channel)", bytes_int8);
    println!("Reduction  : {:.2}x", ratio);
    println!();
    println!("Empreinte logits f32  : {:#018x}", fp_f32);
    println!("Empreinte logits int8 : {:#018x}", fp_int8);
    println!("  (relancer => memes empreintes : reproductible inter-process)");
}
