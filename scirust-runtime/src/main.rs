use scirust_core::autodiff::reverse::{Tape, Tensor};
use scirust_core::data::{DataLoader, MnistDataset};
use scirust_core::nn::{KaimingNormal, Linear, Module, PcgEngine, ReLU, Sequential, Zeros};
use scirust_runtime::{fnv_bytes, fnv_fold_f32, fnv_init, load_weights, save_weights};

fn build_model(seed: u64) -> Sequential {
    let mut rng = PcgEngine::new(seed);
    Sequential::new()
        .add(Linear::new(784, 256, &KaimingNormal, &Zeros, &mut rng))
        .add(ReLU::new())
        .add(Linear::new(256, 10, &KaimingNormal, &Zeros, &mut rng))
}

fn fingerprint(model: &mut Sequential, batches: &[Tensor]) -> u64 {
    let mut fp = fnv_init();
    for x in batches
    {
        let tape = Tape::new();
        let v = tape.input(x.clone());
        let logits = model.forward(&tape, v);
        fp = fnv_fold_f32(fp, &tape.value(logits.idx()).data);
    }
    fp
}

fn main() {
    const GOLDEN: u64 = 0xde2d807686e4b47e;

    let data_dir =
        std::env::var("MNIST_DIR").unwrap_or_else(|_| "/root/scirust/data/mnist".to_string());
    let test = MnistDataset::load_idx(
        format!("{}/t10k-images-idx3-ubyte", data_dir),
        format!("{}/t10k-labels-idx1-ubyte", data_dir),
    )
    .expect("chargement test MNIST");
    let mut loader = DataLoader::new(test.subsample(256), 64, false, 42);
    let batches: Vec<Tensor> = loader.iter().map(|(x, _)| x).collect();

    let mut a = build_model(42);
    let fp_a = fingerprint(&mut a, &batches);
    println!("fp(A, seed=42)             = {:#018x}", fp_a);
    assert_eq!(fp_a, GOLDEN, "A ne reproduit pas l'empreinte d'audit");

    let path = "weights.srt";
    save_weights(&a.state_dict(), path).expect("save_weights");
    let file = std::fs::read(path).unwrap();
    println!(
        "artefact poids             : {} octets, hash = {:#018x}",
        file.len(),
        fnv_bytes(&file)
    );

    let mut b = build_model(999);
    let fp_b_pre = fingerprint(&mut b, &batches);
    println!("fp(B, seed=999, pre-load)  = {:#018x}", fp_b_pre);
    assert_ne!(
        fp_b_pre, GOLDEN,
        "B == A sans load : le test ne prouve rien"
    );

    let sd = load_weights(path).expect("load_weights");
    b.load_state_dict(&sd).expect("load_state_dict");
    let fp_b_post = fingerprint(&mut b, &batches);
    println!("fp(B, post-load)           = {:#018x}", fp_b_post);
    assert_eq!(fp_b_post, GOLDEN, "rechargement NON bit-exact");

    println!("\nOK : persistance fingerprint-stable.");
    println!("Poids figes -> inference bit-exact rejouable. Garantie #1 du runtime posee.");
}
