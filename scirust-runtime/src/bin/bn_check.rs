// Palier 3b : BatchNorm2d (eval mode). Prouve la persistance des running stats.
use scirust_core::autodiff::reverse::{Tape, Tensor};
use scirust_core::nn::{BatchNorm2d, KaimingNormal, Linear, Module, PcgEngine, Sequential, Zeros};
use scirust_runtime::{
    build_model, fnv_fold_f32, fnv_init, load_weights, parse_manifest, save_weights, write_manifest,
    LayerSpec,
};

fn synth(n: usize, cols: usize, seed: u64) -> Tensor {
    let mut s = seed;
    let mut data = Vec::with_capacity(n * cols);
    for _ in 0..n * cols {
        s = s.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = s;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^= z >> 31;
        data.push(((z >> 40) as f32) / (1u32 << 24) as f32);
    }
    Tensor::from_vec(data, n, cols)
}

fn fp_forward(m: &mut Sequential, x: &Tensor) -> u64 {
    let tape = Tape::new();
    let v = tape.input(x.clone());
    let logits = m.forward(&tape, v);
    fnv_fold_f32(fnv_init(), &tape.value(logits.idx()).data)
}

fn main() {
    let x = synth(8, 16, 99); // (N=8, C*H*W=16) avec C=4 -> spatial=4

    // Reference : BatchNorm2d(4) eval, running stats NON triviales (comme apres entrainement)
    let mut bn = BatchNorm2d::new(4);
    bn.set_training(false);
    bn.running_mean = Tensor::from_vec(vec![0.3, -0.2, 0.5, 0.1], 1, 4);
    bn.running_var = Tensor::from_vec(vec![1.5, 0.8, 2.0, 1.2], 1, 4);
    let mut rng = PcgEngine::new(42);
    let mut href = Sequential::new()
        .add(bn)
        .add(Linear::new(16, 4, &KaimingNormal, &Zeros, &mut rng));
    let f_ref = fp_forward(&mut href, &x);
    save_weights(&href.state_dict(), "bn.srt").expect("save");

    let spec = vec![
        LayerSpec::BatchNorm2d { channels: 4 },
        LayerSpec::Linear { in_f: 16, out_f: 4 },
    ];
    let manifest = write_manifest(&spec);
    print!("--- manifeste ---\n{}-----------------\n", manifest);
    std::fs::write("bn.manifest", &manifest).unwrap();
    let parsed = parse_manifest(&std::fs::read_to_string("bn.manifest").unwrap()).expect("parse");
    let mut gen = build_model(&parsed);
    let f_pre = fp_forward(&mut gen, &x); // stats par defaut (0/1) -> doit differer
    gen.load_state_dict(&load_weights("bn.srt").expect("load")).expect("lsd");
    let f_post = fp_forward(&mut gen, &x);

    println!("ref           : {:#018x}", f_ref);
    println!("gen pre-load  : {:#018x} (different : {})", f_pre, f_pre != f_ref);
    println!("gen post-load : {:#018x} (== ref : {})", f_post, f_post == f_ref);
    assert_ne!(f_pre, f_ref, "pre-load == ref : running stats non exercees");
    assert_eq!(f_post, f_ref, "BatchNorm2d eval round-trip NON bit-exact");
    println!("\nOK : BatchNorm2d (eval, running stats) supporte, round-trip SRT1 bit-exact.");
}
