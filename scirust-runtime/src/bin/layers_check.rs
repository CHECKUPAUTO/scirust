// Palier 3 : Sigmoid + LayerNorm dans le manifeste. Round-trip SRT1 bit-exact.
use scirust_core::autodiff::reverse::{Tape, Tensor};
use scirust_core::nn::{
    KaimingNormal, LayerNorm, Linear, Module, PcgEngine, Sequential, Sigmoid, Zeros,
};
use scirust_runtime::{
    LayerSpec, build_model, fnv_fold_f32, fnv_init, load_weights, parse_manifest, save_weights,
    write_manifest,
};

fn synth(n: usize, cols: usize, seed: u64) -> Tensor {
    let mut s = seed;
    let mut data = Vec::with_capacity(n * cols);
    for _ in 0..n * cols
    {
        s = s.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = s;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^= z >> 31;
        data.push(((z >> 40) as f32) / (1u32 << 24) as f32);
    }
    Tensor::from_vec(data, n, cols)
}

fn build_hardcoded(seed: u64) -> Sequential {
    let mut rng = PcgEngine::new(seed);
    Sequential::new()
        .add(Linear::new(8, 16, &KaimingNormal, &Zeros, &mut rng))
        .add(LayerNorm::new(16, 1e-5, &KaimingNormal, &mut rng))
        .add(Sigmoid::new())
        .add(Linear::new(16, 4, &KaimingNormal, &Zeros, &mut rng))
}

fn fp_forward(m: &mut Sequential, x: &Tensor) -> u64 {
    let tape = Tape::new();
    let v = tape.input(x.clone());
    let logits = m.forward(&tape, v);
    fnv_fold_f32(fnv_init(), &tape.value(logits.idx()).data)
}

fn main() {
    let x = synth(16, 8, 7);
    let mut href = build_hardcoded(42);
    let f1 = fp_forward(&mut href, &x);
    let f2 = fp_forward(&mut href, &x);
    let f3 = fp_forward(&mut href, &x);
    assert!(f1 == f2 && f2 == f3, "non deterministe");
    save_weights(&href.state_dict(), "layers.srt").expect("save");

    let spec = vec![
        LayerSpec::Linear { in_f: 8, out_f: 16 },
        LayerSpec::LayerNorm {
            d_model: 16,
            eps: 1e-5,
        },
        LayerSpec::Sigmoid,
        LayerSpec::Linear { in_f: 16, out_f: 4 },
    ];
    let manifest = write_manifest(&spec);
    print!("--- manifeste ---\n{}-----------------\n", manifest);
    std::fs::write("layers.manifest", &manifest).unwrap();
    let parsed =
        parse_manifest(&std::fs::read_to_string("layers.manifest").unwrap()).expect("parse");
    let mut gen = build_model(&parsed);
    gen.load_state_dict(&load_weights("layers.srt").expect("load"))
        .expect("lsd");
    let f_gen = fp_forward(&mut gen, &x);

    println!("hardcode  : {:#018x}", f1);
    println!("manifeste : {:#018x}", f_gen);
    println!("equivalent: {}", f1 == f_gen);
    assert_eq!(f1, f_gen, "Sigmoid+LayerNorm via manifeste NON equivalent");
    println!(
        "\nOK : Sigmoid + LayerNorm supportes (persistance gamma/beta + reconstruction bit-exact)."
    );
}
