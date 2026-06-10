// Palier 1 : les 3 garanties sur archi CNN (conv2d), pas seulement MLP.
// Input synthetique deterministe (splitmix64) -> autonome, pas besoin de CIFAR.
use scirust_core::autodiff::reverse::{Tape, Tensor};
use scirust_core::nn::{
    Conv2d, KaimingNormal, Linear, MaxPool2d, Module, Padding, PcgEngine, ReLU, Sequential, Zeros,
};
use scirust_runtime::{fnv_bytes, fnv_fold_f32, fnv_init, load_weights, save_weights};

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

fn build_cnn(seed: u64) -> Sequential {
    let mut rng = PcgEngine::new(seed);
    Sequential::new()
        .add(
            Conv2d::new(
                3,
                32,
                3,
                1,
                Padding::Same,
                &KaimingNormal,
                Some(&Zeros),
                &mut rng,
            )
            .input_dims(32, 32),
        )
        .add(ReLU::new())
        .add(MaxPool2d::new(2, 2).input_shape(32, 32, 32))
        .add(
            Conv2d::new(
                32,
                64,
                3,
                1,
                Padding::Same,
                &KaimingNormal,
                Some(&Zeros),
                &mut rng,
            )
            .input_dims(16, 16),
        )
        .add(ReLU::new())
        .add(MaxPool2d::new(2, 2).input_shape(64, 16, 16))
        .add(Linear::new(4096, 256, &KaimingNormal, &Zeros, &mut rng))
        .add(ReLU::new())
        .add(Linear::new(256, 10, &KaimingNormal, &Zeros, &mut rng))
}

fn fp_of(model: &mut Sequential, x: &Tensor) -> u64 {
    let tape = Tape::new();
    let v = tape.input(x.clone());
    let logits = model.forward(&tape, v);
    fnv_fold_f32(fnv_init(), &tape.value(logits.idx()).data)
}

fn main() {
    let bench = !std::env::args().any(|a| a == "--no-bench");
    let n = 32usize;
    let x = synth(n, 3 * 32 * 32, 12345);

    // 1. Determinisme : forward x3 bit-exact
    let mut m = build_cnn(42);
    let f1 = fp_of(&mut m, &x);
    let f2 = fp_of(&mut m, &x);
    let f3 = fp_of(&mut m, &x);
    let det = f1 == f2 && f2 == f3;
    if bench
    {
        println!(
            "CNN forward x3 : {:#018x} / {:#018x} / {:#018x} -> {}",
            f1,
            f2,
            f3,
            if det { "BIT-EXACT" } else { "DIVERGE" }
        );
    }
    assert!(det, "CNN forward non deterministe");

    // 2. Round-trip SRT1 des poids conv
    let path = "cnn.srt";
    save_weights(&m.state_dict(), path).expect("save_weights");
    let file = std::fs::read(path).unwrap();
    let mut b = build_cnn(999);
    let f_pre = fp_of(&mut b, &x);
    assert_ne!(f_pre, f1, "B == A sans load : test vide");
    let sd = load_weights(path).expect("load_weights");
    b.load_state_dict(&sd).expect("load_state_dict");
    let f_post = fp_of(&mut b, &x);
    assert_eq!(f_post, f1, "reload CNN NON bit-exact");
    if bench
    {
        println!(
            "artefact CNN   : {} octets, hash {:#018x}",
            file.len(),
            fnv_bytes(&file)
        );
        println!("fp(B pre-load) : {:#018x} (different)", f_pre);
        println!("fp(B post-load): {:#018x} (== A bit-exact)", f_post);
    }

    println!("EMPREINTE CNN  = {:#018x}", f1);

    // 3. Latence sur archi lourde
    if bench
    {
        for _ in 0..5
        {
            let _ = fp_of(&mut m, &x);
        }
        let iters = 80usize;
        let mut lat: Vec<u64> = Vec::with_capacity(iters);
        for _ in 0..iters
        {
            let t = std::time::Instant::now();
            let _ = fp_of(&mut m, &x);
            lat.push(t.elapsed().as_nanos() as u64);
        }
        lat.sort_unstable();
        let us = |v: u64| v as f64 / 1000.0;
        let pc = |q: f64| lat[(((lat.len() - 1) as f64) * q).round() as usize];
        println!(
            "Latence CNN batch={} : p50 {:.1}us  p99 {:.1}us  p99/p50 {:.2}x  ({:.0} batch/s, {:.0} ech/s)",
            n,
            us(pc(0.5)),
            us(pc(0.99)),
            pc(0.99) as f64 / pc(0.5) as f64,
            1.0e9 / pc(0.5) as f64,
            1.0e9 / pc(0.5) as f64 * n as f64
        );
    }
}
