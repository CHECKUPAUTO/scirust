// Palier 2 : runtime generique. Manifeste texte + SRT1 -> n'importe quel Sequential.
use scirust_core::autodiff::reverse::{Tape, Tensor};
use scirust_core::data::{DataLoader, MnistDataset};
use scirust_core::nn::{
    Conv2d, KaimingNormal, Linear, MaxPool2d, Module, Padding, PcgEngine, ReLU, Sequential, Zeros,
};
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

fn build_cnn_hardcoded(seed: u64) -> Sequential {
    let mut rng = PcgEngine::new(seed);
    Sequential::new()
        .add(Conv2d::new(3, 32, 3, 1, Padding::Same, &KaimingNormal, Some(&Zeros), &mut rng).input_dims(32, 32))
        .add(ReLU::new())
        .add(MaxPool2d::new(2, 2).input_shape(32, 32, 32))
        .add(Conv2d::new(32, 64, 3, 1, Padding::Same, &KaimingNormal, Some(&Zeros), &mut rng).input_dims(16, 16))
        .add(ReLU::new())
        .add(MaxPool2d::new(2, 2).input_shape(64, 16, 16))
        .add(Linear::new(4096, 256, &KaimingNormal, &Zeros, &mut rng))
        .add(ReLU::new())
        .add(Linear::new(256, 10, &KaimingNormal, &Zeros, &mut rng))
}

fn fp_forward(model: &mut Sequential, x: &Tensor) -> u64 {
    let tape = Tape::new();
    let v = tape.input(x.clone());
    let logits = model.forward(&tape, v);
    fnv_fold_f32(fnv_init(), &tape.value(logits.idx()).data)
}

fn main() {
    // 1. Equivalence CNN : hardcode vs reconstruit depuis manifeste
    let x = synth(32, 3 * 32 * 32, 12345);
    let mut href = build_cnn_hardcoded(42);
    let f_ref = fp_forward(&mut href, &x);
    save_weights(&href.state_dict(), "cnn.srt").expect("save cnn");

    let cnn_spec = vec![
        LayerSpec::Conv2d { in_c: 3, out_c: 32, kernel: 3, stride: 1, same: true, in_h: 32, in_w: 32 },
        LayerSpec::Relu,
        LayerSpec::MaxPool2d { kernel: 2, stride: 2, c: 32, h: 32, w: 32 },
        LayerSpec::Conv2d { in_c: 32, out_c: 64, kernel: 3, stride: 1, same: true, in_h: 16, in_w: 16 },
        LayerSpec::Relu,
        LayerSpec::MaxPool2d { kernel: 2, stride: 2, c: 64, h: 16, w: 16 },
        LayerSpec::Linear { in_f: 4096, out_f: 256 },
        LayerSpec::Relu,
        LayerSpec::Linear { in_f: 256, out_f: 10 },
    ];
    std::fs::write("cnn.manifest", write_manifest(&cnn_spec)).unwrap();
    let parsed = parse_manifest(&std::fs::read_to_string("cnn.manifest").unwrap()).expect("parse cnn");
    let mut gen = build_model(&parsed);
    gen.load_state_dict(&load_weights("cnn.srt").expect("load cnn")).expect("lsd cnn");
    let f_gen = fp_forward(&mut gen, &x);
    println!("== CNN : hardcode vs manifeste ==");
    println!("  hardcode   : {:#018x}", f_ref);
    println!("  manifeste  : {:#018x}", f_gen);
    println!("  equivalent : {}", f_ref == f_gen);
    assert_eq!(f_ref, f_gen, "reconstruction CNN par manifeste NON equivalente");

    // 2. Coup de grace : MLP MNIST entraine reconstruit depuis manifeste
    let mlp_spec = vec![
        LayerSpec::Linear { in_f: 784, out_f: 256 },
        LayerSpec::Relu,
        LayerSpec::Linear { in_f: 256, out_f: 10 },
    ];
    std::fs::write("mnist.manifest", write_manifest(&mlp_spec)).unwrap();
    let parsed_mlp = parse_manifest(&std::fs::read_to_string("mnist.manifest").unwrap()).expect("parse mlp");
    let mut mlp = build_model(&parsed_mlp);
    let sd = load_weights("mnist_mlp.srt")
        .expect("mnist_mlp.srt absent : lance 'cargo run --release --bin train_artifact'");
    mlp.load_state_dict(&sd).expect("lsd mlp");

    let data_dir = std::env::var("MNIST_DIR").unwrap_or_else(|_| "/root/scirust/data/mnist".to_string());
    let test = MnistDataset::load_idx(
        format!("{}/t10k-images-idx3-ubyte", data_dir),
        format!("{}/t10k-labels-idx1-ubyte", data_dir),
    )
    .expect("chargement test MNIST");
    let mut loader = DataLoader::new(test.subsample(test.n), 64, false, 42);
    let mut correct = 0usize;
    let mut total = 0usize;
    let mut fp = fnv_init();
    for (xb, yb) in loader.iter() {
        let tape = Tape::new();
        let v = tape.input(xb);
        let logits = mlp.forward(&tape, v);
        let scores = tape.value(logits.idx());
        fp = fnv_fold_f32(fp, &scores.data);
        let (bs, _) = scores.shape();
        for i in 0..bs {
            let mut best = scores.data[i * 10];
            let mut pc = 0usize;
            for c in 1..10 {
                if scores.data[i * 10 + c] > best {
                    best = scores.data[i * 10 + c];
                    pc = c;
                }
            }
            let tc = yb.data[i * 10..(i + 1) * 10]
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(j, _)| j)
                .unwrap_or(0);
            if pc == tc {
                correct += 1;
            }
            total += 1;
        }
    }
    println!("== MLP MNIST entraine, reconstruit depuis manifeste ==");
    println!("  accuracy   : {:.2}% ({}/{})", correct as f32 / total as f32 * 100.0, correct, total);
    println!("  empreinte  : {:#018x}  (cible 0xc96d25fa658f5611 : {})", fp, fp == 0xc96d25fa658f5611);
    assert_eq!(fp, 0xc96d25fa658f5611, "MLP reconstruit ne reproduit pas l'empreinte entrainee");

    println!("\nOK : runtime generique. Manifeste + SRT1 -> n'importe quel modele, bit-exact.");
}
