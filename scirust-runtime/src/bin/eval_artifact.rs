// RUNTIME d'inference : charge un artefact fige, evalue accuracy + empreinte.
use scirust_core::autodiff::reverse::Tape;
use scirust_core::data::{DataLoader, MnistDataset};
use scirust_core::nn::{KaimingNormal, Linear, Module, PcgEngine, ReLU, Sequential, Zeros};
use scirust_runtime::{fnv_fold_f32, fnv_init, load_weights};

fn main() {
    let data_dir =
        std::env::var("MNIST_DIR").unwrap_or_else(|_| "/root/scirust/data/mnist".to_string());
    let test = MnistDataset::load_idx(
        format!("{}/t10k-images-idx3-ubyte", data_dir),
        format!("{}/t10k-labels-idx1-ubyte", data_dir),
    )
    .expect("chargement test MNIST");

    // Modele frais (seed quelconque : les poids sont ecrases par le load)
    let mut rng = PcgEngine::new(123);
    let mut model = Sequential::new()
        .add(Linear::new(784, 256, &KaimingNormal, &Zeros, &mut rng))
        .add(ReLU::new())
        .add(Linear::new(256, 10, &KaimingNormal, &Zeros, &mut rng));

    let sd = load_weights("mnist_mlp.srt").expect("load_weights (lance train_artifact d'abord)");
    model.load_state_dict(&sd).expect("load_state_dict");

    let mut loader = DataLoader::new(test.subsample(test.n), 64, false, 42);
    let mut correct = 0usize;
    let mut total = 0usize;
    let mut fp = fnv_init();

    for (xb, yb) in loader.iter() {
        let tape = Tape::new();
        let v = tape.input(xb);
        let logits = model.forward(&tape, v);
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

    println!(
        "Accuracy (artefact fige) : {:.2}% ({}/{})",
        correct as f32 / total as f32 * 100.0,
        correct,
        total
    );
    println!(
        "Empreinte test logits    : {:#018x}",
        fp
    );
    println!("  (relance le binaire : meme empreinte => inference reproductible inter-process)");
}
