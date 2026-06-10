// Outil OFFLINE : entraine le MLP et fige les poids en artefact SRT1.
// Ne fait PAS partie du runtime d'inference (qui est inference-only).
use scirust_core::autodiff::optim::{Adam, Optimizer};
use scirust_core::autodiff::reverse::Tape;
use scirust_core::data::{DataLoader, MnistDataset};
use scirust_core::nn::{
    CrossEntropyLoss, KaimingNormal, Linear, Loss, Module, PcgEngine, ReLU, Sequential, Zeros,
};
use scirust_runtime::save_weights;

fn main() {
    let data_dir =
        std::env::var("MNIST_DIR").unwrap_or_else(|_| "/root/scirust/data/mnist".to_string());
    let train = MnistDataset::load_idx(
        format!("{}/train-images-idx3-ubyte", data_dir),
        format!("{}/train-labels-idx1-ubyte", data_dir),
    )
    .expect("chargement train MNIST");
    let max_train = std::env::var("MNIST_MAX_TRAIN")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(train.n);
    let train_small = train.subsample(max_train);
    println!("Entrainement sur {} images", train_small.n_samples());

    let mut rng = PcgEngine::new(42);
    let mut model = Sequential::new()
        .add(Linear::new(784, 256, &KaimingNormal, &Zeros, &mut rng))
        .add(ReLU::new())
        .add(Linear::new(256, 10, &KaimingNormal, &Zeros, &mut rng));

    let loss_fn = CrossEntropyLoss::new();
    let mut opt = Adam::new(0.001);
    let mut loader = DataLoader::new(train_small, 64, true, 42);
    let n_epochs = 5;

    for epoch in 0..n_epochs
    {
        let mut epoch_loss = 0.0;
        let mut n = 0;
        loader.shuffle_epoch(epoch as u64);
        for (xb, yb) in loader.iter()
        {
            let tape = Tape::new();
            let x = tape.input(xb);
            let t = tape.input(yb);
            let logits = model.forward(&tape, x);
            let loss = loss_fn.forward(&tape, logits, t);
            tape.backward(loss.idx());
            opt.step(&model.parameter_indices(), &tape);
            model.sync(&tape);
            epoch_loss += tape.value(loss.idx()).data[0];
            n += 1;
        }
        println!(
            "  epoch {:>2} : loss = {:.4}",
            epoch + 1,
            epoch_loss / n as f32
        );
    }

    save_weights(&model.state_dict(), "mnist_mlp.srt").expect("save_weights");
    println!("Artefact fige -> mnist_mlp.srt");
}
