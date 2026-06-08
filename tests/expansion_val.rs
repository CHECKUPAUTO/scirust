use scirust_core::autodiff::reverse::{Tape, Tensor};
use scirust_core::nn::{PcgEngine, KaimingNormal, Zeros, Module, Linear, Sequential, ReLU, VAE, ViT, GCN};
use scirust_core::nn::vision::ResNet;
use scirust_core::nn::transformer::MoELayer;
use scirust_core::nn::audio::{AudioEncoder, CTCLoss};
use scirust_core::nn::peft::LoRALinear;
use scirust_learning::nlp::bpe::BpeTokenizer;
use scirust_learning::nlp::tokenization::Tokenizer;
use scirust_learning::rl::tabular::TabularAgent;
use scirust_learning::rl::deep::DQNAgent;
use scirust_learning::rl::ppo::PPOAgent;
use scirust_learning::time_series::nbeats::NBeatsBlock;
use scirust_solvers::scientific::FemSolver1D;
use scirust_core::autodiff::optim::Adam;

#[test]
fn test_bpe_integration() {
    let texts = vec!["the quick brown fox", "jumps over the lazy dog"];
    let tokenizer = BpeTokenizer::train(&texts, 50);
    let encoded = tokenizer.tokenize("the quick dog");
    assert!(encoded.len() > 0);
}

#[test]
fn test_resnet_forward() {
    let mut rng = PcgEngine::new(42);
    let mut model = ResNet::new(&[1, 1], 10, &KaimingNormal, &Zeros, &mut rng);
    let tape = Tape::new();
    let x = tape.input(Tensor::zeros(1, 48));
    let out = model.forward(&tape, x);
    assert_eq!(out.shape(), (1, 10));
}

#[test]
fn test_moe_forward() {
    let mut rng = PcgEngine::new(42);
    let mut moe = MoELayer::new(8, 4, 2,
        || Linear::new(8, 8, &KaimingNormal, &Zeros, &mut PcgEngine::new(0)),
        &KaimingNormal, &Zeros, &mut rng);
    let tape = Tape::new();
    let x = tape.input(Tensor::zeros(1, 8));
    let out = moe.forward(&tape, x);
    assert_eq!(out.shape(), (1, 8));
}

#[test]
fn test_tabular_rl_update() {
    let mut agent = TabularAgent::new(0.1, 0.9, 0.1);
    agent.update_q(&"state1", &"action1", 1.0, &"state2", &["action1", "action2"], false);
    assert!(agent.get_q(&"state1", &"action1") > 0.0);
}

#[test]
fn test_dqn_act() {
    let mut rng = PcgEngine::new(42);
    let model = Sequential::new()
        .add(Linear::new(4, 8, &KaimingNormal, &Zeros, &mut rng))
        .add(ReLU::new())
        .add(Linear::new(8, 2, &KaimingNormal, &Zeros, &mut rng));

    let mut rng2 = PcgEngine::new(43);
    let target_model = Sequential::new()
        .add(Linear::new(4, 8, &KaimingNormal, &Zeros, &mut rng2))
        .add(ReLU::new())
        .add(Linear::new(8, 2, &KaimingNormal, &Zeros, &mut rng2));

    let mut agent = DQNAgent::new(model, target_model, Adam::new(0.001), 0.99, 0.1, 32, 42);
    let state = Tensor::zeros(1, 4);
    let action = agent.act(&state, 2);
    assert!(action < 2);
}

#[test]
fn test_ppo_step() {
    let mut rng = PcgEngine::new(42);
    let actor = Linear::new(4, 2, &KaimingNormal, &Zeros, &mut rng);
    let critic = Linear::new(4, 1, &KaimingNormal, &Zeros, &mut rng);
    let mut agent = PPOAgent::new(actor, critic, 0.001, 0.001, 0.2);

    let state = Tensor::zeros(1, 4);
    agent.train_step(&[state], &[0], &[0.5], &[1.0], &[1.0]);
}

#[test]
fn test_vit_forward() {
    let mut rng = PcgEngine::new(42);
    let mut model = ViT::new(8, 4, 3, 10, 16, 2, 2, 32, &KaimingNormal, &Zeros, &mut rng);
    let tape = Tape::new();
    let x = tape.input(Tensor::zeros(1, 3*8*8));
    let out = model.forward(&tape, x);
    assert_eq!(out.shape(), (1, 10));
}

#[test]
fn test_vae_forward() {
    let mut rng = PcgEngine::new(42);
    let mut model = VAE::new(10, 8, 4, &KaimingNormal, &Zeros, &mut rng);
    let tape = Tape::new();
    let x = tape.input(Tensor::zeros(1, 10));
    let (recon, mu, logvar) = model.forward(&tape, x);
    assert_eq!(recon.shape(), (1, 10));
    assert_eq!(mu.shape(), (1, 4));
    assert_eq!(logvar.shape(), (1, 4));
}

#[test]
fn test_gcn_forward() {
    let mut rng = PcgEngine::new(42);
    let mut model = GCN::new(&[4, 8, 2], &KaimingNormal, &Zeros, &mut rng);
    let tape = Tape::new();
    let x = tape.input(Tensor::zeros(3, 4));
    let adj = tape.input(Tensor::from_vec(vec![1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0], 3, 3));
    let out = model.forward(&tape, x, adj);
    assert_eq!(out.shape(), (3, 2));
}

#[test]
fn test_audio_encoder_and_ctc() {
    let mut rng = PcgEngine::new(42);
    let mut model = AudioEncoder::new(1, 8, 16, &KaimingNormal, &Zeros, &mut rng);
    let tape = Tape::new();
    let x = tape.input(Tensor::zeros(1, 64));
    let logits = model.forward(&tape, x);

    let ctc = CTCLoss;
    let targets = tape.input(Tensor::zeros(1, 1));
    let loss = ctc.forward(&tape, logits, targets);
    assert!(!tape.value(loss.idx()).data[0].is_nan());
}

#[test]
fn test_nbeats_forward() {
    let mut rng = PcgEngine::new(42);
    let mut block = NBeatsBlock::new(10, 16, 5, &KaimingNormal, &Zeros, &mut rng);
    let tape = Tape::new();
    let x = tape.input(Tensor::zeros(1, 10));
    let (back, fore) = block.forward_both(&tape, x);
    assert_eq!(back.shape(), (1, 10));
    assert_eq!(fore.shape(), (1, 5));
}

#[test]
fn test_lora_linear() {
    let mut rng = PcgEngine::new(42);
    let mut lora = LoRALinear::new(8, 4, 2, 1.0, &KaimingNormal, &Zeros, &mut rng);
    let tape = Tape::new();
    let x = tape.input(Tensor::zeros(1, 8));
    let out = lora.forward(&tape, x);
    assert_eq!(out.shape(), (1, 4));
}

#[test]
fn test_fem_solver() {
    let solver = FemSolver1D::new(11, 1.0);
    let u = solver.solve_steady_heat(1.0);
    assert_eq!(u.len(), 11);
    assert_eq!(u[0], 0.0);
    assert_eq!(u[10], 0.0);
    assert!(u[5] > 0.0); // max temperature in the middle
}
