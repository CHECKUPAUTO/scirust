use scirust_core::autodiff::optim::Adam;
use scirust_core::autodiff::reverse::{Tape, Tensor};
use scirust_core::nn::audio::{AudioEncoder, CTCLoss};
use scirust_core::nn::peft::LoRALinear;
use scirust_core::nn::transformer::MoELayer;
use scirust_core::nn::vision::ResNet;
use scirust_core::nn::{
    GCN, KaimingNormal, Linear, Module, PcgEngine, ReLU, Sequential, VAE, ViT, Zeros,
};
use scirust_learning::nlp::bpe::BpeTokenizer;
use scirust_learning::nlp::tokenization::Tokenizer;
use scirust_learning::rl::deep::DQNAgent;
use scirust_learning::rl::ppo::PPOAgent;
use scirust_learning::rl::tabular::TabularAgent;
use scirust_learning::time_series::nbeats::NBeatsBlock;
use scirust_solvers::scientific::FemSolver1D;

#[test]
fn test_bpe_integration() {
    let texts = vec!["the quick brown fox", "jumps over the lazy dog"];
    let tokenizer = BpeTokenizer::train(&texts, 50);
    let encoded = tokenizer.tokenize("the quick dog");
    assert!(!encoded.is_empty());
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
    let mut moe = MoELayer::new(
        8,
        4,
        2,
        || Linear::new(8, 8, &KaimingNormal, &Zeros, &mut PcgEngine::new(0)),
        &KaimingNormal,
        &Zeros,
        &mut rng,
    );
    let tape = Tape::new();
    let x = tape.input(Tensor::zeros(1, 8));
    let out = moe.forward(&tape, x);
    assert_eq!(out.shape(), (1, 8));
}

#[test]
fn test_tabular_rl_update() {
    let mut agent = TabularAgent::new(0.1, 0.9, 0.1);
    agent.update_q(
        &"state1",
        &"action1",
        1.0,
        &"state2",
        &["action1", "action2"],
        false,
    );
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
    let x = tape.input(Tensor::zeros(1, 3 * 8 * 8));
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

    // KL loss must be callable during VAE training
    let kl = model.kl_loss(&tape, mu, logvar);
    let kl_val = tape.value(kl.idx());
    assert!(kl_val.data[0].is_finite());
}

#[test]
fn test_vae_kl_loss_shapes() {
    // Verify kl_loss accepts batched mu/logvar
    let mut rng = PcgEngine::new(42);
    let mut model = VAE::new(20, 12, 6, &KaimingNormal, &Zeros, &mut rng);
    let tape = Tape::new();
    let x = tape.input(Tensor::zeros(3, 20)); // batch of 3
    let (_recon, mu, logvar) = model.forward(&tape, x);
    let kl = model.kl_loss(&tape, mu, logvar);
    let kl_val = tape.value(kl.idx());
    // KL divergence should be a scalar (1x1)
    assert_eq!(kl_val.shape(), (1, 1));
    assert!(kl_val.data[0].is_finite());
}

#[test]
fn test_resnet18_resnet34_constructors() {
    // resnet18 (block_counts=[2,2,2,2]) and resnet34 ([3,4,6,3])
    // both have 4 stages: 64 → 128 → 256 → 512
    let rn18 = ResNet::resnet18(10, &KaimingNormal, &Zeros, &mut PcgEngine::new(42));
    assert_eq!(rn18.out_channels, 512);

    let rn34 = ResNet::resnet34(10, &KaimingNormal, &Zeros, &mut PcgEngine::new(43));
    assert_eq!(rn34.out_channels, 512);
}

#[test]
fn test_infer_step_with_kv_cache() {
    use scirust_core::nn::transformer::MultiHeadAttention;
    let mut rng = PcgEngine::new(42);
    let mut mha = MultiHeadAttention::new(8, 4, 4, false, &KaimingNormal, &Zeros, &mut rng);
    let tape = Tape::new();
    // Single token input: (1, d_model)
    let token = tape.input(Tensor::zeros(1, 8));
    let out = mha.infer_step(&tape, token, 0);
    assert_eq!(out.shape(), (1, 8));
}

#[test]
fn test_flash_attention_forward_basic() {
    use scirust_core::nn::transformer::flash_attention::flash_attention_forward;
    let tape = Tape::new();
    let d_head = 8;
    let seq_len = 4;
    let n_heads = 2;
    let batch = 1;
    let total = batch * n_heads * seq_len;
    let q = tape.input(Tensor::zeros(total, d_head));
    let k = tape.input(Tensor::zeros(total, d_head));
    let v = tape.input(Tensor::zeros(total, d_head));
    let scale = 1.0 / (d_head as f32).sqrt();
    let out = flash_attention_forward(
        &tape, q, k, v, batch, n_heads, seq_len, d_head, scale, 2, false,
    );
    assert_eq!(out.shape(), (total, d_head));
}

#[test]
fn test_pinn_total_loss() {
    use scirust_core::nn::loss::pinn::PinnLossEvaluator;
    use scirust_core::nn::{Linear, Sequential};
    let mut rng = PcgEngine::new(42);
    let mut model = Sequential::new()
        .add(Linear::new(2, 8, &KaimingNormal, &Zeros, &mut rng))
        .add(ReLU::new())
        .add(Linear::new(8, 1, &KaimingNormal, &Zeros, &mut rng));
    let mut pinn = PinnLossEvaluator::new(&mut model, 0.01);
    let tape = Tape::new();
    // coords: (batch, 2) where col0=x, col1=t
    let coords = tape.input(Tensor::zeros(4, 2));
    let targets = tape.input(Tensor::zeros(4, 1));
    let loss = pinn.total_loss(&tape, coords, targets, 0.1);
    let loss_val = tape.value(loss.idx());
    assert!(loss_val.data[0].is_finite());
}

#[test]
fn test_epsilon_greedy_explores_and_exploits() {
    use rand::SeedableRng;
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let agent = TabularAgent::new(0.1, 0.9, 0.5);
    // With epsilon=0.5, both actions should be reachable over many tries
    let actions = vec!["left", "right"];
    let mut counts = std::collections::HashMap::new();
    for _ in 0..500
    {
        let a = agent
            .act_epsilon_greedy(&"s", &actions, &mut rng)
            .expect("action set is non-empty");
        *counts.entry(a).or_insert(0) += 1;
    }
    assert!(counts.contains_key(&"left"));
    assert!(counts.contains_key(&"right"));
}

#[test]
fn test_gcn_forward() {
    let mut rng = PcgEngine::new(42);
    let mut model = GCN::new(&[4, 8, 2], &KaimingNormal, &Zeros, &mut rng);
    let tape = Tape::new();
    let x = tape.input(Tensor::zeros(3, 4));
    let adj = tape.input(Tensor::from_vec(
        vec![1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0],
        3,
        3,
    ));
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
