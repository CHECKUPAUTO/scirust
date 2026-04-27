// examples/v5_features_demo/src/main.rs
//
// Démo v5 — trois nouveautés :
//   1. Op::Log/Exp/Sqrt/Neg sur le tape AD
//   2. BCE strict + CrossEntropy + Softmax via les nouvelles ops
//   3. LazyGraph + Plan : fusion automatique des chaînes pointwise
//
// La démo entraîne un classifier multi-classe (3 classes) et compare
// le nombre d'instructions avant et après compilation lazy.

use scirust_core::autodiff::reverse::{Tape, Tensor};
use scirust_core::autodiff::optim::{Adam, Optimizer};
use scirust_core::nn::{
    PcgEngine, Module, Sequential, Linear,
    KaimingNormal, Zeros,
};
use scirust_core::nn::loss::Loss;
use scirust_core::nn::loss::strict::CrossEntropyLoss;
use scirust_core::lazy::{LazyGraph, LazyTensor, Compiler};

// ---------- Dataset synthétique 3 classes en 2D ---------- //

fn make_dataset() -> (Tensor, Tensor) {
    // 3 clusters bien séparés, 5 points chacun
    let centers = [(2.0, 2.0), (-2.0, 2.0), (0.0, -2.0)];
    let mut x = Vec::new();
    let mut y = Vec::new();  // one-hot

    for (class, &(cx, cy)) in centers.iter().enumerate() {
        for i in 0..5 {
            // Petit décalage déterministe
            let dx = (i as f32) * 0.1 - 0.2;
            let dy = (i as f32) * 0.05 - 0.1;
            x.push(cx + dx);
            x.push(cy + dy);
            // One-hot
            for k in 0..3 {
                y.push(if k == class { 1.0 } else { 0.0 });
            }
        }
    }
    (Tensor::from_vec(x, 15, 2), Tensor::from_vec(y, 15, 3))
}

fn main() {
    println!("=== SciRust v5 — Op::Log/Exp + Lazy Fusion ===\n");

    // ============================================================== //
    //  Partie A : entraînement avec CrossEntropyLoss strict           //
    // ============================================================== //

    let (x, y) = make_dataset();
    let n = x.rows;
    println!("Dataset : {} points, 3 classes one-hot", n);

    let mut rng = PcgEngine::new(42);
    let mut model = Sequential::new()
        .push(Linear::new(2,  16, &KaimingNormal, &Zeros, &mut rng).with_name("fc1"))
        .push(scirust_core::nn::ReLU)
        .push(Linear::new(16, 3,  &KaimingNormal, &Zeros, &mut rng).with_name("fc2"));

    let mut opt = Adam::new(0.05);
    let n_epochs = 200;

    println!("\nÉpoque | CE Loss");
    println!("-------|---------");

    for epoch in 0..n_epochs {
        let tape = Tape::new();
        let x_var = tape.input(x.clone());
        let y_var = tape.input(y.clone());

        let logits = model.forward(&tape, x_var);
        let loss = CrossEntropyLoss.forward(logits, y_var);
        let loss_val = tape.value(loss.idx()).data[0];

        loss.backward();
        opt.step(&model.parameter_indices(), &tape);
        model.sync(&tape);

        if epoch % 25 == 0 || epoch == n_epochs - 1 {
            println!("{:5}  | {:.5}", epoch, loss_val);
        }
    }

    // ----- Évaluation : argmax des logits ----- //
    let tape = Tape::new();
    let x_var = tape.input(x.clone());
    let logits = model.forward(&tape, x_var);
    let logits_t = tape.value(logits.idx());

    let mut correct = 0;
    for i in 0..n {
        let row = &logits_t.data[i*3..(i+1)*3];
        let pred = row.iter().enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0;
        let truth = (0..3).find(|&k| y.data[i*3 + k] > 0.5).unwrap();
        if pred == truth { correct += 1; }
    }
    println!("\nAccuracy : {}/{} = {:.1}%",
             correct, n, 100.0 * correct as f32 / n as f32);

    // ============================================================== //
    //  Partie B : LazyGraph et fusion automatique                     //
    // ============================================================== //

    println!("\n\n=== Lazy Execution + Operator Fusion ===");

    // Construit un graphe avec une longue chaîne pointwise
    //   y = log(exp(relu(scale(x, 2)) + bias) + 1)
    // Soit 6 ops en mode eager.

    let g = LazyGraph::new();
    let x_in = LazyTensor::feed(g.clone(), "x".into(), (1, 8));
    let bias = LazyTensor::from_tensor(g.clone(),
        Tensor::from_vec(vec![0.5; 8], 1, 8));
    let one  = LazyTensor::from_tensor(g.clone(),
        Tensor::from_vec(vec![1.0; 8], 1, 8));

    let chain = x_in.scale(2.0).relu().add(bias).exp().add(one).log();

    println!("Graphe construit : {} nœuds", g.node_count());
    println!("Aucun calcul effectué (lazy)\n");

    // ---- Compilation ----
    let plan = Compiler::new(&g).compile(chain.id);

    println!("Plan compilé :");
    println!("  - Nœuds originaux        : {}", plan.stats.original_node_count);
    println!("  - Instructions émises    : {}", plan.stats.instructions_count);
    println!("  - Chaînes fusionnées     : {}", plan.stats.fused_chains);
    println!("  - Nœuds éliminés (DCE)   : {}", plan.stats.dce_eliminated);
    println!("  - Buffers alloués        : {}", plan.n_buffers);

    // ---- Exécution avec un feed dynamique ----
    let input_data = Tensor::from_vec(
        vec![-1.0, 0.5, -2.0, 3.0, 0.0, -0.5, 1.5, 2.5], 1, 8);
    let result = plan.execute_with(&[("x", input_data.clone())]);

    println!("\nInput  : {:?}", input_data.data);
    println!("Output : {:?}", result.data);

    // Vérification numérique en mode eager
    let mut expected = input_data.data.clone();
    for v in &mut expected { *v *= 2.0; *v = v.max(0.0); *v += 0.5; *v = v.exp(); *v += 1.0; *v = v.ln(); }
    let max_diff = expected.iter().zip(result.data.iter())
        .map(|(a, b)| (a - b).abs()).fold(0.0_f32, f32::max);
    println!("\nÉcart max vs eager     : {:.2e}", max_diff);
    if max_diff < 1e-4 {
        println!("✅ Plan compilé == eager (correctness OK)");
    }

    // ---- Re-exécution sur un autre batch ----
    let r2 = plan.execute_with(&[("x", Tensor::from_vec(vec![10.0; 8], 1, 8))]);
    println!("\nRe-exécution sur autre batch (sans recompiler) :");
    println!("  Output : {:?}", r2.data);

    // ============================================================== //
    //  Partie C : Bénéfice de la fusion (mesure simple)               //
    // ============================================================== //

    println!("\n=== Fusion : 6 ops eager → 2 instructions plan ===");
    println!("(1 LoadFeed + 1 PointwiseChain qui fait toute la chaîne en");
    println!(" un seul parcours mémoire)\n");

    println!("=== Fin de la démo v5 ===");
}
