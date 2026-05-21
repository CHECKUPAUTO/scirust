// Exemple : évaluer une "population" de 1000 MLP via le bridge.
//
// Cet exemple **anticipe** la phase 1 (`scirust-evo`). Il montre comment le
// bridge sera utilisé une fois qu'on aura un trait `Population<T>`.
//
// Pour l'instant, c'est juste : créer 1000 individus avec des seeds
// différents, les évaluer sur la même entrée, mesurer la diversité des
// sorties (proxy de fitness pour cet exemple).
//
// Lancer :
// ```bash
// cargo run --release -p scirust-burn-bridge --example eval_population
// ```

use burn::{
    backend::{ndarray::NdArrayDevice, NdArray},
    module::Module,
    nn::{Linear, LinearConfig, Tanh},
    tensor::{Tensor, TensorData},
};
use scirust_burn_bridge::{InferenceOnly, Policy};
use std::time::Instant;

type B = NdArray<f32>;

#[derive(Clone, Module, Debug)]
struct Individual<BB: burn::tensor::backend::Backend> {
    l1: Linear<BB>,
    l2: Linear<BB>,
    act: Tanh,
}

impl<BB: burn::tensor::backend::Backend> Individual<BB> {
    fn new(device: &BB::Device) -> Self {
        Self {
            l1: LinearConfig::new(4, 16).init(device),
            l2: LinearConfig::new(16, 1).init(device),
            act: Tanh::new(),
        }
    }
}

impl<BB: burn::tensor::backend::Backend> Policy<BB> for Individual<BB> {
    type Input = Tensor<BB, 2>;
    type Output = Tensor<BB, 2>;

    fn forward(&self, input: Tensor<BB, 2>) -> Tensor<BB, 2> {
        let x = self.l1.forward(input);
        let x = self.act.forward(x);
        self.l2.forward(x)
    }
}

/// Évalue un individu sur une entrée fixe et renvoie la valeur scalaire.
/// Sert de "fitness function" stand-in pour cet exemple.
fn eval_fitness(bridge: &InferenceOnly<B, Individual<B>>, input: &Tensor<B, 2>) -> f32 {
    let output = bridge.eval(input.clone());
    let v: Vec<f32> = output.into_data().to_vec().expect("to_vec");
    v[0]
}

fn main() {
    let device = NdArrayDevice::Cpu;

    println!("=== scirust-burn-bridge — exemple population ===");
    println!();

    // ── Créer la population ──────────────────────────────────────────
    let pop_size = 1000;
    print!("Création de {pop_size} individus... ");
    let start = Instant::now();
    let population: Vec<InferenceOnly<B, Individual<B>>> = (0..pop_size)
        .map(|_i| {
            let net = Individual::<B>::new(&device);
            InferenceOnly::new(net, device)
        })
        .collect();
    println!("done en {:?}", start.elapsed());

    // ── Évaluer toute la population ──────────────────────────────────
    let input: Tensor<B, 2> =
        Tensor::from_data(TensorData::from([[0.1f32, 0.5, -0.3, 0.8]]), &device);

    print!("Évaluation séquentielle de la population... ");
    let start = Instant::now();
    let fitnesses: Vec<f32> = population
        .iter()
        .map(|ind| eval_fitness(ind, &input))
        .collect();
    let elapsed = start.elapsed();
    println!("done en {:?}", elapsed);

    let throughput = pop_size as f64 / elapsed.as_secs_f64();
    println!("  → throughput : {throughput:.0} évaluations/s");

    // ── Statistiques de population ───────────────────────────────────
    let min = fitnesses.iter().copied().fold(f32::INFINITY, f32::min);
    let max = fitnesses.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mean: f32 = fitnesses.iter().sum::<f32>() / pop_size as f32;
    let std_dev: f32 = {
        let var = fitnesses.iter().map(|f| (f - mean).powi(2)).sum::<f32>() / pop_size as f32;
        var.sqrt()
    };

    println!();
    println!("--- Statistiques de fitness ---");
    println!("  min   = {min:>8.4}");
    println!("  max   = {max:>8.4}");
    println!("  mean  = {mean:>8.4}");
    println!("  std   = {std_dev:>8.4}");
    println!();

    // ── Top-5 ────────────────────────────────────────────────────────
    let mut indexed: Vec<(usize, f32)> = fitnesses.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    println!("--- Top-5 individus ---");
    for (rank, (idx, fit)) in indexed.iter().take(5).enumerate() {
        println!("  #{rank}: individu {idx:>4} → fitness = {fit:>8.4}");
    }
    println!();

    println!("---");
    println!("En phase 1, ceci sera remplacé par :");
    println!("  let pop = Population::random(1000);");
    println!("  let fitnesses = pop.par_eval(&input);  // rayon");
    println!("  let elites = pop.tournament_select(100);");
    println!("  let next_gen = elites.crossover_and_mutate(0.05);");
    println!();
    println!("Le bridge restera le canal d'évaluation, factorisé.");
}
