// AUDIT evo : oracle = optimum connu de fonctions de test (min = 0) + front Pareto connu (ZDT1).
use scirust_evo::*;

fn rosenbrock(x: &[f64]) -> f64 {
    (0..x.len().saturating_sub(1))
        .map(|i| 100.0 * (x[i + 1] - x[i] * x[i]).powi(2) + (1.0 - x[i]).powi(2))
        .sum()
}
fn ackley(x: &[f64]) -> f64 {
    let n = x.len() as f64;
    let s1: f64 = x.iter().map(|v| v * v).sum();
    let s2: f64 = x
        .iter()
        .map(|v| (2.0 * std::f64::consts::PI * v).cos())
        .sum();
    -20.0 * (-0.2 * (s1 / n).sqrt()).exp() - (s2 / n).exp() + 20.0 + std::f64::consts::E
}

fn ga_min(f: fn(&[f64]) -> f64, dims: usize, gens: usize) -> f64 {
    let ga = GeneticAlgorithm::default();
    let mut pop = ga.init_pop(dims);
    for _ in 0..gens
    {
        ga.evolve(&mut pop, |inds| {
            inds.iter().map(|ind| -f(&ind.genome)).collect()
        });
    }
    pop.iter()
        .map(|ind| f(&ind.genome))
        .fold(f64::INFINITY, f64::min)
}
fn cmaes_min(f: fn(&[f64]) -> f64, dims: usize, steps: usize) -> f64 {
    let mut cma = CmaEs::new(dims);
    let mut theta = vec![2.0; dims];
    let mut best = f(&theta);
    for _ in 0..steps
    {
        let off = cma.step(&mut theta, |x| -f(x));
        for ind in &off
        {
            let v = f(&ind.genome);
            if v < best
            {
                best = v;
            }
        }
        let v = f(&theta);
        if v < best
        {
            best = v;
        }
    }
    best
}
fn openes_min(f: fn(&[f64]) -> f64, dims: usize, steps: usize) -> f64 {
    let openes = OpenEs::new(dims);
    let mut theta = vec![2.0; dims];
    let mut best = f(&theta);
    for _ in 0..steps
    {
        let _ = openes.step(&mut theta, |x| -f(x));
        let v = f(&theta);
        if v < best
        {
            best = v;
        }
    }
    best
}
fn stats(runs: &[f64]) -> (f64, f64) {
    let best = runs.iter().cloned().fold(f64::INFINITY, f64::min);
    let mean = runs.iter().sum::<f64>() / runs.len() as f64;
    (best, mean)
}
fn zdt1_gd(dims: usize, gens: usize) -> (f64, f64, usize) {
    let mut nsga = Nsga2::default();
    nsga.bounds = (0.0, 1.0);
    let mut pop = nsga.init_pop(dims);
    for _ in 0..gens
    {
        nsga.evolve(&mut pop, |inds| {
            inds.iter()
                .map(|ind| {
                    let x = &ind.genome;
                    let f1 = x[0].clamp(0.0, 1.0);
                    let g = 1.0 + 9.0 * x[1..].iter().sum::<f64>() / (x.len() as f64 - 1.0);
                    let h = 1.0 - (f1 / g).sqrt();
                    vec![f1, g * h]
                })
                .collect()
        });
    }
    let front: Vec<&MoIndividual> = pop.iter().filter(|i| i.rank == 1).collect();
    let n = front.len().max(1);
    let (mut sum, mut mx) = (0.0f64, 0.0f64);
    for ind in &front
    {
        let (f1, f2) = (ind.objectives[0], ind.objectives[1]);
        let d = (f2 - (1.0 - f1.sqrt())).abs();
        sum += d;
        if d > mx
        {
            mx = d;
        }
    }
    (sum / n as f64, mx, front.len())
}

fn main() {
    let (dims, runs) = (10usize, 5usize);
    let funcs: [(&str, fn(&[f64]) -> f64); 4] = [
        ("sphere", sphere),
        ("rastrigin", rastrigin),
        ("rosenbrock", rosenbrock),
        ("ackley", ackley),
    ];
    println!();
    println!(
        "=== AUDIT evo : oracle = optimum connu (min=0), dim={}, {} runs ===",
        dims, runs
    );
    println!(
        "{:<11} {:>7}  {:>21}  {:>21}  {:>21}",
        "fonction", "optimum", "GA best/mean", "CMA-ES best/mean", "OpenES best/mean"
    );
    for &(name, f) in &funcs
    {
        let ga: Vec<f64> = (0..runs).map(|_| ga_min(f, dims, 200)).collect();
        let cma: Vec<f64> = (0..runs).map(|_| cmaes_min(f, dims, 200)).collect();
        let oes: Vec<f64> = (0..runs).map(|_| openes_min(f, dims, 400)).collect();
        let ((gb, gm), (cb, cm), (ob, om)) = (stats(&ga), stats(&cma), stats(&oes));
        println!(
            "{:<11} {:>7.1}  {:>9.2e}/{:>9.2e}  {:>9.2e}/{:>9.2e}  {:>9.2e}/{:>9.2e}",
            name, 0.0, gb, gm, cb, cm, ob, om
        );
    }
    println!();
    let (gd_mean, gd_max, fs) = zdt1_gd(dims, 250);
    println!(
        "NSGA-II sur ZDT1 (front connu f2=1-sqrt(f1)) : front rank-1 = {} points",
        fs
    );
    println!(
        "  distance au front connu : moyenne={:.3e}  max={:.3e}",
        gd_mean, gd_max
    );
    println!();
    println!("Note: optimiseurs a graine fixe (StdRng seedable) -> deterministes/reproductibles;");
    println!("les {} runs partagent la meme graine par defaut (utiliser *::seeded(s) pour varier).", runs);
}
