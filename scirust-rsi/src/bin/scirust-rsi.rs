//! `scirust-rsi` — a tiny CLI to run the bounded self-improvement loops on the
//! built-in benchmark objectives. Dependency-free (hand-rolled arg parsing),
//! seeded, and prints an auditable [`Report`].
//!
//! Examples:
//!   scirust-rsi es     --objective rastrigin --dim 4 --seed 7 --max-iters 5000
//!   scirust-rsi refine --objective sphere    --dim 6 --patience 50
//!   scirust-rsi pbt    --objective rosenbrock --pop 24 --steps 8 --json

use rand::Rng;
use rand::rngs::StdRng;
use scirust_rsi::adapters::FnRefine;
use scirust_rsi::evo::OnePlusLambda;
use scirust_rsi::pbt::{Pbt, PbtTask};
use scirust_rsi::refine::SelfRefiner;
use scirust_rsi::{Fitness, Guard, Report, bench};

const USAGE: &str = "\
scirust-rsi — bounded recursive self-improvement on benchmark objectives

USAGE:
    scirust-rsi <method> [options]

METHODS:
    es        (1+λ)-ES with the 1/5 success rule
    refine    Self-Refine (elitist hill-climb)
    pbt       Population-Based Training (neuro-evolution-style)

COMMON OPTIONS:
    --objective <sphere|rastrigin|rosenbrock>   default: sphere
    --dim <N>            problem dimensions          (default 5)
    --start <F>          initial value per coord     (default 3.0)
    --seed <S>           RNG seed                    (default 0)
    --max-iters <N>      iteration cap               (default 2000)
    --target <F>         stop when fitness >= F      (optional)
    --patience <N>       stop after N stalls         (default 0 = off)
    --time-ms <N>        wall-clock budget in ms     (optional)
    --json               print the Report as JSON
    -h, --help           this message

METHOD OPTIONS:
    es:   --lambda <N> (default 16)   --sigma0 <F> (default 1.0)
    pbt:  --pop <N> (default 16)       --steps <N> (default 4)   --sigma0 <F> (default 0.5)
";

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() || args[0] == "-h" || args[0] == "--help"
    {
        print!("{USAGE}");
        return;
    }
    let method = args[0].clone();
    let opts = Opts::parse(&args[1..]);

    let objective = opts.get("objective").unwrap_or_else(|| "sphere".into());
    let obj: fn(&[f64]) -> f64 = match objective.as_str()
    {
        "sphere" => bench::sphere,
        "rastrigin" => bench::rastrigin,
        "rosenbrock" => bench::rosenbrock,
        other =>
        {
            eprintln!("unknown objective '{other}' (sphere|rastrigin|rosenbrock)");
            std::process::exit(2);
        },
    };

    let dim = opts.usize_or("dim", 5);
    let start = opts.f64_or("start", 3.0);
    let seed = opts.u64_or("seed", 0);
    let guard = build_guard(&opts);
    let x0 = vec![start; dim];

    let (best, report) = match method.as_str()
    {
        "es" =>
        {
            let (x, _fit, report) = OnePlusLambda::new(seed)
                .lambda(opts.usize_or("lambda", 16))
                .sigma0(opts.f64_or("sigma0", 1.0))
                .optimize(x0, obj, &guard);
            (x, report)
        },
        "refine" =>
        {
            let step = opts.f64_or("sigma0", 0.5);
            let task = FnRefine::new(
                move |_rng: &mut StdRng| vec![start; dim],
                move |v: &Vec<f64>| obj(v),
                move |v: &Vec<f64>, rng: &mut StdRng| {
                    let mut out = v.clone();
                    let i = rng.gen_range(0..out.len());
                    out[i] += step * rng.gen_range(-1.0..1.0);
                    out
                },
            );
            SelfRefiner::new(seed).run(&task, &guard)
        },
        "pbt" =>
        {
            let task = CliPbt {
                obj,
                dim,
                start,
                sigma0: opts.f64_or("sigma0", 0.5),
            };
            let (x, _hyper, report) = Pbt::new(seed)
                .pop_size(opts.usize_or("pop", 16))
                .steps_per_gen(opts.usize_or("steps", 4))
                .run(&task, &guard);
            (x, report)
        },
        other =>
        {
            eprintln!("unknown method '{other}' (es|refine|pbt)\n");
            print!("{USAGE}");
            std::process::exit(2);
        },
    };

    if opts.flag("json")
    {
        println!("{}", report_json(&report));
    }
    else
    {
        println!("method     : {method}");
        println!("objective  : {objective}  (dim {dim}, seed {seed})");
        println!("best fitness: {:.6}", report.best_fitness);
        println!(
            "iterations : {}   accepted: {}   stop: {:?}   monotone: {}",
            report.iterations,
            report.accepted,
            report.stop_reason,
            report.is_monotone()
        );
        let head: Vec<String> = best.iter().take(8).map(|v| format!("{v:.4}")).collect();
        let tail = if best.len() > 8 { ", …" } else { "" };
        println!("best x     : [{}{}]", head.join(", "), tail);
    }
}

fn build_guard(opts: &Opts) -> Guard {
    let mut g = Guard::new()
        .max_iters(opts.usize_or("max-iters", 2000))
        .patience(opts.usize_or("patience", 0));
    if let Some(t) = opts.get("target").and_then(|s| s.parse::<Fitness>().ok())
    {
        g = g.target(t);
    }
    if let Some(ms) = opts.get("time-ms").and_then(|s| s.parse::<u64>().ok())
    {
        g = g.time_budget(std::time::Duration::from_millis(ms));
    }
    g
}

/// PBT over a continuous objective: each member runs a (1+1)-ES step at its own
/// self-tuned step size σ.
struct CliPbt {
    obj: fn(&[f64]) -> f64,
    dim: usize,
    start: f64,
    sigma0: f64,
}
impl PbtTask for CliPbt {
    type Hyper = f64;
    fn init_member(&self, rng: &mut StdRng) -> (Vec<f64>, f64) {
        let jitter = self.sigma0;
        let params = (0..self.dim)
            .map(|_| self.start + jitter * rng.gen_range(-1.0..1.0))
            .collect();
        (params, self.sigma0)
    }
    fn step(&self, params: &mut Vec<f64>, &sigma: &f64, rng: &mut StdRng) -> Fitness {
        let cur = (self.obj)(params);
        let cand: Vec<f64> = params
            .iter()
            .map(|p| p + sigma * rng.gen_range(-1.0..1.0))
            .collect();
        let cf = (self.obj)(&cand);
        if cf > cur
        {
            *params = cand;
            cf
        }
        else
        {
            cur
        }
    }
    fn perturb(&self, &sigma: &f64, rng: &mut StdRng) -> f64 {
        let factor = if rng.gen_bool(0.5) { 0.7 } else { 1.4 };
        (sigma * factor).clamp(1e-4, 5.0)
    }
}

/// Minimal `--key value` / `--flag` parser (no external deps).
struct Opts {
    map: std::collections::HashMap<String, String>,
    flags: std::collections::HashSet<String>,
}
impl Opts {
    fn parse(args: &[String]) -> Self {
        let mut map = std::collections::HashMap::new();
        let mut flags = std::collections::HashSet::new();
        let mut i = 0;
        while i < args.len()
        {
            let a = &args[i];
            if let Some(key) = a.strip_prefix("--")
            {
                if i + 1 < args.len() && !args[i + 1].starts_with("--")
                {
                    map.insert(key.to_string(), args[i + 1].clone());
                    i += 2;
                }
                else
                {
                    flags.insert(key.to_string());
                    i += 1;
                }
            }
            else
            {
                i += 1;
            }
        }
        Self { map, flags }
    }
    fn get(&self, k: &str) -> Option<String> {
        self.map.get(k).cloned()
    }
    fn flag(&self, k: &str) -> bool {
        self.flags.contains(k)
    }
    fn usize_or(&self, k: &str, d: usize) -> usize {
        self.map.get(k).and_then(|s| s.parse().ok()).unwrap_or(d)
    }
    fn u64_or(&self, k: &str, d: u64) -> u64 {
        self.map.get(k).and_then(|s| s.parse().ok()).unwrap_or(d)
    }
    fn f64_or(&self, k: &str, d: f64) -> f64 {
        self.map.get(k).and_then(|s| s.parse().ok()).unwrap_or(d)
    }
}

/// Hand-rolled compact JSON for a [`Report`] (avoids a serde_json dependency).
fn report_json(r: &Report) -> String {
    let hist: Vec<String> = r.history.iter().map(|v| format!("{v}")).collect();
    format!(
        "{{\"iterations\":{},\"accepted\":{},\"best_fitness\":{},\"stop_reason\":\"{:?}\",\"monotone\":{},\"history\":[{}]}}",
        r.iterations,
        r.accepted,
        r.best_fitness,
        r.stop_reason,
        r.is_monotone(),
        hist.join(",")
    )
}
