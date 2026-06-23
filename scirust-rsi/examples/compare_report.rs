//! Multi-run comparison report: run several `(1+λ)-ES` configurations on the
//! same objective, overlay their convergence curves, and write one
//! self-contained HTML file you can open in a browser.
//!
//! Demonstrates [`scirust_rsi::compare_html`] — the same data is also available
//! programmatically via each [`Report`](scirust_rsi::Report).
//!
//! Run with: `cargo run -p scirust-rsi --example compare_report`

use scirust_rsi::evo::OnePlusLambda;
use scirust_rsi::{Guard, bench, compare_html};

fn main() {
    // Same task, same seed, same budget — vary only the population size λ to see
    // how it changes the convergence curve.
    let guard = Guard::new().max_iters(400);
    let x0 = vec![3.0; 6];

    let runs_owned: Vec<(String, _)> = [4usize, 8, 16, 32]
        .into_iter()
        .map(|lambda| {
            let (_x, _fit, report) = OnePlusLambda::new(0xC0FFEE)
                .lambda(lambda)
                .sigma0(1.0)
                .optimize(x0.clone(), bench::rastrigin, &guard);
            (format!("λ = {lambda}"), report)
        })
        .collect();

    // Borrow into the (&str, &Report) shape `compare_html` expects.
    let runs: Vec<(&str, &_)> = runs_owned
        .iter()
        .map(|(label, r)| (label.as_str(), r))
        .collect();

    println!("=== (1+λ)-ES on rastrigin (dim 6, seed 0xC0FFEE) ===\n");
    println!("  {:<8} {:>12} {:>8}", "λ", "best fitness", "iters");
    for (label, r) in &runs
    {
        println!(
            "  {:<8} {:>12.5} {:>8}",
            label, r.best_fitness, r.iterations
        );
    }

    let html = compare_html("(1+λ)-ES on rastrigin — effect of λ", &runs);
    let path = "compare_report.html";
    match std::fs::write(path, &html)
    {
        Ok(()) => println!("\n  Wrote overlaid comparison chart to {path} — open it in a browser."),
        Err(e) => eprintln!("\n  failed to write {path}: {e}"),
    }
}
