//! Reasoning subcommands: symbolic regression (`symreg`, over
//! `scirust-symreg`) and propositional SAT (`sat`, over
//! `scirust-neuro-symbolic`). Both are deterministic in their input/seed.

use scirust_neuro_symbolic::sat_smt::SatSolver;
use scirust_symbolic::simplify;

fn parse_list(s: &str, what: &str) -> Result<Vec<f64>, u8> {
    s.split(',')
        .map(|t| {
            t.trim().parse::<f64>().map_err(|_| {
                eprintln!("error: `{}` is not a number ({what})", t.trim());
                2u8
            })
        })
        .collect()
}

/// `symreg "<xs>" "<ys>" [--seed N]` — discover a closed-form `y = f(x)`
/// from points by genetic programming + symbolic constant fitting.
pub fn run_symreg(args: &[String]) -> u8 {
    // Optional --seed.
    let mut seed = 42u64;
    let mut pos: Vec<&String> = Vec::new();
    let mut i = 0;
    while i < args.len()
    {
        if args[i] == "--seed" && i + 1 < args.len()
        {
            seed = args[i + 1].parse().unwrap_or(42);
            i += 2;
        }
        else
        {
            pos.push(&args[i]);
            i += 1;
        }
    }
    let (Some(xs_s), Some(ys_s)) = (pos.first(), pos.get(1))
    else
    {
        eprintln!("usage: scirust symreg <x1,x2,..> <y1,y2,..> [--seed N]");
        return 2;
    };
    let xs = match parse_list(xs_s, "x")
    {
        Ok(v) => v,
        Err(c) => return c,
    };
    let ys = match parse_list(ys_s, "y")
    {
        Ok(v) => v,
        Err(c) => return c,
    };
    if xs.len() != ys.len() || xs.is_empty()
    {
        eprintln!("error: xs and ys must be non-empty and the same length");
        return 2;
    }

    let data: Vec<(Vec<f64>, f64)> = xs.iter().zip(&ys).map(|(&x, &y)| (vec![x], y)).collect();
    // Modest GP budget: fast, deterministic, enough for low-order laws.
    let front = scirust_symreg::discover(&data, &["x"], &[seed], 60, 20, 40, 9);
    match front.iter().min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
    {
        Some((size, err, expr)) =>
        {
            println!("best fit: y = {}", simplify(expr));
            println!("  error (MSE): {err:.6}   complexity: {size}");
            println!("  Pareto front: {} candidate(s)", front.len());
            0
        },
        None =>
        {
            println!("no expression discovered");
            1
        },
    }
}

/// `sat "<clauses>"` — DPLL satisfiability. Clauses are `;`-separated, each
/// a `,`-separated list of integer literals (`-v` = ¬v, 1-indexed). Exit 0
/// if SAT (prints a model), 1 if UNSAT.
pub fn run_sat(args: &[String]) -> u8 {
    let Some(spec) = args.first()
    else
    {
        eprintln!("usage: scirust sat \"1,-2;2,3;-1\"   (clauses ;  literals ,  -v = not v)");
        return 2;
    };
    let mut solver = SatSolver::new();
    for clause_s in spec.split(';')
    {
        let mut clause = Vec::new();
        for lit in clause_s.split(',')
        {
            match lit.trim().parse::<i32>()
            {
                Ok(0) | Err(_) =>
                {
                    eprintln!("error: `{}` is not a non-zero literal", lit.trim());
                    return 2;
                },
                Ok(v) => clause.push(v),
            }
        }
        if !clause.is_empty()
        {
            solver.add_clause(clause);
        }
    }
    match solver.solve()
    {
        Ok(Some(model)) =>
        {
            let assignment: Vec<String> = model
                .iter()
                .enumerate()
                .map(|(i, &b)| format!("{}{}", if b { "" } else { "-" }, i + 1))
                .collect();
            println!("SAT  model: {{ {} }}", assignment.join(", "));
            0
        },
        Ok(None) =>
        {
            println!("UNSAT");
            1
        },
        Err(e) =>
        {
            eprintln!("error: {e}");
            2
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn s(v: &[&str]) -> Vec<String> {
        v.iter().map(|x| x.to_string()).collect()
    }

    #[test]
    fn symreg_runs_and_is_deterministic() {
        // y = 2x on a few points; just require it runs and is reproducible.
        let a = run_symreg(&s(&["0,1,2,3", "0,2,4,6", "--seed", "1"]));
        let b = run_symreg(&s(&["0,1,2,3", "0,2,4,6", "--seed", "1"]));
        assert_eq!(a, 0);
        assert_eq!(b, 0);
        assert_eq!(run_symreg(&[]), 2);
        assert_eq!(run_symreg(&s(&["0,1", "0,1,2"])), 2);
    }

    #[test]
    fn sat_solves_and_detects_unsat() {
        // x1 (clause "1") with ¬x1 (clause "-1") → UNSAT.
        assert_eq!(run_sat(&s(&["1;-1"])), 1);
        // (x1 ∨ ¬x2) ∧ (x2) → SAT.
        assert_eq!(run_sat(&s(&["1,-2;2"])), 0);
        assert_eq!(run_sat(&[]), 2);
        assert_eq!(run_sat(&s(&["1,foo"])), 2);
    }
}
