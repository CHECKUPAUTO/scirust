//! Regression symbolique : decouverte de structure par programmation genetique,
//! constantes fittees par la differentiation symbolique de scirust-symbolic
//! (Adam), front de Pareto precision/complexite, multi-variables.
#![allow(dead_code)]
use scirust_symbolic::{Expr, eval, diff, simplify};
use scirust_symbolic::Expr::*;
use std::collections::HashMap;
use std::cell::{Cell, RefCell};

fn b(e: Expr) -> Box<Expr> { Box::new(e) }

struct Rng(u64);
impl Rng {
    fn new(s: u64) -> Self { Rng(s) }
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
    fn f64(&mut self) -> f64 { (self.next() >> 11) as f64 / (1u64 << 53) as f64 }
    fn range(&mut self, n: usize) -> usize { (self.next() % n as u64) as usize }
}

fn size(e: &Expr) -> usize {
    match e {
        Const(_) | Var(_) => 1,
        Neg(a) | Sin(a) | Cos(a) | Exp(a) | Ln(a) | Sqrt(a) | Abs(a) => 1 + size(a),
        Add(a, c) | Sub(a, c) | Mul(a, c) | Div(a, c) | Pow(a, c) => 1 + size(a) + size(c),
    }
}
fn map_children(e: &Expr, f: &dyn Fn(&Expr) -> Expr) -> Expr {
    match e {
        Const(c) => Const(*c), Var(v) => Var(v.clone()),
        Neg(a) => Neg(b(f(a))), Sin(a) => Sin(b(f(a))), Cos(a) => Cos(b(f(a))),
        Exp(a) => Exp(b(f(a))), Ln(a) => Ln(b(f(a))), Sqrt(a) => Sqrt(b(f(a))), Abs(a) => Abs(b(f(a))),
        Add(a, c) => Add(b(f(a)), b(f(c))), Sub(a, c) => Sub(b(f(a)), b(f(c))),
        Mul(a, c) => Mul(b(f(a)), b(f(c))), Div(a, c) => Div(b(f(a)), b(f(c))), Pow(a, c) => Pow(b(f(a)), b(f(c))),
    }
}
fn replace_at(e: &Expr, target: usize, counter: &Cell<usize>, sub: &Expr) -> Expr {
    let here = counter.get(); counter.set(here + 1);
    if here == target { return sub.clone(); }
    map_children(e, &|ch| replace_at(ch, target, counter, sub))
}
fn subtree_at(e: &Expr, target: usize, counter: &Cell<usize>) -> Option<Expr> {
    let here = counter.get(); counter.set(here + 1);
    if here == target { return Some(e.clone()); }
    match e {
        Const(_) | Var(_) => None,
        Neg(a) | Sin(a) | Cos(a) | Exp(a) | Ln(a) | Sqrt(a) | Abs(a) => subtree_at(a, target, counter),
        Add(a, c) | Sub(a, c) | Mul(a, c) | Div(a, c) | Pow(a, c) => {
            let l = subtree_at(a, target, counter);
            if l.is_some() { l } else { subtree_at(c, target, counter) }
        }
    }
}
fn abstract_consts(e: &Expr, idx: &Cell<usize>, inits: &RefCell<Vec<f64>>) -> Expr {
    if let Const(cv) = e {
        let i = idx.get(); idx.set(i + 1); inits.borrow_mut().push(*cv);
        return Var(format!("c{i}"));
    }
    map_children(e, &|ch| abstract_consts(ch, idx, inits))
}
fn substitute(e: &Expr, names: &[String], vals: &[f64]) -> Expr {
    if let Var(v) = e {
        if let Some(p) = names.iter().position(|n| n == v) { return Const(vals[p]); }
    }
    map_children(e, &|ch| substitute(ch, names, vals))
}

/// Fitte les constantes de `expr` (params nommes) aux donnees multi-variables
/// par Adam, gradients via differentiation symbolique.
pub fn fit_constants(expr: &Expr, params: &[&str], data: &[(Vec<f64>, f64)], inputs: &[&str],
                     init: &[f64], lr: f64, max_iters: usize, tol: f64) -> (Vec<f64>, f64, usize) {
    let n = data.len() as f64;
    if params.is_empty() {
        let mut sse = 0.0;
        for (inp, y) in data {
            let mut vars = HashMap::new();
            for (kk, nm) in inputs.iter().enumerate() { vars.insert(nm.to_string(), inp[kk]); }
            let r = eval(expr, &vars).unwrap_or(f64::NAN) - y; sse += r * r;
        }
        return (vec![], sse / n, 0);
    }
    let grads: Vec<Expr> = params.iter().map(|p| diff(expr, p)).collect();
    let mut theta = init.to_vec();
    let (b1, b2, eps) = (0.9_f64, 0.999_f64, 1e-8_f64);
    let mut m = vec![0.0; theta.len()];
    let mut vv = vec![0.0; theta.len()];
    let mut mse = f64::INFINITY;
    let mut it = 0;
    while it < max_iters {
        it += 1;
        let mut g = vec![0.0; theta.len()];
        let mut sse = 0.0;
        for (inp, y) in data {
            let mut vars = HashMap::new();
            for (kk, nm) in inputs.iter().enumerate() { vars.insert(nm.to_string(), inp[kk]); }
            for (nm, &val) in params.iter().zip(theta.iter()) { vars.insert(nm.to_string(), val); }
            let r = eval(expr, &vars).unwrap_or(f64::NAN) - y;
            sse += r * r;
            for (j, ge) in grads.iter().enumerate() {
                g[j] += 2.0 * r * eval(ge, &vars).unwrap_or(f64::NAN) / n;
            }
        }
        mse = sse / n;
        if !mse.is_finite() { break; }
        for j in 0..theta.len() {
            m[j] = b1 * m[j] + (1.0 - b1) * g[j];
            vv[j] = b2 * vv[j] + (1.0 - b2) * g[j] * g[j];
            let mh = m[j] / (1.0 - b1.powi(it as i32));
            let vh = vv[j] / (1.0 - b2.powi(it as i32));
            theta[j] -= lr * mh / (vh.sqrt() + eps);
        }
        if mse < tol { break; }
    }
    (theta, mse, it)
}

fn fitness(tree: &Expr, data: &[(Vec<f64>, f64)], inputs: &[&str], iters: usize) -> f64 {
    let idx = Cell::new(0); let inits = RefCell::new(Vec::new());
    let pexpr = abstract_consts(tree, &idx, &inits);
    let inits = inits.into_inner();
    let names: Vec<String> = (0..inits.len()).map(|i| format!("c{i}")).collect();
    let nr: Vec<&str> = names.iter().map(|s| s.as_str()).collect();
    let (_, mse, _) = fit_constants(&pexpr, &nr, data, inputs, &inits, 0.05, iters, 1e-12);
    mse
}
fn polish(tree: &Expr, data: &[(Vec<f64>, f64)], inputs: &[&str]) -> (Expr, f64) {
    let idx = Cell::new(0); let inits = RefCell::new(Vec::new());
    let pexpr = abstract_consts(tree, &idx, &inits);
    let inits = inits.into_inner();
    let names: Vec<String> = (0..inits.len()).map(|i| format!("c{i}")).collect();
    let nr: Vec<&str> = names.iter().map(|s| s.as_str()).collect();
    let (theta, mse, _) = fit_constants(&pexpr, &nr, data, inputs, &inits, 0.02, 4000, 1e-14);
    (simplify(&substitute(&pexpr, &names, &theta)), mse)
}

fn gen_tree(rng: &mut Rng, depth: usize, maxd: usize, inputs: &[&str]) -> Expr {
    let term = depth >= maxd || (depth > 0 && rng.f64() < 0.3);
    if term {
        if rng.f64() < 0.6 { Var(inputs[rng.range(inputs.len())].to_string()) }
        else { Const(rng.f64() * 4.0 - 2.0) }
    } else {
        match rng.range(7) {
            0 => Add(b(gen_tree(rng, depth+1, maxd, inputs)), b(gen_tree(rng, depth+1, maxd, inputs))),
            1 => Sub(b(gen_tree(rng, depth+1, maxd, inputs)), b(gen_tree(rng, depth+1, maxd, inputs))),
            2 => Mul(b(gen_tree(rng, depth+1, maxd, inputs)), b(gen_tree(rng, depth+1, maxd, inputs))),
            3 => Div(b(gen_tree(rng, depth+1, maxd, inputs)), b(gen_tree(rng, depth+1, maxd, inputs))),
            4 => Sin(b(gen_tree(rng, depth+1, maxd, inputs))),
            5 => Cos(b(gen_tree(rng, depth+1, maxd, inputs))),
            _ => Exp(b(gen_tree(rng, depth+1, maxd, inputs))),
        }
    }
}
fn crossover(rng: &mut Rng, x: &Expr, y: &Expr) -> Expr {
    let ix = rng.range(size(x)); let iy = rng.range(size(y));
    let cy = Cell::new(0); let sub = subtree_at(y, iy, &cy).unwrap();
    let cx = Cell::new(0); replace_at(x, ix, &cx, &sub)
}
fn mutate(rng: &mut Rng, x: &Expr, inputs: &[&str]) -> Expr {
    let ix = rng.range(size(x)); let fresh = gen_tree(rng, 0, 2, inputs);
    let cx = Cell::new(0); replace_at(x, ix, &cx, &fresh)
}
fn tournament(rng: &mut Rng, scores: &[f64], k: usize) -> usize {
    let mut bi = rng.range(scores.len());
    for _ in 1..k { let j = rng.range(scores.len()); if scores[j] < scores[bi] { bi = j; } }
    bi
}
fn dominates(a: (usize, f64), c: (usize, f64)) -> bool {
    a.0 <= c.0 && a.1 <= c.1 && (a.0 < c.0 || a.1 < c.1)
}
fn pareto_insert(arch: &mut Vec<(usize, f64, Expr)>, s: usize, m: f64, e: &Expr) {
    if !m.is_finite() { return; }
    if arch.iter().any(|&(s2, m2, _)| dominates((s2, m2), (s, m))) { return; }
    if arch.iter().any(|&(s2, m2, _)| s2 == s && (m2 - m).abs() < 1e-15) { return; }
    arch.retain(|&(s2, m2, _)| !dominates((s, m), (s2, m2)));
    arch.push((s, m, e.clone()));
}
fn gp(rng: &mut Rng, data: &[(Vec<f64>, f64)], inputs: &[&str], pop_size: usize,
      gens: usize, inner: usize, max_size: usize) -> Vec<(usize, f64, Expr)> {
    let mut pop: Vec<Expr> = (0..pop_size).map(|_| gen_tree(rng, 0, 4, inputs)).collect();
    let mut arch: Vec<(usize, f64, Expr)> = vec![];
    let mut best = f64::INFINITY;
    let mut best_tree = Const(0.0);
    for _ in 0..gens {
        let mut scores = vec![0.0; pop_size];
        for i in 0..pop_size {
            let m = fitness(&pop[i], data, inputs, inner);
            let sz = size(&pop[i]);
            scores[i] = if m.is_finite() { m + 0.005 * sz as f64 } else { f64::INFINITY };
            pareto_insert(&mut arch, sz, m, &pop[i]);
            if m < best { best = m; best_tree = pop[i].clone(); }
        }
        if best < 1e-12 { break; }
        let mut next = Vec::with_capacity(pop_size);
        next.push(best_tree.clone());
        while next.len() < pop_size {
            let pa = tournament(rng, &scores, 4);
            let mut child = if rng.f64() < 0.8 {
                let pb = tournament(rng, &scores, 4); crossover(rng, &pop[pa], &pop[pb])
            } else { pop[pa].clone() };
            if rng.f64() < 0.3 { child = mutate(rng, &child, inputs); }
            if size(&child) > max_size { child = pop[pa].clone(); }
            next.push(child);
        }
        pop = next;
    }
    arch
}

/// Decouvre un front de Pareto (taille, mse, formule) ajustant `data`.
/// `inputs` = noms des variables d'entree. Plusieurs graines = restarts.
pub fn discover(data: &[(Vec<f64>, f64)], inputs: &[&str], seeds: &[u64],
                pop: usize, gens: usize, inner: usize, maxs: usize) -> Vec<(usize, f64, Expr)> {
    let mut arch: Vec<(usize, f64, Expr)> = vec![];
    for &s in seeds {
        let a = gp(&mut Rng::new(s), data, inputs, pop, gens, inner, maxs);
        for (sz, m, e) in a { pareto_insert(&mut arch, sz, m, &e); }
    }
    let mut front: Vec<(usize, f64, Expr)> = vec![];
    for (_, _, tree) in &arch {
        let (pe, pm) = polish(tree, data, inputs);
        pareto_insert(&mut front, size(&pe), pm, &pe);
    }
    front.sort_by_key(|&(s, _, _)| s);
    front
}

#[cfg(test)]
mod tests {
    use super::*;

    // Test rapide (defaut) : le fitter de constantes, cas convexe -> exact.
    #[test]
    fn fit_constants_convex() {
        let xx = Mul(b(Var("x".into())), b(Var("x".into())));
        let term_a = Mul(b(Var("a".into())), b(xx));
        let term_b = Mul(b(Var("b".into())), b(Var("x".into())));
        let expr = Add(b(term_a), b(Add(b(term_b), b(Var("c".into())))));
        let (ta, tb, tc) = (0.5_f64, -1.2_f64, 2.0_f64);
        let mut data = vec![]; let mut x: f64 = -3.0;
        while x <= 3.0 + 1e-9 { data.push((vec![x], ta * x * x + tb * x + tc)); x += 0.2; }
        let (theta, mse, _) = fit_constants(&expr, &["a", "b", "c"], &data, &["x"],
                                            &[0.0, 0.0, 0.0], 0.05, 8000, 1e-14);
        assert!(mse < 1e-6, "mse={mse:.2e}");
        assert!((theta[0]-ta).abs() < 1e-2 && (theta[1]-tb).abs() < 1e-2 && (theta[2]-tc).abs() < 1e-2,
                "constantes non recouvrees");
    }

    // Tests d'integration (recherche GP, lents en debug) :
    //   cargo test -p scirust-symreg --release -- --ignored
    fn min_mse(front: &[(usize, f64, Expr)]) -> f64 {
        front.iter().map(|&(_, m, _)| m).fold(f64::INFINITY, f64::min)
    }
    #[test]
    #[ignore = "integration GP (lent en debug): --release -- --ignored"]
    fn recovers_poly_plus_sin() {
        let mut data = vec![]; let mut x: f64 = -3.0;
        while x <= 3.0 + 1e-9 { data.push((vec![x], x * x + x.sin())); x += 0.2; }
        let front = discover(&data, &["x"], &[1, 2, 3], 200, 22, 35, 25);
        assert!(min_mse(&front) < 1e-6, "x^2+sin(x): {:.2e}", min_mse(&front));
    }
    #[test]
    #[ignore = "integration GP (lent en debug): --release -- --ignored"]
    fn recovers_multivariable() {
        let mut data = vec![];
        let xs: [f64; 6] = [-2.0, -1.2, -0.4, 0.4, 1.2, 2.0];
        let ys: [f64; 5] = [-2.0, -1.0, 0.0, 1.0, 2.0];
        for &xx in &xs { for &yy in &ys { data.push((vec![xx, yy], xx * yy + xx.sin())); } }
        let front = discover(&data, &["x", "y"], &[1, 2, 3], 200, 22, 35, 25);
        assert!(min_mse(&front) < 1e-6, "x*y+sin(x): {:.2e}", min_mse(&front));
    }
    #[test]
    #[ignore = "integration GP (lent en debug): --release -- --ignored"]
    fn rational_appears_parsimonious_on_front() {
        let mut data = vec![]; let mut x: f64 = -4.0;
        while x <= 4.0 + 1e-9 { data.push((vec![x], x / (1.0 + x * x))); x += 0.2; }
        let front = discover(&data, &["x"], &[1, 2, 3], 200, 22, 35, 25);
        assert!(front.iter().any(|&(s, m, _)| s <= 8 && m < 1e-3),
                "pas de forme parcimonieuse precise sur le front");
    }
}
