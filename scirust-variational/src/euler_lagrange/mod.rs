pub mod autodiff;
pub mod constrained;
pub mod symbolic;

use scirust_symbolic::Expr;
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct ELEquation {
    pub coordinate: String,
    pub residual: Expr,
    pub acceleration_deps: Vec<String>,
    pub is_explicit: bool,
}

impl std::fmt::Display for ELEquation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Euler–Lagrange for q[{}]: {} = 0",
            self.coordinate, self.residual
        )
    }
}

#[derive(Debug, Clone)]
pub struct ELDerivation {
    pub equations: Vec<ELEquation>,
    pub lagrangian: Expr,
    pub coordinates: Vec<String>,
    pub time_var: Option<String>,
}

impl ELDerivation {
    pub fn num_coordinates(&self) -> usize {
        self.coordinates.len()
    }

    pub fn is_acceleration_explicit(&self) -> bool {
        self.equations.iter().all(|eq| eq.is_explicit)
    }

    pub fn get_equation(&self, coord: &str) -> Option<&ELEquation> {
        self.equations.iter().find(|eq| eq.coordinate == coord)
    }
}

pub fn substitute(expr: &Expr, from: &str, to: &Expr) -> Expr {
    match expr
    {
        Expr::Const(_) => expr.clone(),
        Expr::Var(v) =>
        {
            if v == from
            {
                to.clone()
            }
            else
            {
                expr.clone()
            }
        },
        Expr::Add(a, b) => substitute(a, from, to) + substitute(b, from, to),
        Expr::Sub(a, b) => substitute(a, from, to) - substitute(b, from, to),
        Expr::Mul(a, b) => substitute(a, from, to) * substitute(b, from, to),
        Expr::Div(a, b) => substitute(a, from, to) / substitute(b, from, to),
        Expr::Neg(a) => -substitute(a, from, to),
        Expr::Pow(a, b) => Expr::Pow(
            Box::new(substitute(a, from, to)),
            Box::new(substitute(b, from, to)),
        ),
        Expr::Sin(a) => Expr::Sin(Box::new(substitute(a, from, to))),
        Expr::Cos(a) => Expr::Cos(Box::new(substitute(a, from, to))),
        Expr::Exp(a) => Expr::Exp(Box::new(substitute(a, from, to))),
        Expr::Ln(a) => Expr::Ln(Box::new(substitute(a, from, to))),
        Expr::Sqrt(a) => Expr::Sqrt(Box::new(substitute(a, from, to))),
        Expr::Abs(a) => Expr::Abs(Box::new(substitute(a, from, to))),
    }
}

pub fn collect_vars_set(expr: &Expr) -> Vec<String> {
    let mut vars = std::collections::BTreeSet::new();
    collect_vars_into(expr, &mut vars);
    vars.into_iter().collect()
}

fn collect_vars_into(expr: &Expr, out: &mut std::collections::BTreeSet<String>) {
    match expr
    {
        Expr::Var(v) =>
        {
            out.insert(v.clone());
        },
        Expr::Add(a, b) | Expr::Sub(a, b) | Expr::Mul(a, b) | Expr::Div(a, b) | Expr::Pow(a, b) =>
        {
            collect_vars_into(a, out);
            collect_vars_into(b, out);
        },
        Expr::Neg(a)
        | Expr::Sin(a)
        | Expr::Cos(a)
        | Expr::Exp(a)
        | Expr::Ln(a)
        | Expr::Sqrt(a)
        | Expr::Abs(a) =>
        {
            collect_vars_into(a, out);
        },
        Expr::Const(_) =>
        {},
    }
}

pub fn make_lagrangian_symbolic(
    coords: &[&str],
    time_label: Option<&str>,
) -> (
    Vec<Expr>,
    Vec<Expr>,
    Vec<Expr>,
    Vec<Expr>,
    HashMap<String, Expr>,
) {
    use scirust_symbolic::Expr;

    let mut q_vars = Vec::new();
    let mut dq_vars = Vec::new();
    let mut ddq_vars = Vec::new();
    let mut accel_vars = Vec::new();
    let mut bindings = HashMap::new();

    for &c in coords
    {
        let q = Expr::Var(c.to_string());
        let dq = Expr::Var(format!("{}_dot", c));
        let ddq = Expr::Var(format!("{}_ddot", c));
        q_vars.push(q);
        dq_vars.push(dq.clone());
        ddq_vars.push(ddq.clone());
        accel_vars.push(ddq);
    }

    if let Some(t) = time_label
    {
        let t_var = Expr::Var(t.to_string());
        bindings.insert(t.to_string(), t_var);
    }

    (q_vars, dq_vars, ddq_vars, accel_vars, bindings)
}
