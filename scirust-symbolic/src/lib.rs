//! Symbolic math engine — expression trees, simplification, differentiation,
//! equation solving, and dual-number autodiff.
//!
//! This replaces the missing scirust_symbolic / scirust_reasoning / scirust_learning
//! crates with a single inline module.

use std::collections::HashMap;
use std::fmt;
use std::ops::{Add, Div, Mul, Neg, Sub};

// ── Expression tree ──

#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    Const(f64),
    Var(String),
    Add(Box<Expr>, Box<Expr>),
    Sub(Box<Expr>, Box<Expr>),
    Mul(Box<Expr>, Box<Expr>),
    Div(Box<Expr>, Box<Expr>),
    Neg(Box<Expr>),
    Pow(Box<Expr>, Box<Expr>),
    Sin(Box<Expr>),
    Cos(Box<Expr>),
    Exp(Box<Expr>),
    Ln(Box<Expr>),
    Sqrt(Box<Expr>),
    Abs(Box<Expr>),
}

impl fmt::Display for Expr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self
        {
            Expr::Const(c) => write!(f, "{c}"),
            Expr::Var(v) => write!(f, "{v}"),
            Expr::Add(a, b) => write!(f, "({} + {})", a, b),
            Expr::Sub(a, b) => write!(f, "({} - {})", a, b),
            Expr::Mul(a, b) => write!(f, "({} * {})", a, b),
            Expr::Div(a, b) => write!(f, "({} / {})", a, b),
            Expr::Neg(a) => write!(f, "(-{})", a),
            Expr::Pow(a, b) => write!(f, "({}^{})", a, b),
            Expr::Sin(a) => write!(f, "sin({})", a),
            Expr::Cos(a) => write!(f, "cos({})", a),
            Expr::Exp(a) => write!(f, "exp({})", a),
            Expr::Ln(a) => write!(f, "ln({})", a),
            Expr::Sqrt(a) => write!(f, "sqrt({})", a),
            Expr::Abs(a) => write!(f, "abs({})", a),
        }
    }
}

// ── Operator impls for ergonomic construction ──

impl Add for Expr {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        match (&self, &rhs)
        {
            (Expr::Const(a), Expr::Const(b)) => Expr::Const(a + b),
            (Expr::Const(0.0), _) => rhs,
            (_, Expr::Const(0.0)) => self,
            _ => Expr::Add(Box::new(self), Box::new(rhs)),
        }
    }
}

impl Sub for Expr {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        match (&self, &rhs)
        {
            (Expr::Const(a), Expr::Const(b)) => Expr::Const(a - b),
            (_, Expr::Const(0.0)) => self,
            _ => Expr::Sub(Box::new(self), Box::new(rhs)),
        }
    }
}

impl Mul for Expr {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        match (&self, &rhs)
        {
            (Expr::Const(0.0), _) | (_, Expr::Const(0.0)) => Expr::Const(0.0),
            (Expr::Const(1.0), _) => rhs,
            (_, Expr::Const(1.0)) => self,
            (Expr::Const(a), Expr::Const(b)) => Expr::Const(a * b),
            _ => Expr::Mul(Box::new(self), Box::new(rhs)),
        }
    }
}

impl Div for Expr {
    type Output = Self;
    fn div(self, rhs: Self) -> Self {
        match (&self, &rhs)
        {
            (Expr::Const(0.0), _) => Expr::Const(0.0),
            (_, Expr::Const(1.0)) => self,
            (Expr::Const(a), Expr::Const(b)) if *b != 0.0 => Expr::Const(a / b),
            _ => Expr::Div(Box::new(self), Box::new(rhs)),
        }
    }
}

impl Neg for Expr {
    type Output = Self;
    fn neg(self) -> Self {
        match &self
        {
            Expr::Const(c) => Expr::Const(-c),
            Expr::Neg(a) => *a.clone(),
            _ => Expr::Neg(Box::new(self)),
        }
    }
}

impl From<f64> for Expr {
    fn from(c: f64) -> Self {
        Expr::Const(c)
    }
}

// ── Parsing ──

/// Parse a mathematical expression string into an Expr tree.
pub fn parse(input: &str) -> Result<Expr, String> {
    let tokens = tokenize(input)?;
    parse_expr(&tokens, 0).map(|(expr, _)| expr)
}

#[derive(Debug, Clone, PartialEq)]
enum Token {
    Num(f64),
    Ident(String),
    Plus,
    Minus,
    Star,
    Slash,
    Caret,
    LParen,
    RParen,
}

fn tokenize(input: &str) -> Result<Vec<Token>, String> {
    let mut tokens = Vec::new();
    let chars: Vec<char> = input.chars().collect();
    let mut i = 0;
    while i < chars.len()
    {
        let c = chars[i];
        if c.is_whitespace()
        {
            i += 1;
            continue;
        }
        if c.is_ascii_digit() || c == '.'
        {
            let start = i;
            while i < chars.len() && (chars[i].is_ascii_digit() || chars[i] == '.')
            {
                i += 1;
            }
            let num: f64 = input[start..i]
                .parse()
                .map_err(|_| format!("Bad number: {}", &input[start..i]))?;
            tokens.push(Token::Num(num));
            continue;
        }
        if c.is_alphabetic() || c == '_'
        {
            let start = i;
            while i < chars.len() && (chars[i].is_alphanumeric() || chars[i] == '_')
            {
                i += 1;
            }
            let ident = &input[start..i];
            // Recognized function names become identifiers too (handled in parse_atom)
            tokens.push(Token::Ident(ident.to_string()));
            continue;
        }
        match c
        {
            '+' => tokens.push(Token::Plus),
            '-' => tokens.push(Token::Minus),
            '*' => tokens.push(Token::Star),
            '/' => tokens.push(Token::Slash),
            '^' => tokens.push(Token::Caret),
            '(' => tokens.push(Token::LParen),
            ')' => tokens.push(Token::RParen),
            other => return Err(format!("Unexpected character: {other}")),
        }
        i += 1;
    }
    Ok(tokens)
}

fn parse_expr(tokens: &[Token], pos: usize) -> Result<(Expr, usize), String> {
    parse_add_sub(tokens, pos)
}

fn parse_add_sub(tokens: &[Token], pos: usize) -> Result<(Expr, usize), String> {
    let (mut lhs, mut pos) = parse_mul_div(tokens, pos)?;
    while pos < tokens.len()
    {
        match tokens[pos]
        {
            Token::Plus =>
            {
                let (rhs, new_pos) = parse_mul_div(tokens, pos + 1)?;
                lhs = lhs + rhs;
                pos = new_pos;
            },
            Token::Minus =>
            {
                let (rhs, new_pos) = parse_mul_div(tokens, pos + 1)?;
                lhs = lhs - rhs;
                pos = new_pos;
            },
            _ => break,
        }
    }
    Ok((lhs, pos))
}

fn parse_mul_div(tokens: &[Token], pos: usize) -> Result<(Expr, usize), String> {
    let (mut lhs, mut pos) = parse_pow(tokens, pos)?;
    while pos < tokens.len()
    {
        match tokens[pos]
        {
            Token::Star =>
            {
                let (rhs, new_pos) = parse_pow(tokens, pos + 1)?;
                lhs = lhs * rhs;
                pos = new_pos;
            },
            Token::Slash =>
            {
                let (rhs, new_pos) = parse_pow(tokens, pos + 1)?;
                lhs = lhs / rhs;
                pos = new_pos;
            },
            _ => break,
        }
    }
    Ok((lhs, pos))
}

fn parse_pow(tokens: &[Token], pos: usize) -> Result<(Expr, usize), String> {
    let (mut base, mut pos) = parse_unary(tokens, pos)?;
    while pos < tokens.len() && tokens[pos] == Token::Caret
    {
        let (exp, new_pos) = parse_unary(tokens, pos + 1)?;
        base = Expr::Pow(Box::new(base), Box::new(exp));
        pos = new_pos;
    }
    Ok((base, pos))
}

fn parse_unary(tokens: &[Token], pos: usize) -> Result<(Expr, usize), String> {
    if pos < tokens.len() && tokens[pos] == Token::Minus
    {
        let (inner, new_pos) = parse_unary(tokens, pos + 1)?;
        return Ok((-inner, new_pos));
    }
    parse_atom(tokens, pos)
}

fn parse_atom(tokens: &[Token], pos: usize) -> Result<(Expr, usize), String> {
    if pos >= tokens.len()
    {
        return Err("Unexpected end of expression".into());
    }
    match &tokens[pos]
    {
        Token::Num(n) => Ok((Expr::Const(*n), pos + 1)),
        Token::Ident(name) =>
        {
            let func = name.as_str();
            // Function call: f(expr)
            if pos + 1 < tokens.len() && tokens[pos + 1] == Token::LParen
            {
                let (arg, after) = parse_expr(tokens, pos + 2)?;
                if after >= tokens.len() || tokens[after] != Token::RParen
                {
                    return Err(format!("Expected ')' after {func}("));
                }
                let expr = match func
                {
                    "sin" => Expr::Sin(Box::new(arg)),
                    "cos" => Expr::Cos(Box::new(arg)),
                    "exp" => Expr::Exp(Box::new(arg)),
                    "ln" => Expr::Ln(Box::new(arg)),
                    "sqrt" => Expr::Sqrt(Box::new(arg)),
                    "abs" => Expr::Abs(Box::new(arg)),
                    _ => return Err(format!("Unknown function: {func}")),
                };
                Ok((expr, after + 1))
            }
            else
            {
                Ok((Expr::Var(name.clone()), pos + 1))
            }
        },
        Token::LParen =>
        {
            let (inner, after) = parse_expr(tokens, pos + 1)?;
            if after >= tokens.len() || tokens[after] != Token::RParen
            {
                return Err("Expected ')'".into());
            }
            Ok((inner, after + 1))
        },
        other => Err(format!("Unexpected token: {other:?}")),
    }
}

// ── Simplification ──

/// Simplify an expression (constant folding + algebraic identities).
pub fn simplify(expr: &Expr) -> Expr {
    match expr
    {
        Expr::Const(_) | Expr::Var(_) => expr.clone(),
        Expr::Add(a, b) => simplify(&simplify(a)) + simplify(&simplify(b)),
        Expr::Sub(a, b) => simplify(&simplify(a)) - simplify(&simplify(b)),
        Expr::Mul(a, b) => simplify(&simplify(a)) * simplify(&simplify(b)),
        Expr::Div(a, b) => simplify(&simplify(a)) / simplify(&simplify(b)),
        Expr::Neg(a) => -simplify(&simplify(a)),
        Expr::Pow(a, b) =>
        {
            let sa = simplify(a);
            let sb = simplify(b);
            match (&sa, &sb)
            {
                (Expr::Const(c), Expr::Const(n)) => Expr::Const(c.powf(*n)),
                (_, Expr::Const(0.0)) => Expr::Const(1.0),
                (_, Expr::Const(1.0)) => sa.clone(),
                _ => Expr::Pow(Box::new(sa), Box::new(sb)),
            }
        },
        Expr::Sin(a) => match simplify(a)
        {
            Expr::Const(c) => Expr::Const(c.sin()),
            sa => Expr::Sin(Box::new(sa)),
        },
        Expr::Cos(a) => match simplify(a)
        {
            Expr::Const(c) => Expr::Const(c.cos()),
            sa => Expr::Cos(Box::new(sa)),
        },
        Expr::Exp(a) => match simplify(a)
        {
            Expr::Const(c) => Expr::Const(c.exp()),
            sa => Expr::Exp(Box::new(sa)),
        },
        Expr::Ln(a) => match simplify(a)
        {
            Expr::Const(c) if c > 0.0 => Expr::Const(c.ln()),
            sa => Expr::Ln(Box::new(sa)),
        },
        Expr::Sqrt(a) => match simplify(a)
        {
            Expr::Const(c) if c >= 0.0 => Expr::Const(c.sqrt()),
            sa => Expr::Sqrt(Box::new(sa)),
        },
        Expr::Abs(a) => match simplify(a)
        {
            Expr::Const(c) => Expr::Const(c.abs()),
            sa => Expr::Abs(Box::new(sa)),
        },
    }
}

// ── Symbolic differentiation ──

/// Symbolically differentiate expr with respect to var.
pub fn diff(expr: &Expr, var: &str) -> Expr {
    match expr
    {
        Expr::Const(_) => Expr::Const(0.0),
        Expr::Var(v) =>
        {
            if v == var
            {
                Expr::Const(1.0)
            }
            else
            {
                Expr::Const(0.0)
            }
        },
        Expr::Add(a, b) => diff(a, var) + diff(b, var),
        Expr::Sub(a, b) => diff(a, var) - diff(b, var),
        Expr::Mul(a, b) =>
        {
            // f'g + fg'
            diff(a, var).clone() * b.as_ref().clone() + a.as_ref().clone() * diff(b, var)
        },
        Expr::Div(a, b) =>
        {
            // (f'g - fg') / g^2
            let num = diff(a, var).clone() * b.as_ref().clone() - a.as_ref().clone() * diff(b, var);
            let den = Expr::Pow(b.clone(), Box::new(Expr::Const(2.0)));
            num / den
        },
        Expr::Neg(a) => -diff(a, var),
        Expr::Pow(a, b) =>
        {
            match b.as_ref()
            {
                Expr::Const(n) =>
                {
                    // x^n → n * x^(n-1) * x'
                    let coef = Expr::Const(*n);
                    let pow = Expr::Pow(a.clone(), Box::new(Expr::Const(n - 1.0)));
                    coef * pow * diff(a, var)
                },
                _ =>
                {
                    // General case: treat as exp(b * ln(a))
                    // derivative = a^b * (b' * ln(a) + b * a' / a)
                    // Stub: fall back to unsimplified form
                    Expr::Mul(
                        Box::new(expr.clone()),
                        Box::new(
                            diff(b, var) * Expr::Ln(a.clone())
                                + b.as_ref().clone() * diff(a, var) / a.as_ref().clone(),
                        ),
                    )
                },
            }
        },
        Expr::Sin(a) =>
        {
            // d/dx sin(u) = cos(u) * du
            Expr::Mul(Box::new(Expr::Cos(a.clone())), Box::new(diff(a, var)))
        },
        Expr::Cos(a) =>
        {
            // d/dx cos(u) = -sin(u) * du
            Expr::Mul(
                Box::new(Expr::Neg(Box::new(Expr::Sin(a.clone())))),
                Box::new(diff(a, var)),
            )
        },
        Expr::Exp(a) =>
        {
            // d/dx e^u = e^u * du
            Expr::Mul(Box::new(expr.clone()), Box::new(diff(a, var)))
        },
        Expr::Ln(a) =>
        {
            // d/dx ln(u) = (1/u) * du
            Expr::Div(Box::new(diff(a, var)), a.clone())
        },
        Expr::Sqrt(a) =>
        {
            // d/dx sqrt(u) = du / (2*sqrt(u))
            Expr::Div(
                Box::new(diff(a, var)),
                Box::new(Expr::Mul(
                    Box::new(Expr::Const(2.0)),
                    Box::new(expr.clone()),
                )),
            )
        },
        Expr::Abs(a) =>
        {
            // d/dx |u| = sign(u) * du  where sign(u) = u / |u|
            Expr::Mul(
                Box::new(Expr::Div(a.clone(), Box::new(Expr::Abs(a.clone())))),
                Box::new(diff(a, var)),
            )
        },
    }
}

// ── Evaluation ──

/// Numerically evaluate an expression with given variable bindings.
pub fn eval(expr: &Expr, vars: &HashMap<String, f64>) -> Result<f64, String> {
    match expr
    {
        Expr::Const(c) => Ok(*c),
        Expr::Var(v) => vars
            .get(v)
            .copied()
            .ok_or_else(|| format!("Undefined variable: {v}")),
        Expr::Add(a, b) => Ok(eval(a, vars)? + eval(b, vars)?),
        Expr::Sub(a, b) => Ok(eval(a, vars)? - eval(b, vars)?),
        Expr::Mul(a, b) => Ok(eval(a, vars)? * eval(b, vars)?),
        Expr::Div(a, b) =>
        {
            let den = eval(b, vars)?;
            if den == 0.0
            {
                return Err("Division by zero".into());
            }
            Ok(eval(a, vars)? / den)
        },
        Expr::Neg(a) => Ok(-eval(a, vars)?),
        Expr::Pow(a, b) => Ok(eval(a, vars)?.powf(eval(b, vars)?)),
        Expr::Sin(a) => Ok(eval(a, vars)?.sin()),
        Expr::Cos(a) => Ok(eval(a, vars)?.cos()),
        Expr::Exp(a) => Ok(eval(a, vars)?.exp()),
        Expr::Ln(a) =>
        {
            let v = eval(a, vars)?;
            if v <= 0.0
            {
                return Err("ln of non-positive number".into());
            }
            Ok(v.ln())
        },
        Expr::Sqrt(a) =>
        {
            let v = eval(a, vars)?;
            if v < 0.0
            {
                return Err("sqrt of negative number".into());
            }
            Ok(v.sqrt())
        },
        Expr::Abs(a) => Ok(eval(a, vars)?.abs()),
    }
}

// ── Equation solving ──

/// Solve a quadratic equation a*x^2 + b*x + c = 0 in `var`.
/// Returns up to 2 real roots.
pub fn solve_quadratic(expr: &Expr, var: &str) -> Vec<f64> {
    // Extract coefficients by evaluating at 3 points
    let points = [0.0_f64, 1.0, -1.0];
    let mut vals = Vec::new();
    for &x in &points
    {
        let mut vars = HashMap::new();
        vars.insert(var.to_string(), x);
        if let Ok(v) = eval(expr, &vars)
        {
            vals.push(v);
        }
        else
        {
            return vec![];
        }
    }
    if vals.len() != 3
    {
        return vec![];
    }

    // Solve the 3x3 system: a*0+b*0+c=vals[0], a+b+c=vals[1], a-b+c=vals[2]
    let c = vals[0];
    let a_plus_b = vals[1] - c;
    let a_minus_b = vals[2] - c;
    let a = (a_plus_b + a_minus_b) / 2.0;
    let b = (a_plus_b - a_minus_b) / 2.0;

    if a.abs() < 1e-12
    {
        // Linear: bx + c = 0
        if b.abs() < 1e-12
        {
            return vec![];
        }
        return vec![-c / b];
    }

    let disc = b * b - 4.0 * a * c;
    if disc < -1e-12
    {
        return vec![];
    }
    let disc = disc.max(0.0);
    let sqrt_disc = disc.sqrt();
    let mut roots = vec![(-b + sqrt_disc) / (2.0 * a), (-b - sqrt_disc) / (2.0 * a)];
    roots.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    roots
}

/// Solve a linear equation: isolate the variable.
/// Tries: ax + b form → root = -b/a
pub fn solve_linear(expr: &Expr, var: &str) -> Option<f64> {
    // Evaluate at two points to find coefficients
    let mut vars0 = HashMap::new();
    vars0.insert(var.to_string(), 0.0);
    let v0 = eval(expr, &vars0).ok()?;

    let mut vars1 = HashMap::new();
    vars1.insert(var.to_string(), 1.0);
    let v1 = eval(expr, &vars1).ok()?;

    let slope = v1 - v0;
    if slope.abs() < 1e-15
    {
        return None;
    }
    Some(-v0 / slope)
}

// ── Proof ──

/// Check if two expressions are equivalent by evaluating at random points.
pub fn prove_equal(a: &Expr, b: &Expr) -> bool {
    let vars = ["x", "y", "z", "u", "v", "w"];
    for i in 0..20
    {
        let mut bindings = HashMap::new();
        for (j, v) in vars.iter().enumerate()
        {
            let val = ((i * 7919 + j * 6271 + 127) as f64 / 1000.0) % 20.0 - 10.0;
            bindings.insert(v.to_string(), val);
        }
        match (eval(a, &bindings), eval(b, &bindings))
        {
            (Ok(va), Ok(vb)) =>
            {
                if (va - vb).abs() > 1e-8
                {
                    return false;
                }
            },
            _ => return false,
        }
    }
    true
}

// ── Code generation ──

/// Generate Rust code for evaluating the expression.
pub fn to_rust_code(expr: &Expr) -> String {
    match expr
    {
        Expr::Const(c) => c.to_string(),
        Expr::Var(v) => v.clone(),
        Expr::Add(a, b) => format!("({} + {})", to_rust_code(a), to_rust_code(b)),
        Expr::Sub(a, b) => format!("({} - {})", to_rust_code(a), to_rust_code(b)),
        Expr::Mul(a, b) => format!("({} * {})", to_rust_code(a), to_rust_code(b)),
        Expr::Div(a, b) => format!("({} / {})", to_rust_code(a), to_rust_code(b)),
        Expr::Neg(a) => format!("(-{})", to_rust_code(a)),
        Expr::Pow(a, b) => format!("({}).powf({})", to_rust_code(a), to_rust_code(b)),
        Expr::Sin(a) => format!("({}).sin()", to_rust_code(a)),
        Expr::Cos(a) => format!("({}).cos()", to_rust_code(a)),
        Expr::Exp(a) => format!("({}).exp()", to_rust_code(a)),
        Expr::Ln(a) => format!("({}).ln()", to_rust_code(a)),
        Expr::Sqrt(a) => format!("({}).sqrt()", to_rust_code(a)),
        Expr::Abs(a) => format!("({}).abs()", to_rust_code(a)),
    }
}

// ── Apply trigonometric identities ──

pub fn apply_trig_identity(expr: &Expr) -> Expr {
    match expr
    {
        // The half-angle identities sin²θ = (1 - cos 2θ)/2 and
        // cos²θ = (1 + cos 2θ)/2 hold ONLY for the square. Matching any other
        // exponent would silently rewrite e.g. sin³ into a false form, so the
        // power must be exactly 2 (any other Pow falls through to `_` below).
        Expr::Pow(a, b) if matches!(b.as_ref(), Expr::Const(n) if (*n - 2.0).abs() < 1e-12) =>
        {
            if let Expr::Sin(inner) = a.as_ref()
            {
                // sin²(x) → (1 - cos(2x)) / 2
                let cos_2x = Expr::Cos(Box::new(Expr::Mul(
                    Box::new(Expr::Const(2.0)),
                    inner.clone(),
                )));
                return Expr::Sub(Box::new(Expr::Const(1.0)), Box::new(cos_2x)) / Expr::Const(2.0);
            }
            if let Expr::Cos(inner) = a.as_ref()
            {
                // cos²(x) → (1 + cos(2x)) / 2
                let cos_2x = Expr::Cos(Box::new(Expr::Mul(
                    Box::new(Expr::Const(2.0)),
                    inner.clone(),
                )));
                return (Expr::Const(1.0) + cos_2x) / Expr::Const(2.0);
            }
            expr.clone()
        },
        _ => expr.clone(),
    }
}

// ── Optimizer (stub — numeric gradient descent) ──

/// Momentum stochastic-gradient-descent optimizer.
pub struct Optimizer {
    pub lr: f64,
    pub momentum: f64,
    pub max_iter: usize,
    velocity: Vec<f64>,
}

impl Optimizer {
    pub fn new(lr: f64, max_iter: usize) -> Self {
        Self {
            lr,
            momentum: 0.9,
            max_iter,
            velocity: Vec::new(),
        }
    }

    pub fn set_momentum(&mut self, m: f64) {
        self.momentum = m;
    }

    /// One in-place momentum update: `v ← momentum·v − lr·grad`, then
    /// `params ← params + v`. The velocity is (re)initialised to match `params`.
    pub fn step(&mut self, params: &mut [f64], grad: &[f64]) {
        if self.velocity.len() != params.len()
        {
            self.velocity = vec![0.0; params.len()];
        }
        for ((p, v), g) in params
            .iter_mut()
            .zip(self.velocity.iter_mut())
            .zip(grad.iter())
        {
            *v = self.momentum * *v - self.lr * *g;
            *p += *v;
        }
    }

    /// Minimize a scalar objective from `x0` with momentum SGD, taking the
    /// gradient by central differences. Runs up to `max_iter` steps and returns
    /// the final point.
    pub fn minimize<F: Fn(&[f64]) -> f64>(&mut self, f: F, x0: &[f64]) -> Vec<f64> {
        let mut x = x0.to_vec();
        self.velocity = vec![0.0; x.len()];
        let eps = 1e-6;
        for _ in 0..self.max_iter
        {
            let mut g = vec![0.0; x.len()];
            for i in 0..x.len()
            {
                let orig = x[i];
                x[i] = orig + eps;
                let fp = f(&x);
                x[i] = orig - eps;
                let fm = f(&x);
                x[i] = orig;
                g[i] = (fp - fm) / (2.0 * eps);
            }
            self.step(&mut x, &g);
        }
        x
    }
}

// ── Pattern discovery / polynomial fit ──

/// Simple polynomial fit using least squares.
pub fn polynomial_fit(xs: &[f64], ys: &[f64], degree: usize) -> Result<Vec<f64>, String> {
    let n = xs.len();
    if n != ys.len() || n == 0
    {
        return Err("Empty or mismatched inputs".into());
    }
    let k = degree + 1;
    // Vandermonde matrix
    let mut v = Vec::with_capacity(n * k);
    for &x in xs
    {
        let mut pow = 1.0;
        for _ in 0..k
        {
            v.push(pow);
            pow *= x;
        }
    }
    // Normal equations: (V^T V) a = V^T y
    let mut vtv = vec![0.0_f64; k * k];
    let mut vty = vec![0.0_f64; k];
    for i in 0..k
    {
        for j in 0..k
        {
            let mut s = 0.0;
            for row in 0..n
            {
                s += v[row * k + i] * v[row * k + j];
            }
            vtv[i * k + j] = s;
        }
        let mut s = 0.0;
        for row in 0..n
        {
            s += v[row * k + i] * ys[row];
        }
        vty[i] = s;
    }
    // Gaussian elimination
    for col in 0..k
    {
        let pivot = vtv[col * k + col];
        if pivot.abs() < 1e-15
        {
            return Err("Singular matrix".into());
        }
        for j in 0..k
        {
            vtv[col * k + j] /= pivot;
        }
        vty[col] /= pivot;
        for row in 0..k
        {
            if row == col
            {
                continue;
            }
            let factor = vtv[row * k + col];
            for j in 0..k
            {
                vtv[row * k + j] -= factor * vtv[col * k + j];
            }
            vty[row] -= factor * vty[col];
        }
    }
    Ok(vty)
}

/// Simple linear regression.
pub fn linear_regression(xs: &[f64], ys: &[f64]) -> Result<(f64, f64), String> {
    let coeffs = polynomial_fit(xs, ys, 1)?;
    if coeffs.len() < 2
    {
        return Err("Fit failed".into());
    }
    Ok((coeffs[0], coeffs[1])) // (intercept, slope)
}

/// Discover patterns in a time series (basic: detect trend).
pub fn discover_patterns(data: &[f64]) -> Vec<String> {
    if data.len() < 3
    {
        return vec![];
    }
    let mut patterns = Vec::new();
    let mut increasing = 0;
    let mut decreasing = 0;
    for w in data.windows(2)
    {
        if w[1] > w[0]
        {
            increasing += 1;
        }
        if w[1] < w[0]
        {
            decreasing += 1;
        }
    }
    let total = (data.len() - 1) as f64;
    if increasing as f64 / total > 0.7
    {
        patterns.push("trend_upward".to_string());
    }
    else if decreasing as f64 / total > 0.7
    {
        patterns.push("trend_downward".to_string());
    }
    let mean: f64 = data.iter().sum::<f64>() / data.len() as f64;
    let var: f64 = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / data.len() as f64;
    if var.sqrt() / mean.abs().max(1e-8) < 0.05
    {
        patterns.push("stable".to_string());
    }
    patterns
}

/// Pattern memory — stores and retrieves numerical patterns.
pub struct PatternMemory {
    patterns: HashMap<String, Vec<f64>>,
}

impl Default for PatternMemory {
    fn default() -> Self {
        Self::new()
    }
}

impl PatternMemory {
    pub fn new() -> Self {
        Self {
            patterns: HashMap::new(),
        }
    }
    pub fn store(&mut self, name: &str, data: Vec<f64>) {
        self.patterns.insert(name.to_string(), data);
    }
    pub fn recall(&self, name: &str) -> Option<&[f64]> {
        self.patterns.get(name).map(|v| v.as_slice())
    }
}

// ── Dual numbers for autodiff ──

/// Dual number: value + derivative component.
/// Represents f(x+ε) = a + bε where ε² = 0.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Dual {
    pub primal: f64,
    pub tangent: f64,
}

impl Dual {
    pub fn new(primal: f64, tangent: f64) -> Self {
        Self { primal, tangent }
    }
    pub fn primal(c: f64) -> Self {
        Self {
            primal: c,
            tangent: 0.0,
        }
    }
    pub fn var(v: f64) -> Self {
        Self {
            primal: v,
            tangent: 1.0,
        }
    }

    pub fn sin(self) -> Self {
        Self {
            primal: self.primal.sin(),
            tangent: self.primal.cos() * self.tangent,
        }
    }
    pub fn cos(self) -> Self {
        Self {
            primal: self.primal.cos(),
            tangent: -self.primal.sin() * self.tangent,
        }
    }
    pub fn exp(self) -> Self {
        let e = self.primal.exp();
        Self {
            primal: e,
            tangent: e * self.tangent,
        }
    }
    pub fn ln(self) -> Self {
        Self {
            primal: self.primal.ln(),
            tangent: self.tangent / self.primal,
        }
    }
    pub fn sqrt(self) -> Self {
        let s = self.primal.sqrt();
        Self {
            primal: s,
            tangent: self.tangent / (2.0 * s),
        }
    }
    pub fn abs(self) -> Self {
        Self {
            primal: self.primal.abs(),
            tangent: self.primal.signum() * self.tangent,
        }
    }
    pub fn powf(self, other: Self) -> Self {
        let v = self.primal.powf(other.primal);
        let d = v * (other.tangent * self.primal.ln() + other.primal * self.tangent / self.primal);
        Self {
            primal: v,
            tangent: d,
        }
    }
}

impl Add for Dual {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self {
            primal: self.primal + rhs.primal,
            tangent: self.tangent + rhs.tangent,
        }
    }
}

impl Sub for Dual {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self {
            primal: self.primal - rhs.primal,
            tangent: self.tangent - rhs.tangent,
        }
    }
}

impl Mul for Dual {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        Self {
            primal: self.primal * rhs.primal,
            tangent: self.primal * rhs.tangent + self.tangent * rhs.primal,
        }
    }
}

impl Div for Dual {
    type Output = Self;
    fn div(self, rhs: Self) -> Self {
        Self {
            primal: self.primal / rhs.primal,
            tangent: (self.tangent * rhs.primal - self.primal * rhs.tangent)
                / (rhs.primal * rhs.primal),
        }
    }
}

impl Neg for Dual {
    type Output = Self;
    fn neg(self) -> Self {
        Self {
            primal: -self.primal,
            tangent: -self.tangent,
        }
    }
}

// ── SIMD operations stubs ──

pub mod ops {
    pub fn add_f32(a: &[f32], b: &[f32], out: &mut [f32]) {
        for i in 0..a.len().min(b.len()).min(out.len())
        {
            out[i] = a[i] + b[i];
        }
    }
    pub fn mul_f32(a: &[f32], b: &[f32], out: &mut [f32]) {
        for i in 0..a.len().min(b.len()).min(out.len())
        {
            out[i] = a[i] * b[i];
        }
    }
    pub fn add_f64(a: &[f64], b: &[f64], out: &mut [f64]) {
        for i in 0..a.len().min(b.len()).min(out.len())
        {
            out[i] = a[i] + b[i];
        }
    }
    pub fn mul_f64(a: &[f64], b: &[f64], out: &mut [f64]) {
        for i in 0..a.len().min(b.len()).min(out.len())
        {
            out[i] = a[i] * b[i];
        }
    }
}

pub fn simd_add_one(data: &mut [f32]) {
    for x in data
    {
        *x += 1.0;
    }
}

// ── GPU dispatch stub ──

pub mod dispatch {
    pub fn gpu_or_cpu<F, G, T>(_on_gpu: F, on_cpu: G) -> T
    where
        F: FnOnce() -> T,
        G: FnOnce() -> T,
    {
        on_cpu()
    }
}

// ── IA Bridge stubs ──

pub struct Pipeline {
    vars: HashMap<String, f64>,
}

#[derive(Debug, Clone)]
pub struct PipelineOutput {
    pub parsed: String,
    pub simplified: String,
    pub derivative: String,
    pub value: f64,
    pub rust_code: String,
}

impl Default for Pipeline {
    fn default() -> Self {
        Self::new()
    }
}

impl Pipeline {
    pub fn new() -> Self {
        Self {
            vars: HashMap::new(),
        }
    }
    pub fn with_vars(mut self, vars: HashMap<String, f64>) -> Self {
        self.vars = vars;
        self
    }
    pub fn run(&self, expr_str: &str) -> Result<PipelineOutput, String> {
        let expr = parse(expr_str)?;
        let simplified = simplify(&expr);
        let derivative = diff(&expr, "x");
        let value = eval(&simplified, &self.vars).unwrap_or(0.0);
        let rust_code = to_rust_code(&simplified);
        Ok(PipelineOutput {
            parsed: expr_str.to_string(),
            simplified: format!("{}", simplified),
            derivative: format!("{}", derivative),
            value,
            rust_code,
        })
    }
}

impl fmt::Display for PipelineOutput {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.simplified)
    }
}

#[derive(Debug)]
pub enum NaturalCommand {
    Evaluate(String),
    Solve(String),
    Derive(String),
}

pub fn parse_natural(input: &str) -> NaturalCommand {
    let lower = input.to_lowercase();
    if lower.contains("solve") || lower.contains("résous") || lower.contains("résoudre")
    {
        NaturalCommand::Solve(input.to_string())
    }
    else if lower.contains("derive") || lower.contains("dérive") || lower.contains("dérivée")
    {
        NaturalCommand::Derive(input.to_string())
    }
    else
    {
        NaturalCommand::Evaluate(input.to_string())
    }
}

// ── Derivative helpers for prelude ──

pub fn derivative_1d<F: Fn(f64) -> f64>(f: F, x: f64) -> f64 {
    let h = 1e-6;
    (f(x + h) - f(x - h)) / (2.0 * h)
}

pub fn gradient_2d<F: Fn(f64, f64) -> f64>(f: F, x: f64, y: f64) -> (f64, f64) {
    let h = 1e-6;
    let dx = (f(x + h, y) - f(x - h, y)) / (2.0 * h);
    let dy = (f(x, y + h) - f(x, y - h)) / (2.0 * h);
    (dx, dy)
}

pub fn gradient_3d<F: Fn(f64, f64, f64) -> f64>(f: F, x: f64, y: f64, z: f64) -> (f64, f64, f64) {
    let h = 1e-6;
    let dx = (f(x + h, y, z) - f(x - h, y, z)) / (2.0 * h);
    let dy = (f(x, y + h, z) - f(x, y - h, z)) / (2.0 * h);
    let dz = (f(x, y, z + h) - f(x, y, z - h)) / (2.0 * h);
    (dx, dy, dz)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_simple() {
        let e = parse("x + 1").unwrap();
        assert_eq!(format!("{}", e), "(x + 1)");
    }

    #[test]
    fn test_parse_mult() {
        let e = parse("2 * x").unwrap();
        assert_eq!(format!("{}", e), "(2 * x)");
    }

    #[test]
    fn test_simplify_const() {
        let e = simplify(&parse("1 + 2").unwrap());
        assert_eq!(e, Expr::Const(3.0));
    }

    #[test]
    fn test_diff_polynomial() {
        let e = parse("x^2").unwrap();
        let d = diff(&e, "x");
        // d(x^2)/dx = 2*x
        assert_eq!(format!("{}", simplify(&d)), "(2 * x)");
    }

    #[test]
    fn test_eval() {
        let e = parse("x^2 + 3*x + 1").unwrap();
        let mut vars = HashMap::new();
        vars.insert("x".to_string(), 2.0);
        let v = eval(&e, &vars).unwrap();
        assert!((v - 11.0).abs() < 1e-10);
    }

    #[test]
    fn test_dual() {
        // f(x) = x^2 at x=3: primal=9, tangent=2*3=6
        let x = Dual::var(3.0);
        let fx = x * x;
        assert_eq!(fx.primal, 9.0);
        assert_eq!(fx.tangent, 6.0);
    }

    #[test]
    fn test_dual_sin() {
        let x = Dual::var(0.0);
        let fx = x.sin();
        assert!((fx.primal - 0.0).abs() < 1e-10);
        assert!((fx.tangent - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_solve_quadratic() {
        // x^2 - 4 = 0 → roots ±2
        let e = parse("x^2 - 4").unwrap();
        let roots = solve_quadratic(&e, "x");
        assert!((roots[0] + 2.0).abs() < 1e-6);
        assert!((roots[1] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_polynomial_fit() {
        // y = 1 + 2x → should find [1, 2]
        let xs = vec![0.0, 1.0, 2.0];
        let ys = vec![1.0, 3.0, 5.0];
        let coeffs = polynomial_fit(&xs, &ys, 1).unwrap();
        assert!((coeffs[0] - 1.0).abs() < 1e-6);
        assert!((coeffs[1] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn solve_linear_finds_the_root() {
        // 2x - 4 = 0 → x = 2
        let e = parse("2*x - 4").unwrap();
        assert!((solve_linear(&e, "x").unwrap() - 2.0).abs() < 1e-9);
    }

    #[test]
    fn prove_equal_distinguishes_equivalent_from_different() {
        assert!(prove_equal(
            &parse("x + x").unwrap(),
            &parse("2*x").unwrap()
        ));
        assert!(!prove_equal(&parse("x").unwrap(), &parse("x + 1").unwrap()));
    }

    #[test]
    fn to_rust_code_emits_evaluable_source() {
        let code = to_rust_code(&parse("x^2 + 1").unwrap());
        assert!(code.contains(".powf(2)"), "got: {code}");
        assert!(code.contains("+ 1"), "got: {code}");
    }

    #[test]
    fn linear_regression_recovers_intercept_and_slope() {
        // y = 1 + 2x
        let (intercept, slope) =
            linear_regression(&[0.0, 1.0, 2.0, 3.0], &[1.0, 3.0, 5.0, 7.0]).unwrap();
        assert!((intercept - 1.0).abs() < 1e-6, "intercept {intercept}");
        assert!((slope - 2.0).abs() < 1e-6, "slope {slope}");
    }

    #[test]
    fn discover_patterns_detects_trend_and_stability() {
        assert!(
            discover_patterns(&[1.0, 2.0, 3.0, 4.0, 5.0]).contains(&"trend_upward".to_string())
        );
        assert!(discover_patterns(&[5.0, 5.0, 5.0, 5.0]).contains(&"stable".to_string()));
    }

    #[test]
    fn pattern_memory_round_trips() {
        let mut mem = PatternMemory::new();
        mem.store("ramp", vec![1.0, 2.0, 3.0]);
        assert_eq!(mem.recall("ramp"), Some(&[1.0, 2.0, 3.0][..]));
        assert!(mem.recall("absent").is_none());
    }

    #[test]
    fn finite_difference_helpers_match_known_gradients() {
        // d/dx x^2 at 3 = 6
        assert!((derivative_1d(|x| x * x, 3.0) - 6.0).abs() < 1e-4);
        // ∇(x² + y²) at (1,2) = (2,4)
        let (gx, gy) = gradient_2d(|x, y| x * x + y * y, 1.0, 2.0);
        assert!((gx - 2.0).abs() < 1e-4 && (gy - 4.0).abs() < 1e-4);
        // ∇(x² + y² + z²) at (1,2,3) = (2,4,6)
        let (a, b, c) = gradient_3d(|x, y, z| x * x + y * y + z * z, 1.0, 2.0, 3.0);
        assert!((a - 2.0).abs() < 1e-4 && (b - 4.0).abs() < 1e-4 && (c - 6.0).abs() < 1e-4);
    }

    #[test]
    fn optimizer_minimizes_a_quadratic_bowl() {
        // minimize (x-3)² + (y+1)² → (3, -1)
        let mut opt = Optimizer::new(0.1, 1000);
        let x = opt.minimize(|p| (p[0] - 3.0).powi(2) + (p[1] + 1.0).powi(2), &[0.0, 0.0]);
        assert!((x[0] - 3.0).abs() < 1e-2, "x0 = {}", x[0]);
        assert!((x[1] + 1.0).abs() < 1e-2, "x1 = {}", x[1]);
    }

    #[test]
    fn optimizer_step_applies_a_momentum_update() {
        // First step from zero velocity: v = -lr·grad, x += v.
        let mut opt = Optimizer::new(0.1, 1);
        let mut p = vec![1.0];
        opt.step(&mut p, &[2.0]); // v = -0.1·2 = -0.2 → p = 0.8
        assert!((p[0] - 0.8).abs() < 1e-12);
    }

    #[test]
    fn pipeline_parses_simplifies_and_evaluates() {
        let mut vars = HashMap::new();
        vars.insert("x".to_string(), 2.0);
        let out = Pipeline::new().with_vars(vars).run("x^2 + 1").unwrap();
        assert!((out.value - 5.0).abs() < 1e-9, "value {}", out.value);
        assert!(!out.rust_code.is_empty());
        assert!(!out.simplified.is_empty());
    }

    #[test]
    fn parse_natural_dispatches_intents() {
        assert!(matches!(
            parse_natural("solve x^2 = 4"),
            NaturalCommand::Solve(_)
        ));
        assert!(matches!(
            parse_natural("derive x^2"),
            NaturalCommand::Derive(_)
        ));
        assert!(matches!(
            parse_natural("2 + 2"),
            NaturalCommand::Evaluate(_)
        ));
    }

    #[test]
    fn simplify_drops_identities() {
        assert_eq!(
            simplify(&parse("x * 1").unwrap()),
            Expr::Var("x".to_string())
        );
        assert_eq!(
            simplify(&parse("x + 0").unwrap()),
            Expr::Var("x".to_string())
        );
    }

    #[test]
    fn trig_identity_rewrites_sin_squared_preserving_value() {
        // sin²(x) → (1 - cos 2x)/2 is a TRUE identity: the rewrite must
        // change the expression yet evaluate identically everywhere.
        let sin2 = parse("sin(x)^2").unwrap();
        let out = apply_trig_identity(&sin2);
        assert_ne!(out, sin2, "the identity should fire on the square");
        assert!(prove_equal(&out, &sin2), "sin² rewrite changed the value");
        // Hand check at x = 0.7: sin(0.7)² = 0.41501642…
        let mut b = HashMap::new();
        b.insert("x".to_string(), 0.7);
        assert!((eval(&out, &b).unwrap() - 0.7_f64.sin().powi(2)).abs() < 1e-12);
    }

    #[test]
    fn trig_identity_rewrites_cos_squared_preserving_value() {
        // cos²(x) → (1 + cos 2x)/2, also a true identity.
        let cos2 = parse("cos(x)^2").unwrap();
        let out = apply_trig_identity(&cos2);
        assert_ne!(out, cos2, "the identity should fire on the square");
        assert!(prove_equal(&out, &cos2), "cos² rewrite changed the value");
        let mut b = HashMap::new();
        b.insert("x".to_string(), 0.7);
        assert!((eval(&out, &b).unwrap() - 0.7_f64.cos().powi(2)).abs() < 1e-12);
    }

    #[test]
    fn trig_identity_leaves_non_square_powers_untouched() {
        // The half-angle identity is FALSE for any exponent ≠ 2; firing on
        // sin³ or cos⁵ would silently corrupt the expression. Regression
        // guard for the `Pow(_, _)` wildcard that used to match any power.
        let sin3 = parse("sin(x)^3").unwrap();
        assert_eq!(apply_trig_identity(&sin3), sin3, "must not rewrite a cube");
        let cos5 = parse("cos(x)^5").unwrap();
        assert_eq!(
            apply_trig_identity(&cos5),
            cos5,
            "must not rewrite a fifth power"
        );
    }

    #[test]
    fn trig_identity_leaves_unrelated_expressions_untouched() {
        // A square of a non-trig base, and non-Pow nodes, are returned
        // verbatim — the rewriter is a single, exact rule.
        let poly = parse("x^2").unwrap();
        assert_eq!(apply_trig_identity(&poly), poly, "x² is not a trig square");
        let e = parse("sin(x) + 1").unwrap();
        assert_eq!(apply_trig_identity(&e), e);
    }
}

pub mod prelude;
