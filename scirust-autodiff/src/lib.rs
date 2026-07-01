use std::cell::RefCell;
use std::ops::{Add, Div, Mul, Neg, Sub};

/// Dual number for forward-mode automatic differentiation.
///
/// A dual number `x + ε·x'` where `ε² = 0`.
/// When evaluating a function with dual numbers, the derivative
/// propagates automatically through the computation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Dual {
    pub value: f64,
    pub deriv: f64,
}

impl Dual {
    /// Create a new dual number.
    /// `value` is the primal value, `deriv` is the derivative (seed).
    pub fn new(value: f64, deriv: f64) -> Self {
        Dual { value, deriv }
    }

    /// Create a primal (deriv = 0).
    pub fn primal(value: f64) -> Self {
        Dual { value, deriv: 0.0 }
    }

    /// Create a variable with unit derivative (deriv = 1).
    pub fn var(value: f64) -> Self {
        Dual { value, deriv: 1.0 }
    }

    /// Extract the primal value.
    pub fn val(self) -> f64 {
        self.value
    }

    /// Extract the derivative.
    pub fn grad(self) -> f64 {
        self.deriv
    }
}

/// Combine a local derivative `factor` with a seed `deriv` via the chain rule.
///
/// This is `factor * deriv`, except that a zero seed always contributes exactly
/// `0.0`. Without this guard, a constant (a primal with `deriv == 0`) evaluated
/// at a domain edge where `factor` is non-finite (e.g. `1 / (2·√0)`) would turn
/// `0 · ∞` into `NaN` and poison the gradient of unrelated variables when
/// partial derivatives are taken one variable at a time.
#[inline]
fn chain(factor: f64, deriv: f64) -> f64 {
    if deriv == 0.0 { 0.0 } else { factor * deriv }
}

// ---------------------------------------------------------------------------
// Arithmetic operators
// ---------------------------------------------------------------------------

impl Add for Dual {
    type Output = Dual;
    fn add(self, rhs: Dual) -> Dual {
        Dual {
            value: self.value + rhs.value,
            deriv: self.deriv + rhs.deriv,
        }
    }
}

impl Sub for Dual {
    type Output = Dual;
    fn sub(self, rhs: Dual) -> Dual {
        Dual {
            value: self.value - rhs.value,
            deriv: self.deriv - rhs.deriv,
        }
    }
}

impl Mul for Dual {
    type Output = Dual;
    fn mul(self, rhs: Dual) -> Dual {
        // product rule: (f·g)' = f'·g + f·g'
        Dual {
            value: self.value * rhs.value,
            deriv: self.deriv * rhs.value + self.value * rhs.deriv,
        }
    }
}

impl Div for Dual {
    type Output = Dual;
    fn div(self, rhs: Dual) -> Dual {
        // quotient rule: (f/g)' = f'/g - f·g'/g²
        // Split into per-operand chain-rule contributions so that a constant
        // operand (seed 0) never turns a non-finite factor into a NaN.
        let denom = rhs.value * rhs.value;
        Dual {
            value: self.value / rhs.value,
            deriv: chain(1.0 / rhs.value, self.deriv) - chain(self.value / denom, rhs.deriv),
        }
    }
}

impl Neg for Dual {
    type Output = Dual;
    fn neg(self) -> Dual {
        Dual {
            value: -self.value,
            deriv: -self.deriv,
        }
    }
}

// ---------------------------------------------------------------------------
// Scalar ops (f64 on left and right)
// ---------------------------------------------------------------------------

impl Add<f64> for Dual {
    type Output = Dual;
    fn add(self, rhs: f64) -> Dual {
        Dual {
            value: self.value + rhs,
            deriv: self.deriv,
        }
    }
}

impl Add<Dual> for f64 {
    type Output = Dual;
    fn add(self, rhs: Dual) -> Dual {
        Dual {
            value: self + rhs.value,
            deriv: rhs.deriv,
        }
    }
}

impl Sub<f64> for Dual {
    type Output = Dual;
    fn sub(self, rhs: f64) -> Dual {
        Dual {
            value: self.value - rhs,
            deriv: self.deriv,
        }
    }
}

impl Sub<Dual> for f64 {
    type Output = Dual;
    fn sub(self, rhs: Dual) -> Dual {
        Dual {
            value: self - rhs.value,
            deriv: -rhs.deriv,
        }
    }
}

impl Mul<f64> for Dual {
    type Output = Dual;
    fn mul(self, rhs: f64) -> Dual {
        Dual {
            value: self.value * rhs,
            deriv: self.deriv * rhs,
        }
    }
}

impl Mul<Dual> for f64 {
    type Output = Dual;
    fn mul(self, rhs: Dual) -> Dual {
        Dual {
            value: self * rhs.value,
            deriv: self * rhs.deriv,
        }
    }
}

impl Div<f64> for Dual {
    type Output = Dual;
    fn div(self, rhs: f64) -> Dual {
        Dual {
            value: self.value / rhs,
            deriv: self.deriv / rhs,
        }
    }
}

impl Div<Dual> for f64 {
    type Output = Dual;
    fn div(self, rhs: Dual) -> Dual {
        let denom = rhs.value * rhs.value;
        Dual {
            value: self / rhs.value,
            deriv: -chain(self / denom, rhs.deriv),
        }
    }
}

// ---------------------------------------------------------------------------
// Math functions
// ---------------------------------------------------------------------------

impl Dual {
    pub fn powi(self, n: i32) -> Dual {
        // d/dx(x^n) = n·x^(n-1)
        let pow_val = self.value.powi(n);
        // x^0 is the constant 1, so its derivative is 0 everywhere (including
        // x = 0, where n·x^(n-1) would otherwise evaluate to 0·∞ = NaN).
        let pow_deriv = if n == 0
        {
            0.0
        }
        else
        {
            chain(n as f64 * self.value.powi(n - 1), self.deriv)
        };
        Dual {
            value: pow_val,
            deriv: pow_deriv,
        }
    }

    pub fn powf(self, n: f64) -> Dual {
        let pow_val = self.value.powf(n);
        let pow_deriv = chain(n * self.value.powf(n - 1.0), self.deriv);
        Dual {
            value: pow_val,
            deriv: pow_deriv,
        }
    }

    pub fn sqrt(self) -> Dual {
        let s = self.value.sqrt();
        Dual {
            value: s,
            deriv: chain(1.0 / (2.0 * s), self.deriv),
        }
    }

    pub fn exp(self) -> Dual {
        let e = self.value.exp();
        Dual {
            value: e,
            deriv: e * self.deriv,
        }
    }

    pub fn ln(self) -> Dual {
        Dual {
            value: self.value.ln(),
            deriv: chain(1.0 / self.value, self.deriv),
        }
    }

    pub fn sin(self) -> Dual {
        Dual {
            value: self.value.sin(),
            deriv: self.deriv * self.value.cos(),
        }
    }

    pub fn cos(self) -> Dual {
        Dual {
            value: self.value.cos(),
            deriv: -self.deriv * self.value.sin(),
        }
    }

    pub fn tan(self) -> Dual {
        let c = self.value.cos();
        Dual {
            value: self.value.tan(),
            deriv: self.deriv / (c * c),
        }
    }

    pub fn abs(self) -> Dual {
        Dual {
            value: self.value.abs(),
            deriv: if self.value > 0.0
            {
                self.deriv
            }
            else if self.value < 0.0
            {
                -self.deriv
            }
            else
            {
                0.0
            },
        }
    }
}

// ---------------------------------------------------------------------------
// Reverse-mode AutoDiff
// ---------------------------------------------------------------------------

/// A node in the computation graph for reverse-mode AD.
#[derive(Debug)]
pub struct Node {
    pub value: f64,
    pub grad: f64,
    pub deps: Vec<(usize, f64)>, // (index_in_tape, partial_derivative)
}

/// Tape (Wengert list) for reverse-mode AD.
pub struct Tape {
    pub nodes: RefCell<Vec<Node>>,
}

impl Default for Tape {
    fn default() -> Self {
        Self::new()
    }
}

impl Tape {
    pub fn new() -> Self {
        Tape {
            nodes: RefCell::new(Vec::new()),
        }
    }

    pub fn var(&self, value: f64) -> Var<'_> {
        let mut nodes = self.nodes.borrow_mut();
        let idx = nodes.len();
        nodes.push(Node {
            value,
            grad: 0.0,
            deps: Vec::new(),
        });
        Var { tape: self, idx }
    }

    pub fn backward(&self, out_idx: usize) {
        let mut nodes = self.nodes.borrow_mut();
        for node in nodes.iter_mut()
        {
            node.grad = 0.0;
        }
        nodes[out_idx].grad = 1.0;

        for i in (0..nodes.len()).rev()
        {
            let grad = nodes[i].grad;
            let deps = nodes[i].deps.clone();
            for (dep_idx, partial) in deps
            {
                nodes[dep_idx].grad += grad * partial;
            }
        }
    }
}

/// A variable in reverse-mode AD.
#[derive(Clone, Copy)]
pub struct Var<'a> {
    pub tape: &'a Tape,
    pub idx: usize,
}

impl<'a> Var<'a> {
    pub fn value(&self) -> f64 {
        self.tape.nodes.borrow()[self.idx].value
    }

    pub fn grad(&self) -> f64 {
        self.tape.nodes.borrow()[self.idx].grad
    }

    fn push_op(&self, value: f64, deps: Vec<(usize, f64)>) -> Var<'a> {
        let mut nodes = self.tape.nodes.borrow_mut();
        let idx = nodes.len();
        nodes.push(Node {
            value,
            grad: 0.0,
            deps,
        });
        Var {
            tape: self.tape,
            idx,
        }
    }

    pub fn powi(self, n: i32) -> Var<'a> {
        let val = self.value().powi(n);
        let deriv = n as f64 * self.value().powi(n - 1);
        self.push_op(val, vec![(self.idx, deriv)])
    }

    pub fn exp(self) -> Var<'a> {
        let val = self.value().exp();
        self.push_op(val, vec![(self.idx, val)])
    }

    pub fn sin(self) -> Var<'a> {
        let val = self.value().sin();
        let deriv = self.value().cos();
        self.push_op(val, vec![(self.idx, deriv)])
    }

    pub fn cos(self) -> Var<'a> {
        let val = self.value().cos();
        let deriv = -self.value().sin();
        self.push_op(val, vec![(self.idx, deriv)])
    }
}

impl<'a> Add for Var<'a> {
    type Output = Var<'a>;
    fn add(self, rhs: Var<'a>) -> Var<'a> {
        self.push_op(
            self.value() + rhs.value(),
            vec![(self.idx, 1.0), (rhs.idx, 1.0)],
        )
    }
}

impl<'a> Sub for Var<'a> {
    type Output = Var<'a>;
    fn sub(self, rhs: Var<'a>) -> Var<'a> {
        self.push_op(
            self.value() - rhs.value(),
            vec![(self.idx, 1.0), (rhs.idx, -1.0)],
        )
    }
}

impl<'a> Mul for Var<'a> {
    type Output = Var<'a>;
    fn mul(self, rhs: Var<'a>) -> Var<'a> {
        self.push_op(
            self.value() * rhs.value(),
            vec![(self.idx, rhs.value()), (rhs.idx, self.value())],
        )
    }
}

impl<'a> Div for Var<'a> {
    type Output = Var<'a>;
    fn div(self, rhs: Var<'a>) -> Var<'a> {
        let val = self.value() / rhs.value();
        let d_lhs = 1.0 / rhs.value();
        let d_rhs = -self.value() / (rhs.value() * rhs.value());
        self.push_op(val, vec![(self.idx, d_lhs), (rhs.idx, d_rhs)])
    }
}

impl<'a> Neg for Var<'a> {
    type Output = Var<'a>;
    fn neg(self) -> Var<'a> {
        self.push_op(-self.value(), vec![(self.idx, -1.0)])
    }
}

// ---------------------------------------------------------------------------
// Utility: gradient extraction helpers
// ---------------------------------------------------------------------------

/// Evaluate `f` with a dual-number seed to obtain the exact derivative.
pub fn derivative_1d<F>(f: F, x: f64) -> f64
where
    F: Fn(Dual) -> Dual,
{
    let x_dual = Dual::var(x);
    f(x_dual).grad()
}

/// Evaluate `f` with respect to each variable and return all partial derivatives.
pub fn gradient_2d<F>(f: F, x: f64, y: f64) -> (f64, f64)
where
    F: Fn(Dual, Dual) -> Dual,
{
    let dx = f(Dual::var(x), Dual::primal(y)).grad();
    let dy = f(Dual::primal(x), Dual::var(y)).grad();
    (dx, dy)
}

/// Evaluate `f` with respect to each variable and return all partial derivatives.
pub fn gradient_3d<F>(f: F, x: f64, y: f64, z: f64) -> (f64, f64, f64)
where
    F: Fn(Dual, Dual, Dual) -> Dual,
{
    let dx = f(Dual::var(x), Dual::primal(y), Dual::primal(z)).grad();
    let dy = f(Dual::primal(x), Dual::var(y), Dual::primal(z)).grad();
    let dz = f(Dual::primal(x), Dual::primal(y), Dual::var(z)).grad();
    (dx, dy, dz)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_square() {
        let x = Dual::var(3.0);
        let y = x * x;
        assert!((y.val() - 9.0).abs() < 1e-12);
        assert!((y.grad() - 6.0).abs() < 1e-12);
    }

    #[test]
    fn test_sin() {
        let x = Dual::var(std::f64::consts::PI / 2.0);
        let y = x.sin();
        assert!((y.val() - 1.0).abs() < 1e-12);
        assert!((y.grad() - 0.0).abs() < 1e-12); // cos(π/2) = 0
    }

    #[test]
    fn test_rosenbrock() {
        let x = Dual::var(1.0);
        let y = Dual::primal(1.0);
        let f = (Dual::primal(1.0) - x).powi(2) + Dual::primal(100.0) * (y - x * x).powi(2);
        assert!((f.grad()).abs() < 1e-10);
    }

    #[test]
    fn test_derivative_1d() {
        let d = derivative_1d(|x| x * x + x.sin(), 1.0);
        // d/dx(x² + sin(x)) = 2x + cos(x) = 2 + cos(1) ≈ 2.5403
        let expected = 2.0 + 1.0f64.cos();
        assert!((d - expected).abs() < 1e-10);
    }

    #[test]
    fn test_reverse_mode_simple() {
        let tape = Tape::new();
        let x = tape.var(3.0);
        let y = tape.var(2.0);
        let z = x * x + x * y;
        // z = x^2 + xy
        // dz/dx = 2x + y = 2(3) + 2 = 8
        // dz/dy = x = 3
        tape.backward(z.idx);
        assert_eq!(x.grad(), 8.0);
        assert_eq!(y.grad(), 3.0);
    }

    #[test]
    fn test_reverse_mode_complex() {
        let tape = Tape::new();
        let x = tape.var(1.0);
        let y = (x.sin() * x.exp()) / x.powi(2);
        // d/dx(sin(x)e^x / x^2)
        // = ( (cos(x)e^x + sin(x)e^x)x^2 - 2x sin(x)e^x ) / x^4
        // at x=1:
        // = ( (cos(1)e + sin(1)e) - 2 sin(1)e ) / 1
        // = e * (cos(1) - sin(1))
        tape.backward(y.idx);
        let expected = 1.0f64.exp() * (1.0f64.cos() - 1.0f64.sin());
        assert!((x.grad() - expected).abs() < 1e-10);
    }

    // -- Domain-edge derivative regressions (formerly NaN/Inf) -------------

    #[test]
    fn test_sqrt_zero_constant_no_nan() {
        // sqrt(0) has value 0 but the naive derivative 0/(2·0) = NaN.
        // A held-constant operand (primal) must not poison the seed.
        let d = Dual::primal(0.0).sqrt();
        assert_eq!(d.val(), 0.0);
        assert!(
            d.grad().is_finite(),
            "sqrt(0) constant deriv was {}",
            d.grad()
        );
        assert_eq!(d.grad(), 0.0);
    }

    #[test]
    fn test_sqrt_zero_partial_not_poisoned() {
        // f(x, y) = x + sqrt(y); ∂f/∂x = 1 for all y, even at y = 0.
        let (dx, _dy) = gradient_2d(|x, y| x + y.sqrt(), 5.0, 0.0);
        assert_eq!(dx, 1.0);
    }

    #[test]
    fn test_powi_zero_exponent_at_zero() {
        // x^0 ≡ 1, so d/dx(x^0) = 0 everywhere, including x = 0
        // (naive n·x^(n-1) = 0·∞ = NaN there).
        let d = Dual::var(0.0).powi(0);
        assert_eq!(d.val(), 1.0);
        assert_eq!(d.grad(), 0.0);
    }

    #[test]
    fn test_powf_zero_constant_no_nan() {
        // primal(0)^0.5 has value 0 but naive deriv 0.5·0^(-0.5)·0 = NaN.
        let d = Dual::primal(0.0).powf(0.5);
        assert_eq!(d.val(), 0.0);
        assert_eq!(d.grad(), 0.0);
    }

    #[test]
    fn test_ln_zero_partial_not_poisoned() {
        // f(x, y) = x + ln(y); ∂f/∂x = 1 regardless of y, even y = 0
        // (where ln(y) itself is -∞ but must not corrupt ∂f/∂x).
        let (dx, _dy) = gradient_2d(|x, y| x + y.ln(), 3.0, 0.0);
        assert_eq!(dx, 1.0);
    }

    #[test]
    fn test_div_constant_denominator_zero_not_poisoned() {
        // f(x, y) = x + a/y with a a constant. ∂f/∂x = 1 even at y = 0,
        // where the a/y term's derivative factor is non-finite.
        let (dx, _dy) = gradient_2d(|x, y| x + Dual::primal(2.0) / y, 4.0, 0.0);
        assert_eq!(dx, 1.0);
    }

    #[test]
    fn test_div_regular_case_unchanged() {
        // Refactored quotient rule must still be correct on ordinary input.
        // d/dx((x+1)/x) at x = 2 = -1/x² = -0.25.
        let x = Dual::var(2.0);
        let f = (x + 1.0) / x;
        assert!((f.val() - 1.5).abs() < 1e-12);
        assert!((f.grad() - (-0.25)).abs() < 1e-12);
    }
}
