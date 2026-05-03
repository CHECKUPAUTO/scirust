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
        // quotient rule: (f/g)' = (f'·g - f·g') / g²
        let denom = rhs.value * rhs.value;
        Dual {
            value: self.value / rhs.value,
            deriv: (self.deriv * rhs.value - self.value * rhs.deriv) / denom,
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
            deriv: (-self * rhs.deriv) / denom,
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
        let pow_deriv = n as f64 * self.value.powi(n - 1) * self.deriv;
        Dual {
            value: pow_val,
            deriv: pow_deriv,
        }
    }

    pub fn powf(self, n: f64) -> Dual {
        let pow_val = self.value.powf(n);
        let pow_deriv = n * self.value.powf(n - 1.0) * self.deriv;
        Dual {
            value: pow_val,
            deriv: pow_deriv,
        }
    }

    pub fn sqrt(self) -> Dual {
        let s = self.value.sqrt();
        Dual {
            value: s,
            deriv: self.deriv / (2.0 * s),
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
            deriv: self.deriv / self.value,
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
            deriv: if self.value > 0.0 {
                self.deriv
            } else if self.value < 0.0 {
                -self.deriv
            } else {
                0.0
            },
        }
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
}
