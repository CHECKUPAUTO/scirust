use core::ops::{Add, AddAssign, Div, Mul, MulAssign, Neg, Sub, SubAssign};

/// A simple complex number with `f64` real and imaginary parts.
/// Used by the FFT and signal analysis routines.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Complex {
    pub re: f64,
    pub im: f64,
}

impl Complex {
    #[inline]
    pub const fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }

    #[inline]
    pub const fn zero() -> Self {
        Self { re: 0.0, im: 0.0 }
    }

    /// Magnitude (absolute value).
    #[inline]
    pub fn mag(&self) -> f64 {
        f64::sqrt(self.re * self.re + self.im * self.im)
    }

    /// Squared magnitude (faster, avoids sqrt).
    #[inline]
    pub fn mag_sq(&self) -> f64 {
        self.re * self.re + self.im * self.im
    }

    /// Phase angle in radians.
    #[inline]
    pub fn phase(&self) -> f64 {
        f64::atan2(self.im, self.re)
    }

    /// Complex conjugate.
    #[inline]
    pub fn conj(&self) -> Self {
        Self {
            re: self.re,
            im: -self.im,
        }
    }

    /// Euler's formula: e^(i*theta).
    #[inline]
    pub fn cis(theta: f64) -> Self {
        Self {
            re: f64::cos(theta),
            im: f64::sin(theta),
        }
    }
}

impl Add for Complex {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        Self {
            re: self.re + rhs.re,
            im: self.im + rhs.im,
        }
    }
}

impl AddAssign for Complex {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        self.re += rhs.re;
        self.im += rhs.im;
    }
}

impl Sub for Complex {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        Self {
            re: self.re - rhs.re,
            im: self.im - rhs.im,
        }
    }
}

impl SubAssign for Complex {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        self.re -= rhs.re;
        self.im -= rhs.im;
    }
}

impl Mul for Complex {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        Self {
            re: self.re * rhs.re - self.im * rhs.im,
            im: self.re * rhs.im + self.im * rhs.re,
        }
    }
}

impl MulAssign for Complex {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        let re = self.re * rhs.re - self.im * rhs.im;
        let im = self.re * rhs.im + self.im * rhs.re;
        self.re = re;
        self.im = im;
    }
}

impl Mul<f64> for Complex {
    type Output = Self;
    #[inline]
    fn mul(self, scalar: f64) -> Self::Output {
        Self {
            re: self.re * scalar,
            im: self.im * scalar,
        }
    }
}

impl Mul<Complex> for f64 {
    type Output = Complex;
    #[inline]
    fn mul(self, c: Complex) -> Self::Output {
        Complex {
            re: self * c.re,
            im: self * c.im,
        }
    }
}

impl Neg for Complex {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self::Output {
        Self {
            re: -self.re,
            im: -self.im,
        }
    }
}

impl Div for Complex {
    type Output = Self;
    #[inline]
    fn div(self, rhs: Self) -> Self::Output {
        let denom = rhs.mag_sq();
        Self {
            re: (self.re * rhs.re + self.im * rhs.im) / denom,
            im: (self.im * rhs.re - self.re * rhs.im) / denom,
        }
    }
}

impl Div<f64> for Complex {
    type Output = Self;
    #[inline]
    fn div(self, scalar: f64) -> Self::Output {
        Self {
            re: self.re / scalar,
            im: self.im / scalar,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: Complex, b: Complex, tol: f64) -> bool {
        (a.re - b.re).abs() < tol && (a.im - b.im).abs() < tol
    }

    #[test]
    fn neg_flips_both_parts() {
        let c = Complex::new(3.0, -4.0);
        assert_eq!(-c, Complex::new(-3.0, 4.0));
    }

    #[test]
    fn div_by_complex_is_inverse_of_mul() {
        let a = Complex::new(3.0, 2.0);
        let b = Complex::new(1.0, -5.0);
        let product = a * b;
        assert!(close(product / b, a, 1e-12));
    }

    #[test]
    fn div_by_scalar_matches_componentwise() {
        let a = Complex::new(6.0, -9.0);
        assert!(close(a / 3.0, Complex::new(2.0, -3.0), 1e-12));
    }

    #[test]
    fn div_matches_known_value() {
        // (4+2i)/(1+i) = (4+2i)(1-i)/2 = (4-4i+2i-2i^2)/2 = (6-2i)/2 = 3-i
        let z = Complex::new(4.0, 2.0) / Complex::new(1.0, 1.0);
        assert!(close(z, Complex::new(3.0, -1.0), 1e-12));
    }
}
