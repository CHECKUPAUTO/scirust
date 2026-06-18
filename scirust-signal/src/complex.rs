use core::ops::{Add, AddAssign, Mul, MulAssign, Sub, SubAssign};

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
