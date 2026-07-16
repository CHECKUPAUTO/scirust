// scirust-simd/src/dsp/fft.rs
//
// # Transformée de Fourier rapide générique (radix-2 Cooley–Tukey)
//
// FFT **générique sur le scalaire** : le même code transforme des signaux en
// `f32`, `f64` **et** `FixedI32<FRAC>` (virgule fixe déterministe, **bit-à-bit**
// reproductible sur toute architecture). Algorithme radix-2 à **entrelacement
// temporel** (DIT), itératif et **en place** (permutation par inversion de bits
// puis papillons).
//
// Les facteurs de rotation (« twiddles ») `Wₙᵏ = e^{−2iπk/n}` sont calculés via
// [`RealScalar`] (`sin`/`cos`) : c'est ce qui rend la FFT disponible en virgule
// fixe. La longueur doit être une **puissance de 2**.
//
// ## Précision en virgule fixe
//
// Chaque étage (`log₂ n`) accumule l'arrondi des twiddles et des papillons ; les
// magnitudes croissent jusqu'à `≈ n·max|x|`. Pour un signal dans `[−1, 1]` et
// `n ≤ 2¹⁵`, aucun débordement en `Q16_16` (bin ≤ n < 32768). Pour de longues
// FFT ou une meilleure fidélité, préférer un `FRAC` large. Aucune allocation
// dans le cœur (`fft`/`ifft` opèrent en place) ; aucun `unsafe`.

use core::ops::{Add, Mul, Sub};

use crate::fixed::{NumericScalar, RealScalar};

/// Nombre complexe `re + i·im`, générique sur le scalaire. Les opérateurs
/// `+ − *` réalisent l'arithmétique complexe usuelle.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Complex<T> {
    /// Partie réelle.
    pub re: T,
    /// Partie imaginaire.
    pub im: T,
}

impl<T: NumericScalar> Complex<T> {
    /// Construit `re + i·im`.
    #[inline]
    pub fn new(re: T, im: T) -> Self {
        Self { re, im }
    }

    /// Complexe réel pur `re + 0i`.
    #[inline]
    pub fn from_real(re: T) -> Self {
        Self::new(re, T::zero())
    }

    /// Zéro complexe.
    #[inline]
    pub fn zero() -> Self {
        Self::new(T::zero(), T::zero())
    }

    /// Multiplication par un scalaire réel.
    #[inline]
    pub fn scale(self, s: T) -> Self {
        Self::new(self.re * s, self.im * s)
    }

    /// Conjugué `re − i·im`.
    #[inline]
    pub fn conj(self) -> Self {
        Self::new(self.re, -self.im)
    }

    /// Module au carré `re² + im²`.
    #[inline]
    pub fn norm_sqr(self) -> T {
        self.re * self.re + self.im * self.im
    }
}

impl<T: NumericScalar> Add for Complex<T> {
    type Output = Self;
    #[inline]
    fn add(self, r: Self) -> Self {
        Self::new(self.re + r.re, self.im + r.im)
    }
}
impl<T: NumericScalar> Sub for Complex<T> {
    type Output = Self;
    #[inline]
    fn sub(self, r: Self) -> Self {
        Self::new(self.re - r.re, self.im - r.im)
    }
}
/// Produit complexe `(ac − bd) + (ad + bc)i`.
impl<T: NumericScalar> Mul for Complex<T> {
    type Output = Self;
    #[inline]
    fn mul(self, r: Self) -> Self {
        Self::new(
            self.re * r.re - self.im * r.im,
            self.re * r.im + self.im * r.re,
        )
    }
}

/// Inverse l'ordre des `bits` bits de poids faible de `x`.
#[inline]
fn reverse_bits(x: usize, bits: u32) -> usize {
    let mut r = 0usize;
    let mut v = x;
    for _ in 0..bits
    {
        r = (r << 1) | (v & 1);
        v >>= 1;
    }
    r
}

/// Permutation par inversion de bits (mise en place radix-2).
fn bit_reverse_permute<T: NumericScalar>(data: &mut [Complex<T>]) {
    let n = data.len();
    let bits = n.trailing_zeros();
    for i in 0..n
    {
        let j = reverse_bits(i, bits);
        if j > i
        {
            data.swap(i, j);
        }
    }
}

/// FFT directe **en place** (radix-2 DIT). `data.len()` doit être une puissance
/// de 2. Convention `X[k] = Σₙ x[n]·e^{−2iπkn/N}` (non normalisée).
pub fn fft<T: RealScalar>(data: &mut [Complex<T>]) {
    let n = data.len();
    assert!(
        n.is_power_of_two(),
        "fft: la longueur doit être une puissance de 2"
    );
    if n <= 1
    {
        return;
    }
    bit_reverse_permute(data);

    let neg_two_pi = T::from_i32(-2) * T::pi();
    let mut len = 2usize;
    while len <= n
    {
        let half = len / 2;
        let inv_len = T::from_i32(len as i32).recip();
        for start in (0..n).step_by(len)
        {
            for k in 0..half
            {
                // Wₗₑₙᵏ = e^{−2iπk/len}.
                let angle = neg_two_pi * T::from_i32(k as i32) * inv_len;
                let w = Complex::new(angle.cos(), angle.sin());
                let t = w * data[start + k + half];
                let u = data[start + k];
                data[start + k] = u + t;
                data[start + k + half] = u - t;
            }
        }
        len <<= 1;
    }
}

/// FFT inverse **en place**, normalisée par `1/N` : `ifft(fft(x)) ≈ x`.
///
/// Implémentée via `ifft(X) = conj(fft(conj(X))) / N`.
pub fn ifft<T: RealScalar>(data: &mut [Complex<T>]) {
    let n = data.len();
    if n == 0
    {
        return;
    }
    for c in data.iter_mut()
    {
        *c = c.conj();
    }
    fft(data);
    let inv_n = T::from_i32(n as i32).recip();
    for c in data.iter_mut()
    {
        *c = c.conj().scale(inv_n);
    }
}
