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

use core::ops::{Add, Div, Mul, Sub};

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

impl<T: RealScalar> Complex<T> {
    /// Réciproque complexe `1/z = z̄ / |z|²` (`z = 0` : saturation, cf.
    /// [`RealScalar::recip`] — pas d'infini en virgule fixe).
    #[inline]
    pub fn recip(self) -> Self {
        self.conj().scale(self.norm_sqr().recip())
    }
}

/// Division complexe `a/b = a·b⁻¹` (cf. [`Complex::recip`]) — utilisée par
/// [`super::freqz`] pour évaluer une fonction de transfert rationnelle
/// `H(z) = N(z)/D(z)`.
impl<T: RealScalar> Div for Complex<T> {
    type Output = Self;
    #[inline]
    #[allow(clippy::suspicious_arithmetic_impl)] // a/b = a·b⁻¹, cf. Complex::recip.
    fn div(self, r: Self) -> Self {
        self * r.recip()
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

/// Plan de FFT de longueur fixe : **précalcule** les facteurs de rotation et la
/// permutation par inversion de bits, une seule fois. Chaque transformation
/// réutilise ces tables (aucun `sin`/`cos` recalculé, **aucune allocation par
/// transformation**) — nettement plus rapide sur le chemin virgule fixe.
///
/// Les twiddles sont calculés avec **exactement** la même expression d'angle que
/// [`fft`], donc [`Plan::fft`] produit un résultat **bit-à-bit identique** à la
/// fonction libre (vérifié par test en virgule fixe).
#[derive(Debug, Clone)]
pub struct Plan<T> {
    n: usize,
    /// `stages[s]` = les `len/2` twiddles de l'étage `len = 2^{s+1}`.
    stages: Vec<Vec<Complex<T>>>,
    /// Permutation par inversion de bits.
    rev: Vec<usize>,
    /// `1/n` (normalisation de l'inverse).
    inv_n: T,
}

impl<T: RealScalar> Plan<T> {
    /// Prépare un plan pour une longueur `n` (puissance de 2).
    #[must_use]
    pub fn new(n: usize) -> Self {
        assert!(
            n.is_power_of_two(),
            "Plan: la longueur doit être une puissance de 2"
        );
        let bits = n.trailing_zeros();
        let rev = (0..n).map(|i| reverse_bits(i, bits)).collect();

        let neg_two_pi = T::from_i32(-2) * T::pi();
        let mut stages = Vec::new();
        let mut len = 2usize;
        while len <= n
        {
            let half = len / 2;
            let inv_len = T::from_i32(len as i32).recip();
            let twiddles = (0..half)
                .map(|k| {
                    let angle = neg_two_pi * T::from_i32(k as i32) * inv_len;
                    Complex::new(angle.cos(), angle.sin())
                })
                .collect();
            stages.push(twiddles);
            len <<= 1;
        }
        Self {
            n,
            stages,
            rev,
            inv_n: T::from_i32(n.max(1) as i32).recip(),
        }
    }

    /// Longueur du plan.
    #[inline]
    pub fn len(&self) -> usize {
        self.n
    }

    /// `true` si le plan est de longueur nulle.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    /// FFT directe **en place** via les tables précalculées. Panique si
    /// `data.len() != self.len()`.
    pub fn fft(&self, data: &mut [Complex<T>]) {
        assert_eq!(data.len(), self.n, "Plan::fft: longueur incompatible");
        if self.n <= 1
        {
            return;
        }
        for i in 0..self.n
        {
            let j = self.rev[i];
            if j > i
            {
                data.swap(i, j);
            }
        }
        let mut len = 2usize;
        for twiddles in &self.stages
        {
            let half = len / 2;
            for start in (0..self.n).step_by(len)
            {
                for (k, &w) in twiddles.iter().enumerate()
                {
                    let t = w * data[start + k + half];
                    let u = data[start + k];
                    data[start + k] = u + t;
                    data[start + k + half] = u - t;
                }
            }
            len <<= 1;
        }
    }

    /// FFT inverse **en place**, normalisée `1/N`, via les tables précalculées.
    pub fn ifft(&self, data: &mut [Complex<T>]) {
        assert_eq!(data.len(), self.n, "Plan::ifft: longueur incompatible");
        if self.n == 0
        {
            return;
        }
        for c in data.iter_mut()
        {
            *c = c.conj();
        }
        self.fft(data);
        for c in data.iter_mut()
        {
            *c = c.conj().scale(self.inv_n);
        }
    }
}

// ------------------------------------------------------------------ //
//  FFT à entrée réelle (rfft / irfft)                                 //
// ------------------------------------------------------------------ //

/// FFT d'un signal **réel** de longueur `n` (puissance de 2), renvoyant les
/// `n/2 + 1` bins non redondants `X[0..=n/2]` (les autres s'obtiennent par
/// symétrie hermitienne `X[n−k] = conj(X[k])`).
///
/// **Deux fois moins de travail** qu'une FFT complexe : les échantillons pairs
/// et impairs sont empaquetés dans une FFT complexe de longueur `n/2`, puis
/// recombinés. Générique et **déterministe bit-à-bit** en virgule fixe.
#[must_use]
pub fn rfft<T: RealScalar>(input: &[T]) -> Vec<Complex<T>> {
    let n = input.len();
    assert!(
        n.is_power_of_two() && n >= 2,
        "rfft: longueur = puissance de 2 ≥ 2"
    );
    let m = n / 2;

    // Empaquetage : z[k] = x[2k] + i·x[2k+1], puis FFT complexe de longueur m.
    let mut z: Vec<Complex<T>> = (0..m)
        .map(|k| Complex::new(input[2 * k], input[2 * k + 1]))
        .collect();
    fft(&mut z);

    let half = T::from_i32(2).recip();
    let two_pi = T::from_i32(2) * T::pi();
    let inv_n = T::from_i32(n as i32).recip();
    let mut out = Vec::with_capacity(m + 1);
    for k in 0..=m
    {
        let k1 = k % m;
        let kc = (m - k1) % m;
        let zk = z[k1];
        let zc = z[kc].conj();
        // Xe = (Zk + conj(Z_{m−k}))/2 ; Xo = (Zk − conj(Z_{m−k}))/(2i).
        let xe = (zk + zc).scale(half);
        let d = zk - zc;
        let xo = Complex::new(d.im * half, -(d.re * half)); // (1/2i)·d
        // Wₙᵏ = e^{−2iπk/n}.
        let angle = two_pi * T::from_i32(k as i32) * inv_n;
        let w = Complex::new(angle.cos(), -(angle.sin()));
        out.push(xe + w * xo);
    }
    out
}

/// FFT inverse réelle : reconstruit le signal réel de longueur `n` depuis ses
/// `n/2 + 1` bins (réciproque de [`rfft`], `irfft(rfft(x), x.len()) ≈ x`).
///
/// Panique si `spectrum.len() != n/2 + 1` ou si `n` n'est pas une puissance de 2.
#[must_use]
pub fn irfft<T: RealScalar>(spectrum: &[Complex<T>], n: usize) -> Vec<T> {
    assert!(
        n.is_power_of_two() && n >= 2,
        "irfft: longueur = puissance de 2 ≥ 2"
    );
    let m = n / 2;
    assert_eq!(spectrum.len(), m + 1, "irfft: {} bins attendus", m + 1);

    let half = T::from_i32(2).recip();
    let two_pi = T::from_i32(2) * T::pi();
    let inv_n = T::from_i32(n as i32).recip();

    // Reconstruit Z[k] = Xe[k] + i·Xo[k] à partir de X[k] et X[k+m].
    let mut z = vec![Complex::zero(); m];
    for (k, zk) in z.iter_mut().enumerate()
    {
        let xk = spectrum[k];
        // X[k+m] = X[m] (k=0) sinon conj(X[m−k]) (symétrie hermitienne).
        let xkm = if k == 0
        {
            spectrum[m]
        }
        else
        {
            spectrum[m - k].conj()
        };
        let xe = (xk + xkm).scale(half);
        let d = (xk - xkm).scale(half);
        // W⁻ᵏ = e^{+2iπk/n}.
        let angle = two_pi * T::from_i32(k as i32) * inv_n;
        let winv = Complex::new(angle.cos(), angle.sin());
        let xo = winv * d;
        // Z[k] = Xe + i·Xo.
        *zk = xe + Complex::new(-xo.im, xo.re);
    }
    ifft(&mut z); // normalisée 1/m

    // Désentrelacement : x[2k] = Re(z[k]), x[2k+1] = Im(z[k]).
    let mut out = vec![T::zero(); n];
    for (k, zk) in z.iter().enumerate()
    {
        out[2 * k] = zk.re;
        out[2 * k + 1] = zk.im;
    }
    out
}
