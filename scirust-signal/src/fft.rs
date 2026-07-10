use core::f64::consts::PI;

use crate::Complex;

/// Bit-reversal permutation for a slice of Complex values.
/// `n` must be a power of 2.
fn bit_reverse(buf: &mut [Complex], n: usize) {
    let mut j = 0usize;
    for i in 1..n
    {
        let mut bit = n >> 1;
        while j & bit != 0
        {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if i < j
        {
            buf.swap(i, j);
        }
    }
}

/// In-place radix-2 Cooley-Tukey forward FFT.
///
/// `n` must be a power of 2. The output is in standard order (not bit-reversed).
pub fn fft(buf: &mut [Complex]) {
    let n = buf.len();
    assert!(
        n.is_power_of_two(),
        "FFT size must be a power of 2, got {}",
        n
    );
    if n <= 1
    {
        return;
    }

    bit_reverse(buf, n);

    let mut len = 2usize;
    while len <= n
    {
        let half = len / 2;
        let ang = -2.0 * PI / len as f64;
        let wlen = Complex::cis(ang);
        for chunk in buf.chunks_mut(len)
        {
            let mut w = Complex::new(1.0, 0.0);
            for i in 0..half
            {
                let even = chunk[i];
                let odd = chunk[i + half];
                let t = w * odd;
                chunk[i] = even + t;
                chunk[i + half] = even - t;
                w *= wlen;
            }
        }
        len <<= 1;
    }
}

/// In-place radix-2 Cooley-Tukey inverse FFT.
///
/// `n` must be a power of 2. The output is divided by `n` (true inverse).
pub fn ifft(buf: &mut [Complex]) {
    let n = buf.len();
    assert!(
        n.is_power_of_two(),
        "IFFT size must be a power of 2, got {}",
        n
    );
    if n <= 1
    {
        return;
    }

    // Conjugate, forward FFT, conjugate, scale
    for c in buf.iter_mut()
    {
        *c = c.conj();
    }
    fft(buf);
    let scale = 1.0 / n as f64;
    for c in buf.iter_mut()
    {
        *c = c.conj() * scale;
    }
}

/// FFT radix-2 **portable** : identique à [`fft`], mais les twiddle factors
/// passent par `scirust_core::portable_f32::sincos_small_f64` (Cody–Waite +
/// polynômes portables, erreur absolue ≤ ~2⁻⁵²) au lieu de la libm — le
/// spectre est donc **bit-identique inter-plates-formes** (le reste de
/// l'algorithme — bit-reversal, papillons, accumulation de `w` — n'utilise
/// que des opérations IEEE de base en ordre fixe). Voie de référence pour
/// l'analyse spectrale reproductible (cartographie volet 111 : « FFT
/// portable »).
pub fn fft_portable(buf: &mut [Complex]) {
    let n = buf.len();
    assert!(
        n.is_power_of_two(),
        "FFT size must be a power of 2, got {}",
        n
    );
    if n <= 1
    {
        return;
    }

    bit_reverse(buf, n);

    let mut len = 2usize;
    while len <= n
    {
        let half = len / 2;
        let ang = -2.0 * PI / len as f64;
        let (s, c) = scirust_core::portable_f32::sincos_small_f64(ang);
        let wlen = Complex::new(c, s);
        for chunk in buf.chunks_mut(len)
        {
            let mut w = Complex::new(1.0, 0.0);
            for i in 0..half
            {
                let even = chunk[i];
                let odd = chunk[i + half];
                let t = w * odd;
                chunk[i] = even + t;
                chunk[i + half] = even - t;
                w *= wlen;
            }
        }
        len <<= 1;
    }
}

/// IFFT **portable** (cf. [`fft_portable`]) : conjugaison, FFT portable,
/// conjugaison, division par n — que des opérations IEEE de base.
pub fn ifft_portable(buf: &mut [Complex]) {
    let n = buf.len();
    assert!(
        n.is_power_of_two(),
        "IFFT size must be a power of 2, got {}",
        n
    );
    if n <= 1
    {
        return;
    }
    for c in buf.iter_mut()
    {
        *c = c.conj();
    }
    fft_portable(buf);
    let scale = 1.0 / n as f64;
    for c in buf.iter_mut()
    {
        *c = c.conj() * scale;
    }
}

/// Forward FFT of a real-valued signal.
///
/// Returns the positive-frequency half-spectrum (DC to Nyquist).
/// Input length `n` must be a power of 2.
pub fn fft_real(signal: &[f64]) -> Vec<Complex> {
    let n = signal.len();
    assert!(
        n.is_power_of_two(),
        "FFT size must be a power of 2, got {}",
        n
    );

    let mut buf: Vec<Complex> = signal.iter().map(|&x| Complex::new(x, 0.0)).collect();
    fft(&mut buf);

    // Return positive frequencies only: 0..=n/2
    buf.truncate(n / 2 + 1);
    buf
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-10;

    /// FFT portable ≈ FFT libm (les twiddles diffèrent d'ulps), aller-retour
    /// exact, et empreinte du spectre FIGÉE — le contrat de portabilité de
    /// l'analyse spectrale.
    #[test]
    fn fft_portable_matches_and_is_fingerprinted() {
        // signal déterministe dérivé d'entiers (identique partout)
        let n = 256usize;
        let signal: Vec<Complex> = (0..n)
            .map(|i| {
                let a = ((i.wrapping_mul(2_654_435_761)) % 2048) as f64 / 1024.0 - 1.0;
                let b = ((i.wrapping_mul(40_503)) % 2048) as f64 / 1024.0 - 1.0;
                Complex::new(a, b)
            })
            .collect();

        let mut libm = signal.clone();
        fft(&mut libm);
        let mut portable = signal.clone();
        fft_portable(&mut portable);
        for i in 0..n
        {
            assert!(
                (libm[i].re - portable[i].re).abs() < 1e-9
                    && (libm[i].im - portable[i].im).abs() < 1e-9,
                "bin {i}: libm {:?} vs portable {:?}",
                libm[i],
                portable[i]
            );
        }

        // aller-retour : ifft_portable(fft_portable(x)) ≈ x
        let mut round = signal.clone();
        fft_portable(&mut round);
        ifft_portable(&mut round);
        for i in 0..n
        {
            assert!(
                (round[i].re - signal[i].re).abs() < EPS
                    && (round[i].im - signal[i].im).abs() < EPS,
                "roundtrip {i}"
            );
        }

        // contrat de portabilité : empreinte FNV des bits f64 du spectre
        let mut fp = 0xcbf2_9ce4_8422_2325u64;
        let mut fold = |v: f64| {
            let b = v.to_bits();
            for half in [b as u32, (b >> 32) as u32]
            {
                fp ^= half as u64;
                fp = fp.wrapping_mul(0x0000_0100_0000_01b3);
            }
        };
        for c in &portable
        {
            fold(c.re);
            fold(c.im);
        }
        assert_eq!(
            fp, 0x0acd_e0a6_7b42_7c67,
            "empreinte fft portable : 0x{fp:016x}"
        );
    }

    #[test]
    fn test_fft_dc() {
        let mut buf = vec![Complex::new(1.0, 0.0); 8];
        fft(&mut buf);
        // All energy should be in bin 0
        assert!((buf[0].re - 8.0).abs() < EPS);
        for (i, c) in buf.iter().enumerate().take(8).skip(1)
        {
            assert!(c.mag() < EPS, "bin {} has magnitude {}", i, c.mag());
        }
    }

    #[test]
    fn test_fft_roundtrip() {
        let original: Vec<Complex> = (0..16)
            .map(|i| Complex::new((i as f64).sin(), 0.0))
            .collect();
        let mut freq = original.clone();
        fft(&mut freq);
        ifft(&mut freq);
        for (i, (a, b)) in original.iter().zip(freq.iter()).enumerate()
        {
            assert!(
                (a.re - b.re).abs() < EPS,
                "mismatch at {}: {} vs {}",
                i,
                a.re,
                b.re
            );
            assert!(
                (a.im - b.im).abs() < EPS,
                "mismatch at {}: {} vs {}",
                i,
                a.im,
                b.im
            );
        }
    }

    #[test]
    fn test_fft_sine() {
        // 32-point FFT of sin(2*pi*4*t) — should have energy only at bin 4
        let n = 32;
        let signal: Vec<f64> = (0..n)
            .map(|i| (2.0 * PI * 4.0 * i as f64 / n as f64).sin())
            .collect();
        let spec = fft_real(&signal);
        // Bin 4 should dominate
        let mag4 = spec[4].mag();
        let mag5 = spec[5].mag();
        assert!(mag4 > 10.0, "bin 4 magnitude too low: {}", mag4);
        assert!(mag5 < 1.0, "bin 5 has unexpected energy: {}", mag5);
        // DC should be near zero
        assert!(spec[0].mag() < 1.0);
    }

    #[test]
    fn test_power_of_two_assertion() {
        let result = std::panic::catch_unwind(|| {
            let mut buf = vec![Complex::zero(); 7];
            fft(&mut buf);
        });
        assert!(result.is_err());
    }
}
