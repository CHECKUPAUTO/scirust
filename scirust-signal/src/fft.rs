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
