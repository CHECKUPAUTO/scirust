use core::f64::consts::PI;

/// Generate a Hanning (Hann) window of length `n`.
///
/// w[k] = 0.5 * (1 - cos(2*pi*k/(n-1)))
pub fn hanning(n: usize) -> Vec<f64> {
    if n == 0
    {
        return Vec::new();
    }
    if n == 1
    {
        return vec![1.0];
    }
    let denom = (n - 1) as f64;
    (0..n)
        .map(|k| 0.5 * (1.0 - (2.0 * PI * k as f64 / denom).cos()))
        .collect()
}

/// Generate a Hamming window of length `n`.
///
/// w[k] = 0.54 - 0.46 * cos(2*pi*k/(n-1))
pub fn hamming(n: usize) -> Vec<f64> {
    if n == 0
    {
        return Vec::new();
    }
    if n == 1
    {
        return vec![1.0];
    }
    let denom = (n - 1) as f64;
    (0..n)
        .map(|k| 0.54 - 0.46 * (2.0 * PI * k as f64 / denom).cos())
        .collect()
}

/// Generate a Blackman window of length `n`.
///
/// w[k] = 0.42 - 0.5*cos(2*pi*k/(n-1)) + 0.08*cos(4*pi*k/(n-1))
pub fn blackman(n: usize) -> Vec<f64> {
    if n == 0
    {
        return Vec::new();
    }
    if n == 1
    {
        return vec![1.0];
    }
    let denom = (n - 1) as f64;
    (0..n)
        .map(|k| {
            let a = 2.0 * PI * k as f64 / denom;
            0.42 - 0.5 * a.cos() + 0.08 * (2.0 * a).cos()
        })
        .collect()
}

/// Generate a Blackman-Harris (4-term) window of length `n`.
pub fn blackman_harris(n: usize) -> Vec<f64> {
    if n == 0
    {
        return Vec::new();
    }
    if n == 1
    {
        return vec![1.0];
    }
    let denom = (n - 1) as f64;
    (0..n)
        .map(|k| {
            let a = 2.0 * PI * k as f64 / denom;
            0.35875 - 0.48829 * a.cos() + 0.14128 * (2.0 * a).cos() - 0.01168 * (3.0 * a).cos()
        })
        .collect()
}

/// Generate a Flat-top window of length `n`.
/// Used when amplitude accuracy is critical (e.g., calibration).
pub fn flattop(n: usize) -> Vec<f64> {
    if n == 0
    {
        return Vec::new();
    }
    if n == 1
    {
        return vec![1.0];
    }
    let denom = (n - 1) as f64;
    (0..n)
        .map(|k| {
            let a = 2.0 * PI * k as f64 / denom;
            1.0 - 1.93 * a.cos() + 1.29 * (2.0 * a).cos() - 0.388 * (3.0 * a).cos()
                + 0.028 * (4.0 * a).cos()
        })
        .collect()
}

/// Apply a window function to a signal in-place.
///
/// `signal[i] *= window[i]`
pub fn apply_window(signal: &mut [f64], window: &[f64]) {
    let len = signal.len().min(window.len());
    for i in 0..len
    {
        signal[i] *= window[i];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-10;

    #[test]
    fn test_hanning_ends() {
        let w = hanning(8);
        assert!(w[0].abs() < EPS);
        assert!(w[7].abs() < EPS);
        assert!(w[3] > 0.8); // center should be ~1
    }

    #[test]
    fn test_hamming_ends() {
        let w = hamming(8);
        assert!((w[0] - 0.08).abs() < EPS);
        assert!((w[7] - 0.08).abs() < EPS);
    }

    #[test]
    fn test_apply_window() {
        let mut sig = vec![2.0; 4];
        let win = vec![1.0, 0.5, 0.25, 0.0];
        apply_window(&mut sig, &win);
        assert!((sig[0] - 2.0).abs() < EPS);
        assert!((sig[1] - 1.0).abs() < EPS);
        assert!((sig[2] - 0.5).abs() < EPS);
        assert!((sig[3] - 0.0).abs() < EPS);
    }

    #[test]
    fn test_empty_windows() {
        assert!(hanning(0).is_empty());
        assert!(hamming(0).is_empty());
        assert!(blackman(0).is_empty());
        assert!(blackman_harris(0).is_empty());
        assert!(flattop(0).is_empty());
    }

    #[test]
    fn test_single_element() {
        assert!((hanning(1)[0] - 1.0).abs() < EPS);
        assert!((hamming(1)[0] - 1.0).abs() < EPS);
        assert!((blackman(1)[0] - 1.0).abs() < EPS);
    }
}
