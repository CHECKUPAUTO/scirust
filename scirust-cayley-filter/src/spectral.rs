//! Deterministic STFT application of 16D Cayley transforms.
//!
//! Eight consecutive positive-frequency complex bins are encoded as one
//! 16-dimensional real vector:
//!
//! `[re₀, im₀, re₁, im₁, ..., re₇, im₇]`.
//!
//! DC, Nyquist and incomplete final groups remain unchanged.

use crate::operator::{Matrix16, matrix_vector_mul};
use crate::projector::CayleyProjector;
use crate::scalar::{SEDENION_DIMENSION, Sedenion};
use crate::soft::SoftCayleyFilter;
use scirust_signal::Complex;
use scirust_signal::fft::{fft, ifft};
use scirust_signal::windows::hanning;

/// Number of complex Fourier bins encoded by one sedenion.
pub const SPECTRAL_COMPLEX_BINS: usize = SEDENION_DIMENSION / 2;

/// STFT analysis, Cayley transformation and weighted overlap-add synthesis.
#[derive(Clone, Debug, PartialEq)]
pub struct SpectralBlockFilter {
    transform: Matrix16,
    frame_len: usize,
    hop: usize,
}

impl SpectralBlockFilter {
    /// Constructs a spectral Cayley filter.
    ///
    /// `frame_len` is rounded up to a power of two with a minimum of four.
    /// `hop` is clamped to `[1, frame_len / 2]`.
    #[must_use]
    pub fn new(transform: Matrix16, frame_len: usize, hop: usize) -> Self {
        let frame_len = frame_len.max(4).next_power_of_two();
        let hop = hop.clamp(1, frame_len / 2);

        Self {
            transform,
            frame_len,
            hop,
        }
    }

    /// Constructs a spectral filter from a hard Cayley projector.
    #[must_use]
    pub fn from_hard(projector: &CayleyProjector, frame_len: usize, hop: usize) -> Self {
        Self::new(*projector.projection(), frame_len, hop)
    }

    /// Constructs a spectral filter from a soft Cayley transform.
    #[must_use]
    pub fn from_soft(filter: &SoftCayleyFilter, frame_len: usize, hop: usize) -> Self {
        Self::new(*filter.transform(), frame_len, hop)
    }

    /// Applies the spectral transform and returns a length-preserving signal.
    #[must_use]
    pub fn apply(&self, signal: &[f64]) -> Vec<f64> {
        if signal.len() < 2
        {
            return signal.to_vec();
        }

        let window = hanning(self.frame_len);
        let padded_len = signal.len() + 2 * self.frame_len;

        let padded: Vec<f64> = (0..padded_len)
            .map(|index| {
                let source = mirror_index(index as isize - self.frame_len as isize, signal.len());

                signal[source]
            })
            .collect();

        let mut accumulator = vec![0.0; padded_len];
        let mut normalization = vec![0.0; padded_len];

        let mut offset = 0;

        while offset + self.frame_len <= padded_len
        {
            let mut spectrum: Vec<Complex> = (0..self.frame_len)
                .map(|index| Complex::new(window[index] * padded[offset + index], 0.0))
                .collect();

            fft(&mut spectrum);
            self.transform_positive_bins(&mut spectrum);
            ifft(&mut spectrum);

            for index in 0..self.frame_len
            {
                accumulator[offset + index] += window[index] * spectrum[index].re;

                normalization[offset + index] += window[index] * window[index];
            }

            offset += self.hop;
        }

        (0..signal.len())
            .map(|index| {
                let padded_index = self.frame_len + index;
                let weight = normalization[padded_index];

                if weight > 1.0e-12
                {
                    accumulator[padded_index] / weight
                }
                else
                {
                    0.0
                }
            })
            .collect()
    }

    /// Returns the normalized FFT frame length.
    #[must_use]
    pub const fn frame_len(&self) -> usize {
        self.frame_len
    }

    /// Returns the normalized frame hop.
    #[must_use]
    pub const fn hop(&self) -> usize {
        self.hop
    }

    /// Returns the number of complete Cayley groups transformed per frame.
    #[must_use]
    pub const fn groups_per_frame(&self) -> usize {
        let positive_bins_without_nyquist = self.frame_len / 2 - 1;

        positive_bins_without_nyquist / SPECTRAL_COMPLEX_BINS
    }

    /// Returns the underlying real 16 × 16 transform.
    #[must_use]
    pub const fn transform(&self) -> &Matrix16 {
        &self.transform
    }

    fn transform_positive_bins(&self, spectrum: &mut [Complex]) {
        let nyquist = self.frame_len / 2;
        let mut first_bin = 1;

        while first_bin + SPECTRAL_COMPLEX_BINS <= nyquist
        {
            let coordinates: Sedenion = core::array::from_fn(|coordinate| {
                let local_bin = coordinate / 2;
                let bin = first_bin + local_bin;

                if coordinate.is_multiple_of(2)
                {
                    spectrum[bin].re
                }
                else
                {
                    spectrum[bin].im
                }
            });

            let transformed = matrix_vector_mul(&self.transform, &coordinates);

            for local_bin in 0..SPECTRAL_COMPLEX_BINS
            {
                let bin = first_bin + local_bin;

                let value =
                    Complex::new(transformed[2 * local_bin], transformed[2 * local_bin + 1]);

                spectrum[bin] = value;
                spectrum[self.frame_len - bin] = value.conj();
            }

            first_bin += SPECTRAL_COMPLEX_BINS;
        }
    }
}

fn mirror_index(index: isize, length: usize) -> usize {
    if length <= 1
    {
        return 0;
    }

    let period = 2 * (length - 1);
    let mut wrapped = index % period as isize;

    if wrapped < 0
    {
        wrapped += period as isize;
    }

    let wrapped = wrapped as usize;

    if wrapped < length
    {
        wrapped
    }
    else
    {
        period - wrapped
    }
}

#[cfg(test)]
mod tests {
    use super::SpectralBlockFilter;
    use crate::{CayleyProjector, basis_vector, left_multiplication_matrix};

    fn assert_close(left: &[f64], right: &[f64]) {
        assert_eq!(left.len(), right.len());

        let maximum_error = left
            .iter()
            .zip(right)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);

        assert!(
            maximum_error <= 1.0e-10,
            "maximum reconstruction error: {maximum_error}",
        );
    }

    #[test]
    fn parameters_are_normalized() {
        let identity = left_multiplication_matrix(basis_vector(0).expect("e0 exists"));

        let filter = SpectralBlockFilter::new(identity, 48, 100);

        assert_eq!(filter.frame_len(), 64);
        assert_eq!(filter.hop(), 32);
        assert_eq!(filter.groups_per_frame(), 3);
    }

    #[test]
    fn identity_transform_reconstructs_input() {
        let identity = left_multiplication_matrix(basis_vector(0).expect("e0 exists"));

        let filter = SpectralBlockFilter::new(identity, 64, 16);

        let signal: Vec<f64> = (0..257)
            .map(|index| {
                let x = index as f64;
                (0.071 * x).sin() + 0.3 * (0.193 * x).cos()
            })
            .collect();

        let output = filter.apply(&signal);

        assert_close(&output, &signal);
    }

    #[test]
    fn output_is_length_preserving_and_finite() {
        let mut multiplier = [0.0; 16];
        multiplier[1] = 1.0;
        multiplier[10] = -1.0;

        let projector = CayleyProjector::new(multiplier, 5.0e-2).expect("valid projector");

        let filter = SpectralBlockFilter::from_hard(&projector, 64, 16);

        let signal: Vec<f64> = (0..311)
            .map(|index| {
                let x = index as f64;
                (0.11 * x).sin() + 0.05 * (0.79 * x).cos()
            })
            .collect();

        let output = filter.apply(&signal);

        assert_eq!(output.len(), signal.len());
        assert!(output.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn empty_and_single_sample_inputs_are_preserved() {
        let identity = left_multiplication_matrix(basis_vector(0).expect("e0 exists"));

        let filter = SpectralBlockFilter::new(identity, 64, 16);

        assert!(filter.apply(&[]).is_empty());
        assert_eq!(filter.apply(&[3.5]), vec![3.5]);
    }
}
