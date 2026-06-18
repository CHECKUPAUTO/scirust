use serde::{Deserialize, Serialize};

/// Geometry parameters for a rolling-element bearing.
/// All dimensions in millimetres.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BearingGeometry {
    /// Pitch diameter (centre of rolling elements)
    pub pitch_diameter: f64,
    /// Rolling element diameter
    pub ball_diameter: f64,
    /// Number of rolling elements
    pub n_balls: usize,
    /// Contact angle in degrees (0 for radial, >0 for angular contact)
    pub contact_angle_deg: f64,
}

impl BearingGeometry {
    pub fn contact_angle_rad(&self) -> f64 {
        self.contact_angle_deg.to_radians()
    }
}

/// Ball Pass Frequency Outer race (BPFO).
///
/// Frequency at which a rolling element passes over a defect on the outer race.
/// shaft_freq: shaft rotation frequency in Hz.
pub fn bpfo(geo: &BearingGeometry, shaft_freq: f64) -> f64 {
    let theta = geo.contact_angle_rad();
    let ratio = geo.ball_diameter / geo.pitch_diameter * theta.cos();
    (geo.n_balls as f64 / 2.0) * shaft_freq * (1.0 - ratio)
}

/// Ball Pass Frequency Inner race (BPFI).
///
/// Frequency at which a rolling element passes over a defect on the inner race.
pub fn bpfi(geo: &BearingGeometry, shaft_freq: f64) -> f64 {
    let theta = geo.contact_angle_rad();
    let ratio = geo.ball_diameter / geo.pitch_diameter * theta.cos();
    (geo.n_balls as f64 / 2.0) * shaft_freq * (1.0 + ratio)
}

/// Ball Spin Frequency (BSF).
///
/// Frequency at which a single rolling element spins about its own axis.
pub fn bsf(geo: &BearingGeometry, shaft_freq: f64) -> f64 {
    let theta = geo.contact_angle_rad();
    let pd = geo.pitch_diameter;
    let bd = geo.ball_diameter;
    let ratio_sq = bd / pd * theta.cos();
    (pd / (2.0 * bd)) * shaft_freq * (1.0 - ratio_sq * ratio_sq)
}

/// Fundamental Train Frequency (FTF) — cage frequency.
///
/// Frequency at which the cage/retainer rotates.
pub fn ftf(geo: &BearingGeometry, shaft_freq: f64) -> f64 {
    let theta = geo.contact_angle_rad();
    let ratio = geo.ball_diameter / geo.pitch_diameter * theta.cos();
    (shaft_freq / 2.0) * (1.0 - ratio)
}

/// A detected bearing fault with its frequency and confidence.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BearingFault {
    /// Type of fault: "BPFO", "BPFI", "BSF", "FTF"
    pub fault_type: String,
    /// Expected fault frequency in Hz
    pub expected_frequency: f64,
    /// Detected peak frequency in Hz
    pub detected_frequency: f64,
    /// Relative amplitude of the detected peak
    pub amplitude: f64,
    /// Harmonic number (1 = fundamental, 2 = 2nd harmonic, etc.)
    pub harmonic: usize,
}

/// Detect bearing faults by searching for characteristic frequencies in an
/// envelope spectrum.
///
/// `envelope_spectrum`: magnitude spectrum of the envelope signal (usually after
///   band-pass filtering and Hilbert transform).
/// `freq_resolution`: frequency spacing between bins in Hz (sample_rate / n_fft).
/// `geo`: bearing geometry parameters.
/// `shaft_freq`: shaft rotation frequency in Hz.
/// `threshold_factor`: peaks must exceed this multiple of the spectrum mean (default ~3).
///
/// Returns detected faults sorted by amplitude descending.
pub fn detect_bearing_faults(
    envelope_spectrum: &[f64],
    freq_resolution: f64,
    geo: &BearingGeometry,
    shaft_freq: f64,
    threshold_factor: f64,
) -> Vec<BearingFault> {
    let mean: f64 = envelope_spectrum.iter().sum::<f64>() / envelope_spectrum.len() as f64;
    let threshold = mean * threshold_factor;

    let candidates = vec![
        ("BPFO", bpfo(geo, shaft_freq)),
        ("BPFI", bpfi(geo, shaft_freq)),
        ("BSF", bsf(geo, shaft_freq)),
        ("FTF", ftf(geo, shaft_freq)),
    ];

    let mut faults = Vec::new();

    for &(fault_type, base_freq) in &candidates
    {
        // Check up to 5 harmonics
        for h in 1..=5
        {
            let target_freq = base_freq * h as f64;
            let bin = (target_freq / freq_resolution).round() as usize;
            if bin >= envelope_spectrum.len()
            {
                break;
            }
            // Check bin and nearest neighbors
            let mut best_amp = envelope_spectrum[bin];
            let mut best_bin = bin;
            for offset in [-1isize, 0, 1].iter()
            {
                let idx = (bin as isize + offset) as usize;
                if idx < envelope_spectrum.len() && envelope_spectrum[idx] > best_amp
                {
                    best_amp = envelope_spectrum[idx];
                    best_bin = idx;
                }
            }
            if best_amp > threshold
            {
                faults.push(BearingFault {
                    fault_type: fault_type.to_string(),
                    expected_frequency: target_freq,
                    detected_frequency: best_bin as f64 * freq_resolution,
                    amplitude: best_amp,
                    harmonic: h,
                });
            }
        }
    }

    faults.sort_by(|a, b| {
        b.amplitude
            .partial_cmp(&a.amplitude)
            .unwrap_or(core::cmp::Ordering::Equal)
    });
    faults
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_skf_6205_frequencies() {
        // SKF 6205 deep-groove ball bearing parameters
        let geo = BearingGeometry {
            pitch_diameter: 39.04,
            ball_diameter: 7.94,
            n_balls: 9,
            contact_angle_deg: 0.0,
        };
        let shaft = 29.53; // Hz (≈ 1772 RPM)
        // Known values from bearing data sheet:
        // BPFO ≈ 105.8 Hz, BPFI ≈ 159.9 Hz, BSF ≈ 69.6 Hz, FTF ≈ 11.8 Hz
        let bpo = bpfo(&geo, shaft);
        let bpi = bpfi(&geo, shaft);
        let bsp = bsf(&geo, shaft);
        let cage = ftf(&geo, shaft);

        assert!((bpo - 105.8).abs() < 1.0, "BPFO: {}", bpo);
        assert!((bpi - 159.9).abs() < 1.0, "BPFI: {}", bpi);
        assert!((bsp - 69.6).abs() < 1.0, "BSF: {}", bsp);
        assert!((cage - 11.8).abs() < 0.5, "FTF: {}", cage);
    }

    #[test]
    fn test_detect_bearing_faults_synthetic() {
        let geo = BearingGeometry {
            pitch_diameter: 39.04,
            ball_diameter: 7.94,
            n_balls: 9,
            contact_angle_deg: 0.0,
        };
        let shaft = 29.53;
        let bpfo_freq = bpfo(&geo, shaft);

        // Create a synthetic spectrum with a peak at BPFO
        let n_bins = 512;
        let freq_res = 1.0; // 1 Hz per bin
        let mut spectrum = vec![1.0; n_bins]; // background noise floor
        let bpfo_bin = bpfo_freq.round() as usize;
        spectrum[bpfo_bin] = 20.0; // strong peak

        let faults = detect_bearing_faults(&spectrum, freq_res, &geo, shaft, 3.0);
        assert!(!faults.is_empty());
        assert_eq!(faults[0].fault_type, "BPFO");
    }

    #[test]
    fn test_no_false_positives_on_flat_spectrum() {
        let geo = BearingGeometry {
            pitch_diameter: 39.04,
            ball_diameter: 7.94,
            n_balls: 9,
            contact_angle_deg: 0.0,
        };
        let spectrum = vec![1.0; 512];
        let faults = detect_bearing_faults(&spectrum, 1.0, &geo, 29.53, 5.0);
        assert!(faults.is_empty());
    }
}
