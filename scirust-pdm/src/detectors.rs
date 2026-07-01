use scirust_signal::bearing::BearingGeometry;
use serde::{Deserialize, Serialize};

/// Fault type classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FaultType {
    NoFault,
    Imbalance,
    Misalignment,
    BearingOuterRace,
    BearingInnerRace,
    BearingRollingElement,
    Looseness,
    Resonance,
    Cavitation,
    ElectricalFault,
    Unknown,
}

impl FaultType {
    pub fn label_en(&self) -> &'static str {
        match self
        {
            FaultType::NoFault => "no_fault",
            FaultType::Imbalance => "imbalance",
            FaultType::Misalignment => "misalignment",
            FaultType::BearingOuterRace => "bearing_outer_race_fault",
            FaultType::BearingInnerRace => "bearing_inner_race_fault",
            FaultType::BearingRollingElement => "bearing_rolling_element_fault",
            FaultType::Looseness => "mechanical_looseness",
            FaultType::Resonance => "resonance",
            FaultType::Cavitation => "cavitation",
            FaultType::ElectricalFault => "electrical_fault",
            FaultType::Unknown => "unknown",
        }
    }

    pub fn label_fr(&self) -> &'static str {
        match self
        {
            FaultType::NoFault => "pas_de_defaut",
            FaultType::Imbalance => "desequilibre",
            FaultType::Misalignment => "desalignement",
            FaultType::BearingOuterRace => "defaut_bague_exterieure",
            FaultType::BearingInnerRace => "defaut_bague_interieure",
            FaultType::BearingRollingElement => "defaut_element_roulant",
            FaultType::Looseness => "desserrage_mecanique",
            FaultType::Resonance => "resonance",
            FaultType::Cavitation => "cavitation",
            FaultType::ElectricalFault => "defaut_electrique",
            FaultType::Unknown => "inconnu",
        }
    }
}

/// Fault severity per ISO 10816 vibration severity classes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FaultSeverity {
    Normal,
    Alert,
    Danger,
    Critical,
}

impl FaultSeverity {
    pub fn from_rms(rms_mms: f64) -> Self {
        // ISO 10816 velocity RMS thresholds (mm/s)
        if rms_mms < 2.8
        {
            FaultSeverity::Normal
        }
        else if rms_mms < 4.5
        {
            FaultSeverity::Alert
        }
        else if rms_mms < 7.1
        {
            FaultSeverity::Danger
        }
        else
        {
            FaultSeverity::Critical
        }
    }
}

/// A fault detection report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FaultReport {
    pub fault_type: FaultType,
    pub severity: FaultSeverity,
    pub confidence: f32,
    pub label_en: String,
    pub label_fr: String,
    pub details: Vec<(String, f64)>,
    pub timestamp: f64,
}

// ----------------------------------------------------------------------
// Imbalance Detector
// ----------------------------------------------------------------------

/// Detects rotor imbalance by checking if 1x RPM component dominates
/// the vibration spectrum.
///
/// **Imbalance signature:** Strong peak at 1x shaft frequency, with
/// harmonics decreasing in amplitude. Radial vibration > axial.
#[derive(Debug, Clone)]
pub struct ImbalanceDetector {
    /// Shaft frequency in Hz
    pub shaft_freq: f64,
    /// Minimum ratio of 1x component to overall RMS
    pub min_ratio: f64,
    /// Frequency resolution of the spectrum (Hz per bin)
    pub freq_resolution: f64,
}

impl ImbalanceDetector {
    pub fn new(shaft_freq: f64, min_ratio: f64, freq_resolution: f64) -> Self {
        Self {
            shaft_freq,
            min_ratio,
            freq_resolution,
        }
    }

    /// Analyze a vibration magnitude spectrum.
    ///
    /// `spectrum`: positive-frequency half-spectrum magnitudes
    /// `sample_rate`: sample rate in Hz
    pub fn detect(
        &self,
        spectrum: &[f64],
        _sample_rate: f64,
        timestamp: f64,
    ) -> Option<FaultReport> {
        let bin_1x = (self.shaft_freq / self.freq_resolution).round() as usize;
        let bin_2x = (2.0 * self.shaft_freq / self.freq_resolution).round() as usize;
        let bin_3x = (3.0 * self.shaft_freq / self.freq_resolution).round() as usize;

        if bin_1x >= spectrum.len()
        {
            return None;
        }

        let amp_1x = spectrum[bin_1x];
        let amp_2x = if bin_2x < spectrum.len()
        {
            spectrum[bin_2x]
        }
        else
        {
            0.0
        };
        let amp_3x = if bin_3x < spectrum.len()
        {
            spectrum[bin_3x]
        }
        else
        {
            0.0
        };

        // Overall RMS estimate from spectrum: sqrt(mean of squared magnitudes).
        // Using the mean (not the raw L2 energy) keeps `min_ratio` independent of
        // the spectrum length, so the threshold means the same thing regardless of
        // how many bins the half-spectrum has.
        let total_rms: f64 =
            (spectrum.iter().map(|m| m * m).sum::<f64>() / spectrum.len() as f64).sqrt();
        if total_rms < f64::EPSILON
        {
            return None;
        }

        let ratio_1x = amp_1x / total_rms;
        let ratio_2x = amp_2x / amp_1x.max(f64::EPSILON);

        // Imbalance: 1x >> 2x, and 1x is significant fraction of total
        let is_imbalance = ratio_1x > self.min_ratio && ratio_2x < 0.5 && amp_1x > amp_3x;

        if is_imbalance
        {
            // Estimate severity from 1x amplitude (convert to mm/s heuristic)
            let rms_mms = amp_1x * 0.5; // rough scaling
            let severity = FaultSeverity::from_rms(rms_mms);
            let confidence =
                ((ratio_1x - self.min_ratio) / (1.0 - self.min_ratio)).clamp(0.0, 1.0) as f32;
            Some(FaultReport {
                fault_type: FaultType::Imbalance,
                severity,
                confidence: confidence.max(0.5),
                label_en: "imbalance".to_string(),
                label_fr: "desequilibre".to_string(),
                details: vec![
                    ("1x_amplitude".to_string(), amp_1x),
                    ("2x_amplitude".to_string(), amp_2x),
                    ("3x_amplitude".to_string(), amp_3x),
                    ("1x_to_rms_ratio".to_string(), ratio_1x),
                ],
                timestamp,
            })
        }
        else
        {
            None
        }
    }
}

// ----------------------------------------------------------------------
// Misalignment Detector
// ----------------------------------------------------------------------

/// Detects shaft misalignment by checking for dominant 2x and 3x RPM
/// components in the vibration spectrum.
///
/// **Misalignment signature:** Strong peaks at 2x and possibly 3x shaft
/// frequency. Axial vibration is significant.
#[derive(Debug, Clone)]
pub struct MisalignmentDetector {
    pub shaft_freq: f64,
    /// Minimum ratio of 2x to 1x for misalignment
    pub min_2x_to_1x_ratio: f64,
    pub freq_resolution: f64,
}

impl MisalignmentDetector {
    pub fn new(shaft_freq: f64, freq_resolution: f64) -> Self {
        Self {
            shaft_freq,
            min_2x_to_1x_ratio: 0.5,
            freq_resolution,
        }
    }

    pub fn detect(
        &self,
        spectrum: &[f64],
        _sample_rate: f64,
        timestamp: f64,
    ) -> Option<FaultReport> {
        let bin_1x = (self.shaft_freq / self.freq_resolution).round() as usize;
        let bin_2x = (2.0 * self.shaft_freq / self.freq_resolution).round() as usize;
        let bin_3x = (3.0 * self.shaft_freq / self.freq_resolution).round() as usize;

        if bin_2x >= spectrum.len()
        {
            return None;
        }

        let amp_1x = if bin_1x < spectrum.len()
        {
            spectrum[bin_1x]
        }
        else
        {
            0.0
        };
        let amp_2x = spectrum[bin_2x];
        let amp_3x = if bin_3x < spectrum.len()
        {
            spectrum[bin_3x]
        }
        else
        {
            0.0
        };

        let ratio = amp_2x / amp_1x.max(f64::EPSILON);
        let is_misalignment = ratio > self.min_2x_to_1x_ratio && amp_3x > 0.3 * amp_1x;

        if is_misalignment
        {
            let rms_mms = amp_2x * 0.5;
            let severity = FaultSeverity::from_rms(rms_mms);
            let confidence = ((ratio - self.min_2x_to_1x_ratio) / 2.0).clamp(0.0, 1.0) as f32;
            Some(FaultReport {
                fault_type: FaultType::Misalignment,
                severity,
                confidence: confidence.max(0.5),
                label_en: "misalignment".to_string(),
                label_fr: "desalignement".to_string(),
                details: vec![
                    ("1x_amplitude".to_string(), amp_1x),
                    ("2x_amplitude".to_string(), amp_2x),
                    ("3x_amplitude".to_string(), amp_3x),
                    ("2x_to_1x_ratio".to_string(), ratio),
                ],
                timestamp,
            })
        }
        else
        {
            None
        }
    }
}

// ----------------------------------------------------------------------
// Bearing Fault Detector
// ----------------------------------------------------------------------

/// Detects bearing faults (outer race, inner race, rolling element)
/// by checking for characteristic fault frequencies in the envelope spectrum.
#[derive(Debug, Clone)]
pub struct BearingFaultDetector {
    pub bearing: BearingGeometry,
    pub shaft_freq: f64,
    pub freq_resolution: f64,
    /// Peak detection threshold (multiple of mean)
    pub threshold_factor: f64,
}

impl BearingFaultDetector {
    pub fn new(bearing: BearingGeometry, shaft_freq: f64, freq_resolution: f64) -> Self {
        Self {
            bearing,
            shaft_freq,
            freq_resolution,
            threshold_factor: 3.0,
        }
    }

    /// Detect from envelope spectrum magnitudes.
    ///
    /// `envelope_spectrum`: positive-frequency magnitudes of the envelope signal
    pub fn detect(&self, envelope_spectrum: &[f64], timestamp: f64) -> Option<FaultReport> {
        let faults = scirust_signal::bearing::detect_bearing_faults(
            envelope_spectrum,
            self.freq_resolution,
            &self.bearing,
            self.shaft_freq,
            self.threshold_factor,
        );
        if faults.is_empty()
        {
            return None;
        }
        let f = &faults[0];
        let ft = match f.fault_type.as_str()
        {
            "BPFO" => FaultType::BearingOuterRace,
            "BPFI" => FaultType::BearingInnerRace,
            "BSF" => FaultType::BearingRollingElement,
            _ => FaultType::Unknown,
        };
        let confidence = (f.amplitude / 10.0).clamp(0.0, 1.0) as f32;
        let rms_mms = f.amplitude * 0.3;
        let severity = FaultSeverity::from_rms(rms_mms);
        Some(FaultReport {
            fault_type: ft,
            severity,
            confidence,
            label_en: f.fault_type.to_lowercase().to_string(),
            label_fr: f.fault_type.to_lowercase().to_string(),
            details: vec![
                ("expected_freq".to_string(), f.expected_frequency),
                ("detected_freq".to_string(), f.detected_frequency),
                ("amplitude".to_string(), f.amplitude),
                ("harmonic".to_string(), f.harmonic as f64),
            ],
            timestamp,
        })
    }
}

// ----------------------------------------------------------------------
// Cavitation Detector
// ----------------------------------------------------------------------

/// Detects pump cavitation by checking for high-frequency broadband
/// vibration energy (> 5 kHz) and characteristic "crackling" patterns.
///
/// **Cavitation signature:** Broadband high-frequency noise,
/// increased kurtosis (> 4), reduced flow efficiency.
#[derive(Debug, Clone)]
pub struct CavitationDetector {
    /// Minimum high-frequency band power ratio
    pub min_hf_ratio: f64,
    /// Minimum kurtosis for cavitation
    pub min_kurtosis: f64,
    /// High-frequency cutoff (Hz)
    pub hf_cutoff: f64,
}

impl CavitationDetector {
    pub fn new(hf_cutoff: f64) -> Self {
        Self {
            min_hf_ratio: 0.4,
            min_kurtosis: 4.0,
            hf_cutoff,
        }
    }

    /// Detect cavitation from time-domain signal and spectrum.
    ///
    /// `signal`: time-domain vibration signal
    /// `spectrum`: positive-frequency magnitude spectrum
    /// `sample_rate`: sample rate in Hz
    pub fn detect(
        &self,
        signal: &[f64],
        spectrum: &[f64],
        sample_rate: f64,
        timestamp: f64,
    ) -> Option<FaultReport> {
        let kurt = scirust_signal::kurtosis(signal);
        if kurt < self.min_kurtosis
        {
            return None;
        }

        // Compute high-frequency band power ratio
        let n = spectrum.len();
        if n < 2
        {
            return None;
        }
        // Frequency spacing per bin, using the same convention as the other
        // detectors (`freq_resolution = sample_rate / n_fft`). For a positive-
        // frequency half-spectrum of `n` bins that includes DC and Nyquist,
        // `n_fft = 2 * (n - 1)`, so bin `k` maps to `k * freq_resolution` Hz and
        // the Nyquist bin lands at `sample_rate / 2`. Deriving it as `nyquist / n`
        // would be off by one bin and shift the high-frequency cutoff.
        let nyquist = sample_rate / 2.0;
        let freq_resolution = nyquist / (n - 1) as f64;
        let hf_bin = (self.hf_cutoff / freq_resolution).round() as usize;
        if hf_bin >= n
        {
            return None;
        }

        let lf_power: f64 = spectrum[..hf_bin].iter().map(|m| m * m).sum();
        let hf_power: f64 = spectrum[hf_bin..].iter().map(|m| m * m).sum();
        let total = lf_power + hf_power;
        if total < f64::EPSILON
        {
            return None;
        }
        let hf_ratio = hf_power / total;

        if hf_ratio > self.min_hf_ratio
        {
            let confidence =
                ((hf_ratio - self.min_hf_ratio) / (1.0 - self.min_hf_ratio)).clamp(0.0, 1.0) as f32;
            Some(FaultReport {
                fault_type: FaultType::Cavitation,
                severity: FaultSeverity::Alert,
                confidence: confidence.max(0.5),
                label_en: "cavitation".to_string(),
                label_fr: "cavitation".to_string(),
                details: vec![
                    ("kurtosis".to_string(), kurt),
                    ("hf_band_ratio".to_string(), hf_ratio),
                    ("hf_cutoff_hz".to_string(), self.hf_cutoff),
                ],
                timestamp,
            })
        }
        else
        {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_synthetic_spectrum(n_bins: usize, peaks: &[(usize, f64)]) -> Vec<f64> {
        let mut s = vec![1.0; n_bins]; // baseline noise
        for (bin, amp) in peaks
        {
            s[*bin] = *amp;
        }
        s
    }

    #[test]
    fn test_imbalance_detector_detects() {
        // 1x peak dominates, 2x is small
        let spectrum = make_synthetic_spectrum(200, &[(20, 15.0), (40, 2.0), (60, 1.0)]);
        // shaft_freq = 20 Hz, freq_resolution = 1 Hz → 1x at bin 20
        let det = ImbalanceDetector::new(20.0, 0.3, 1.0);
        let report = det.detect(&spectrum, 200.0, 0.0);
        assert!(report.is_some(), "should detect imbalance");
        let r = report.unwrap();
        assert_eq!(r.fault_type, FaultType::Imbalance);
    }

    #[test]
    fn test_imbalance_detector_no_false_positive() {
        // No dominant 1x peak
        let spectrum = make_synthetic_spectrum(200, &[(10, 2.0), (20, 1.5), (30, 2.0)]);
        let det = ImbalanceDetector::new(20.0, 0.3, 1.0);
        let report = det.detect(&spectrum, 200.0, 0.0);
        assert!(
            report.is_none(),
            "should not detect imbalance on flat spectrum"
        );
    }

    #[test]
    fn test_imbalance_ratio_is_rms_not_energy() {
        // Regression: the "1x to overall RMS" ratio must use RMS
        // (sqrt of the *mean* squared magnitude), not the raw L2 energy
        // (sqrt of the *sum*). With the L2 energy, the ratio shrinks by
        // sqrt(n) as the spectrum grows, so a fixed `min_ratio` becomes
        // spectrum-length dependent and this clear imbalance goes undetected.
        //
        // Baseline noise of 1.0 everywhere → RMS ≈ 1.0 regardless of length,
        // so a 1x peak of 12.0 gives ratio ≈ 12 (>> 5.0). The buggy energy
        // norm would give 12 / sqrt(~1143) ≈ 0.35 (< 5.0) → no detection.
        let spectrum = make_synthetic_spectrum(1000, &[(20, 12.0), (40, 1.0), (60, 1.0)]);
        let det = ImbalanceDetector::new(20.0, 5.0, 1.0);
        let report = det.detect(&spectrum, 2000.0, 0.0);
        assert!(
            report.is_some(),
            "imbalance must be detected using RMS, not L2 energy, of the spectrum"
        );
        let r = report.unwrap();
        assert_eq!(r.fault_type, FaultType::Imbalance);
        // Reported ratio should be ~12 (peak / RMS≈1), not the tiny L2-based value.
        let ratio = r
            .details
            .iter()
            .find(|(k, _)| k == "1x_to_rms_ratio")
            .map(|(_, v)| *v)
            .expect("ratio detail present");
        assert!(
            ratio > 5.0,
            "1x_to_rms_ratio should be RMS-based (~12), got {ratio}"
        );
    }

    #[test]
    fn test_misalignment_detector_detects() {
        // 2x peak dominant
        let spectrum = make_synthetic_spectrum(200, &[(20, 3.0), (40, 10.0), (60, 5.0)]);
        let det = MisalignmentDetector::new(20.0, 1.0);
        let report = det.detect(&spectrum, 200.0, 0.0);
        assert!(report.is_some());
        assert_eq!(report.unwrap().fault_type, FaultType::Misalignment);
    }

    #[test]
    fn test_bearing_fault_detector() {
        let bearing = BearingGeometry {
            pitch_diameter: 39.04,
            ball_diameter: 7.94,
            n_balls: 9,
            contact_angle_deg: 0.0,
        };
        let shaft = 29.53; // Hz
        let bpfo_freq = scirust_signal::bearing::bpfo(&bearing, shaft);
        let freq_res = 1.0;

        let mut spectrum = vec![0.5; 300]; // baseline
        let bpfo_bin = bpfo_freq.round() as usize;
        if bpfo_bin < spectrum.len()
        {
            spectrum[bpfo_bin] = 20.0;
        }
        let det = BearingFaultDetector::new(bearing, shaft, freq_res);
        let report = det.detect(&spectrum, 0.0);
        assert!(report.is_some());
        assert_eq!(report.unwrap().fault_type, FaultType::BearingOuterRace);
    }

    #[test]
    fn test_cavitation_detector_detects() {
        // High kurtosis signal (> 4) + broadband high-frequency content
        let n = 256;
        let mut signal = vec![0.01; n]; // low baseline
        // Add strong impulses every 16 samples → high kurtosis
        for i in (0..n).step_by(16)
        {
            signal[i] = 10.0;
        }
        // Spectrum: mostly high-frequency energy (all bins have similar magnitude)
        let spectrum = vec![0.5; n];
        let det = CavitationDetector::new(1000.0);
        // sample_rate must be high enough so hf_bin < n
        let report = det.detect(&signal, &spectrum, 20000.0, 0.0);
        // May or may not detect depending on exact kurtosis — check kurtosis first
        let k = scirust_signal::kurtosis(&signal);
        if k >= det.min_kurtosis
        {
            assert!(
                report.is_some(),
                "should detect cavitation with kurtosis={}",
                k
            );
        }
        else
        {
            // If kurtosis is borderline, just verify no crash
            println!("Kurtosis {} below threshold {}", k, det.min_kurtosis);
        }
    }

    #[test]
    fn test_cavitation_bin_spacing_matches_freq_resolution() {
        // Regression: bin spacing must follow the crate-wide convention
        // `freq_resolution = sample_rate / n_fft`. For a half-spectrum of `n`
        // bins that includes DC and Nyquist, `n_fft = 2*(n-1)`, so bin `k` is at
        // `k * nyquist / (n-1)` Hz. Deriving it as `nyquist / n` is off by one
        // bin and misplaces the high-frequency cutoff.
        //
        // n = 11, sample_rate = 20 → nyquist = 10.
        //   correct: freq_resolution = 10/(11-1) = 1.0 Hz/bin → cutoff 5 Hz = bin 5
        //   buggy:   bin_spacing     = 10/11    ≈ 0.909      → cutoff 5 Hz = bin 6
        // Energy sits in bin 5. With the correct spacing bin 5 is in the HF band
        // (>= cutoff) so hf_ratio = 0.5 and cavitation is flagged; with the buggy
        // spacing bin 5 falls in the LF band so hf_ratio = 0 and it is missed.
        let n = 11;
        let mut spectrum = vec![0.0; n];
        spectrum[0] = 1.0; // guaranteed low-frequency energy
        spectrum[5] = 1.0; // energy exactly at the cutoff bin

        // High-kurtosis signal (periodic impulses) to pass the kurtosis gate.
        let mut signal = vec![0.0; 64];
        for i in (0..64).step_by(16)
        {
            signal[i] = 10.0;
        }
        assert!(
            scirust_signal::kurtosis(&signal) >= 4.0,
            "test signal must clear the kurtosis gate"
        );

        let det = CavitationDetector::new(5.0);
        let report = det.detect(&signal, &spectrum, 20.0, 0.0);
        assert!(
            report.is_some(),
            "cutoff bin must be placed with freq_resolution = nyquist/(n-1)"
        );
        assert_eq!(report.unwrap().fault_type, FaultType::Cavitation);
    }

    #[test]
    fn test_fault_type_labels() {
        assert_eq!(FaultType::Imbalance.label_en(), "imbalance");
        assert_eq!(FaultType::Imbalance.label_fr(), "desequilibre");
        assert_eq!(
            FaultType::BearingOuterRace.label_en(),
            "bearing_outer_race_fault"
        );
    }

    #[test]
    fn test_fault_severity_from_rms() {
        assert_eq!(FaultSeverity::from_rms(2.0), FaultSeverity::Normal);
        assert_eq!(FaultSeverity::from_rms(3.5), FaultSeverity::Alert);
        assert_eq!(FaultSeverity::from_rms(5.0), FaultSeverity::Danger);
        assert_eq!(FaultSeverity::from_rms(10.0), FaultSeverity::Critical);
    }
}
