//! Real-data validation on a **different domain**: rolling-element bearing vibration
//! (Case Western Reserve University Bearing Data Center). This is the cross-domain
//! check of the classifier's periodic-feature robustness (`detect::periodic_impulse_train`,
//! added for the ECG QRS case): a **bearing outer-race fault** produces sharp,
//! high-crest impact impulses that *look* like impulsive noise (they trip the
//! `kurtosis > 4 ∧ crest > 5` gate), yet they are the **diagnostic signature** — a
//! periodic train at the bearing's characteristic defect frequency — and must not be
//! filtered away as noise.
//!
//! Data (drive-end accelerometer, 12 kHz), committed as `tests/data/cwru_bearing.csv`:
//! * `normal_g` — record 97, a healthy bearing (baseline).
//! * `or_fault_g` — record 130, a 0.007 in outer-race fault (impacts at BPFO).
//! * Source: <https://engineering.case.edu/bearingdatacenter> (Case Western Reserve
//!   University Bearing Data Center; data freely distributed for research, cite CWRU).
//!
//! Deterministic: real recorded vibration from the fixture, no RNG.

use scirust_signal::denoise::{NoiseType, classify};

fn load_fixture() -> (Vec<f64>, Vec<f64>) {
    let raw = include_str!("data/cwru_bearing.csv");
    let (mut normal, mut fault) = (Vec::new(), Vec::new());
    for line in raw.lines()
    {
        if line.starts_with('#') || line.is_empty()
        {
            continue;
        }
        let mut it = line.split(',');
        normal.push(it.next().unwrap().trim().parse::<f64>().unwrap());
        fault.push(it.next().unwrap().trim().parse::<f64>().unwrap());
    }
    (normal, fault)
}

#[test]
fn bearing_fault_impulses_are_recognized_as_a_periodic_signature_not_noise() {
    // The outer-race fault's impacts genuinely *look* impulsive: the high-pass residual
    // trips both halves of the impulsive gate (excess kurtosis > 4 AND crest > 5). A
    // naive classifier would call that Impulsive and route to a spike remover (Hampel),
    // destroying the diagnostic fault signature.
    let (_, fault) = load_fixture();
    let p = classify(&fault, 12_000.0);
    assert!(
        p.residual_kurtosis > 4.0 && p.crest_factor > 5.0,
        "the fault should reach the impulsive gate (kurt {:.2} > 4, crest {:.2} > 5)",
        p.residual_kurtosis,
        p.crest_factor
    );
    // But the impacts recur periodically (at the bearing's defect frequency), so the
    // energy-envelope periodicity veto recognizes them as a legitimate repeated
    // feature — the verdict is NOT Impulsive, and the signature is preserved.
    assert_ne!(
        p.dominant,
        NoiseType::Impulsive,
        "the periodic bearing-fault signature was mislabeled as impulsive noise"
    );
}

#[test]
fn healthy_bearing_is_not_impulsive() {
    // The healthy baseline is smooth (sub-Gaussian residual): it must not be read as
    // impulsive noise either.
    let (normal, _) = load_fixture();
    let p = classify(&normal, 12_000.0);
    assert_ne!(p.dominant, NoiseType::Impulsive);
}

#[test]
fn fixture_is_well_formed_bearing_vibration() {
    let (normal, fault) = load_fixture();
    assert_eq!(normal.len(), 4096);
    assert_eq!(fault.len(), 4096);
    assert!(normal.iter().chain(&fault).all(|v| v.is_finite()));
    // The fault carries far more impact energy than the healthy baseline — the
    // signature of a developed defect (and evidence the two columns are not swapped).
    let rms = |x: &[f64]| (x.iter().map(|&v| v * v).sum::<f64>() / x.len() as f64).sqrt();
    assert!(
        rms(&fault) > 3.0 * rms(&normal),
        "fault RMS {:.3} should dwarf healthy RMS {:.3}",
        rms(&fault),
        rms(&normal)
    );
}
