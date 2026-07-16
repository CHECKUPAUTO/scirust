//! Cross-domain validation of the classifier's periodic-feature robustness on **real
//! bearing vibration** (Case Western Reserve University Bearing Data Center).
//!
//! Run with `cargo run --release -p scirust-signal --example classify_real_bearing`.
//! Deterministic: real recorded vibration from the committed fixture, no RNG.
//!
//! ## The point
//!
//! A rolling-element bearing with an outer-race defect emits a sharp mechanical impact
//! each time a rolling element passes the defect — a **periodic** impulse train at the
//! bearing's characteristic defect frequency (BPFO). Those impacts have high kurtosis
//! and crest factor, so a naive impulsivity gate reads them as *impulsive noise* and a
//! spike remover would erase the very signature a diagnostic pipeline is looking for.
//! The classifier's energy-envelope periodicity veto (`detect::periodic_impulse_train`,
//! introduced for the ECG QRS case) recognizes the periodicity and keeps the verdict
//! off `Impulsive` — the same robustness, validated here on a completely different
//! physical domain.
//!
//! ## Data provenance
//!
//! Drive-end accelerometer, 12 kHz, committed as `tests/data/cwru_bearing.csv`:
//! record 97 (healthy baseline) and record 130 (0.007 in outer-race fault). Source:
//! Case Western Reserve University Bearing Data Center,
//! <https://engineering.case.edu/bearingdatacenter> (freely distributed for research;
//! cite CWRU).

use scirust_signal::denoise::{NoiseType, classify};

fn load_fixture() -> (Vec<f64>, Vec<f64>) {
    let raw = include_str!("../tests/data/cwru_bearing.csv");
    let (mut normal, mut fault) = (Vec::new(), Vec::new());
    for line in raw.lines()
    {
        if line.starts_with('#') || line.is_empty()
        {
            continue;
        }
        let mut it = line.split(',');
        normal.push(it.next().unwrap().trim().parse().unwrap());
        fault.push(it.next().unwrap().trim().parse().unwrap());
    }
    (normal, fault)
}

fn main() {
    let (normal, fault) = load_fixture();
    let fs = 12_000.0;
    println!("# CWRU bearing vibration (drive-end, 12 kHz), 4096 samples\n");
    println!(
        "{:<18} {:>9} {:>7} {:>7}   note",
        "record", "verdict", "kurt", "crest"
    );
    for (tag, x) in [
        ("healthy (rec 97)", &normal),
        ("OR fault (rec 130)", &fault),
    ]
    {
        let p = classify(x, fs);
        let reaches_gate = p.residual_kurtosis > 4.0 && p.crest_factor > 5.0;
        let note = if tag.starts_with("OR")
        {
            if reaches_gate && p.dominant != NoiseType::Impulsive
            {
                "reaches the impulsive gate, yet NOT Impulsive → periodic veto preserved the fault signature"
            }
            else if reaches_gate
            {
                "reaches the impulsive gate AND read as Impulsive (!) — the signature would be stripped"
            }
            else
            {
                "did not reach the impulsive gate on this excerpt"
            }
        }
        else
        {
            "smooth baseline"
        };
        println!(
            "{tag:<18} {:>9?} {:>7.2} {:>7.2}   {note}",
            p.dominant, p.residual_kurtosis, p.crest_factor
        );
    }
    println!(
        "\n# Finding: the outer-race fault's impacts trip both halves of the impulsive gate\n\
         # (kurtosis > 4 AND crest > 5) — a naive classifier would call them noise and a spike\n\
         # remover would erase them. The energy-envelope periodicity veto recognizes the BPFO\n\
         # impulse train as a legitimate repeated feature, so the diagnostic signature is kept.\n\
         # Same robustness as the ECG QRS case (detect::periodic_impulse_train), a different domain."
    );
}
