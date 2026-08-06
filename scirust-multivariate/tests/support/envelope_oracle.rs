// AUTO-GENERATED from the ITD reference (itd_research.ood_shift + ood.reference).
// Do not edit by hand. Regenerate with:
//   PYTHONPATH=/path/to/itd-simulator python3 gen_envelope_oracle.py <out>
// Included via `mod envelope_oracle { include!(...) }` in tests/envelope.rs.

pub const N_ROWS: usize = 12;
pub const N_FEATURES: usize = 4;
pub const SEVERITY_K: usize = 3;
pub const S_LOW: f64 = 1.5;
pub const S_HIGH: f64 = 3.0;

// ---- calibration matrix (row-major) and fitted per-axis statistics ----
pub static CALIBRATION: &[f64] = &[5.609434159508862, -1.0399841062404955, 0.7504511958064572, 7.25, 1.0979296226923272, -1.302179506862318, 0.12784040316728537, 7.25, 4.966397684991422, -0.85304392757358, 0.8793979748628286, 7.25, 5.132061395122432, 1.1272412069680329, 0.4675093422520456, 7.25, 5.737501568164998, -0.9588826008289989, 0.8784503013072725, 7.25, 4.630275272909479, -0.6809295444039414, 1.2225413386740303, 7.25, 4.1433443556737855, -0.3521335504882296, 0.5323091855533487, 7.25, 5.825465223191976, 0.43082100300788273, 2.1416476008704612, 7.25, 3.975514541856925, -0.8137727282478777, 0.6159794225754956, 7.25, 4.77210508469025, -0.840156476962528, -0.8244812156912396, 7.25, 6.486508342406885, 0.543154268305195, -0.6655097072886943, 7.25, 5.233371618281456, 0.21868859672901295, 0.8714287779481898, 7.25];
pub static FITTED_MEAN: &[f64] = &[4.800825739124234, -0.3767647805498205, 0.5831303850031234, 7.25];
/// Population std with the reference floor: `std < 1e-9` is replaced by `1.0`.
pub static FITTED_SCALE: &[f64] = &[1.3102678000033559, 0.7346104734934548, 0.7576671641655893, 1.0];

// ---- probes: per-axis |z|, top-k severity, arg-max attribution ----
pub static PROBES: &[f64] = &[5.609434159508862, -1.0399841062404955, 0.7504511958064572, 7.25, 4.800825739124234, 7.623235219450179, 0.5831303850031234, 7.25, 8.800825739124235, -6.376764780549821, 5.583130385003123, 7.25];
pub static PROBE_DEVIATIONS: &[f64] = &[0.617132177393474, 0.902817683141274, 0.22083682481819367, 0.0, 0.0, 10.890125159740563, 0.0, 0.0, 3.0528110360261898, 8.167593869805422, 6.599203761860851, 0.0];
pub static PROBE_SEVERITIES: &[f64] = &[0.5802622284509805, 3.630041719913521, 5.939869555897488];
pub static PROBE_ATTRIBUTIONS: &[usize] = &[1, 1, 1];

// ---- three-state policy over a severity sweep ----
pub static SWEEP: &[f64] = &[0.0, 1.5, 2.0, 2.5, 3.0, 4.0];
pub static SWEEP_DISCOUNTS: &[f64] = &[1.0, 1.0, 0.6666666666666667, 0.33333333333333337, 0.0, 0.0];
/// State codes: 0 accept, 1 accept-with-reduced-confidence, 2 abstain.
pub static SWEEP_STATES: &[usize] = &[0, 0, 1, 1, 2, 2];
pub static SWEEP_CONFIDENCES: &[f64] = &[1.0, 1.0, 0.6666666666666667, 0.33333333333333337, 0.0, 0.0];
/// Degenerate band `s_high <= s_low` collapses to a hard step at `s_low`.
pub static DEGENERATE_STATES: &[usize] = &[0, 0, 0, 2, 2, 2];
pub static DEGENERATE_CONFIDENCES: &[f64] = &[1.0, 1.0, 1.0, 0.0, 0.0, 0.0];

// ---- monotone-separation rank agreement ----
pub static SEP_SCORES: &[f64] = &[0.1, 0.4, 0.3, 0.9, 0.8, 1.5];
pub static SEP_LEVELS: &[usize] = &[0, 0, 1, 1, 2, 2];
pub const SEPARATION: f64 = 0.8333333333333334;
