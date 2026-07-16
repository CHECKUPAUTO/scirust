//! **Reproducible VST-selection benchmark** — the CANR "benchmark tool"
//! deliverable (`docs/research/CANR_CERTIFIED_ADAPTIVE_REPRESENTATIONS_2026-07-16.md`
//! §9), built on [`super::autotune_vst`]. It sweeps a grid of
//! **(noise model × inner denoiser)** and, for each cell, runs the data-driven
//! VST autotuner (select on a dev record, validate on a disjoint held-out
//! record) and records which transform won and whether it beat direct denoising.
//! The whole run is deterministic (fixed seeds, Philox noise) and emits a
//! machine-readable table (CSV) and a human-readable one (Markdown).
//!
//! It answers the report's Phase-1 question — *when* does variance stabilization
//! actually help, and by how much — with numbers instead of folklore, and keeps
//! the honest negatives in the table:
//!
//! * signal-dependent noise (multiplicative `σ ∝ level`, Poisson-like
//!   `σ = √level`) paired with a denoiser that assumes **homoscedastic** noise
//!   (a globally-calibrated Wiener filter) is where a VST pays off;
//! * a **linear** smoother, or genuinely homoscedastic noise, is where it does
//!   not — the `beats_baseline = false` cells are the pre-registered kill signal
//!   (CANR §13), left in the table rather than hidden.
//!
//! Named inner denoisers ([`InnerDenoiser`]) wrap this crate's existing routines
//! so callers need not hand-roll them; [`super::autotune_vst`] still accepts any
//! closure for bespoke denoisers.

use scirust_core::philox::Philox4x32;

use super::VstKind;
use super::transform::{ThresholdMode, wavelet_denoise};
use super::{DenoiseCase, autotune_vst, default_vst_candidates};
use super::{estimate_noise_std, gaussian_smooth, stft_wiener_auto, wiener_white};

/// A named inner Gaussian denoiser applied in the (variance-stabilized) domain.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InnerDenoiser {
    /// Linear Gaussian smoother (`σ = 4`). A VST provably cannot help a linear
    /// smoother of a smooth signal — the benchmark's negative control.
    GaussianSmooth,
    /// Wiener filter calibrated with a **single global** noise σ (robust MAD).
    /// Assumes homoscedastic noise, so it is the estimator a VST is meant to help.
    WienerGlobal,
    /// Soft-threshold wavelet shrinkage (VisuShrink-style, raw-MAD calibration).
    WaveletSoft,
    /// Short-time Wiener with automatic noise-floor tracking — the strongest
    /// identity-domain baseline of the toolkit (cf. [`super::vst_denoise_auto`]).
    StftWienerAuto,
}

impl InnerDenoiser {
    /// Human-readable name (stable; used as a table key).
    pub fn name(self) -> &'static str {
        match self
        {
            InnerDenoiser::GaussianSmooth => "gaussian_smooth",
            InnerDenoiser::WienerGlobal => "wiener_global",
            InnerDenoiser::WaveletSoft => "wavelet_soft",
            InnerDenoiser::StftWienerAuto => "stft_wiener_auto",
        }
    }

    /// Apply the denoiser to a (transformed-domain) signal.
    pub fn apply(self, s: &[f64]) -> Vec<f64> {
        match self
        {
            InnerDenoiser::GaussianSmooth => gaussian_smooth(s, 4.0),
            InnerDenoiser::WienerGlobal => wiener_white(s, estimate_noise_std(s).max(1e-6)),
            InnerDenoiser::WaveletSoft =>
            {
                let levels = ((s.len() as f64).log2() as usize)
                    .saturating_sub(2)
                    .clamp(1, 6);
                wavelet_denoise(s, levels, ThresholdMode::Soft)
            },
            InnerDenoiser::StftWienerAuto => stft_wiener_auto(s),
        }
    }
}

/// A synthetic noise model with a deterministic (Philox) generator.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum NoiseModel {
    /// Multiplicative noise `y = x·(1 + α·z)` (`σ ∝ level`) — the signed-log /
    /// Box–Cox regime.
    Multiplicative {
        /// Relative noise strength.
        alpha: f64,
    },
    /// Poisson-like noise `y = x + √x·z` (`σ = √level`) — the Anscombe regime.
    PoissonLike,
    /// Additive homoscedastic Gaussian `y = x + σ·z` — the negative control (no
    /// VST should help).
    AdditiveGaussian {
        /// Noise standard deviation.
        sigma: f64,
    },
}

impl NoiseModel {
    /// Stable name (table key).
    pub fn name(self) -> &'static str {
        match self
        {
            NoiseModel::Multiplicative { .. } => "multiplicative",
            NoiseModel::PoissonLike => "poisson_like",
            NoiseModel::AdditiveGaussian { .. } => "additive_gaussian",
        }
    }

    /// Generate a noisy realization of `clean` deterministically from `seed`.
    pub fn corrupt(self, clean: &[f64], seed: u64) -> Vec<f64> {
        let rng = Philox4x32::new(seed);
        clean
            .iter()
            .enumerate()
            .map(|(i, &x)| {
                // One standard normal per sample via Box–Muller on two Philox draws.
                let u1 = (rng.f32_at(0, 2 * i as u64) as f64).max(1e-12);
                let u2 = rng.f32_at(0, 2 * i as u64 + 1) as f64;
                let z = (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos();
                match self
                {
                    NoiseModel::Multiplicative { alpha } => (x * (1.0 + alpha * z)).max(1e-6),
                    NoiseModel::PoissonLike => (x + x.max(0.0).sqrt() * z).max(1e-6),
                    NoiseModel::AdditiveGaussian { sigma } => x + sigma * z,
                }
            })
            .collect()
    }
}

/// A smooth positive clean reference signal (slow sinusoid well above zero).
pub fn clean_reference(n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| {
            let t = i as f64 / n as f64;
            20.0 + 15.0 * (2.0 * std::f64::consts::PI * 3.0 * t).sin()
        })
        .collect()
}

/// One benchmark cell: the autotuner's verdict for a (noise, denoiser) pair.
#[derive(Debug, Clone, Copy)]
pub struct BenchRow {
    /// The noise model exercised.
    pub noise: NoiseModel,
    /// The inner denoiser used.
    pub denoiser: InnerDenoiser,
    /// The empirically-selected transform (`None` if no candidate was chosen).
    pub chosen: Option<VstKind>,
    /// The winner's SNR (dB) on the held-out record.
    pub eval_snr_db: f64,
    /// The identity (direct-denoise) baseline's held-out SNR (dB).
    pub baseline_snr_db: f64,
    /// Whether the chosen VST beat direct denoising on held-out data.
    pub beats_baseline: bool,
}

impl BenchRow {
    /// Held-out gain of the chosen VST over direct denoising, in dB.
    pub fn gain_db(&self) -> f64 {
        self.eval_snr_db - self.baseline_snr_db
    }
}

/// A full benchmark table (one [`BenchRow`] per grid cell), plus renderers.
#[derive(Debug, Clone)]
pub struct BenchTable {
    /// The rows, in (noise, denoiser) input order.
    pub rows: Vec<BenchRow>,
}

impl BenchTable {
    /// Machine-readable CSV (stable column order; deterministic for a given run).
    pub fn to_csv(&self) -> String {
        let mut out = String::from(
            "noise,denoiser,chosen,eval_snr_db,baseline_snr_db,gain_db,beats_baseline\n",
        );
        for r in &self.rows
        {
            out.push_str(&format!(
                "{},{},{},{:.3},{:.3},{:.3},{}\n",
                r.noise.name(),
                r.denoiser.name(),
                r.chosen.map_or("none".to_string(), kind_name),
                r.eval_snr_db,
                r.baseline_snr_db,
                r.gain_db(),
                r.beats_baseline,
            ));
        }
        out
    }

    /// Human-readable Markdown table.
    pub fn to_markdown(&self) -> String {
        let mut out = String::from(
            "| noise | denoiser | chosen VST | eval SNR (dB) | baseline (dB) | gain (dB) | beats |\n\
             |---|---|---|--:|--:|--:|:-:|\n",
        );
        for r in &self.rows
        {
            out.push_str(&format!(
                "| {} | {} | {} | {:.2} | {:.2} | {:+.2} | {} |\n",
                r.noise.name(),
                r.denoiser.name(),
                r.chosen.map_or("none".to_string(), kind_name),
                r.eval_snr_db,
                r.baseline_snr_db,
                r.gain_db(),
                if r.beats_baseline { "✓" } else { "·" },
            ));
        }
        out
    }

    /// Find the row for a (noise, denoiser) cell, if present.
    pub fn cell(&self, noise: &str, denoiser: InnerDenoiser) -> Option<&BenchRow> {
        self.rows
            .iter()
            .find(|r| r.noise.name() == noise && r.denoiser == denoiser)
    }
}

/// Short stable label for a [`VstKind`].
fn kind_name(k: VstKind) -> String {
    match k
    {
        VstKind::Identity => "identity".into(),
        VstKind::Anscombe => "anscombe".into(),
        VstKind::SignedLog => "signed_log".into(),
        VstKind::SignedSqrt => "signed_sqrt".into(),
        VstKind::BoxCox(l) => format!("boxcox({l:.2})"),
        VstKind::Gat { gain, sigma } => format!("gat({gain:.2},{sigma:.2})"),
    }
}

/// Run the VST-selection benchmark over the grid `noises × denoisers`.
///
/// For each cell: generate a dev record (`dev_seed`) and a disjoint held-out
/// record (`eval_seed`) of `clean` corrupted by the noise model, then
/// [`autotune_vst`] the `candidates` with that inner denoiser. Deterministic for
/// fixed seeds.
pub fn vst_benchmark(
    clean: &[f64],
    noises: &[NoiseModel],
    denoisers: &[InnerDenoiser],
    candidates: &[VstKind],
    dev_seed: u64,
    eval_seed: u64,
) -> BenchTable {
    let mut rows = Vec::with_capacity(noises.len() * denoisers.len());
    for &noise in noises
    {
        let dev = DenoiseCase::new(noise.corrupt(clean, dev_seed), clean.to_vec());
        let eval = DenoiseCase::new(noise.corrupt(clean, eval_seed), clean.to_vec());
        for &den in denoisers
        {
            let r = autotune_vst(&dev, &eval, candidates, move |s: &[f64]| den.apply(s));
            rows.push(BenchRow {
                noise,
                denoiser: den,
                chosen: r.kind,
                eval_snr_db: r.eval_snr_db,
                baseline_snr_db: r.baseline_snr_db,
                beats_baseline: r.beats_baseline,
            });
        }
    }
    BenchTable { rows }
}

/// The standard benchmark grid: the three noise models × four named denoisers,
/// over the default VST candidate set, on a 1024-sample reference. Fully
/// deterministic — the reproducible artifact behind the report's Phase-1 claims.
pub fn vst_benchmark_default() -> BenchTable {
    let clean = clean_reference(1024);
    let noises = [
        NoiseModel::Multiplicative { alpha: 0.35 },
        NoiseModel::PoissonLike,
        NoiseModel::AdditiveGaussian { sigma: 1.5 },
    ];
    let denoisers = [
        InnerDenoiser::GaussianSmooth,
        InnerDenoiser::WienerGlobal,
        InnerDenoiser::WaveletSoft,
        InnerDenoiser::StftWienerAuto,
    ];
    vst_benchmark(&clean, &noises, &denoisers, &default_vst_candidates(), 1, 2)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn table_is_well_formed_and_complete() {
        let table = vst_benchmark_default();
        assert_eq!(table.rows.len(), 3 * 4, "3 noise models × 4 denoisers");
        // Every cell reached a decision and produced finite SNRs.
        for r in &table.rows
        {
            assert!(
                r.chosen.is_some(),
                "{:?}/{:?} chose nothing",
                r.noise,
                r.denoiser
            );
            assert!(r.eval_snr_db.is_finite() && r.baseline_snr_db.is_finite());
        }
    }

    #[test]
    fn vst_pays_off_for_signal_dependent_noise_with_homoscedastic_denoisers() {
        let table = vst_benchmark_default();
        // Denoisers that assume homoscedastic noise (a globally-calibrated Wiener
        // filter and the noise-floor-tracking STFT Wiener) benefit from
        // stabilization on both signal-dependent noise models — all four cells
        // beat direct denoising with a real margin, none choosing Identity.
        for noise in ["multiplicative", "poisson_like"]
        {
            for den in [InnerDenoiser::WienerGlobal, InnerDenoiser::StftWienerAuto]
            {
                let cell = table.cell(noise, den).unwrap();
                assert!(
                    cell.beats_baseline && cell.chosen != Some(VstKind::Identity),
                    "{noise}/{} should benefit from a VST: chose {:?}, gain {:.2} dB",
                    den.name(),
                    cell.chosen,
                    cell.gain_db()
                );
                assert!(
                    cell.gain_db() > 0.3,
                    "{noise}/{} gain only {:.2} dB",
                    den.name(),
                    cell.gain_db()
                );
            }
        }
    }

    #[test]
    fn honest_negatives_are_kept_in_the_table() {
        let table = vst_benchmark_default();
        // (a) A linear smoother of a smooth signal cannot be helped by a VST.
        for noise in ["multiplicative", "poisson_like", "additive_gaussian"]
        {
            let cell = table.cell(noise, InnerDenoiser::GaussianSmooth).unwrap();
            assert!(
                cell.gain_db() <= 0.1,
                "{noise}/gaussian_smooth should show no VST gain, got {:.2} dB",
                cell.gain_db()
            );
        }
        // (b) Raw-MAD wavelet thresholding measures no better stabilized on
        // signal-dependent noise (this crate's documented VisuShrink negative).
        for noise in ["multiplicative", "poisson_like"]
        {
            let cell = table.cell(noise, InnerDenoiser::WaveletSoft).unwrap();
            assert!(
                cell.gain_db() <= 0.1,
                "{noise}/wavelet_soft VST gain should be nil, got {:.2} dB",
                cell.gain_db()
            );
        }
    }

    #[test]
    fn renderers_are_deterministic_and_nonempty() {
        let a = vst_benchmark_default();
        let b = vst_benchmark_default();
        // Reproducible: identical bytes across runs.
        assert_eq!(a.to_csv(), b.to_csv());
        assert_eq!(a.to_markdown(), b.to_markdown());
        // Well-shaped: header + one line per row.
        assert_eq!(a.to_csv().lines().count(), 1 + a.rows.len());
        assert_eq!(a.to_markdown().lines().count(), 2 + a.rows.len());
    }
}
