//! Systematic denoising **quality benchmark** for `scirust_signal::denoise` — the
//! regression / selection-guidance harness behind the module's method table.
//!
//! The benchmark crosses three axes and reports the full matrix:
//!
//! * **Methods** — a representative denoiser from each family exported by the module: the
//!   plain moving average, the rank filters, the wavelet-shrinkage family (universal, TI,
//!   level-dependent, Bayes and SURE), the Wiener filters (global and short-time), the
//!   variational smoothers and the Kalman smoother, up to the three automatic entry points
//!   (`denoise_auto`, `denoise_best`, `denoise_cascade`). It is a selection, not an
//!   exhaustive enumeration of every exported function.
//! * **Noise types** — white Gaussian, colored AR(1) (pole 0.9), impulsive spikes over a
//!   small white floor, a tonal 50 Hz hum with two harmonics, slow baseline drift, a mixed
//!   disturbance (spikes + hum + white), and non-stationary white noise whose level ramps
//!   4x across the record.
//! * **Input SNR levels** — each noise vector is rescaled so the observation sits at 0, 10
//!   and 20 dB SNR against the clean reference, exactly by construction.
//!
//! Two clean references are used (n = 2048 samples at fs = 1000 Hz): a smooth three-tone
//! sine mix and a piecewise step + ramp signal, so both the "smooth" and the "edgy" regime
//! are covered. Every table cell prints `SNR-out (gain)` in dB against the clean reference;
//! the input SNR is the column label, so the gain is `SNR-out − SNR-in`. A final table names
//! the winner per noise type by the average gain over both references and all three input
//! levels, and a handful of qualitative sanity checks turn the run into a coarse regression
//! test: the rank family must win impulsive noise, the notch-routing automatic pipeline must
//! win tonal interference, and the short-time Wiener filter must beat the global one on
//! non-stationary noise. The `main` process exits non-zero when a check fails, so the example
//! can be run directly as a manual regression gate. (A plain `cargo test --workspace` compiles
//! but does *not* execute an example's `main` or its `#[cfg(test)]` tests; the same qualitative
//! assertions are duplicated as an integration test under `tests/` so CI covers them there.)
//!
//! No denoiser is ever shown the clean reference: the σ-dependent methods estimate their own
//! noise level from the observation via `estimate_noise_std` (no oracle leakage).
//!
//! Fully deterministic: noise comes from a fixed-seed LCG (copied below, because the
//! module's `testutil` helpers are `#[cfg(test)]`-only and invisible to examples) and no
//! system time or thread scheduling enters the computation, so two runs print byte-identical
//! reports.
//!
//! Run with:
//!
//! ```text
//! cargo run -p scirust-signal --example denoise_benchmark
//! ```

use core::f64::consts::PI;
use scirust_signal::denoise::{
    ThresholdMode, Wavelet, denoise_auto, denoise_best, denoise_cascade, estimate_noise_std,
    hampel_filter, kalman_smooth_auto, median_filter, moving_average, savitzky_golay,
    stft_wiener_auto, tikhonov_smooth, total_variation, wavelet_denoise, wavelet_denoise_bayes,
    wavelet_denoise_leveldep, wavelet_denoise_sure, wavelet_denoise_ti, wiener_white,
};
use std::process::ExitCode;

/// Record length shared by every fixture.
const N: usize = 2048;
/// Sample rate in Hz shared by every fixture.
const FS: f64 = 1000.0;
/// Input SNR levels (dB) the noise is scaled to for every (reference, noise-type) pair.
const SNR_TARGETS: [f64; 3] = [0.0, 10.0, 20.0];

// ---------------------------------------------------------------------------
// Deterministic helpers, copied from `denoise::testutil` — that module is
// `#[cfg(test)]`-only, so an example cannot reuse it directly.
// ---------------------------------------------------------------------------

/// Deterministic 64-bit LCG (Knuth's MMIX multiplier) so the benchmark reproduces exactly
/// without a `rand` dependency.
struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed)
    }
    fn next_u64(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0
    }
    /// Uniform in [0, 1).
    fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
    /// Standard normal via Box-Muller.
    fn gauss(&mut self) -> f64 {
        let u1 = self.uniform().max(1.0e-12);
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
    }
}

/// Signal-to-noise ratio in dB of an estimate `est` against a clean reference.
fn snr_db(clean: &[f64], est: &[f64]) -> f64 {
    let sig: f64 = clean.iter().map(|&x| x * x).sum();
    let err: f64 = clean
        .iter()
        .zip(est.iter())
        .map(|(&c, &e)| (c - e) * (c - e))
        .sum();
    10.0 * (sig / err.max(1.0e-30)).log10()
}

/// Sum of squares of a slice.
fn energy(x: &[f64]) -> f64 {
    x.iter().map(|&v| v * v).sum()
}

// ---------------------------------------------------------------------------
// Clean references.
// ---------------------------------------------------------------------------

/// Smooth reference: three tones at 5.0, 11.3 and 23.7 Hz with decreasing amplitudes and
/// arbitrary phases. Non-integer bin frequencies keep the spectrum realistic (leakage), and
/// everything sits well below the 50 Hz hum band so tonal noise is separable in principle.
fn clean_sine_mix(n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| {
            let t = i as f64 / FS;
            (2.0 * PI * 5.0 * t).sin()
                + 0.7 * (2.0 * PI * 11.3 * t + 0.6).sin()
                + 0.4 * (2.0 * PI * 23.7 * t + 1.1).sin()
        })
        .collect()
}

/// Edgy reference: flat at 0.5, step up to 2.0, ramp down to -1.0, flat at -1.0 — one
/// discontinuity, one slope corner, long constant stretches. The regime where smoothing
/// linear filters blur and edge-preserving methods (rank, wavelet, TV) are supposed to win.
fn clean_step_ramp(n: usize) -> Vec<f64> {
    let q = n / 4;
    (0..n)
        .map(|i| {
            if i < q
            {
                0.5
            }
            else if i < 2 * q
            {
                2.0
            }
            else if i < 3 * q
            {
                2.0 - 3.0 * ((i - 2 * q) as f64) / q as f64
            }
            else
            {
                -1.0
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Unit-scale noise generators. Each returns an unscaled vector; `add_noise_at_snr`
// rescales it so the observation hits the target input SNR exactly.
// ---------------------------------------------------------------------------

/// White Gaussian noise, unit scale.
fn noise_white(n: usize, seed: u64) -> Vec<f64> {
    let mut rng = Lcg::new(seed);
    (0..n).map(|_| rng.gauss()).collect()
}

/// Colored AR(1) noise `x_k = rho * x_{k-1} + w_k` — power concentrated at low frequencies,
/// the regime level-dependent wavelet thresholds were built for.
fn noise_ar1(n: usize, seed: u64, rho: f64) -> Vec<f64> {
    let mut rng = Lcg::new(seed);
    let mut state = 0.0;
    (0..n)
        .map(|_| {
            state = rho * state + rng.gauss();
            state
        })
        .collect()
}

/// Impulsive noise: a small white floor (sigma 0.05) plus rare large spikes — probability
/// 1/64 per sample, signed amplitude in [4, 8]. Spikes carry ~99 % of the power.
fn noise_impulsive(n: usize, seed: u64) -> Vec<f64> {
    let mut rng = Lcg::new(seed);
    (0..n)
        .map(|_| {
            let floor = 0.05 * rng.gauss();
            if rng.uniform() < 1.0 / 64.0
            {
                let sign = if rng.uniform() < 0.5 { -1.0 } else { 1.0 };
                floor + sign * (4.0 + 4.0 * rng.uniform())
            }
            else
            {
                floor
            }
        })
        .collect()
}

/// Tonal interference: 50 Hz plus its second and third harmonics (100, 150 Hz) with
/// decreasing amplitudes — the classic mains-hum stack.
fn noise_tonal(n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| {
            let t = i as f64 / FS;
            (2.0 * PI * 50.0 * t).sin()
                + 0.5 * (2.0 * PI * 100.0 * t + 0.4).sin()
                + 0.25 * (2.0 * PI * 150.0 * t + 1.3).sin()
        })
        .collect()
}

/// Baseline drift: one slow 0.6 Hz sine — a hair over one cycle across the 2.048 s record,
/// well below the slowest (5 Hz) signal component.
fn noise_drift(n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| {
            let t = i as f64 / FS;
            (2.0 * PI * 0.6 * t + 0.5).sin()
        })
        .collect()
}

/// Mixed disturbance: sparse signed spikes (probability 1/128, amplitude [4, 6]) + 50 Hz hum
/// + a white floor — the multi-family case the cascade was built for.
fn noise_mixed(n: usize, seed: u64) -> Vec<f64> {
    let mut rng = Lcg::new(seed);
    (0..n)
        .map(|i| {
            let t = i as f64 / FS;
            let mut v = 0.35 * rng.gauss() + 0.8 * (2.0 * PI * 50.0 * t).sin();
            if rng.uniform() < 1.0 / 128.0
            {
                let sign = if rng.uniform() < 0.5 { -1.0 } else { 1.0 };
                v += sign * (4.0 + 2.0 * rng.uniform());
            }
            v
        })
        .collect()
}

/// Non-stationary white noise: sigma ramps linearly from 1x to 4x across the record — the
/// regime where a single global Wiener gain must lose to per-frame short-time gains.
fn noise_nonstationary(n: usize, seed: u64) -> Vec<f64> {
    let mut rng = Lcg::new(seed);
    (0..n)
        .map(|i| (1.0 + 3.0 * i as f64 / (n - 1).max(1) as f64) * rng.gauss())
        .collect()
}

/// Return `clean + alpha * noise` with `alpha` chosen so that `snr_db(clean, noisy)` equals
/// `target_snr_db` **exactly**: `alpha = sqrt(E_s / (E_n * 10^(target/10)))` with `E_s`,
/// `E_n` the energies of the clean reference and the unit noise vector.
fn add_noise_at_snr(clean: &[f64], noise: &[f64], target_snr_db: f64) -> Vec<f64> {
    let es = energy(clean);
    let en = energy(noise);
    if en <= 0.0
    {
        return clean.to_vec();
    }
    let alpha = (es / (en * 10.0_f64.powf(target_snr_db / 10.0))).sqrt();
    clean
        .iter()
        .zip(noise.iter())
        .map(|(&c, &w)| c + alpha * w)
        .collect()
}

// ---------------------------------------------------------------------------
// Method catalog.
// ---------------------------------------------------------------------------

/// A boxed denoiser: input samples in, same-length estimate out.
type DenoiseFn = Box<dyn Fn(&[f64]) -> Vec<f64>>;

/// One benchmark entry: a display name, whether it is an automatic detect-then-denoise
/// pipeline (`meta`) rather than a single fixed method, and the denoiser itself.
struct Method {
    name: &'static str,
    meta: bool,
    run: DenoiseFn,
}

impl Method {
    fn fixed(name: &'static str, run: impl Fn(&[f64]) -> Vec<f64> + 'static) -> Self {
        Self {
            name,
            meta: false,
            run: Box::new(run),
        }
    }
    fn auto(name: &'static str, run: impl Fn(&[f64]) -> Vec<f64> + 'static) -> Self {
        Self {
            name,
            meta: true,
            run: Box::new(run),
        }
    }
}

/// The benchmarked method matrix. Order matters twice: ties in the winner tables resolve to
/// the earlier entry, and the fixed methods are listed before the automatic pipelines so
/// that an automatic pipeline which merely reproduces a fixed method (e.g. `denoise_auto`
/// routing impulsive noise to the very same Hampel filter) does not steal its win.
fn method_catalog() -> Vec<Method> {
    vec![
        Method::fixed("moving_average(5)", |x| moving_average(x, 5)),
        Method::fixed("savitzky_golay(2,5)", |x| savitzky_golay(x, 2, 5)),
        Method::fixed("median_filter(3)", |x| median_filter(x, 3)),
        Method::fixed("hampel_filter(3,3.0)", |x| hampel_filter(x, 3, 3.0)),
        Method::fixed("wavelet_denoise(0,Soft)", |x| {
            wavelet_denoise(x, 0, ThresholdMode::Soft)
        }),
        Method::fixed("wavelet_denoise_ti(0,Soft,Db4,15)", |x| {
            wavelet_denoise_ti(x, 0, ThresholdMode::Soft, Wavelet::Db4, 15)
        }),
        Method::fixed("wavelet_denoise_leveldep(0,Soft,Db4)", |x| {
            wavelet_denoise_leveldep(x, 0, ThresholdMode::Soft, Wavelet::Db4)
        }),
        Method::fixed("wavelet_denoise_bayes(0,Db4)", |x| {
            wavelet_denoise_bayes(x, 0, Wavelet::Db4)
        }),
        Method::fixed("wavelet_denoise_sure(0,Db4)", |x| {
            wavelet_denoise_sure(x, 0, Wavelet::Db4)
        }),
        // The noise level is re-estimated from the observation itself — never the oracle.
        Method::fixed("wiener_white(sigma_hat)", |x| {
            wiener_white(x, estimate_noise_std(x))
        }),
        Method::fixed("stft_wiener_auto", stft_wiener_auto),
        Method::fixed("total_variation(1.0,8)", |x| total_variation(x, 1.0, 8)),
        Method::fixed("tikhonov_smooth(10)", |x| tikhonov_smooth(x, 10.0)),
        Method::fixed("kalman_smooth_auto", |x| kalman_smooth_auto(x).output),
        Method::auto("denoise_auto", |x| denoise_auto(x, FS).output),
        Method::auto("denoise_best", |x| denoise_best(x, FS).output),
        Method::auto("denoise_cascade(4)", |x| denoise_cascade(x, FS, 4).output),
    ]
}

// ---------------------------------------------------------------------------
// Markdown table rendering.
// ---------------------------------------------------------------------------

/// Render an aligned markdown table: the first column is left-aligned (method names), every
/// other column right-aligned (numbers). All pipes line up, so the raw terminal output reads
/// as cleanly as the rendered markdown.
fn render_table(headers: &[String], rows: &[Vec<String>]) -> String {
    let mut widths: Vec<usize> = headers.iter().map(|h| h.len()).collect();
    for row in rows
    {
        for (i, cell) in row.iter().enumerate()
        {
            widths[i] = widths[i].max(cell.len());
        }
    }
    let mut out = String::new();
    out.push('|');
    for (i, h) in headers.iter().enumerate()
    {
        if i == 0
        {
            out.push_str(&format!(" {:<w$} |", h, w = widths[i]));
        }
        else
        {
            out.push_str(&format!(" {:>w$} |", h, w = widths[i]));
        }
    }
    out.push('\n');
    out.push('|');
    for (i, &w) in widths.iter().enumerate()
    {
        if i == 0
        {
            out.push(':');
            out.push_str(&"-".repeat(w + 1));
        }
        else
        {
            out.push_str(&"-".repeat(w + 1));
            out.push(':');
        }
        out.push('|');
    }
    out.push('\n');
    for row in rows
    {
        out.push('|');
        for (i, cell) in row.iter().enumerate()
        {
            if i == 0
            {
                out.push_str(&format!(" {:<w$} |", cell, w = widths[i]));
            }
            else
            {
                out.push_str(&format!(" {:>w$} |", cell, w = widths[i]));
            }
        }
        out.push('\n');
    }
    out
}

/// Index of the largest value; ties resolve to the earliest entry (deterministic).
fn argmax(v: &[f64]) -> usize {
    let mut best = 0;
    for (i, &x) in v.iter().enumerate()
    {
        if x > v[best]
        {
            best = i;
        }
    }
    best
}

/// Index of the largest value excluding `winner`; ties resolve to the earliest entry.
/// Returns `winner` itself only in the degenerate single-entry case.
fn runner_up(v: &[f64], winner: usize) -> usize {
    let mut best: Option<usize> = None;
    for (i, &x) in v.iter().enumerate()
    {
        if i == winner
        {
            continue;
        }
        if best.is_none_or(|b| x > v[b])
        {
            best = Some(i);
        }
    }
    best.unwrap_or(winner)
}

// ---------------------------------------------------------------------------
// The benchmark itself.
// ---------------------------------------------------------------------------

fn main() -> ExitCode {
    let cleans: [(&str, Vec<f64>); 2] = [("sine", clean_sine_mix(N)), ("step", clean_step_ramp(N))];
    let noises: [(&str, Vec<f64>); 7] = [
        ("white gaussian", noise_white(N, 0xBEEF_0001)),
        ("colored AR(1) 0.9", noise_ar1(N, 0xBEEF_0002, 0.9)),
        (
            "impulsive (white 0.05 + spikes)",
            noise_impulsive(N, 0xBEEF_0003),
        ),
        ("tonal (50 Hz + 2 harmonics)", noise_tonal(N)),
        ("baseline drift (slow sine)", noise_drift(N)),
        ("mixed (spikes + hum + white)", noise_mixed(N, 0xBEEF_0006)),
        (
            "non-stationary (sigma ramp 4x)",
            noise_nonstationary(N, 0xBEEF_0007),
        ),
    ];
    let methods = method_catalog();
    let cells_per_noise = cleans.len() * SNR_TARGETS.len();

    println!("# Denoising quality benchmark");
    println!();
    println!(
        "Matrix: {} methods x {} noise types x {} input SNR levels ({} dB), on two clean",
        methods.len(),
        noises.len(),
        SNR_TARGETS.len(),
        SNR_TARGETS
            .iter()
            .map(|t| format!("{t:.0}"))
            .collect::<Vec<_>>()
            .join("/"),
    );
    println!("references (n = {N}, fs = {FS:.0} Hz): 'sine' = smooth three-tone mix, 'step' =");
    println!("step + ramp. Each cell is `SNR-out (gain)` in dB against the clean reference;");
    println!("the input SNR is the column label, so gain = SNR-out - SNR-in. No denoiser sees");
    println!("the clean reference: sigma-dependent methods estimate their own noise level from");
    println!("the observation (estimate_noise_std). Deterministic (fixed-seed LCG).");

    // avg_gain[noise][method]: mean gain over the 2 x 3 grid, feeding the winner tables and
    // the sanity checks.
    let mut avg_gain = vec![vec![0.0_f64; methods.len()]; noises.len()];
    let mut all_finite = true;

    for (ni, (noise_name, noise)) in noises.iter().enumerate()
    {
        // Precompute the 2 x 3 grid of observations once per noise type; every method sees
        // the exact same noisy records (a paired comparison).
        let mut headers = vec!["method".to_string()];
        let mut grid: Vec<(&[f64], Vec<f64>)> = Vec::new();
        for (clean_name, clean) in cleans.iter()
        {
            for &target in SNR_TARGETS.iter()
            {
                let noisy = add_noise_at_snr(clean, noise, target);
                let realized = snr_db(clean, &noisy);
                assert!(
                    (realized - target).abs() < 1.0e-6,
                    "SNR scaling failed: wanted {target} dB, realized {realized} dB"
                );
                headers.push(format!("{clean_name} @ {target:.0} dB"));
                grid.push((clean.as_slice(), noisy));
            }
        }

        let mut rows: Vec<Vec<String>> = Vec::new();
        for (mi, method) in methods.iter().enumerate()
        {
            let mut row = vec![method.name.to_string()];
            let mut gain_sum = 0.0;
            for (clean, noisy) in grid.iter()
            {
                let out = (method.run)(noisy);
                let s_in = snr_db(clean, noisy);
                let s_out = snr_db(clean, &out);
                all_finite &= s_out.is_finite() && out.iter().all(|v| v.is_finite());
                let gain = s_out - s_in;
                gain_sum += gain;
                row.push(format!("{s_out:.1} ({gain:+.1})"));
            }
            avg_gain[ni][mi] = gain_sum / grid.len() as f64;
            rows.push(row);
        }

        println!();
        println!("## Noise: {noise_name}");
        println!();
        print!("{}", render_table(&headers, &rows));
    }

    // ------------------------------------------------------------------
    // Winner per noise type.
    // ------------------------------------------------------------------
    println!();
    println!(
        "## Winner per noise type (average gain over {cells_per_noise} cells: 2 references x \
         {} SNR levels)",
        SNR_TARGETS.len()
    );
    println!();
    let headers: Vec<String> = [
        "noise type",
        "winner",
        "avg gain (dB)",
        "runner-up",
        "runner-up gain (dB)",
    ]
    .map(String::from)
    .to_vec();
    let mut rows: Vec<Vec<String>> = Vec::new();
    for (ni, (noise_name, _)) in noises.iter().enumerate()
    {
        let w = argmax(&avg_gain[ni]);
        let r = runner_up(&avg_gain[ni], w);
        rows.push(vec![
            noise_name.to_string(),
            methods[w].name.to_string(),
            format!("{:+.1}", avg_gain[ni][w]),
            methods[r].name.to_string(),
            format!("{:+.1}", avg_gain[ni][r]),
        ]);
    }
    print!("{}", render_table(&headers, &rows));

    // ------------------------------------------------------------------
    // Qualitative sanity checks — the regression part of the harness.
    // ------------------------------------------------------------------
    let method_index = |name: &str| {
        methods
            .iter()
            .position(|m| m.name == name)
            .expect("method name")
    };
    let noise_index = |prefix: &str| {
        noises
            .iter()
            .position(|(n, _)| n.starts_with(prefix))
            .expect("noise name")
    };

    // 1. A rank filter must be the best FIXED method on impulsive noise. The automatic
    //    pipelines are excluded: they route impulsive noise to the Hampel filter themselves
    //    (and the cascade may add further stages), so beating them is not the claim. Both
    //    rank entrants are accepted: the Hampel filter removes the spikes with the least
    //    signal distortion, but it deliberately leaves the small white floor untouched, so
    //    the plain median — which also smooths the floor — can edge it on total SNR.
    let imp = noise_index("impulsive");
    let mut best_fixed: Option<usize> = None;
    for (i, method) in methods.iter().enumerate()
    {
        if !method.meta && best_fixed.is_none_or(|b| avg_gain[imp][i] > avg_gain[imp][b])
        {
            best_fixed = Some(i);
        }
    }
    let best_fixed = best_fixed.expect("at least one fixed method");
    let rank_wins = best_fixed == method_index("median_filter(3)")
        || best_fixed == method_index("hampel_filter(3,3.0)");

    // 2. Tonal interference must be won overall by one of the automatic pipelines — the only
    //    entrants with a notch treatment (no fixed method in this matrix notches).
    let tonal_winner = argmax(&avg_gain[noise_index("tonal")]);

    // 3. The short-time Wiener filter must beat the single global Wiener gain when the noise
    //    level ramps — non-stationarity is the whole point of the STFT variant.
    let ns = noise_index("non-stationary");
    let stft_beats_wiener = avg_gain[ns][method_index("stft_wiener_auto")]
        > avg_gain[ns][method_index("wiener_white(sigma_hat)")];

    let checks = [
        (
            "every SNR value in the matrix is finite".to_string(),
            all_finite,
        ),
        (
            format!(
                "a rank filter is the best fixed method on impulsive noise (best: {})",
                methods[best_fixed].name
            ),
            rank_wins,
        ),
        (
            format!(
                "a notch-routing automatic pipeline wins tonal noise (winner: {})",
                methods[tonal_winner].name
            ),
            methods[tonal_winner].meta,
        ),
        (
            "stft_wiener_auto beats wiener_white(sigma_hat) on non-stationary noise".to_string(),
            stft_beats_wiener,
        ),
    ];
    println!();
    println!("## Sanity checks");
    println!();
    let mut ok = true;
    for (label, pass) in checks.iter()
    {
        println!("- [{}] {label}", if *pass { "PASS" } else { "FAIL" });
        ok &= *pass;
    }

    // Selection guidance the matrix consistently shows — worth stating next to the raw
    // numbers, because several of these are documented sharp edges of specific methods.
    println!();
    println!("## Notes");
    println!();
    println!(
        "- Impulsive: hampel removes the spikes with the least distortion but leaves the white\n\
         \x20 floor untouched by design, so the plain median (which also smooths the floor) can\n\
         \x20 edge it on total SNR; the cascade (rank stage, then a broadband stage) tops both."
    );
    println!(
        "- wavelet_denoise_leveldep wipes sustained tones (its per-band MAD reads a dense band\n\
         \x20 as noise) — the documented caveat explains its large negative cells on the sine\n\
         \x20 reference; prefer the global or Bayes rule on tonal content."
    );
    println!(
        "- Baseline drift at input SNR >= 0 dB never trips the classifier's Baseline gate (the\n\
         \x20 drift would need ~1.5x the signal power), and no fixed method here can separate a\n\
         \x20 sub-signal-band drift: detrend explicitly (fft_highpass, signal minus a stiff\n\
         \x20 tikhonov trend) when drift is visible."
    );
    println!(
        "- On the 'step' reference at high input SNR the classifier may read the step's own\n\
         \x20 low-frequency energy as drift and subtract a trend — the negative cells of the\n\
         \x20 automatic pipelines. Those pipelines are tuned for noise-dominated records; on\n\
         \x20 nearly-clean data prefer a fixed method."
    );
    if ok
    {
        ExitCode::SUCCESS
    }
    else
    {
        ExitCode::FAILURE
    }
}

// ---------------------------------------------------------------------------
// Tests. Run with: cargo test -p scirust-signal --example denoise_benchmark
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lcg_is_deterministic_and_seed_is_live() {
        let seq = |seed: u64| {
            let mut rng = Lcg::new(seed);
            (0..64).map(|_| rng.next_u64()).collect::<Vec<_>>()
        };
        assert_eq!(seq(42), seq(42), "same seed must reproduce exactly");
        assert_ne!(seq(42), seq(43), "the seed must matter");
        // Rough sanity of the Gaussian: zero-mean, unit-ish variance.
        let mut rng = Lcg::new(7);
        let draws: Vec<f64> = (0..4096).map(|_| rng.gauss()).collect();
        let mean = draws.iter().sum::<f64>() / draws.len() as f64;
        let var = draws.iter().map(|&x| (x - mean) * (x - mean)).sum::<f64>() / draws.len() as f64;
        assert!(mean.abs() < 0.1, "mean {mean}");
        assert!((var - 1.0).abs() < 0.15, "var {var}");
    }

    #[test]
    fn snr_db_matches_its_definition() {
        // E_s = 25, err = 1 → 10·log10(25).
        let clean = [3.0, 4.0];
        let est = [3.0, 3.0];
        assert!((snr_db(&clean, &est) - 10.0 * 25.0_f64.log10()).abs() < 1.0e-12);
        // A perfect estimate is clamped by the 1e-30 error floor, not infinite/NaN.
        assert!(snr_db(&clean, &clean) > 300.0);
        assert!(snr_db(&clean, &clean).is_finite());
    }

    #[test]
    fn add_noise_at_snr_hits_the_target_exactly() {
        let cleans = [clean_sine_mix(N), clean_step_ramp(N)];
        let noises = [
            noise_white(N, 1),
            noise_ar1(N, 2, 0.9),
            noise_impulsive(N, 3),
            noise_tonal(N),
            noise_drift(N),
            noise_mixed(N, 6),
            noise_nonstationary(N, 7),
        ];
        for clean in cleans.iter()
        {
            for noise in noises.iter()
            {
                for &target in SNR_TARGETS.iter()
                {
                    let noisy = add_noise_at_snr(clean, noise, target);
                    assert_eq!(noisy.len(), clean.len());
                    let realized = snr_db(clean, &noisy);
                    assert!(
                        (realized - target).abs() < 1.0e-9,
                        "target {target} dB, realized {realized} dB"
                    );
                }
            }
        }
        // Degenerate: an all-zero noise vector cannot be scaled — the clean copy comes back.
        let zeros = vec![0.0; N];
        assert_eq!(add_noise_at_snr(&cleans[0], &zeros, 10.0), cleans[0]);
    }

    #[test]
    fn noise_generator_parameters_are_live() {
        // rho = 0 collapses AR(1) to white noise from the same seed — pins both the seed and
        // the rho plumbing at once.
        assert_eq!(noise_ar1(64, 9, 0.0), noise_white(64, 9));
        assert_ne!(noise_ar1(64, 9, 0.9), noise_white(64, 9));
        assert_ne!(noise_white(64, 1), noise_white(64, 2), "seed is ignored");
        // The non-stationary ramp: the last quarter must carry far more power than the first
        // (sigma ratio 4 → power ratio ~7 between the end quarters).
        let ns = noise_nonstationary(N, 11);
        let q = N / 4;
        let p_first = energy(&ns[..q]) / q as f64;
        let p_last = energy(&ns[N - q..]) / q as f64;
        assert!(p_last > 3.0 * p_first, "{p_last} vs {p_first}");
        // Impulsive: rare spikes (expected ~N/64) over a floor that never reaches 2.0.
        let imp = noise_impulsive(N, 12);
        let n_spikes = imp.iter().filter(|v| v.abs() > 2.0).count();
        assert!((8..=96).contains(&n_spikes), "{n_spikes} spikes");
        // Tonal and drift are deterministic (no RNG at all).
        assert_eq!(noise_tonal(N), noise_tonal(N));
        assert_eq!(noise_drift(N), noise_drift(N));
    }

    #[test]
    fn clean_references_have_the_advertised_shape() {
        let sine = clean_sine_mix(N);
        let step = clean_step_ramp(N);
        assert_eq!(sine.len(), N);
        assert_eq!(step.len(), N);
        assert!(sine.iter().all(|v| v.is_finite()));
        // The piecewise reference really steps and ramps: flat, high flat, ramp end, low flat.
        let q = N / 4;
        assert_eq!(step[0], 0.5);
        assert_eq!(step[q - 1], 0.5);
        assert_eq!(step[q], 2.0);
        assert_eq!(step[2 * q - 1], 2.0);
        assert!((step[3 * q - 1] - -1.0).abs() < 0.01, "ramp must reach -1");
        assert_eq!(step[N - 1], -1.0);
    }

    #[test]
    fn method_catalog_wrappers_match_their_functions() {
        // Same idiom as `denoiser_wrappers_match_their_functions` in the module tests: pin
        // every closure against the direct call so a transposed or ignored parameter (many
        // are same-typed) cannot compile silently.
        let mut rng = Lcg::new(77);
        let obs: Vec<f64> = (0..512)
            .map(|i| (2.0 * PI * 4.0 * i as f64 / 512.0).sin() + 0.3 * rng.gauss())
            .collect();
        let methods = method_catalog();
        let by_name = |name: &str| {
            methods
                .iter()
                .find(|m| m.name == name)
                .unwrap_or_else(|| panic!("method {name} missing from the catalog"))
        };
        assert_eq!(
            (by_name("moving_average(5)").run)(&obs),
            moving_average(&obs, 5)
        );
        assert_eq!(
            (by_name("savitzky_golay(2,5)").run)(&obs),
            savitzky_golay(&obs, 2, 5)
        );
        assert_eq!(
            (by_name("median_filter(3)").run)(&obs),
            median_filter(&obs, 3)
        );
        assert_eq!(
            (by_name("hampel_filter(3,3.0)").run)(&obs),
            hampel_filter(&obs, 3, 3.0)
        );
        assert_eq!(
            (by_name("wavelet_denoise(0,Soft)").run)(&obs),
            wavelet_denoise(&obs, 0, ThresholdMode::Soft)
        );
        assert_eq!(
            (by_name("wavelet_denoise_ti(0,Soft,Db4,15)").run)(&obs),
            wavelet_denoise_ti(&obs, 0, ThresholdMode::Soft, Wavelet::Db4, 15)
        );
        assert_eq!(
            (by_name("wavelet_denoise_leveldep(0,Soft,Db4)").run)(&obs),
            wavelet_denoise_leveldep(&obs, 0, ThresholdMode::Soft, Wavelet::Db4)
        );
        assert_eq!(
            (by_name("wavelet_denoise_bayes(0,Db4)").run)(&obs),
            wavelet_denoise_bayes(&obs, 0, Wavelet::Db4)
        );
        assert_eq!(
            (by_name("wavelet_denoise_sure(0,Db4)").run)(&obs),
            wavelet_denoise_sure(&obs, 0, Wavelet::Db4)
        );
        assert_eq!(
            (by_name("wiener_white(sigma_hat)").run)(&obs),
            wiener_white(&obs, estimate_noise_std(&obs))
        );
        assert_eq!(
            (by_name("stft_wiener_auto").run)(&obs),
            stft_wiener_auto(&obs)
        );
        assert_eq!(
            (by_name("total_variation(1.0,8)").run)(&obs),
            total_variation(&obs, 1.0, 8)
        );
        assert_eq!(
            (by_name("tikhonov_smooth(10)").run)(&obs),
            tikhonov_smooth(&obs, 10.0)
        );
        assert_eq!(
            (by_name("kalman_smooth_auto").run)(&obs),
            kalman_smooth_auto(&obs).output
        );
        assert_eq!(
            (by_name("denoise_auto").run)(&obs),
            denoise_auto(&obs, FS).output
        );
        assert_eq!(
            (by_name("denoise_best").run)(&obs),
            denoise_best(&obs, FS).output
        );
        assert_eq!(
            (by_name("denoise_cascade(4)").run)(&obs),
            denoise_cascade(&obs, FS, 4).output
        );
        // The meta flags drive the sanity checks: exactly the three automatic pipelines.
        let metas: Vec<&str> = methods.iter().filter(|m| m.meta).map(|m| m.name).collect();
        assert_eq!(
            metas,
            vec!["denoise_auto", "denoise_best", "denoise_cascade(4)"]
        );
    }

    #[test]
    fn render_table_aligns_every_line() {
        let headers: Vec<String> = ["method", "col a", "b"].map(String::from).to_vec();
        let rows = vec![
            vec!["x".to_string(), "1.0 (+0.1)".to_string(), "yy".to_string()],
            vec![
                "a much longer name".to_string(),
                "2".to_string(),
                "z".to_string(),
            ],
        ];
        let table = render_table(&headers, &rows);
        let lines: Vec<&str> = table.lines().collect();
        assert_eq!(lines.len(), 2 + rows.len());
        let pipes = |line: &str| -> Vec<usize> {
            line.char_indices()
                .filter(|&(_, c)| c == '|')
                .map(|(i, _)| i)
                .collect()
        };
        let reference = pipes(lines[0]);
        assert_eq!(reference.len(), headers.len() + 1);
        for line in lines.iter()
        {
            assert_eq!(line.len(), lines[0].len(), "ragged line: {line}");
            assert!(line.starts_with('|') && line.ends_with('|'));
            assert_eq!(pipes(line), reference, "pipes misaligned: {line}");
        }
    }

    #[test]
    fn argmax_and_runner_up_prefer_earlier_on_ties() {
        assert_eq!(argmax(&[1.0, 3.0, 3.0]), 1);
        assert_eq!(argmax(&[2.0]), 0);
        assert_eq!(runner_up(&[1.0, 3.0, 3.0], 1), 2);
        assert_eq!(runner_up(&[5.0, 1.0, 4.0], 0), 2);
        assert_eq!(
            runner_up(&[5.0], 0),
            0,
            "single entry falls back to the winner"
        );
    }

    #[test]
    fn every_method_degrades_gracefully_on_edge_inputs() {
        // Module-wide convention: degenerate inputs return same-length finite output, never
        // panic. Exercised here end-to-end through every catalog entry, including the
        // automatic pipelines.
        let methods = method_catalog();
        let cases: Vec<Vec<f64>> = vec![
            vec![],
            vec![1.5],
            vec![1.5, -0.5],
            vec![1.5, -0.5, 2.0],
            vec![3.5; 64],
        ];
        for case in cases.iter()
        {
            for method in methods.iter()
            {
                let out = (method.run)(case);
                assert_eq!(
                    out.len(),
                    case.len(),
                    "{} changed the length on input of len {}",
                    method.name,
                    case.len()
                );
                assert!(
                    out.iter().all(|v| v.is_finite()),
                    "{} produced non-finite output on input of len {}",
                    method.name,
                    case.len()
                );
            }
        }
    }

    #[test]
    fn benchmark_oracle_snr_improves_where_it_must() {
        // A miniature of the main matrix as a behavioral oracle: on white noise at 10 dB the
        // Wiener filter must improve the SNR, and on impulsive noise the Hampel filter must.
        let clean = clean_sine_mix(N);
        let noisy_w = add_noise_at_snr(&clean, &noise_white(N, 21), 10.0);
        let wiener = wiener_white(&noisy_w, estimate_noise_std(&noisy_w));
        assert!(
            snr_db(&clean, &wiener) > snr_db(&clean, &noisy_w) + 1.0,
            "wiener gained too little on white noise"
        );
        let noisy_i = add_noise_at_snr(&clean, &noise_impulsive(N, 22), 10.0);
        let hampel = hampel_filter(&noisy_i, 3, 3.0);
        assert!(
            snr_db(&clean, &hampel) > snr_db(&clean, &noisy_i) + 10.0,
            "hampel gained too little on impulsive noise"
        );
    }
}
