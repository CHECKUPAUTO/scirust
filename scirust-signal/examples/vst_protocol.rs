//! **VST experimental protocol (P1-P5)** — the reproducible measurement harness
//! implementing §9 (protocole expérimental) and §10 (métriques, seuil de matérialité
//! +0.5 dB) of the research report `TSHF_RESEARCH_2026-07-16.md`, exercising the
//! `denoise::vst` module (Anscombe + exact unbiased inverse, SignedLog + Duan
//! smearing, and the Generalized Anscombe Transformation `VstKind::Gat` for mixed
//! Poisson-Gaussian noise). Run with:
//!
//! ```text
//! cargo run --release -p scirust-signal --example vst_protocol
//! ```
//!
//! Blocks (n = 4096, fixed seeds; SNR and bias are measured in ORIGINAL coordinates,
//! as §10 requires):
//!
//! * **P1** — Poisson low count (§9.1a), slow intensity λ ∈ [1, ~21]: raw
//!   observation vs identity pipeline vs VST-naive (algebraic Anscombe inverse) vs
//!   VST-corrected (exact unbiased inverse), over four inner denoisers; plus the
//!   §10 retransformation-bias metric (mean(x̂) − mean(λ), cf. E4).
//! * **P2** — multiplicative 10-40 % noise (§9.1b) on a strong ×10 level range:
//!   identity vs SignedLog + smearing inverse.
//! * **P3** — mixed Poisson-Gaussian CCD model (§9.1c, the Starck-Murtagh case):
//!   identity vs GAT-corrected across four (gain, σ) calibrations.
//! * **P4** — crossover sweeps (§9.3, the report's open question): VST gain vs
//!   (a) multiplicative noise fraction and (b) level dynamic range, locating where
//!   the gain crosses the +0.5 dB materiality threshold of §10.
//! * **P5** — carrier-regime sweep: the round-5 measured limitation (module docs of
//!   `denoise::vst`, "Known limitation: fast carriers") — the Anscombe gain as the
//!   intensity carrier speeds up from 3 to 40 cycles per record.
//!
//! Every number is deterministic: fixed-seed LCG (the same generator as the crate's
//! test fixtures), Knuth's Poisson sampler, no system randomness — repeated runs
//! print bit-identical tables. The final summary is computed from the measured
//! numbers, never hardcoded.

use scirust_signal::denoise::{
    ThresholdMode, VstKind, collab1d_auto, moving_average, stft_wiener_auto, vst_denoise,
    vst_forward, vst_inverse_naive, wavelet_denoise,
};
use std::f64::consts::PI;

/// Record length shared by every block (§9: n = 4096).
const N: usize = 4096;
/// §10 materiality threshold: any |gain| below this is declared null.
const MATERIALITY_DB: f64 = 0.5;

/// Deterministic 64-bit LCG — same multiplier/increment as the crate's test
/// fixtures, so the protocol is reproducible bit-for-bit without a `rand`
/// dependency.
struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed)
    }
    /// Uniform in [0, 1).
    fn uniform(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
    /// Standard normal via Box-Muller.
    fn gauss(&mut self) -> f64 {
        let u1 = self.uniform().max(1.0e-12);
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
    }
}

/// Poisson sampler — Knuth's product-of-uniforms algorithm; adequate for the λ ≲ 30
/// range of every fixture below (the largest intensity probed is ≈ 21).
fn poisson(rng: &mut Lcg, lambda: f64) -> f64 {
    if lambda <= 0.0
    {
        return 0.0;
    }
    let l = (-lambda).exp();
    let mut k: u64 = 0;
    let mut p = 1.0;
    loop
    {
        k += 1;
        p *= rng.uniform();
        if p <= l
        {
            break;
        }
    }
    (k - 1) as f64
}

/// SNR (dB) of `est` against `clean` — always in original coordinates (§10).
fn snr(clean: &[f64], est: &[f64]) -> f64 {
    let sig: f64 = clean.iter().map(|&x| x * x).sum();
    let err: f64 = clean
        .iter()
        .zip(est)
        .map(|(&c, &e)| (c - e) * (c - e))
        .sum();
    10.0 * (sig / err.max(1.0e-30)).log10()
}

fn mean(x: &[f64]) -> f64 {
    x.iter().sum::<f64>() / x.len() as f64
}

/// §10 verdict rule applied to a measured gain — never a hardcoded expectation.
fn verdict(gain_db: f64) -> &'static str {
    if gain_db >= MATERIALITY_DB
    {
        "material gain (>= +0.5 dB)"
    }
    else if gain_db <= -MATERIALITY_DB
    {
        "material LOSS (<= -0.5 dB)"
    }
    else
    {
        "null (|gain| < 0.5 dB, §10)"
    }
}

/// P1/P3 slow intensity: `λᵢ = 10.5 + 9.5·sin(2π·3·i/n) + i/n`, so λ ∈ [1, ~21] —
/// the low-count §9.1a profile (3 cycles + ramp, same shape family as the vst.rs
/// acceptance-gate fixture).
fn p1_intensity(n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| 10.5 + 9.5 * (2.0 * PI * 3.0 * i as f64 / n as f64).sin() + i as f64 / n as f64)
        .collect()
}

/// Poisson counts drawn per-sample from `lambda` with a fixed seed.
fn poisson_counts(lambda: &[f64], seed: u64) -> Vec<f64> {
    let mut rng = Lcg::new(seed);
    lambda.iter().map(|&l| poisson(&mut rng, l)).collect()
}

/// `(clean, noisy)` multiplicative fixture: levels sweep `[lo, hi]` over 3 slow
/// cycles (the vst.rs `strong_multiplicative_fixture` style), `x = s·(1 + f·g)`.
fn multiplicative_fixture(
    n: usize,
    lo: f64,
    hi: f64,
    fraction: f64,
    seed: u64,
) -> (Vec<f64>, Vec<f64>) {
    let amp = 0.5 * (hi - lo);
    let mid = lo + amp;
    let clean: Vec<f64> = (0..n)
        .map(|i| mid + amp * (2.0 * PI * 3.0 * i as f64 / n as f64).sin())
        .collect();
    let mut rng = Lcg::new(seed);
    let noisy = clean
        .iter()
        .map(|&s| s * (1.0 + fraction * rng.gauss()))
        .collect();
    (clean, noisy)
}

/// `(clean = gain·λ, noisy)` mixed Poisson-Gaussian CCD fixture (§9.1c):
/// `x = gain·p + σ·g`, `p ~ Poisson(λᵢ)` on the P1 slow intensity.
fn gat_fixture(lambda: &[f64], gain: f64, sigma: f64, seed: u64) -> (Vec<f64>, Vec<f64>) {
    let mut rng = Lcg::new(seed);
    let noisy = lambda
        .iter()
        .map(|&l| gain * poisson(&mut rng, l) + sigma * rng.gauss())
        .collect();
    let clean = lambda.iter().map(|&l| gain * l).collect();
    (clean, noisy)
}

/// P5 pure-Poisson carrier intensity: `λᵢ = 6.5 + 5.5·sin(2π·c·i/n)`, so λ ∈ [1, 12]
/// at every carrier speed `c` (cycles per record).
fn carrier_intensity(n: usize, cycles: f64) -> Vec<f64> {
    (0..n)
        .map(|i| 6.5 + 5.5 * (2.0 * PI * cycles * i as f64 / n as f64).sin())
        .collect()
}

/// An inner Gaussian denoiser, by reference — the columns of the P1 table.
type Denoiser<'a> = &'a dyn Fn(&[f64]) -> Vec<f64>;

/// Identity pipeline: the denoiser applied directly in the original coordinates.
fn arm_identity(noisy: &[f64], f: Denoiser) -> Vec<f64> {
    f(noisy)
}

/// VST-naive pipeline: `φ → denoiser → algebraic φ⁻¹` (the biased inversion the
/// report's §9.2 compares against).
fn arm_naive(noisy: &[f64], kind: VstKind, f: Denoiser) -> Vec<f64> {
    vst_inverse_naive(kind, &f(&vst_forward(kind, noisy)))
}

/// VST-corrected pipeline: `φ → denoiser → bias-corrected φ⁻¹` (exact unbiased for
/// Anscombe/GAT, Duan smearing for SignedLog) — [`vst_denoise`].
fn arm_corrected(noisy: &[f64], kind: VstKind, f: Denoiser) -> Vec<f64> {
    vst_denoise(noisy, kind, f)
}

/// Right-aligned `%.2f` cells, 10 columns wide each.
fn cells(vals: &[f64]) -> String {
    let mut s = String::new();
    for v in vals
    {
        s.push_str(&format!(" {v:>9.2}"));
    }
    s
}

fn main() {
    // ============ P1: Poisson low count (§9.1a) ============
    println!("## P1 Poisson low count (§9.1a) — λ = 10.5 + 9.5·sin(2π·3·i/n) + i/n ∈ [1, ~21],");
    println!("##    n = 4096, seed 7; SNR dB in original coordinates (§10)");
    let ma9 = |x: &[f64]| moving_average(x, 9);
    let wav = |x: &[f64]| wavelet_denoise(x, 0, ThresholdMode::Soft);
    let denoisers: [(&str, Denoiser); 4] = [
        ("MA9", &ma9),
        ("wavelet", &wav),
        ("stft", &stft_wiener_auto),
        ("collab1d", &collab1d_auto),
    ];
    let lambda = p1_intensity(N);
    let noisy_p1 = poisson_counts(&lambda, 7);
    println!("raw observation = {:.2} dB", snr(&lambda, &noisy_p1));
    let mut header = String::new();
    for (name, _) in &denoisers
    {
        header.push_str(&format!(" {name:>9}"));
    }
    println!("{:<24}{header}", "pipeline");
    let mut p1_identity = [0.0f64; 4];
    let mut p1_naive = [0.0f64; 4];
    let mut p1_corrected = [0.0f64; 4];
    for (j, (_, f)) in denoisers.iter().enumerate()
    {
        p1_identity[j] = snr(&lambda, &arm_identity(&noisy_p1, *f));
        p1_naive[j] = snr(&lambda, &arm_naive(&noisy_p1, VstKind::Anscombe, *f));
        p1_corrected[j] = snr(&lambda, &arm_corrected(&noisy_p1, VstKind::Anscombe, *f));
    }
    println!("{:<24}{}", "identity", cells(&p1_identity));
    println!("{:<24}{}", "vst-naive (Anscombe)", cells(&p1_naive));
    println!("{:<24}{}", "vst-corrected (exact)", cells(&p1_corrected));
    // §10 retransformation-bias metric on the strongest smoother (measured as the
    // best corrected-arm column, not assumed).
    let best = p1_corrected
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(j, _)| j)
        .unwrap_or(0);
    let (best_name, best_f) = denoisers[best];
    let bias_naive = mean(&arm_naive(&noisy_p1, VstKind::Anscombe, best_f)) - mean(&lambda);
    let bias_corr = mean(&arm_corrected(&noisy_p1, VstKind::Anscombe, best_f)) - mean(&lambda);
    println!(
        "retransformation bias mean(x̂) − mean(λ) on the strongest smoother ({best_name}): \
         naive {bias_naive:+.4}, corrected {bias_corr:+.4}"
    );
    let p1_gain = p1_corrected[best] - p1_identity[best];

    // ============ P2: multiplicative 10-40 % (§9.1b) ============
    println!("\n## P2 multiplicative 10-40 % (§9.1b) — levels [4, 40] (×10 strong regime),");
    println!("##    x = s·(1 + f·g), n = 4096, seed 9; identity vs SignedLog + smearing.");
    println!("##    Inner denoisers kept: stft_wiener_auto + wavelet(0, Soft) — the strongest");
    println!("##    baseline and the classic literature beneficiary; MA9/collab1d omitted for");
    println!("##    table readability.");
    println!(
        "{:<10} {:>9} {:>9} {:>9} {:>9} {:>9}",
        "fraction", "raw", "id-stft", "vst-stft", "id-wav", "vst-wav"
    );
    let mut p2_gain_stft: Vec<(f64, f64)> = Vec::new();
    let mut p2_gain_wav: Vec<f64> = Vec::new();
    for &fr in &[0.10, 0.20, 0.30, 0.40]
    {
        let (clean, noisy) = multiplicative_fixture(N, 4.0, 40.0, fr, 9);
        let id_stft = snr(&clean, &arm_identity(&noisy, &stft_wiener_auto));
        let vs_stft = snr(
            &clean,
            &arm_corrected(&noisy, VstKind::SignedLog, &stft_wiener_auto),
        );
        let id_wav = snr(&clean, &arm_identity(&noisy, &wav));
        let vs_wav = snr(&clean, &arm_corrected(&noisy, VstKind::SignedLog, &wav));
        println!(
            "{:<10.2} {:>9.2} {:>9.2} {:>9.2} {:>9.2} {:>9.2}",
            fr,
            snr(&clean, &noisy),
            id_stft,
            vs_stft,
            id_wav,
            vs_wav
        );
        p2_gain_stft.push((fr, vs_stft - id_stft));
        p2_gain_wav.push(vs_wav - id_wav);
    }

    // ============ P3: mixed Poisson-Gaussian / GAT (§9.1c) ============
    println!("\n## P3 mixed Poisson-Gaussian / GAT (§9.1c, Starck-Murtagh) — x = gain·p + σ·g");
    println!("##    on the P1 slow intensity, stft inner denoiser, n = 4096, seed 7");
    println!(
        "{:<12} {:>9} {:>9} {:>9} {:>9}",
        "(gain, σ)", "raw", "identity", "GAT-corr", "gain dB"
    );
    let mut p3_gains: Vec<f64> = Vec::new();
    for &(g, s) in &[(1.0, 0.5), (1.3, 1.5), (2.0, 1.0), (4.0, 2.0)]
    {
        let (clean, noisy) = gat_fixture(&lambda, g, s, 7);
        let kind = VstKind::Gat { gain: g, sigma: s };
        let s_id = snr(&clean, &arm_identity(&noisy, &stft_wiener_auto));
        let s_gat = snr(&clean, &arm_corrected(&noisy, kind, &stft_wiener_auto));
        println!(
            "({g:>3.1}, {s:>3.1})  {:>9.2} {s_id:>9.2} {s_gat:>9.2} {:>+9.2}",
            snr(&clean, &noisy),
            s_gat - s_id
        );
        p3_gains.push(s_gat - s_id);
    }
    println!("note: (1.0, 0.5), (2.0, 1.0) and (4.0, 2.0) share σ/gain = 0.5 — they are exact");
    println!("      rescalings of one calibration, the GAT normalizes the gain out, and every");
    println!("      pipeline stage is scale-equivariant, so those rows coincide by construction");
    println!("      (same seed). The read-noise-dominated mix is the (1.3, 1.5) row.");

    // ============ P4a: crossover sweep — noise fraction (§9.3) ============
    println!("\n## P4a crossover sweep, multiplicative fraction (§9.3) — levels [4, 40] fixed");
    println!("##     (×10), SignedLog-corrected vs identity, stft inner, n = 4096, seed 9");
    println!(
        "{:<10} {:>9} {:>9} {:>9} {:>9}",
        "fraction", "raw", "identity", "vst-corr", "gain dB"
    );
    let p4a_fractions = [0.02, 0.05, 0.10, 0.15, 0.20, 0.30];
    let mut p4a_threshold: Option<f64> = None;
    for &fr in &p4a_fractions
    {
        let (clean, noisy) = multiplicative_fixture(N, 4.0, 40.0, fr, 9);
        let s_id = snr(&clean, &arm_identity(&noisy, &stft_wiener_auto));
        let s_vst = snr(
            &clean,
            &arm_corrected(&noisy, VstKind::SignedLog, &stft_wiener_auto),
        );
        let gain = s_vst - s_id;
        println!(
            "{:<10.2} {:>9.2} {s_id:>9.2} {s_vst:>9.2} {gain:>+9.2}",
            fr,
            snr(&clean, &noisy)
        );
        if p4a_threshold.is_none() && gain >= MATERIALITY_DB
        {
            p4a_threshold = Some(fr);
        }
    }
    match p4a_threshold
    {
        Some(fr) if fr == p4a_fractions[0] => println!(
            "-> seuil ≈ {fr:.2} or below: gain already >= +0.5 dB at the smallest probed \
             fraction"
        ),
        Some(fr) =>
        {
            println!("-> seuil ≈ {fr:.2} (first fraction with gain >= +0.5 dB, §10 materiality)")
        },
        None => println!("-> seuil not reached: gain stays < +0.5 dB over the whole sweep"),
    }

    // ============ P4b: crossover sweep — level dynamic range (§9.3) ============
    println!("\n## P4b crossover sweep, level dynamic range (§9.3) — 30 % multiplicative noise");
    println!("##     fixed, levels [4, 4·R], SignedLog-corrected vs identity, stft, seed 9");
    println!(
        "{:<10} {:>9} {:>9} {:>9} {:>9}",
        "range", "raw", "identity", "vst-corr", "gain dB"
    );
    let p4b_ranges = [1.5, 2.0, 3.0, 5.0, 10.0];
    let mut p4b_threshold: Option<f64> = None;
    let mut p4b_gain_x2 = f64::NAN;
    for &r in &p4b_ranges
    {
        let (clean, noisy) = multiplicative_fixture(N, 4.0, 4.0 * r, 0.30, 9);
        let s_id = snr(&clean, &arm_identity(&noisy, &stft_wiener_auto));
        let s_vst = snr(
            &clean,
            &arm_corrected(&noisy, VstKind::SignedLog, &stft_wiener_auto),
        );
        let gain = s_vst - s_id;
        println!(
            "x{:<9.1} {:>9.2} {s_id:>9.2} {s_vst:>9.2} {gain:>+9.2}",
            r,
            snr(&clean, &noisy)
        );
        if p4b_threshold.is_none() && gain >= MATERIALITY_DB
        {
            p4b_threshold = Some(r);
        }
        if r == 2.0
        {
            p4b_gain_x2 = gain;
        }
    }
    match p4b_threshold
    {
        Some(r) if r == p4b_ranges[0] => println!(
            "-> seuil ≈ x{r:.1} or below: gain already >= +0.5 dB at the smallest probed range"
        ),
        Some(r) => println!(
            "-> seuil ≈ x{r:.1} (first dynamic range with gain >= +0.5 dB, §10 materiality)"
        ),
        None => println!("-> seuil not reached: gain stays < +0.5 dB over the whole sweep"),
    }

    // ============ P5: carrier-regime sweep (round-5 limitation) ============
    println!("\n## P5 carrier-regime sweep (vst.rs module docs, 'Known limitation: fast");
    println!("##    carriers') — pure Poisson, λ = 6.5 + 5.5·sin(2π·c·i/n) ∈ [1, 12],");
    println!("##    Anscombe-corrected vs identity, stft inner, n = 4096, seed 7");
    println!(
        "{:<10} {:>9} {:>9} {:>9} {:>9}",
        "cycles", "raw", "identity", "ansc-corr", "gain dB"
    );
    let mut p5_gains: Vec<(f64, f64)> = Vec::new();
    for &c in &[3.0, 8.0, 16.0, 40.0]
    {
        let lam = carrier_intensity(N, c);
        let noisy = poisson_counts(&lam, 7);
        let s_id = snr(&lam, &arm_identity(&noisy, &stft_wiener_auto));
        let s_vst = snr(
            &lam,
            &arm_corrected(&noisy, VstKind::Anscombe, &stft_wiener_auto),
        );
        let gain = s_vst - s_id;
        println!(
            "{:<10.0} {:>9.2} {s_id:>9.2} {s_vst:>9.2} {gain:>+9.2}",
            c,
            snr(&lam, &noisy)
        );
        p5_gains.push((c, gain));
    }

    // ============ Summary (computed from the measurements above) ============
    println!("\n## Summary — computed from the numbers above (§10: |gain| < 0.5 dB = null)");
    println!(
        "P1 : corrected Anscombe vs identity on {best_name}: {p1_gain:+.2} dB — {}",
        verdict(p1_gain)
    );
    println!(
        "P1 : |bias| naive {:.4} vs corrected {:.4} — {}",
        bias_naive.abs(),
        bias_corr.abs(),
        if bias_corr.abs() < bias_naive.abs()
        {
            "exact unbiased inverse reduces the retransformation bias"
        }
        else
        {
            "correction did NOT reduce the bias — investigate"
        }
    );
    let (fr_lo, g_lo) = p2_gain_stft[0];
    let (fr_hi, g_hi) = p2_gain_stft[p2_gain_stft.len() - 1];
    println!(
        "P2 : SignedLog gain (stft) {g_lo:+.2} dB at {:.0} % -> {g_hi:+.2} dB at {:.0} % — {}",
        fr_lo * 100.0,
        fr_hi * 100.0,
        verdict(g_hi),
    );
    let wav_min = p2_gain_wav.iter().copied().fold(f64::INFINITY, f64::min);
    let wav_max = p2_gain_wav
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    println!(
        "P2 : wavelet arm gain spans {wav_min:+.2}..{wav_max:+.2} dB — {}",
        if wav_max < 0.0
        {
            "stabilization never helps VisuShrink here (its raw-domain MAD threshold is \
             accidentally level-adaptive; cf. the vst.rs module docs)"
        }
        else
        {
            "stabilization helps the wavelet arm on at least one fraction"
        }
    );
    let p3_min = p3_gains.iter().copied().fold(f64::INFINITY, f64::min);
    let p3_max = p3_gains.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    println!(
        "P3 : GAT gain across the 4 calibrations: min {p3_min:+.2} dB, max {p3_max:+.2} dB — \
         worst case {}",
        verdict(p3_min)
    );
    match p4a_threshold
    {
        Some(fr) if fr == p4a_fractions[0] => println!(
            "P4a: gain already material at the smallest probed fraction ({fr:.2}) — at a ×10 \
             level range the crossover lies at or below 2 %"
        ),
        Some(fr) => println!("P4a: materiality crossover at fraction ≈ {fr:.2}"),
        None => println!("P4a: no materiality crossover measured (gain always < +0.5 dB)"),
    }
    match p4b_threshold
    {
        Some(r) => println!(
            "P4b: materiality crossover at dynamic range ≈ x{r:.1}; at x2.0 the gain is \
             {p4b_gain_x2:+.2} dB — {}",
            verdict(p4b_gain_x2)
        ),
        None => println!(
            "P4b: no materiality crossover measured; at x2.0 the gain is {p4b_gain_x2:+.2} dB \
             — {}",
            verdict(p4b_gain_x2)
        ),
    }
    let (c_slow, g_slow) = p5_gains[0];
    let (c_fast, g_fast) = p5_gains[p5_gains.len() - 1];
    println!(
        "P5 : Anscombe gain {g_slow:+.2} dB at {c_slow:.0} cycles -> {g_fast:+.2} dB at \
         {c_fast:.0} cycles — {}",
        if g_fast < 0.0 && g_slow > g_fast
        {
            "gain collapses and turns NEGATIVE on fast carriers (module-doc limitation confirmed)"
        }
        else if g_fast < g_slow
        {
            "gain shrinks as the carrier speeds up (collapse without sign flip)"
        }
        else
        {
            "no collapse measured — revisit the module-doc limitation note"
        }
    );
}
