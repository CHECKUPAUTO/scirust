//! **TSHF falsification experiments** — the reproducible measurement harness behind
//! the research report `TSHF_RESEARCH_2026-07-16.md` (Transformed-Scalar Hypercomplex
//! Filters). Run with:
//!
//! ```text
//! cargo run -p scirust-signal --example tshf_experiments
//! ```
//!
//! The program prints six experiment blocks, fully deterministic (fixed-seed LCG):
//!
//! * **E1** — noise statistics after each candidate scalar transform φ: the std of
//!   `φ(s+n) − φ(s)` across signal levels, under additive Gaussian, multiplicative
//!   and Poisson-like (signal-dependent) noise. Flat rows = variance-stabilized.
//! * **E2** — invertibility and conditioning: non-injectivity of `1/Γ(x+1)`,
//!   non-monotonicity of `ln Γ(x+1)` below x ≈ 0.4616, and the worst-case noise
//!   amplification `max |dφ⁻¹/dy|` each inverse applies at reconstruction.
//! * **E3** — end-to-end `φ → filter → φ⁻¹` SNR versus filtering directly, for
//!   moving-average / median / wavelet filters under the three noise models.
//!   (The median row is constant by mathematics: medians commute with monotone φ.)
//! * **E4** — retransformation (Jensen) bias of the naive algebraic inverse.
//! * **E5** — hypercomplex embedding: componentwise-linear-filter identity, vector
//!   median vs per-channel median on correlated impulses, and the ordering
//!   (non-)commutation of coordinate-coupled transforms.
//! * **E6** — step-edge preservation through the φ round trip.
//!
//! The harness never assumes the hypothesis is true: every block is designed so a
//! negative result is measurable (and several are — see the report's conclusions).

use scirust_signal::denoise::*;
use scirust_special::{gamma, ln_gamma};
use std::f64::consts::PI;

struct Lcg(u64);
impl Lcg {
    fn new(s: u64) -> Self {
        Self(s)
    }
    fn u(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
    fn g(&mut self) -> f64 {
        let a = self.u().max(1e-12);
        let b = self.u();
        (-2.0 * a.ln()).sqrt() * (2.0 * PI * b).cos()
    }
}
fn snr(c: &[f64], e: &[f64]) -> f64 {
    let s: f64 = c.iter().map(|x| x * x).sum();
    let d: f64 = c.iter().zip(e).map(|(a, b)| (a - b) * (a - b)).sum();
    10.0 * (s / d.max(1e-30)).log10()
}
fn std_of(x: &[f64]) -> f64 {
    let n = x.len() as f64;
    let m = x.iter().sum::<f64>() / n;
    (x.iter().map(|&a| (a - m) * (a - m)).sum::<f64>() / n).sqrt()
}

#[derive(Clone, Copy)]
struct Phi {
    name: &'static str,
    f: fn(f64) -> f64,
    inv: fn(f64) -> f64,
    mono: bool,
}
fn id(x: f64) -> f64 {
    x
}
fn slog(x: f64) -> f64 {
    x.signum() * (1.0 + x.abs()).ln()
}
fn slog_i(y: f64) -> f64 {
    y.signum() * (y.abs().exp() - 1.0)
}
fn spow(x: f64) -> f64 {
    x.signum() * x.abs().sqrt()
}
fn spow_i(y: f64) -> f64 {
    y.signum() * y * y
}
fn th(x: f64) -> f64 {
    x.tanh()
}
fn th_i(y: f64) -> f64 {
    y.clamp(-0.999_999_999, 0.999_999_999).atanh()
}
fn sg(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}
fn sg_i(y: f64) -> f64 {
    let y = y.clamp(1e-12, 1.0 - 1e-12);
    (y / (1.0 - y)).ln()
}
fn at(x: f64) -> f64 {
    x.atan()
}
fn at_i(y: f64) -> f64 {
    y.clamp(-PI / 2.0 + 1e-9, PI / 2.0 - 1e-9).tan()
}
fn ss(x: f64) -> f64 {
    x / (1.0 + x.abs())
}
fn ss_i(y: f64) -> f64 {
    let y = y.clamp(-0.999_999_999, 0.999_999_999);
    y / (1.0 - y.abs())
}
fn lgam(x: f64) -> f64 {
    ln_gamma(x + 1.0)
} // valid/monotone only for x >= 0.4616
fn lgam_i(y: f64) -> f64 {
    // Newton on lnGamma(x+1)=y, x>=0.47
    let mut x = y.max(0.0) + 1.0;
    for _ in 0..60
    {
        let fx = ln_gamma(x + 1.0) - y;
        let d = (ln_gamma(x + 1.0 + 1e-6) - ln_gamma(x + 1.0 - 1e-6)) / 2e-6;
        if d.abs() < 1e-14
        {
            break;
        }
        x -= fx / d;
        if x < 0.47
        {
            x = 0.47;
        }
    }
    x
}
fn rgam(x: f64) -> f64 {
    1.0 / gamma(x + 1.0)
}
fn ansc(x: f64) -> f64 {
    2.0 * (x.max(0.0) + 0.375).sqrt()
}
fn ansc_i(y: f64) -> f64 {
    (y / 2.0) * (y / 2.0) - 0.375
}

fn main() {
    let phis = [
        Phi {
            name: "identity",
            f: id,
            inv: id,
            mono: true,
        },
        Phi {
            name: "signed-log",
            f: slog,
            inv: slog_i,
            mono: true,
        },
        Phi {
            name: "signed-sqrt",
            f: spow,
            inv: spow_i,
            mono: true,
        },
        Phi {
            name: "tanh",
            f: th,
            inv: th_i,
            mono: true,
        },
        Phi {
            name: "sigmoid",
            f: sg,
            inv: sg_i,
            mono: true,
        },
        Phi {
            name: "atan",
            f: at,
            inv: at_i,
            mono: true,
        },
        Phi {
            name: "softsign",
            f: ss,
            inv: ss_i,
            mono: true,
        },
        Phi {
            name: "logGamma(x+1)",
            f: lgam,
            inv: lgam_i,
            mono: false,
        }, // monotone only x>0.4616
        Phi {
            name: "anscombe",
            f: ansc,
            inv: ansc_i,
            mono: true,
        },
    ];

    // ============ E2: invertibility & conditioning ============
    println!("## E2 invertibility / conditioning");
    // recip-gamma collision:
    println!(
        "recipGamma: phi(0)={:.6} phi(1)={:.6} phi(0.4616)={:.6}  -> NON-INJECTIVE on x>=0",
        rgam(0.0),
        rgam(1.0),
        rgam(0.4616)
    );
    // logGamma monotonicity failure below 0.4616:
    println!(
        "logGamma:  phi(0)={:.6} phi(0.2)={:.6} phi(0.4616)={:.6} phi(1)={:.6} -> non-monotone on [0,1]",
        lgam(0.0),
        lgam(0.2),
        lgam(0.4616),
        lgam(1.0)
    );
    // max |(phi^-1)'| over y-range reached by x in [-3,3] (amplification of noise at inversion)
    for p in &phis
    {
        let mut maxamp: f64 = 0.0;
        let mut roundtrip: f64 = 0.0;
        for i in 0..6001
        {
            let x = -3.0 + i as f64 * 0.001;
            if (p.name == "logGamma(x+1)" || p.name == "anscombe") && x < 0.5
            {
                continue;
            }
            let y = (p.f)(x);
            let dinv = ((p.inv)(y + 1e-6) - (p.inv)(y - 1e-6)) / 2e-6;
            if dinv.is_finite()
            {
                maxamp = maxamp.max(dinv.abs());
            }
            roundtrip = roundtrip.max(((p.inv)(y) - x).abs());
        }
        println!(
            "{:<14} max|dphi^-1/dy| over x in [-3,3]: {:>12.2}   max roundtrip err: {:.2e}",
            p.name, maxamp, roundtrip
        );
    }

    // ============ E1: noise statistics after transform ============
    println!("\n## E1 noise std after transform, vs signal level (additive Gaussian sigma=0.3)");
    println!(
        "{:<14} {:>8} {:>8} {:>8} {:>8}   (level 0.6 / 1.2 / 1.8 / 2.4) — flat = level-independent",
        "phi", "s=0.6", "s=1.2", "s=1.8", "s=2.4"
    );
    for p in &phis
    {
        let mut row = String::new();
        for lvl in [0.6f64, 1.2, 1.8, 2.4]
        {
            let mut r = Lcg::new(42);
            let devs: Vec<f64> = (0..20000)
                .map(|_| (p.f)(lvl + 0.3 * r.g()) - (p.f)(lvl))
                .collect();
            row += &format!(" {:>8.4}", std_of(&devs));
        }
        println!("{:<14}{}", p.name, row);
    }
    println!("-- multiplicative noise x = s*(1+0.3 g):");
    for p in &[phis[0], phis[1], phis[7], phis[8]]
    {
        let mut row = String::new();
        for lvl in [0.6f64, 1.2, 1.8, 2.4]
        {
            let mut r = Lcg::new(42);
            let devs: Vec<f64> = (0..20000)
                .map(|_| (p.f)(lvl * (1.0 + 0.3 * r.g())) - (p.f)(lvl))
                .collect();
            row += &format!(" {:>8.4}", std_of(&devs));
        }
        println!("{:<14}{}", p.name, row);
    }
    println!("-- Poisson-like noise x = s + 0.3*sqrt(s)*g:");
    for p in &[phis[0], phis[2], phis[8]]
    {
        let mut row = String::new();
        for lvl in [0.6f64, 1.2, 1.8, 2.4]
        {
            let mut r = Lcg::new(42);
            let devs: Vec<f64> = (0..20000)
                .map(|_| (p.f)(lvl + 0.3 * lvl.sqrt() * r.g()) - (p.f)(lvl))
                .collect();
            row += &format!(" {:>8.4}", std_of(&devs));
        }
        println!("{:<14}{}", p.name, row);
    }

    // ============ E3: end-to-end phi -> filter -> phi^-1 vs direct ============
    let n = 2048;
    let clean: Vec<f64> = (0..n)
        .map(|i| 2.0 + (2.0 * PI * 4.0 * i as f64 / n as f64).sin())
        .collect(); // positive signal (offset 2)
    let pipe = |p: &Phi, filt: &dyn Fn(&[f64]) -> Vec<f64>, obs: &[f64]| -> Vec<f64> {
        let t: Vec<f64> = obs.iter().map(|&x| (p.f)(x)).collect();
        let f = filt(&t);
        f.iter().map(|&y| (p.inv)(y)).collect()
    };
    let ma = |x: &[f64]| moving_average(x, 5);
    let md = |x: &[f64]| median_filter(x, 2);
    let wv = |x: &[f64]| wavelet_denoise(x, 0, ThresholdMode::Soft);
    type NoiseGen = Box<dyn Fn(&mut Lcg, f64) -> f64>;
    let noises: Vec<(&str, NoiseGen)> = vec![
        (
            "additive g=0.3",
            Box::new(|r: &mut Lcg, s: f64| s + 0.3 * r.g()),
        ),
        (
            "multiplicative 0.25",
            Box::new(|r: &mut Lcg, s: f64| s * (1.0 + 0.25 * r.g())),
        ),
        (
            "poisson-like 0.35",
            Box::new(|r: &mut Lcg, s: f64| s + 0.35 * s.sqrt() * r.g()),
        ),
    ];
    for (nm, gen) in &noises
    {
        println!("\n## E3 noise = {nm}  (SNR dB; higher better; raw first)");
        let mut r = Lcg::new(7);
        let obs: Vec<f64> = clean.iter().map(|&s| gen(&mut r, s)).collect();
        println!("raw = {:.2} dB", snr(&clean, &obs));
        println!("{:<14} {:>8} {:>8} {:>8}", "phi", "MA5", "med2", "wavelet");
        for p in &phis
        {
            // skip logGamma if signal range dips below 0.5 (ours doesn't: clean>=1, noise could... obs min?)
            let ok = obs.iter().all(|&x| x.is_finite() && (p.mono || x > 0.5));
            if !ok && !p.mono
            {
                println!("{:<14}   (domain violated: obs below 0.5)", p.name);
                continue;
            }
            let a = snr(&clean, &pipe(p, &ma, &obs));
            let b = snr(&clean, &pipe(p, &md, &obs));
            let c = snr(&clean, &pipe(p, &wv, &obs));
            println!("{:<14} {:>8.2} {:>8.2} {:>8.2}", p.name, a, b, c);
        }
    }

    // ============ E4: Jensen / retransformation bias ============
    println!(
        "\n## E4 retransformation bias: mean(phi^-1(MA9(phi(x)))) - mean(clean), flat signal s=2, g=0.4"
    );
    for p in &phis
    {
        let mut r = Lcg::new(13);
        let obs: Vec<f64> = (0..20000).map(|_| 2.0 + 0.4 * r.g()).collect();
        let out = pipe(p, &|x: &[f64]| moving_average(x, 9), &obs);
        let bias = out.iter().sum::<f64>() / out.len() as f64 - 2.0;
        println!("{:<14} bias = {:+.5}", p.name, bias);
    }

    // ============ E5: hypercomplex embedding ============
    println!("\n## E5 hypercomplex embedding");
    // (a) componentwise linear filter == per-channel filter (exact)
    let mut r = Lcg::new(21);
    let base: Vec<f64> = (0..512)
        .map(|i| (2.0 * PI * 3.0 * i as f64 / 512.0).sin())
        .collect();
    let chans: Vec<Vec<f64>> = (0..4)
        .map(|c| {
            base.iter()
                .map(|&b| b * (1.0 + 0.2 * c as f64) + 0.3 * r.g())
                .collect()
        })
        .collect();
    // quaternion-as-4-channels, componentwise MA == per-channel MA trivially (same code path) -> state as identity
    println!(
        "(a) any R-linear filter applied componentwise to a quaternion embedding IS per-channel filtering (algebraic identity; no experiment needed)."
    );
    // (b) vector median vs per-channel median under CORRELATED impulses
    let mut r = Lcg::new(23);
    let mut noisy: Vec<Vec<f64>> = chans.clone();
    for i in (7..512).step_by(41)
    {
        for ch in noisy.iter_mut()
        {
            ch[i] += if r.u() > 0.5 { 6.0 } else { -6.0 };
        }
    } // impulses hit all channels
    // per-channel median
    let pc: Vec<Vec<f64>> = noisy.iter().map(|ch| median_filter(ch, 2)).collect();
    // vector median: in window, pick the SAMPLE VECTOR minimizing sum of distances to others
    let mut vm = vec![vec![0.0; 512]; 4];
    for i in 0..512usize
    {
        let lo = i.saturating_sub(2);
        let hi = (i + 2).min(511);
        let mut best = lo;
        let mut bestd = f64::INFINITY;
        for a in lo..=hi
        {
            let mut d = 0.0;
            for b in lo..=hi
            {
                let mut e = 0.0;
                for ch in &noisy
                {
                    let t = ch[a] - ch[b];
                    e += t * t;
                }
                d += e.sqrt();
            }
            if d < bestd
            {
                bestd = d;
                best = a;
            }
        }
        for (vmc, nc) in vm.iter_mut().zip(&noisy)
        {
            vmc[i] = nc[best];
        }
    }
    let clean_c: Vec<Vec<f64>> = (0..4)
        .map(|c| base.iter().map(|&b| b * (1.0 + 0.2 * c as f64)).collect())
        .collect();
    let snr4 = |est: &Vec<Vec<f64>>| -> f64 {
        let mut s = 0.0;
        for c in 0..4
        {
            s += snr(&clean_c[c], &est[c]);
        }
        s / 4.0
    };
    println!(
        "(b) correlated impulses: per-channel median {:.2} dB, vector median (joint) {:.2} dB",
        snr4(&pc),
        snr4(&vm)
    );
    // channel-correlation preservation: corr between ch0/ch1 clean vs filtered
    let corr = |a: &[f64], b: &[f64]| -> f64 {
        let n = a.len() as f64;
        let ma = a.iter().sum::<f64>() / n;
        let mb = b.iter().sum::<f64>() / n;
        let mut num = 0.0;
        let mut da = 0.0;
        let mut db = 0.0;
        for i in 0..a.len()
        {
            let x = a[i] - ma;
            let y = b[i] - mb;
            num += x * y;
            da += x * x;
            db += y * y;
        }
        num / (da * db).sqrt()
    };
    println!(
        "    corr(ch0,ch1): clean {:.4}, per-channel median {:.4}, vector median {:.4}",
        corr(&clean_c[0], &clean_c[1]),
        corr(&pc[0], &pc[1]),
        corr(&vm[0], &vm[1])
    );
    // (c) ordering: phi componentwise commutes with embedding (identity); norm-coupled transform does not:
    let x4 = [0.5, 0.4, 0.3, 0.2];
    let normx = (x4.iter().map(|v| v * v).sum::<f64>()).sqrt();
    let phi_then_embed: Vec<f64> = x4.iter().map(|&v| th(v)).collect();
    let embed_then_normphi: Vec<f64> = x4.iter().map(|&v| v * th(normx) / normx).collect();
    println!(
        "(c) componentwise tanh then embed = embed then componentwise tanh (identical by construction)."
    );
    println!(
        "    norm-coupled tanh (v*tanh|v|/|v|): {:?} vs componentwise: {:?} -> ordering matters iff the transform couples coordinates",
        embed_then_normphi
            .iter()
            .map(|v| format!("{v:.3}"))
            .collect::<Vec<_>>(),
        phi_then_embed
            .iter()
            .map(|v| format!("{v:.3}"))
            .collect::<Vec<_>>()
    );

    // ============ E6: edge preservation through the round trip ============
    println!("\n## E6 step edge through phi->MA5->phi^-1 (additive g=0.25, step 0->3): SNR dB");
    let step: Vec<f64> = (0..1024).map(|i| if i < 512 { 0.5 } else { 3.5 }).collect();
    let mut r = Lcg::new(31);
    let sobs: Vec<f64> = step.iter().map(|&s| s + 0.25 * r.g()).collect();
    for p in &phis
    {
        if !p.mono && sobs.iter().any(|&x| x < 0.5)
        {
            println!("{:<14} (domain)", p.name);
            continue;
        }
        let out = pipe(p, &ma, &sobs);
        println!(
            "{:<14} {:>7.2}   (direct MA5 = {:.2})",
            p.name,
            snr(&step, &out),
            snr(&step, &ma(&sobs))
        );
    }
}
