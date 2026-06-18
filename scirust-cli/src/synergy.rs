//! Ecosystem-synergy demos (CCOS / SLHAv2): the compressed elastic KV-cache, the
//! statistically-guaranteed guard, and the hash-chained attestation log. All
//! deterministic in their `--seed`.

use scirust_core::nn::PcgEngine;
use scirust_core::nn::elastic_kv_cache::{ElasticKvCache, cosine_similarity};
use scirust_core::nn::guard::{GuardVerdict, StatisticalGuard};
use scirust_core::nn::paged_attention::contiguous_attention;
use scirust_runtime::attest::{AttestationLog, attest_and_record};
use scirust_runtime::vinfer::{P, VModel};

fn flag_u64(args: &[String], name: &str, default: u64) -> u64 {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn flag_f32(args: &[String], name: &str, default: f32) -> f32 {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn hex8(h: &[u8; 32]) -> String {
    let s: String = h.iter().take(8).map(|b| format!("{b:02x}")).collect();
    format!("{s}…")
}

/// `kvcache [--seed N] [--budget B]` — compress a KV sequence into the elastic cache
/// (SLHAv2-style INT4 tiles + per-group scales) and report the compression ratio and
/// the cosine fidelity of attention versus full precision; with `--budget` show the
/// bounded soft-paging (CCOS).
pub fn run_kvcache(args: &[String]) -> u8 {
    let seed = flag_u64(args, "--seed", 1);
    let budget = flag_u64(args, "--budget", 0) as usize;
    let (d, n, group) = (64usize, 128usize, 16usize);
    let mut rng = PcgEngine::new(seed);
    let mut cache = ElasticKvCache::new_grouped(d, budget, group);
    let (mut keys, mut values) = (Vec::new(), Vec::new());
    for _ in 0..n
    {
        let k: Vec<f32> = (0..d).map(|_| rng.float_signed()).collect();
        let v: Vec<f32> = (0..d).map(|_| rng.float_signed()).collect();
        cache.append(&k, &v);
        keys.extend(&k);
        values.extend(&v);
    }
    let raw = n * 2 * d * std::mem::size_of::<f32>();
    let compressed = cache.compressed_bytes();
    let resident = cache.len();
    let q: Vec<f32> = (0..d).map(|_| rng.float_signed()).collect();
    let approx = cache.attention(&q);
    let kr = &keys[(n - resident) * d..];
    let vr = &values[(n - resident) * d..];
    let exact = contiguous_attention(kr, vr, &q, d, resident);
    let cos = cosine_similarity(&approx, &exact);

    println!("Elastic compressed KV-cache — pure Rust, deterministic (seed {seed})");
    println!(
        "  sequence: {n} tokens · d = {d} · INT4 tiles (base+residual), per-group scale (g={group})"
    );
    println!(
        "  compression: {} B → {} B  ({:.2}× smaller than f32)",
        raw,
        compressed,
        raw as f32 / compressed as f32
    );
    if budget > 0
    {
        println!(
            "  elastic budget: {budget} tiles resident · {} evicted (soft-paging, CCOS-style)",
            cache.evicted()
        );
    }
    else
    {
        println!("  budget: unbounded ({resident} tiles resident)");
    }
    println!("  attention fidelity vs full precision: cosine = {cos:.5}");
    0
}

/// `guard [--seed N] [--alpha A]` — calibrate the statistically-guaranteed guard,
/// measure its distribution-free coverage on fresh data, and show its Accept /
/// Abstain / Reject verdicts (for a CCOS-style response guard).
pub fn run_guard(args: &[String]) -> u8 {
    let seed = flag_u64(args, "--seed", 1);
    let alpha = flag_f32(args, "--alpha", 0.1);
    if alpha <= 0.0 || alpha >= 1.0 || !alpha.is_finite()
    {
        eprintln!("usage: scirust guard [--seed N] [--alpha A]");
        eprintln!("error: --alpha must be in (0, 1)");
        return 2;
    }
    let softmax = |l: &[f32]| -> Vec<f32> {
        let m = l.iter().cloned().fold(f32::MIN, f32::max);
        let e: Vec<f32> = l.iter().map(|&x| (x - m).exp()).collect();
        let s: f32 = e.iter().sum();
        e.iter().map(|&x| x / s).collect()
    };
    let mut rng = PcgEngine::new(seed);
    let sample = |rng: &mut PcgEngine| -> (Vec<f32>, usize) {
        let y = (rng.next_u32() % 3) as usize;
        let mut lg = [0.0f32; 3];
        for v in lg.iter_mut()
        {
            *v = rng.float_signed();
        }
        lg[y] += 1.5;
        (softmax(&lg), y)
    };
    let (cal_p, cal_y): (Vec<Vec<f32>>, Vec<usize>) = (0..2000).map(|_| sample(&mut rng)).unzip();
    let guard = StatisticalGuard::calibrate(&cal_p, &cal_y, alpha);
    let n = 5000;
    let mut covered = 0usize;
    for _ in 0..n
    {
        let (p, y) = sample(&mut rng);
        if guard.covers(&p, y)
        {
            covered += 1;
        }
    }
    let verdict = |p: &[f32]| match guard.decide(p)
    {
        GuardVerdict::Accept(c) => format!("ACCEPT class {c}"),
        GuardVerdict::Abstain => "ABSTAIN (ambiguous — flag for review)".to_string(),
        GuardVerdict::Reject => "REJECT (out of distribution)".to_string(),
    };
    println!("Statistical guard — pure Rust, deterministic (seed {seed})");
    println!(
        "  conformal coverage target: {:.0}% (alpha {alpha}) · distribution-free",
        100.0 * (1.0 - alpha)
    );
    println!(
        "  empirical coverage on {n} fresh points: {:.1}%  (≥ {:.0}% guaranteed)",
        100.0 * covered as f32 / n as f32,
        100.0 * (1.0 - alpha)
    );
    println!("  verdicts:");
    println!(
        "    confident [0.95, 0.03, 0.02]  -> {}",
        verdict(&[0.95, 0.03, 0.02])
    );
    println!(
        "    ambiguous [0.50, 0.48, 0.02]  -> {}",
        verdict(&[0.50, 0.48, 0.02])
    );
    println!(
        "    flat      [0.34, 0.33, 0.33]  -> {}",
        verdict(&[0.34, 0.33, 0.33])
    );
    0
}

/// `attest [--seed N]` — record verifiable inferences into the hash-chained
/// attestation log (the CCOS event-log bridge), verify the chain, reject a forged
/// inference, and show that any change yields a different chain head.
pub fn run_attest(args: &[String]) -> u8 {
    let seed = flag_u64(args, "--seed", 1);
    let mut rng = PcgEngine::new(seed);
    let (out, inn, batch) = (4usize, 6usize, 3usize);
    let w: Vec<i64> = (0..out * inn)
        .map(|_| (rng.next_u32() as i64) % P)
        .collect();
    let model = VModel::new(w, out, inn);

    let mut log = AttestationLog::new();
    for _ in 0..4
    {
        let x: Vec<i64> = (0..inn * batch)
            .map(|_| (rng.next_u32() as i64) % P)
            .collect();
        let y = model.infer(&x, batch);
        attest_and_record(&mut log, &model, &x, batch, &y, 2);
    }

    // A forged output is rejected (and not chained).
    let x: Vec<i64> = (0..inn * batch)
        .map(|_| (rng.next_u32() as i64) % P)
        .collect();
    let mut forged = model.infer(&x, batch);
    forged[0] = (forged[0] + 1) % P;
    let rejected = attest_and_record(&mut log, &model, &x, batch, &forged, 2).is_none();

    // Any altered inference yields a different chain head (tamper-evidence).
    let mut alt = AttestationLog::new();
    for i in 0..4usize
    {
        let xx: Vec<i64> = (0..inn * batch).map(|j| ((i * 7 + j) as i64) % P).collect();
        let yy = model.infer(&xx, batch);
        let yy2 = if i == 1
        {
            let mut t = yy.clone();
            t[0] = (t[0] + 1) % P;
            t
        }
        else
        {
            yy
        };
        alt.record(model.commit(), &xx, &yy2);
    }

    println!("Hash-chained attestation log — pure Rust, deterministic (seed {seed})");
    println!(
        "  recorded {} verifiable inferences (Freivalds over GF(p), #80)",
        log.len()
    );
    println!(
        "  chain head: {}  ·  verify_chain: {}",
        hex8(&log.head()),
        if log.verify_chain() { "OK" } else { "FAIL" }
    );
    println!(
        "  forged inference rejected: {rejected}  (log still {} entries)",
        log.len()
    );
    println!(
        "  a single altered output ⇒ different head: {}  (tamper-evident)",
        hex8(&alt.head())
    );
    0
}
