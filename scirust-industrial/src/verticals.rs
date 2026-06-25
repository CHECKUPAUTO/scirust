//! CLI commands for the SciRust industrial verticals.
//!
//! Each command runs a small, fully deterministic scenario against the real
//! crate API and prints a report — no stubs, no randomness without a fixed seed.
//! They exist so an evaluator can exercise navigation, state estimation, water
//! diagnostics, OT security, and GMP batch comparison from the command line.
//!
//! Every scenario is split into a `*_run` function that returns the computed
//! result and a thin `pub fn` that prints it. The `*_run` functions carry the
//! numeric oracles exercised by the test module at the bottom of this file, so
//! the verticals are verified end-to-end rather than merely "doesn't panic".

use scirust_estimation::{Imm, ImmModel, KalmanFilter, Mat, UdFilter};
use scirust_func_safety::{AuditLog, GoldenBatch};
use scirust_ids::{Attestation, FirmwareBaseline, PlcBaseline, PlcRung, PlcVerdict};
use scirust_nav::tdoa_locate_2d;
use scirust_water::{joukowsky_surge, korteweg_wave_speed, locate_leak};
use std::collections::BTreeSet;

// --- deterministic helpers ---------------------------------------------------

/// splitmix64 → uniform in [0, 1).
fn unif(state: &mut u64) -> f64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    ((z >> 11) as f64 + 0.5) / ((1u64 << 53) as f64)
}

/// Box–Muller standard normal from the deterministic stream.
fn gauss(state: &mut u64) -> f64 {
    let u1 = unif(state).max(1e-12);
    let u2 = unif(state);
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

fn dist2(a: [f64; 2], b: [f64; 2]) -> f64 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt()
}

// --- navigation --------------------------------------------------------------

/// Result of the TDOA multilateration scenario.
#[derive(Debug, Clone)]
pub struct TdoaRun {
    pub source: [f64; 2],
    pub recovered: [f64; 2],
    pub error: f64,
    pub iterations: usize,
    pub converged: bool,
    pub residual_rms: f64,
}

/// NAV-TDOA: locate an emitter from time-difference-of-arrival across sensors.
pub fn nav_tdoa_run(speed: f64) -> Result<TdoaRun, String> {
    let sensors = [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]];
    let src = [3.0, 7.0];
    let t0 = 0.123; // unknown emission time — cancels in the differences
    let arrivals: Vec<f64> = sensors
        .iter()
        .map(|s| t0 + dist2(*s, src) / speed)
        .collect();

    let sol = tdoa_locate_2d(&sensors, &arrivals, speed, None, 50)
        .ok_or("TDOA solve failed (degenerate sensor geometry)")?;
    Ok(TdoaRun {
        source: src,
        recovered: sol.position,
        error: dist2(sol.position, src),
        iterations: sol.iterations,
        converged: sol.converged,
        residual_rms: sol.residual_rms,
    })
}

/// NAV-TDOA: locate an emitter from time-difference-of-arrival across sensors.
pub fn nav_tdoa(speed: f64) -> Result<(), String> {
    println!("=== Navigation — TDOA multilateration ===\n");
    let sensors = [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]];
    let r = nav_tdoa_run(speed)?;
    println!("wave speed       : {speed} m/s");
    println!("sensors          : {sensors:?}");
    println!("true source      : {:?}\n", r.source);
    println!(
        "recovered source : [{:.4}, {:.4}]",
        r.recovered[0], r.recovered[1]
    );
    println!("localization err : {:.2e} m", r.error);
    println!(
        "Gauss–Newton     : {} iters, converged={}, residual RMS {:.2e} m",
        r.iterations, r.converged, r.residual_rms
    );
    println!("\nSame geometry locates a partial-discharge / acoustic-emission source.");
    Ok(())
}

/// Result of the GNSS/INS fusion scenario.
#[derive(Debug, Clone)]
pub struct FusionRun {
    pub steps: usize,
    pub outage_start: usize,
    pub outage_end: usize,
    pub truth: [f64; 2],
    pub fused: [f64; 2],
    pub error: f64,
    pub uncertainty_at_outage_end: f64,
    pub uncertainty_final: f64,
}

/// NAV-FUSION: loosely-coupled GNSS/INS fusion with an optional GNSS outage.
pub fn nav_fusion_run(steps: usize, outage: usize) -> Result<FusionRun, String> {
    use scirust_nav::GnssInsFusion;
    if steps == 0
    {
        return Err("steps must be > 0".into());
    }
    let dt = 1.0;
    let vel = [1.0, 0.5];
    let mut truth = [0.0, 0.0];
    let mut f = GnssInsFusion::new([0.0, 0.0], vel, 0.3, 2.0, [5.0, 5.0, 1.0, 1.0]);
    let mut seed = 0x6E5Du64;

    let outage_start = steps / 2;
    let outage_end = (outage_start + outage).min(steps);
    let mut unc_mid = 0.0;

    for k in 0..steps
    {
        truth[0] += vel[0] * dt;
        truth[1] += vel[1] * dt;
        f.predict([0.0, 0.0], dt); // acceleration carried as process noise
        let in_outage = k >= outage_start && k < outage_end;
        if !in_outage
        {
            let fix = [
                truth[0] + 0.8 * gauss(&mut seed),
                truth[1] + 0.8 * gauss(&mut seed),
            ];
            f.update_gnss(fix);
        }
        if k == outage_end.saturating_sub(1)
        {
            unc_mid = f.position_uncertainty();
        }
    }

    Ok(FusionRun {
        steps,
        outage_start,
        outage_end,
        truth,
        fused: f.position(),
        error: dist2(f.position(), truth),
        uncertainty_at_outage_end: unc_mid,
        uncertainty_final: f.position_uncertainty(),
    })
}

/// NAV-FUSION: loosely-coupled GNSS/INS fusion with an optional GNSS outage.
pub fn nav_fusion(steps: usize, outage: usize) -> Result<(), String> {
    println!("=== Navigation — GNSS/INS fusion ===\n");
    let dt = 1.0;
    let r = nav_fusion_run(steps, outage)?;
    println!("steps            : {}  (dt = {dt}s)", r.steps);
    println!(
        "GNSS outage      : steps [{}, {}) — {} step(s) dead-reckoned",
        r.outage_start,
        r.outage_end,
        r.outage_end - r.outage_start
    );
    println!("true position    : [{:.2}, {:.2}]", r.truth[0], r.truth[1]);
    println!(
        "fused position   : [{:.2}, {:.2}]  (error {:.3} m)",
        r.fused[0], r.fused[1], r.error
    );
    println!(
        "uncertainty σ    : {:.3} m at outage end → {:.3} m after re-acquisition",
        r.uncertainty_at_outage_end, r.uncertainty_final
    );
    println!("\nThe covariance grows while GNSS is lost and is pulled back when fixes resume.");
    Ok(())
}

// --- state estimation --------------------------------------------------------

/// Result of the IMM tracking scenario.
#[derive(Debug, Clone)]
pub struct ImmRun {
    pub steps: usize,
    pub maneuver_at: usize,
    pub mu_quiet_before: f64,
    pub mu_quiet_after: f64,
    pub peak_maneuver: f64,
    pub peak_step: usize,
}

/// TRACK-IMM: an Interacting Multiple Models filter shifts probability onto the
/// maneuver model when the target maneuvers.
pub fn track_imm_run(steps: usize) -> Result<ImmRun, String> {
    if steps < 4
    {
        return Err("steps must be >= 4 to bracket a maneuver".into());
    }
    let dt = 1.0;
    let a = Mat::new(2, 2, vec![1.0, dt, 0.0, 1.0]); // constant-velocity
    let h = Mat::new(1, 2, vec![1.0, 0.0]); // measure position
    let r = Mat::new(1, 1, vec![0.25]);
    let q_quiet = Mat::new(2, 2, vec![1e-4, 0.0, 0.0, 1e-4]);
    let q_agile = Mat::new(2, 2, vec![1.0, 0.0, 0.0, 1.0]);
    let models = vec![
        ImmModel {
            a: a.clone(),
            q: q_quiet,
            h: h.clone(),
            r: r.clone(),
        },
        ImmModel {
            a: a.clone(),
            q: q_agile,
            h: h.clone(),
            r,
        },
    ];
    let p0 = Mat::new(2, 2, vec![1.0, 0.0, 0.0, 1.0]);
    let pi = vec![vec![0.97, 0.03], vec![0.03, 0.97]];
    let mut imm = Imm::new(models, vec![0.0, 1.0], p0, vec![0.5, 0.5], pi);

    let mut pos = 0.0;
    let mut vel = 1.0;
    let mut seed = 0xA1B2u64;
    let maneuver_at = steps / 2;
    let mut mu_quiet_before = 0.5;
    let mut peak_maneuver = 0.0_f64; // maneuver-model probability spikes at the reversal
    let mut peak_step = 0;
    for k in 0..steps
    {
        if k == maneuver_at
        {
            vel = -1.0; // abrupt velocity reversal
        }
        pos += vel * dt;
        if k == maneuver_at.saturating_sub(1)
        {
            mu_quiet_before = imm.mode_probabilities()[0];
        }
        imm.step(pos + 0.3 * gauss(&mut seed));
        let m = imm.mode_probabilities()[1];
        if m > peak_maneuver
        {
            peak_maneuver = m;
            peak_step = k;
        }
    }
    Ok(ImmRun {
        steps,
        maneuver_at,
        mu_quiet_before,
        mu_quiet_after: imm.mode_probabilities()[0],
        peak_maneuver,
        peak_step,
    })
}

/// TRACK-IMM: an Interacting Multiple Models filter shifts probability onto the
/// maneuver model when the target maneuvers.
pub fn track_imm(steps: usize) -> Result<(), String> {
    println!("=== Estimation — Interacting Multiple Models (IMM) ===\n");
    let r = track_imm_run(steps)?;
    println!(
        "steps            : {}, velocity reversal at step {}",
        r.steps, r.maneuver_at
    );
    println!(
        "P(quiet model)   : {:.3} before maneuver → {:.3} once steady again",
        r.mu_quiet_before, r.mu_quiet_after
    );
    println!(
        "P(maneuver model): peaks at {:.3} (step {}) when the target maneuvers",
        r.peak_maneuver, r.peak_step
    );
    if r.peak_maneuver > 0.5
    {
        println!(
            "\n✓ The filter swings onto the maneuver model exactly when the target maneuvers,"
        );
        println!(
            "  then relaxes back to the quiet model once motion is steady — the point of IMM."
        );
    }
    Ok(())
}

/// Result of the UD square-root filter scenario.
#[derive(Debug, Clone)]
pub struct UdRun {
    pub steps: usize,
    pub max_state_diff: f64,
    pub min_variance: f64,
}

/// TRACK-UD: the Bierman–Thornton square-root filter agrees with a textbook
/// Kalman filter while keeping the covariance positive-semidefinite by factor.
pub fn track_ud_run(steps: usize) -> Result<UdRun, String> {
    if steps == 0
    {
        return Err("steps must be > 0".into());
    }
    let dt = 1.0;
    let phi = Mat::new(2, 2, vec![1.0, dt, 0.0, 1.0]);
    let g = Mat::identity(2);
    let qd = vec![1e-3, 1e-3];
    let q = Mat::new(2, 2, vec![1e-3, 0.0, 0.0, 1e-3]);
    let h = Mat::new(1, 2, vec![1.0, 0.0]);
    let r = 0.25;
    let p0 = Mat::new(2, 2, vec![1.0, 0.0, 0.0, 1.0]);

    let mut ud = UdFilter::new(vec![0.0, 0.0], &p0);
    let mut kf = KalmanFilter::new(
        vec![0.0, 0.0],
        p0,
        phi.clone(),
        q,
        h,
        Mat::new(1, 1, vec![r]),
    );

    let mut pos = 0.0;
    let vel = 0.7;
    let mut seed = 0xC0FFu64;
    let mut max_state_diff: f64 = 0.0;
    let mut min_var: f64 = f64::INFINITY;
    for _ in 0..steps
    {
        pos += vel * dt;
        let z = pos + 0.2 * gauss(&mut seed);
        ud.predict(&phi, &g, &qd);
        kf.predict();
        ud.update(&[1.0, 0.0], r, z);
        kf.update(&[z]);
        for (a, b) in ud.x.iter().zip(kf.state())
        {
            max_state_diff = max_state_diff.max((a - b).abs());
        }
        for v in ud.variances()
        {
            min_var = min_var.min(v);
        }
    }
    Ok(UdRun {
        steps,
        max_state_diff,
        min_variance: min_var,
    })
}

/// TRACK-UD: the Bierman–Thornton square-root filter agrees with a textbook
/// Kalman filter while keeping the covariance positive-semidefinite by factor.
pub fn track_ud(steps: usize) -> Result<(), String> {
    println!("=== Estimation — UD square-root Kalman filter ===\n");
    let r = track_ud_run(steps)?;
    println!("steps                 : {}", r.steps);
    println!(
        "max |UD − Kalman| state: {:.2e}  (should be ~0)",
        r.max_state_diff
    );
    println!(
        "min UD variance        : {:.2e}  (stays ≥ 0 by construction)",
        r.min_variance
    );
    println!("\nThe UD form carries P = U·D·Uᵀ, so the covariance can never go indefinite.");
    Ok(())
}

// --- water networks ----------------------------------------------------------

/// Result of the acoustic leak-location scenario.
#[derive(Debug, Clone)]
pub struct LeakRun {
    pub true_leak: f64,
    pub located: f64,
    pub error: f64,
    pub lag_samples: i64,
    pub peak_corr: f64,
}

/// WATER-LEAK: locate a leak on a pipe segment by acoustic cross-correlation.
pub fn water_leak_run(
    pipe_length: f64,
    wave_speed: f64,
    sample_rate: f64,
    leak_at: f64,
) -> Result<LeakRun, String> {
    if !(0.0..=pipe_length).contains(&leak_at)
    {
        return Err(format!(
            "leak position {leak_at} must lie within [0, {pipe_length}] m"
        ));
    }
    let d_a = leak_at;
    let d_b = pipe_length - leak_at;
    let delay_a = (d_a / wave_speed * sample_rate).round() as usize;
    let delay_b = (d_b / wave_speed * sample_rate).round() as usize;

    // Broadband leak noise reaching each sensor with its own delay.
    let mut seed = 0x1EA4u64;
    let src: Vec<f64> = (0..3000).map(|_| 2.0 * unif(&mut seed) - 1.0).collect();
    let total = src.len() + delay_a.max(delay_b) + 8;
    let mut a = vec![0.0; total];
    let mut b = vec![0.0; total];
    for (i, &v) in src.iter().enumerate()
    {
        a[i + delay_a] = v;
        b[i + delay_b] = v;
    }

    let loc = locate_leak(&a, &b, pipe_length, wave_speed, sample_rate);
    Ok(LeakRun {
        true_leak: leak_at,
        located: loc.dist_from_a,
        error: (loc.dist_from_a - leak_at).abs(),
        lag_samples: loc.lag_samples as i64,
        peak_corr: loc.peak_corr,
    })
}

/// WATER-LEAK: locate a leak on a pipe segment by acoustic cross-correlation.
pub fn water_leak(
    pipe_length: f64,
    wave_speed: f64,
    sample_rate: f64,
    leak_at: f64,
) -> Result<(), String> {
    println!("=== Water — acoustic leak correlation ===\n");
    let r = water_leak_run(pipe_length, wave_speed, sample_rate, leak_at)?;
    println!(
        "pipe length      : {pipe_length} m   wave speed: {wave_speed} m/s   fs: {sample_rate} Hz"
    );
    println!("true leak        : {:.2} m from sensor A", r.true_leak);
    println!(
        "located leak     : {:.2} m from sensor A  (error {:.3} m)",
        r.located, r.error
    );
    println!(
        "correlation peak : lag {} samples, normalized corr {:.3}",
        r.lag_samples, r.peak_corr
    );
    Ok(())
}

/// Result of the water-hammer transient scenario.
#[derive(Debug, Clone)]
pub struct SurgeRun {
    pub delta_pressure: f64,
    pub head_m: f64,
    pub korteweg_speed: f64,
    pub free_fluid_speed: f64,
}

/// WATER-SURGE: water-hammer pressure surge (Joukowsky) and wave speed (Korteweg).
#[allow(clippy::too_many_arguments)]
pub fn water_surge_run(
    rho: f64,
    wave_speed: f64,
    delta_v: f64,
    bulk: f64,
    e_pipe: f64,
    diameter: f64,
    wall: f64,
) -> Result<SurgeRun, String> {
    let dp = joukowsky_surge(rho, wave_speed, delta_v);
    Ok(SurgeRun {
        delta_pressure: dp,
        head_m: dp / (rho * 9.806_65),
        korteweg_speed: korteweg_wave_speed(bulk, rho, e_pipe, diameter, wall),
        free_fluid_speed: (bulk / rho).sqrt(),
    })
}

/// WATER-SURGE: water-hammer pressure surge (Joukowsky) and wave speed (Korteweg).
#[allow(clippy::too_many_arguments)]
pub fn water_surge(
    rho: f64,
    wave_speed: f64,
    delta_v: f64,
    bulk: f64,
    e_pipe: f64,
    diameter: f64,
    wall: f64,
) -> Result<(), String> {
    println!("=== Water — water-hammer transient ===\n");
    let r = water_surge_run(rho, wave_speed, delta_v, bulk, e_pipe, diameter, wall)?;
    println!("fluid density    : {rho} kg/m³");
    println!("velocity change  : {delta_v} m/s");
    println!(
        "Joukowsky surge  : {:.3} MPa  (Δp = ρ·c·Δv at c = {wave_speed} m/s)",
        r.delta_pressure / 1e6
    );
    println!("  as head        : {:.1} m of water column", r.head_m);
    println!(
        "Korteweg speed   : {:.0} m/s  (free-fluid {:.0} m/s; the elastic wall lowers it)",
        r.korteweg_speed, r.free_fluid_speed
    );
    println!("  pipe: K={bulk:.2e} Pa, E={e_pipe:.2e} Pa, D={diameter} m, wall={wall} m");
    Ok(())
}

// --- OT cybersecurity --------------------------------------------------------

/// Result of the firmware-attestation scenario.
#[derive(Debug, Clone)]
pub struct FirmwareRun {
    pub clean: Attestation,
    pub tampered: Attestation,
    pub flipped_byte: usize,
}

/// OT-FIRMWARE: capture a firmware baseline, then attest a clean and a tampered
/// image against it.
pub fn ot_firmware_run(
    size: usize,
    block: usize,
    tamper_block: usize,
) -> Result<FirmwareRun, String> {
    if size == 0 || block == 0
    {
        return Err("size and block must be > 0".into());
    }
    let mut seed = 0xF1A7u64;
    let fw: Vec<u8> = (0..size).map(|_| (unif(&mut seed) * 256.0) as u8).collect();
    let base = FirmwareBaseline::capture(&fw, block);

    let clean = base.attest(&fw);

    let mut bad = fw;
    let idx = (tamper_block * block).min(size - 1);
    bad[idx] ^= 0xFF; // flip one byte
    let tampered = base.attest(&bad);
    Ok(FirmwareRun {
        clean,
        tampered,
        flipped_byte: idx,
    })
}

/// OT-FIRMWARE: capture a firmware baseline, then attest a clean and a tampered
/// image against it.
pub fn ot_firmware(size: usize, block: usize, tamper_block: usize) -> Result<(), String> {
    println!("=== OT security — firmware attestation ===\n");
    let r = ot_firmware_run(size, block, tamper_block)?;
    println!("firmware         : {size} bytes, block size {block}");
    match r.clean
    {
        Attestation::Intact => println!("clean image      : INTACT ✓"),
        other => println!("clean image      : unexpected {other:?}"),
    }
    match r.tampered
    {
        Attestation::Tampered { first_bad_block } => println!(
            "tampered image   : TAMPERED — first altered block #{first_bad_block} (byte {} flipped)",
            r.flipped_byte
        ),
        Attestation::SizeMismatch { expected, actual } =>
        {
            println!("tampered image   : SIZE MISMATCH expected {expected}, got {actual}")
        },
        Attestation::Intact => println!("tampered image   : INTACT (tamper not detected!)"),
    }
    println!("\nDigest, not signature — use with a signed boot chain for full trust.");
    Ok(())
}

fn plc_golden() -> Vec<PlcRung> {
    vec![
        PlcRung::new(vec![0x01, 0x02, 0x03], Some(10)),
        PlcRung::new(vec![0x11, 0x12], Some(11)),
        PlcRung::new(vec![0x21, 0x22, 0x23], None),
    ]
}

/// Result of the PLC ladder-integrity scenario.
#[derive(Debug, Clone)]
pub struct PlcRun {
    pub golden_verdict: PlcVerdict,
    pub tampered_verdict: PlcVerdict,
    pub unauthorized_writes: Vec<u16>,
}

/// OT-PLC: verify PLC ladder integrity and flag unauthorized writes to a
/// safety-critical output (the Stuxnet pattern).
pub fn ot_plc_run() -> PlcRun {
    let base = PlcBaseline::capture(&plc_golden());
    let golden_verdict = base.verify(&plc_golden());

    // Attacker appends a rung that drives a safety-critical output (#99).
    let mut tampered = plc_golden();
    tampered.push(PlcRung::new(vec![0xDE, 0xAD, 0xBE, 0xEF], Some(99)));
    let tampered_verdict = base.verify(&tampered);

    let critical: BTreeSet<u16> = [99u16].into_iter().collect();
    let unauthorized_writes = base.unauthorized_critical_writes(&tampered, &critical);
    PlcRun {
        golden_verdict,
        tampered_verdict,
        unauthorized_writes,
    }
}

/// OT-PLC: verify PLC ladder integrity and flag unauthorized writes to a
/// safety-critical output (the Stuxnet pattern).
pub fn ot_plc() -> Result<(), String> {
    println!("=== OT security — PLC ladder integrity ===\n");
    let base = PlcBaseline::capture(&plc_golden());
    println!(
        "golden program   : {} rungs, chain {:#018x}",
        plc_golden().len(),
        base.chain_digest()
    );
    let r = ot_plc_run();

    match r.golden_verdict
    {
        PlcVerdict::Intact => println!("re-verify golden : INTACT ✓"),
        other => println!("re-verify golden : unexpected {other:?}"),
    }
    match r.tampered_verdict
    {
        PlcVerdict::RungCountChanged { expected, actual } =>
        {
            println!("modified program : RUNG COUNT CHANGED {expected} → {actual}")
        },
        PlcVerdict::RungModified { rung } => println!("modified program : RUNG #{rung} MODIFIED"),
        PlcVerdict::Intact => println!("modified program : INTACT (change not detected!)"),
    }

    let critical: BTreeSet<u16> = [99u16].into_iter().collect();
    println!("critical outputs : {critical:?}");
    println!(
        "unauthorized writes to safety-critical outputs: {:?}",
        r.unauthorized_writes
    );
    if !r.unauthorized_writes.is_empty()
    {
        println!("\n⚠ Stuxnet-style write to a critical output the golden logic never drives.");
    }
    Ok(())
}

// --- pharma / GMP ------------------------------------------------------------

/// Result of the golden-batch comparator scenario.
#[derive(Debug, Clone)]
pub struct BatchRun {
    pub reference_steps: usize,
    pub candidate_steps: usize,
    pub conforming: bool,
    pub worst_ratio: f64,
    pub worst_variable: usize,
    pub worst_step: usize,
    pub dtw_distance: f64,
    pub audit_entries: usize,
    pub audit_chain_valid: bool,
}

/// GOLDEN-BATCH: DTW-align a candidate batch to the golden reference, check the
/// per-variable tolerance, and write the RELEASE/REJECT verdict to a hash-chained
/// 21 CFR Part 11 audit log.
pub fn golden_batch_run(lag: usize) -> Result<BatchRun, String> {
    // Golden temperature trajectory: ramp 20→37 °C then hold.
    let mut reference: Vec<Vec<f64>> = Vec::new();
    for k in 0..40
    {
        let temp = if k < 30
        {
            20.0 + (37.0 - 20.0) * (k as f64 / 30.0)
        }
        else
        {
            37.0
        };
        reference.push(vec![temp]);
    }
    let tolerance = vec![0.5]; // ±0.5 °C
    let gb = GoldenBatch::new(reference.clone(), tolerance);

    // Candidate: identical profile but delayed to start (a phase lag DTW absorbs).
    let mut candidate: Vec<Vec<f64>> = Vec::new();
    for _ in 0..lag
    {
        candidate.push(vec![20.0]);
    }
    candidate.extend(reference.iter().cloned());

    let report = gb.compare(&candidate);
    let mut log = AuditLog::new(64);
    gb.record_audit(&mut log, "BATCH-2026-001", &report, 1_750_000_000.0);

    Ok(BatchRun {
        reference_steps: reference.len(),
        candidate_steps: candidate.len(),
        conforming: report.conforming,
        worst_ratio: report.worst_ratio,
        worst_variable: report.worst_variable,
        worst_step: report.worst_step,
        dtw_distance: report.dtw_distance,
        audit_entries: log.len(),
        audit_chain_valid: log.verify_chain(),
    })
}

/// GOLDEN-BATCH: DTW-align a candidate batch to the golden reference, check the
/// per-variable tolerance, and write the RELEASE/REJECT verdict to a hash-chained
/// 21 CFR Part 11 audit log.
pub fn golden_batch(lag: usize) -> Result<(), String> {
    println!("=== GMP — golden-batch comparator (21 CFR Part 11) ===\n");
    let r = golden_batch_run(lag)?;
    println!(
        "golden steps     : {}, candidate steps: {} (lag {lag})",
        r.reference_steps, r.candidate_steps
    );
    println!("conforming       : {}", r.conforming);
    println!(
        "worst deviation  : {:.3}× tolerance on variable {} at step {}",
        r.worst_ratio, r.worst_variable, r.worst_step
    );
    println!("DTW distance     : {:.4}", r.dtw_distance);
    println!(
        "audit log        : {} entry(ies), chain intact: {}",
        r.audit_entries, r.audit_chain_valid
    );
    println!(
        "decision         : {}",
        if r.conforming { "RELEASE" } else { "REJECT" }
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nav_tdoa_recovers_the_source() {
        let r = nav_tdoa_run(1480.0).unwrap();
        // Noise-free arrivals → Gauss–Newton should recover the source to mm.
        assert!(r.converged, "TDOA did not converge");
        assert!(r.error < 1e-3, "localization error too large: {}", r.error);
        assert!(
            r.residual_rms < 1e-6,
            "residual RMS too large: {}",
            r.residual_rms
        );
    }

    #[test]
    fn nav_fusion_uncertainty_recovers_after_outage() {
        let r = nav_fusion_run(40, 8).unwrap();
        // Covariance is pulled back below its outage-end value once fixes resume.
        assert!(
            r.uncertainty_final < r.uncertainty_at_outage_end,
            "uncertainty did not recover: end={} final={}",
            r.uncertainty_at_outage_end,
            r.uncertainty_final
        );
        // The fused track stays near truth despite the dead-reckoned gap.
        assert!(r.error < 5.0, "fusion error too large: {}", r.error);
    }

    #[test]
    fn track_imm_swings_onto_the_maneuver_model() {
        let r = track_imm_run(40).unwrap();
        // Quiet model dominates before the reversal; maneuver model spikes at it.
        assert!(
            r.mu_quiet_before > 0.5,
            "quiet model not dominant pre-maneuver"
        );
        assert!(
            r.peak_maneuver > 0.5,
            "maneuver model never took over: peak={}",
            r.peak_maneuver
        );
        // The spike happens at/after the reversal, not before.
        assert!(r.peak_step >= r.maneuver_at.saturating_sub(1));
    }

    #[test]
    fn track_ud_matches_kalman_and_stays_psd() {
        let r = track_ud_run(60).unwrap();
        // UD square-root form is algebraically the Kalman filter → states agree.
        assert!(
            r.max_state_diff < 1e-6,
            "UD and Kalman states diverged: {}",
            r.max_state_diff
        );
        // Variances stay non-negative by construction (P = U·D·Uᵀ).
        assert!(
            r.min_variance >= 0.0,
            "negative variance: {}",
            r.min_variance
        );
    }

    #[test]
    fn water_leak_is_located_near_truth() {
        let r = water_leak_run(100.0, 1200.0, 8000.0, 30.0).unwrap();
        // Cross-correlation lag resolves the leak to within sample quantization.
        assert!(r.error < 2.0, "leak location error too large: {}", r.error);
        assert!(
            r.peak_corr > 0.5,
            "correlation peak too weak: {}",
            r.peak_corr
        );
    }

    #[test]
    fn water_surge_obeys_joukowsky_and_korteweg() {
        let (rho, c, dv) = (1000.0, 1200.0, 2.0);
        let r = water_surge_run(rho, c, dv, 2.2e9, 2.0e11, 0.5, 0.01).unwrap();
        // Joukowsky is exactly Δp = ρ·c·Δv.
        assert!((r.delta_pressure - rho * c * dv).abs() < 1e-6);
        // The elastic wall lowers the wave speed below the free-fluid value.
        assert!(
            r.korteweg_speed < r.free_fluid_speed,
            "Korteweg speed {} not below free-fluid {}",
            r.korteweg_speed,
            r.free_fluid_speed
        );
    }

    #[test]
    fn ot_firmware_passes_clean_and_catches_tamper() {
        let r = ot_firmware_run(4096, 256, 5).unwrap();
        assert!(
            matches!(r.clean, Attestation::Intact),
            "clean image flagged"
        );
        match r.tampered
        {
            Attestation::Tampered { first_bad_block } =>
            {
                // The flipped byte sits in block tamper_block (5).
                assert_eq!(first_bad_block, 5, "wrong block reported tampered");
            },
            other => panic!("tamper not detected: {other:?}"),
        }
    }

    #[test]
    fn ot_plc_detects_unauthorized_critical_write() {
        let r = ot_plc_run();
        assert!(
            matches!(r.golden_verdict, PlcVerdict::Intact),
            "golden not intact"
        );
        // Appending a rung changes the rung count, which the baseline catches.
        assert!(
            matches!(r.tampered_verdict, PlcVerdict::RungCountChanged { .. }),
            "tamper not detected: {:?}",
            r.tampered_verdict
        );
        // The appended rung drives critical output #99 — flagged as unauthorized.
        assert_eq!(r.unauthorized_writes, vec![99]);
    }

    #[test]
    fn golden_batch_releases_a_lagged_but_conforming_run() {
        let r = golden_batch_run(5).unwrap();
        // DTW absorbs the pure phase lag → the batch conforms and is RELEASEd.
        assert!(r.conforming, "lagged-but-identical batch wrongly rejected");
        assert!(
            r.worst_ratio <= 1.0,
            "worst ratio exceeds tolerance: {}",
            r.worst_ratio
        );
        // The 21 CFR Part 11 audit log holds exactly one hash-chained entry.
        assert_eq!(r.audit_entries, 1);
        assert!(r.audit_chain_valid, "audit chain broken");
    }

    #[test]
    fn invalid_inputs_are_rejected() {
        assert!(water_leak_run(100.0, 1200.0, 8000.0, 150.0).is_err());
        assert!(ot_firmware_run(0, 256, 1).is_err());
        assert!(track_imm_run(2).is_err());
        assert!(nav_fusion_run(0, 0).is_err());
    }
}
