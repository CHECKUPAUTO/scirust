//! CLI commands for the SciRust industrial verticals.
//!
//! Each command runs a small, fully deterministic scenario against the real
//! crate API and prints a report — no stubs, no randomness without a fixed seed.
//! They exist so an evaluator can exercise navigation, state estimation, water
//! diagnostics, OT security, and GMP batch comparison from the command line.

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

/// NAV-TDOA: locate an emitter from time-difference-of-arrival across sensors.
pub fn nav_tdoa(speed: f64) -> Result<(), String> {
    println!("=== Navigation — TDOA multilateration ===\n");
    let sensors = [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]];
    let src = [3.0, 7.0];
    let t0 = 0.123; // unknown emission time — cancels in the differences
    let arrivals: Vec<f64> = sensors
        .iter()
        .map(|s| t0 + dist2(*s, src) / speed)
        .collect();

    println!("wave speed       : {speed} m/s");
    println!("sensors          : {sensors:?}");
    println!("true source      : {src:?}\n");

    let sol = tdoa_locate_2d(&sensors, &arrivals, speed, None, 50)
        .ok_or("TDOA solve failed (degenerate sensor geometry)")?;
    let err = dist2(sol.position, src);
    println!(
        "recovered source : [{:.4}, {:.4}]",
        sol.position[0], sol.position[1]
    );
    println!("localization err : {err:.2e} m");
    println!(
        "Gauss–Newton     : {} iters, converged={}, residual RMS {:.2e} m",
        sol.iterations, sol.converged, sol.residual_rms
    );
    println!("\nSame geometry locates a partial-discharge / acoustic-emission source.");
    Ok(())
}

/// NAV-FUSION: loosely-coupled GNSS/INS fusion with an optional GNSS outage.
pub fn nav_fusion(steps: usize, outage: usize) -> Result<(), String> {
    use scirust_nav::GnssInsFusion;
    println!("=== Navigation — GNSS/INS fusion ===\n");
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

    let err = dist2(f.position(), truth);
    println!("steps            : {steps}  (dt = {dt}s)");
    println!(
        "GNSS outage      : steps [{outage_start}, {outage_end}) — {} step(s) dead-reckoned",
        outage_end - outage_start
    );
    println!("true position    : [{:.2}, {:.2}]", truth[0], truth[1]);
    println!(
        "fused position   : [{:.2}, {:.2}]  (error {:.3} m)",
        f.position()[0],
        f.position()[1],
        err
    );
    println!(
        "uncertainty σ    : {:.3} m at outage end → {:.3} m after re-acquisition",
        unc_mid,
        f.position_uncertainty()
    );
    println!("\nThe covariance grows while GNSS is lost and is pulled back when fixes resume.");
    Ok(())
}

// --- state estimation --------------------------------------------------------

/// TRACK-IMM: an Interacting Multiple Models filter shifts probability onto the
/// maneuver model when the target maneuvers.
pub fn track_imm(steps: usize) -> Result<(), String> {
    println!("=== Estimation — Interacting Multiple Models (IMM) ===\n");
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
    let mu = imm.mode_probabilities();
    println!("steps            : {steps}, velocity reversal at step {maneuver_at}");
    println!(
        "P(quiet model)   : {:.3} before maneuver → {:.3} once steady again",
        mu_quiet_before, mu[0]
    );
    println!(
        "P(maneuver model): peaks at {:.3} (step {peak_step}) when the target maneuvers",
        peak_maneuver
    );
    if peak_maneuver > 0.5
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

/// TRACK-UD: the Bierman–Thornton square-root filter agrees with a textbook
/// Kalman filter while keeping the covariance positive-semidefinite by factor.
pub fn track_ud(steps: usize) -> Result<(), String> {
    println!("=== Estimation — UD square-root Kalman filter ===\n");
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
    println!("steps                 : {steps}");
    println!("max |UD − Kalman| state: {max_state_diff:.2e}  (should be ~0)");
    println!("min UD variance        : {min_var:.2e}  (stays ≥ 0 by construction)");
    println!("\nThe UD form carries P = U·D·Uᵀ, so the covariance can never go indefinite.");
    Ok(())
}

// --- water networks ----------------------------------------------------------

/// WATER-LEAK: locate a leak on a pipe segment by acoustic cross-correlation.
pub fn water_leak(
    pipe_length: f64,
    wave_speed: f64,
    sample_rate: f64,
    leak_at: f64,
) -> Result<(), String> {
    println!("=== Water — acoustic leak correlation ===\n");
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
    println!(
        "pipe length      : {pipe_length} m   wave speed: {wave_speed} m/s   fs: {sample_rate} Hz"
    );
    println!("true leak        : {leak_at:.2} m from sensor A");
    println!(
        "located leak     : {:.2} m from sensor A  (error {:.3} m)",
        loc.dist_from_a,
        (loc.dist_from_a - leak_at).abs()
    );
    println!(
        "correlation peak : lag {} samples, normalized corr {:.3}",
        loc.lag_samples, loc.peak_corr
    );
    Ok(())
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
    let dp = joukowsky_surge(rho, wave_speed, delta_v);
    let head = dp / (rho * 9.806_65);
    let c_k = korteweg_wave_speed(bulk, rho, e_pipe, diameter, wall);
    let c_free = (bulk / rho).sqrt();
    println!("fluid density    : {rho} kg/m³");
    println!("velocity change  : {delta_v} m/s");
    println!(
        "Joukowsky surge  : {:.3} MPa  (Δp = ρ·c·Δv at c = {wave_speed} m/s)",
        dp / 1e6
    );
    println!("  as head        : {head:.1} m of water column");
    println!(
        "Korteweg speed   : {c_k:.0} m/s  (free-fluid {c_free:.0} m/s; the elastic wall lowers it)"
    );
    println!("  pipe: K={bulk:.2e} Pa, E={e_pipe:.2e} Pa, D={diameter} m, wall={wall} m");
    Ok(())
}

// --- OT cybersecurity --------------------------------------------------------

/// OT-FIRMWARE: capture a firmware baseline, then attest a clean and a tampered
/// image against it.
pub fn ot_firmware(size: usize, block: usize, tamper_block: usize) -> Result<(), String> {
    println!("=== OT security — firmware attestation ===\n");
    if size == 0 || block == 0
    {
        return Err("size and block must be > 0".into());
    }
    let mut seed = 0xF1A7u64;
    let fw: Vec<u8> = (0..size).map(|_| (unif(&mut seed) * 256.0) as u8).collect();
    let base = FirmwareBaseline::capture(&fw, block);

    println!("firmware         : {size} bytes, block size {block}");
    println!("baseline chain   : {:#018x}", base.chain_digest());
    match base.attest(&fw)
    {
        Attestation::Intact => println!("clean image      : INTACT ✓"),
        other => println!("clean image      : unexpected {other:?}"),
    }

    let mut bad = fw;
    let idx = (tamper_block * block).min(size - 1);
    bad[idx] ^= 0xFF; // flip one byte
    match base.attest(&bad)
    {
        Attestation::Tampered { first_bad_block } => println!(
            "tampered image   : TAMPERED — first altered block #{first_bad_block} (byte {idx} flipped)"
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

/// OT-PLC: verify PLC ladder integrity and flag unauthorized writes to a
/// safety-critical output (the Stuxnet pattern).
pub fn ot_plc() -> Result<(), String> {
    println!("=== OT security — PLC ladder integrity ===\n");
    fn golden() -> Vec<PlcRung> {
        vec![
            PlcRung::new(vec![0x01, 0x02, 0x03], Some(10)),
            PlcRung::new(vec![0x11, 0x12], Some(11)),
            PlcRung::new(vec![0x21, 0x22, 0x23], None),
        ]
    }
    let base = PlcBaseline::capture(&golden());
    println!(
        "golden program   : {} rungs, chain {:#018x}",
        golden().len(),
        base.chain_digest()
    );

    match base.verify(&golden())
    {
        PlcVerdict::Intact => println!("re-verify golden : INTACT ✓"),
        other => println!("re-verify golden : unexpected {other:?}"),
    }

    // Attacker appends a rung that drives a safety-critical output (#99).
    let mut tampered = golden();
    tampered.push(PlcRung::new(vec![0xDE, 0xAD, 0xBE, 0xEF], Some(99)));
    match base.verify(&tampered)
    {
        PlcVerdict::RungCountChanged { expected, actual } =>
        {
            println!("modified program : RUNG COUNT CHANGED {expected} → {actual}")
        },
        PlcVerdict::RungModified { rung } => println!("modified program : RUNG #{rung} MODIFIED"),
        PlcVerdict::Intact => println!("modified program : INTACT (change not detected!)"),
    }

    let critical: BTreeSet<u16> = [99u16].into_iter().collect();
    let writes = base.unauthorized_critical_writes(&tampered, &critical);
    println!("critical outputs : {critical:?}");
    println!("unauthorized writes to safety-critical outputs: {writes:?}");
    if !writes.is_empty()
    {
        println!("\n⚠ Stuxnet-style write to a critical output the golden logic never drives.");
    }
    Ok(())
}

// --- pharma / GMP ------------------------------------------------------------

/// GOLDEN-BATCH: DTW-align a candidate batch to the golden reference, check the
/// per-variable tolerance, and write the RELEASE/REJECT verdict to a hash-chained
/// 21 CFR Part 11 audit log.
pub fn golden_batch(lag: usize) -> Result<(), String> {
    println!("=== GMP — golden-batch comparator (21 CFR Part 11) ===\n");
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

    println!(
        "golden steps     : {}, candidate steps: {} (lag {lag})",
        reference.len(),
        candidate.len()
    );
    println!("conforming       : {}", report.conforming);
    println!(
        "worst deviation  : {:.3}× tolerance on variable {} at step {}",
        report.worst_ratio, report.worst_variable, report.worst_step
    );
    println!("DTW distance     : {:.4}", report.dtw_distance);
    println!(
        "audit log        : {} entry(ies), chain intact: {}",
        log.len(),
        log.verify_chain()
    );
    println!(
        "decision         : {}",
        if report.conforming
        {
            "RELEASE"
        }
        else
        {
            "REJECT"
        }
    );
    Ok(())
}
