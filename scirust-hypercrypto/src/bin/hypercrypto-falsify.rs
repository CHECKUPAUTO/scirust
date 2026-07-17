//! `hypercrypto-falsify` — the Phase-1 structural falsification CLI.
//!
//! EXPERIMENTAL research tool. It never prints "secure" / "passed security".
//! Its purpose is to *break* SciRust-HyperCrypto v0.1, not to endorse it.
//!
//! Subcommands: `controls`, `matrix-lifting`, `affinity`, `degree`,
//! `invariants`, `zero-divisors`, `reduced-rounds`, `exhaustive-nano2`,
//! `report`. Run `hypercrypto-falsify help` for options.

use std::path::PathBuf;

use scirust_hypercrypto::algebra::Oct;
use scirust_hypercrypto::algebra::word::{W2, WidthTag};
use scirust_hypercrypto::analysis::battery::{
    Flags, analyze_v01_tag, decide, degree_compare_nano2, validate_controls_tag,
    zero_divisor_examples_json,
};
use scirust_hypercrypto::analysis::report::{Json, sha256_hex, write_result_file};
use scirust_hypercrypto::fixtures::{Fixture, FixtureId};
use scirust_hypercrypto::permutation::{State, Variant, feistel};
use scirust_hypercrypto::{EXPERIMENTAL_BANNER, SPEC_VERSION};

struct Opts {
    width: WidthTag,
    fixture: FixtureId,
    seed: u64,
    sample: usize,
    rounds: u32,
    out: PathBuf,
    limit: u64,
    full: bool,
}

impl Opts {
    fn parse(args: &[String]) -> Opts {
        let mut o = Opts {
            width: WidthTag::Mini8,
            fixture: FixtureId::PseudoRandom(0x5C1_0001),
            seed: 0xA11CE,
            sample: 100_000,
            rounds: 4,
            out: PathBuf::from("target/hypercrypto-falsification"),
            limit: 1 << 24,
            full: false,
        };
        let mut i = 0;
        while i < args.len()
        {
            let a = &args[i];
            let mut next = || {
                i += 1;
                args.get(i).cloned().unwrap_or_default()
            };
            match a.as_str()
            {
                "--width" => o.width = WidthTag::parse(&next()).unwrap_or(o.width),
                "--fixture" => o.fixture = FixtureId::parse(&next()).unwrap_or(o.fixture),
                "--seed" =>
                {
                    let s = next();
                    let s = s.trim_start_matches("0x");
                    o.seed = u64::from_str_radix(s, 16).unwrap_or(o.seed);
                },
                "--sample" => o.sample = next().parse().unwrap_or(o.sample),
                "--rounds" => o.rounds = next().parse().unwrap_or(o.rounds),
                "--out" => o.out = PathBuf::from(next()),
                "--limit" => o.limit = next().parse().unwrap_or(o.limit),
                "--full" => o.full = true,
                _ =>
                {},
            }
            i += 1;
        }
        o
    }
}

fn git_commit() -> String {
    std::env::var("HYPERCRYPTO_GIT_COMMIT").unwrap_or_else(|_| "unknown".to_string())
}

/// Shared metadata header for every run (spec §"Command-line tool").
fn meta(subcommand: &str, o: &Opts, coverage: &str) -> Vec<(&'static str, Json)> {
    vec![
        ("spec_version", Json::s(SPEC_VERSION)),
        ("git_commit", Json::s(git_commit())),
        ("subcommand", Json::s(subcommand.to_string())),
        ("coefficient_width_bits", Json::U64(o.width.bits() as u64)),
        ("variant_domain", Json::s(o.width.variant_name())),
        ("state_width_bits", Json::U64((o.width.bits() as u64) * 16)),
        ("round_count", Json::U64(o.rounds as u64)),
        ("fixture", Json::s(o.fixture.label())),
        ("coverage", Json::s(coverage.to_string())),
        ("sample_seed", Json::s(format!("0x{:016x}", o.seed))),
        ("sample_count", Json::U64(o.sample as u64)),
    ]
}

fn emit(o: &Opts, name: &str, meta_fields: Vec<(&'static str, Json)>, extra: Vec<(&str, Json)>) {
    let mut pairs: Vec<(String, Json)> = meta_fields
        .into_iter()
        .map(|(k, v)| (k.to_string(), v))
        .collect();
    for (k, v) in extra
    {
        pairs.push((k.to_string(), v));
    }
    let doc = Json::Obj(pairs);
    match write_result_file(&o.out, name, &doc)
    {
        Ok((path, fp)) =>
        {
            println!("  result_file : {}", path.display());
            println!("  fingerprint : {fp}");
        },
        Err(e) => eprintln!("  (could not write result file: {e})"),
    }
}

fn banner(sub: &str) {
    println!("{EXPERIMENTAL_BANNER}");
    println!("== hypercrypto-falsify :: {sub} ==");
    println!("  spec        : {SPEC_VERSION}");
    println!("  git_commit  : {}", git_commit());
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let sub = args.get(1).cloned().unwrap_or_else(|| "help".to_string());
    let rest = if args.len() > 2 { &args[2..] } else { &[] };
    let o = Opts::parse(rest);

    match sub.as_str()
    {
        "controls" => cmd_controls(&o),
        "matrix-lifting" => cmd_section(&o, "matrix-lifting", "exp1_matrix_lifting"),
        "affinity" => cmd_section(&o, "affinity", "exp3_full_round_affinity"),
        "degree" => cmd_degree(&o),
        "invariants" => cmd_section(&o, "invariants", "exp5_invariants"),
        "zero-divisors" => cmd_zero_divisors(&o),
        "reduced-rounds" => cmd_reduced_rounds(&o),
        "exhaustive-nano2" => cmd_exhaustive_nano2(&o),
        "report" => cmd_report(&o),
        _ => print_help(),
    }
}

fn print_help() {
    println!("{EXPERIMENTAL_BANNER}");
    println!(
        "hypercrypto-falsify <subcommand> [options]\n\
         \n\
         subcommands:\n\
         \x20 controls          validate the analysis harness against Controls A/B\n\
         \x20 matrix-lifting    L_a / R_a matrices, det/rank/kernel (Experiment 1)\n\
         \x20 affinity          full-round affinity over Z/2^k and GF(2) (Experiment 3)\n\
         \x20 degree            algebraic degree by exact ANF, v0.1 vs controls (Experiment 4)\n\
         \x20 invariants        norm / conjugation invariants (Experiment 5)\n\
         \x20 zero-divisors     zero-divisor fibers and kernels (Experiment 6)\n\
         \x20 reduced-rounds    degree/roundtrip/differential by round count\n\
         \x20 exhaustive-nano2  full 2^32 NANO-2 sweep (DISABLED unless invoked; costly)\n\
         \x20 report            full battery + Phase-1 verdict\n\
         \n\
         options: --width nano2|nano4|mini8|mini16|full64  --fixture <id>\n\
         \x20        --seed 0xHEX  --sample N  --rounds N  --out DIR  --limit N  --full\n"
    );
}

fn cmd_controls(o: &Opts) {
    banner("controls (harness validation)");
    let (a_broken, b_broken, json) = validate_controls_tag(o.width, o.seed, o.sample);
    let status = if a_broken && b_broken
    {
        "control break successfully recovered"
    }
    else
    {
        "HARNESS FAILURE — controls NOT recovered (results untrustworthy)"
    };
    println!("  control A recovered : {a_broken}");
    println!("  control B recovered : {b_broken}");
    println!("  status              : {status}");
    let cov = if o.width == WidthTag::Nano2
    {
        "exhaustive"
    }
    else
    {
        "sampled"
    };
    emit(
        o,
        "controls",
        meta("controls", o, cov),
        vec![("controls", json), ("status", Json::s(status))],
    );
}

fn cmd_section(o: &Opts, name: &str, section_key: &str) {
    banner(name);
    let mut flags = Flags {
        roundtrip_ok: true,
        ..Default::default()
    };
    let json = analyze_v01_tag(o.width, o.seed, o.sample, &mut flags);
    // slice out the requested section
    let section = if let Json::Obj(pairs) = &json
    {
        pairs
            .iter()
            .find(|(k, _)| k == section_key)
            .map(|(_, v)| v.clone())
            .unwrap_or(Json::Null)
    }
    else
    {
        Json::Null
    };
    println!("  detected relation   : see '{section_key}' in result file");
    println!("  (a failed model excludes only that model — not a security claim)");
    let cov = if o.width == WidthTag::Nano2
    {
        "exhaustive (single-octonion domain)"
    }
    else
    {
        "sampled"
    };
    emit(o, name, meta(name, o, cov), vec![(section_key, section)]);
}

fn cmd_degree(o: &Opts) {
    banner("degree (algebraic degree, exact ANF)");
    let compare = degree_compare_nano2();
    let mut flags = Flags {
        roundtrip_ok: true,
        ..Default::default()
    };
    let json = analyze_v01_tag(WidthTag::Nano2, o.seed, o.sample, &mut flags);
    let deg_section = if let Json::Obj(pairs) = &json
    {
        pairs
            .iter()
            .find(|(k, _)| k == "exp4_algebraic_degree")
            .map(|(_, v)| v.clone())
            .unwrap_or(Json::Null)
    }
    else
    {
        Json::Null
    };
    println!("  exact ANF at NANO-2 (v0.1 vs Controls A-D) written to result file");
    emit(
        o,
        "degree",
        meta("degree", o, "exhaustive (NANO-2 exact ANF)"),
        vec![
            ("degree_by_round", deg_section),
            ("v01_vs_controls", compare),
        ],
    );
}

fn cmd_zero_divisors(o: &Opts) {
    banner("zero-divisors (Experiment 6)");
    let examples = zero_divisor_examples_json(5);
    let mut flags = Flags {
        roundtrip_ok: true,
        ..Default::default()
    };
    let json = analyze_v01_tag(o.width, o.seed, o.sample, &mut flags);
    let section = if let Json::Obj(pairs) = &json
    {
        pairs
            .iter()
            .find(|(k, _)| k == "exp6_zero_divisors")
            .map(|(_, v)| v.clone())
            .unwrap_or(Json::Null)
    }
    else
    {
        Json::Null
    };
    println!("  explicit zero-divisor pairs (W2, exhaustive) written to result file");
    println!("  note: zero divisors may weaken F; the outer Feistel stays invertible");
    emit(
        o,
        "zero-divisors",
        meta(
            "zero-divisors",
            o,
            "W2 exhaustive examples + sampled kernels",
        ),
        vec![("explicit_pairs_w2", examples), ("kernel_summary", section)],
    );
}

fn cmd_reduced_rounds(o: &Opts) {
    banner("reduced-rounds");
    // degree growth + roundtrip across reduced round counts at NANO-2
    let fx = Fixture::new(o.fixture);
    let mut rows = Vec::new();
    for r in [1u32, 2, 4, 6, 8]
    {
        let deg = scirust_hypercrypto::analysis::degree::octfn_degree::<W2>(
            scirust_hypercrypto::analysis::degree::feistel_branch_after::<W2>(&fx, Variant::V01, r),
        )
        .map(|d| d.max_degree)
        .unwrap_or(0);
        let rt = scirust_hypercrypto::analysis::battery::roundtrip_ok::<W2>(&fx, r, o.seed);
        rows.push(Json::obj(vec![
            ("rounds", Json::U64(r as u64)),
            ("branch_max_degree_nano2", Json::U64(deg as u64)),
            ("forward_inverse_roundtrip_ok", Json::Bool(rt)),
        ]));
    }
    println!("  degree growth + round-trip by round count (NANO-2) in result file");
    emit(
        o,
        "reduced-rounds",
        meta("reduced-rounds", o, "exhaustive (NANO-2 exact ANF)"),
        vec![("by_round", Json::Arr(rows))],
    );
}

/// Full `2^32` NANO-2 state sweep (spec §"NANO-2 execution policy"). DISABLED by
/// default: it runs only when this subcommand is explicitly invoked, and the
/// full `2^32` sweep requires `--full`. Deterministic progress, checked
/// counters, final exact count + fingerprint. No wall-clock guarantee is implied.
fn cmd_exhaustive_nano2(o: &Opts) {
    banner("exhaustive-nano2");
    let total: u64 = if o.full
    {
        1u64 << 32
    }
    else
    {
        o.limit.min(1u64 << 32)
    };
    println!("  ⚠ COST WARNING: sweeping {total} NANO-2 states (32-bit state).");
    println!("    This is a research command; runtime is NOT bounded or guaranteed.");
    println!("    Use --full for the complete 2^32 sweep, or --limit N for a prefix.");
    let fx = Fixture::new(o.fixture);
    let rounds = o.rounds.max(1);

    let mut count: u64 = 0;
    let mut roundtrip_ok = true;
    // rolling fingerprint: fold each output state's 32 bits with a mixing constant
    let mut acc: u64 = 0xcbf2_9ce4_8422_2325; // FNV offset basis
    let progress_step = (total / 16).max(1);
    let mut first_bad: Option<u64> = None;

    for code in 0..total
    {
        // decode 32-bit code -> (L,R) each 8 x 2-bit
        let l = Oct::<W2>::from_u64s(std::array::from_fn(|i| (code >> (2 * i)) & 3));
        let r = Oct::<W2>::from_u64s(std::array::from_fn(|i| (code >> (16 + 2 * i)) & 3));
        let s = State::new(l, r);
        let enc = feistel::forward(s, &fx, Variant::V01, rounds, true);
        let dec = feistel::inverse(enc, &fx, Variant::V01, rounds, true);
        if dec != s && first_bad.is_none()
        {
            roundtrip_ok = false;
            first_bad = Some(code);
        }
        // fold encoded state into the fingerprint (order-independent-safe via mul-mix)
        let mut enc_code: u64 = 0;
        let ec = enc.l.to_u64s();
        let er = enc.r.to_u64s();
        for i in 0..8
        {
            enc_code |= ec[i] << (2 * i);
            enc_code |= er[i] << (16 + 2 * i);
        }
        acc ^= enc_code;
        acc = acc.wrapping_mul(0x0000_0100_0000_01b3); // FNV prime
        count = count.checked_add(1).expect("counter overflow");
        if code % progress_step == 0
        {
            println!("    progress: {count}/{total} ({}%)", count * 100 / total);
        }
    }

    let fingerprint = sha256_hex(&format!("nano2-sweep:{count}:{acc:016x}"));
    println!("  states_swept      : {count}");
    println!("  roundtrip_ok      : {roundtrip_ok}");
    if let Some(bad) = first_bad
    {
        println!("  first_mismatch    : code 0x{bad:08x}  (KILL: forward/inverse mismatch)");
    }
    println!("  sweep_fingerprint : {fingerprint}");
    let kill = if roundtrip_ok
    {
        "no forward/inverse mismatch found in swept prefix"
    }
    else
    {
        "kill criterion triggered: forward/inverse mismatch"
    };
    emit(
        o,
        "exhaustive-nano2",
        meta(
            "exhaustive-nano2",
            o,
            if o.full { "exhaustive-2^32" } else { "prefix" },
        ),
        vec![
            ("states_swept", Json::U64(count)),
            ("full_sweep", Json::Bool(o.full)),
            ("roundtrip_ok", Json::Bool(roundtrip_ok)),
            ("sweep_fingerprint", Json::s(fingerprint)),
            ("kill_criterion_status", Json::s(kill)),
        ],
    );
}

fn cmd_report(o: &Opts) {
    banner("report (full battery + verdict)");
    let mut flags = Flags {
        roundtrip_ok: true,
        ..Default::default()
    };

    // 1. harness validation at NANO-2 (exhaustive) and MINI-8 (sampled)
    let (a2, b2, cj2) = validate_controls_tag(WidthTag::Nano2, o.seed, o.sample);
    let (a8, b8, cj8) = validate_controls_tag(WidthTag::Mini8, o.seed, o.sample);
    flags.control_a_broken = a2 && a8;
    flags.control_b_broken = b2 && b8;

    // 2. v0.1 battery at NANO-2 (exhaustive) and MINI-8 (sampled)
    let v2 = analyze_v01_tag(WidthTag::Nano2, o.seed, o.sample, &mut flags);
    let v8 = analyze_v01_tag(WidthTag::Mini8, o.seed, o.sample, &mut flags);

    // 3. control-degree comparison + explicit zero divisors
    let degcmp = degree_compare_nano2();
    let zd = zero_divisor_examples_json(5);

    // 4. verdict
    let (verdict, reasons) = decide(&flags);

    println!();
    println!("  control_A_broken  : {}", flags.control_a_broken);
    println!("  control_B_broken  : {}", flags.control_b_broken);
    println!(
        "  full_round_ring_affine : {}",
        flags.full_round_ring_affine
    );
    println!("  full_round_gf2_affine  : {}", flags.full_round_gf2_affine);
    println!("  multiround_affine      : {}", flags.multiround_affine);
    println!(
        "  norm_invariant_survives: {}",
        flags.norm_invariant_survives
    );
    println!("  roundtrip_ok           : {}", flags.roundtrip_ok);
    println!();
    for r in &reasons
    {
        println!("  - {r}");
    }
    println!();
    println!("  {}", verdict.line());

    let doc_fields: Vec<(&str, Json)> = vec![
        (
            "harness_validation",
            Json::obj(vec![
                ("control_A_broken", Json::Bool(flags.control_a_broken)),
                ("control_B_broken", Json::Bool(flags.control_b_broken)),
                ("nano2", cj2),
                ("mini8", cj8),
            ]),
        ),
        ("v01_nano2_exhaustive", v2),
        ("v01_mini8_sampled", v8),
        ("degree_v01_vs_controls_nano2", degcmp),
        ("zero_divisor_examples_w2", zd),
        (
            "gating_flags",
            Json::obj(vec![
                (
                    "full_round_ring_affine",
                    Json::Bool(flags.full_round_ring_affine),
                ),
                (
                    "full_round_gf2_affine",
                    Json::Bool(flags.full_round_gf2_affine),
                ),
                ("multiround_affine", Json::Bool(flags.multiround_affine)),
                (
                    "norm_invariant_survives",
                    Json::Bool(flags.norm_invariant_survives),
                ),
                ("roundtrip_ok", Json::Bool(flags.roundtrip_ok)),
            ]),
        ),
        (
            "verdict_reasons",
            Json::Arr(reasons.iter().map(|r| Json::s(r.clone())).collect()),
        ),
        ("phase1_verdict", Json::s(verdict.line())),
    ];
    emit(
        o,
        "phase1-report",
        meta("report", o, "NANO-2 exhaustive + MINI-8 sampled"),
        doc_fields,
    );
}
