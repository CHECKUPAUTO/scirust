//! Binaire de **preuve cross-platform** de la voie f32 portable
//! ([`scirust_core::portable_f32`]).
//!
//! À exécuter sur CHAQUE plate-forme cible (x86-64 Debian, Jetson/aarch64, …),
//! de préférence via `scripts/proof-portable-f32.sh` qui archive le bundle
//! d'évidence. Le binaire recalcule les empreintes FNV-1a des balayages de
//! l'espace des bits f32 et les goldens ponctuels, puis les compare aux
//! constantes **commises dans le dépôt** (contrat de portabilité) :
//!
//! - le code de sortie vaut 0 si et seulement si TOUT coïncide (`verdict=PASS`) ;
//! - les lignes commençant par `#` sont du contexte machine (exclues de la
//!   comparaison) ; toutes les autres lignes sont canoniques : leur SHA-256
//!   doit être identique sur toutes les plates-formes conformes.
//!
//! `--full` ajoute le balayage **exhaustif** (pas 1 : les 2³² entrées
//! possibles de chaque fonction — quelques minutes en release).

use scirust_core::portable_f32 as pf;
use std::process::ExitCode;
use std::time::Instant;

fn hex_list(bits: &[u32]) -> String {
    bits.iter()
        .map(|b| format!("{b:08x}"))
        .collect::<Vec<_>>()
        .join(",")
}

fn main() -> ExitCode {
    let full = std::env::args().any(|a| a == "--full");
    let mut ok = true;
    let check_fp = |name: &str, got: u64, want: u64| -> bool {
        let pass = got == want;
        println!(
            "{name}.fp=0x{got:016x} attendu=0x{want:016x} {}",
            if pass { "OK" } else { "ECART" }
        );
        pass
    };

    println!("PROOF-PORTABLE-F32 v1");
    println!(
        "# arch={} os={} family={}",
        std::env::consts::ARCH,
        std::env::consts::OS,
        std::env::consts::FAMILY
    );

    // --- Goldens ponctuels ---
    let exp_bits: Vec<u32> = pf::PROOF_EXP_GOLDEN_INPUTS
        .iter()
        .map(|&x| pf::exp_f32(x).to_bits())
        .collect();
    let exp_golden_ok = exp_bits == pf::PROOF_EXP_GOLDEN_BITS.to_vec();
    println!(
        "exp.golden.bits={} {}",
        hex_list(&exp_bits),
        if exp_golden_ok { "OK" } else { "ECART" }
    );
    ok &= exp_golden_ok;

    let ln_bits: Vec<u32> = pf::PROOF_LN_GOLDEN_INPUTS
        .iter()
        .map(|&x| pf::ln_f32(x).to_bits())
        .collect();
    let ln_golden_ok = ln_bits == pf::PROOF_LN_GOLDEN_BITS.to_vec();
    println!(
        "ln.golden.bits={} {}",
        hex_list(&ln_bits),
        if ln_golden_ok { "OK" } else { "ECART" }
    );
    ok &= ln_golden_ok;

    // --- Balayage-contrat (pas 65 537) ---
    println!("contract.step={}", pf::PROOF_STEP_CONTRACT);
    ok &= check_fp(
        "exp.contract",
        pf::sweep_fingerprint(pf::exp_f32, pf::PROOF_STEP_CONTRACT),
        pf::PROOF_EXP_FP_CONTRACT,
    );
    ok &= check_fp(
        "ln.contract",
        pf::sweep_fingerprint(pf::ln_f32, pf::PROOF_STEP_CONTRACT),
        pf::PROOF_LN_FP_CONTRACT,
    );
    ok &= check_fp(
        "tanh.contract",
        pf::sweep_fingerprint(pf::tanh_f32, pf::PROOF_STEP_CONTRACT),
        pf::PROOF_TANH_FP_CONTRACT,
    );
    ok &= check_fp(
        "sigmoid.contract",
        pf::sweep_fingerprint(pf::sigmoid_f32, pf::PROOF_STEP_CONTRACT),
        pf::PROOF_SIGMOID_FP_CONTRACT,
    );

    // --- Balayage dense (pas 257, ≈ 16,7 M d'entrées par fonction) ---
    println!("dense.step={}", pf::PROOF_STEP_DENSE);
    let t = Instant::now();
    ok &= check_fp(
        "exp.dense",
        pf::sweep_fingerprint(pf::exp_f32, pf::PROOF_STEP_DENSE),
        pf::PROOF_EXP_FP_DENSE,
    );
    ok &= check_fp(
        "ln.dense",
        pf::sweep_fingerprint(pf::ln_f32, pf::PROOF_STEP_DENSE),
        pf::PROOF_LN_FP_DENSE,
    );
    ok &= check_fp(
        "tanh.dense",
        pf::sweep_fingerprint(pf::tanh_f32, pf::PROOF_STEP_DENSE),
        pf::PROOF_TANH_FP_DENSE,
    );
    ok &= check_fp(
        "sigmoid.dense",
        pf::sweep_fingerprint(pf::sigmoid_f32, pf::PROOF_STEP_DENSE),
        pf::PROOF_SIGMOID_FP_DENSE,
    );
    println!("# duree_dense_s={:.1}", t.elapsed().as_secs_f64());

    // --- Composites (softmax, GEMM) ---
    ok &= check_fp(
        "softmax",
        pf::proof_softmax_fingerprint(),
        pf::PROOF_SOFTMAX_FP,
    );
    ok &= check_fp("gemm", pf::proof_gemm_fingerprint(), pf::PROOF_GEMM_FP);

    // --- Balayage exhaustif optionnel (pas 1 : les 2³² entrées) ---
    if full
    {
        println!("exhaustive.step=1");
        let t = Instant::now();
        ok &= check_fp(
            "exp.exhaustive",
            pf::sweep_fingerprint(pf::exp_f32, 1),
            pf::PROOF_EXP_FP_EXHAUSTIVE,
        );
        ok &= check_fp(
            "ln.exhaustive",
            pf::sweep_fingerprint(pf::ln_f32, 1),
            pf::PROOF_LN_FP_EXHAUSTIVE,
        );
        ok &= check_fp(
            "tanh.exhaustive",
            pf::sweep_fingerprint(pf::tanh_f32, 1),
            pf::PROOF_TANH_FP_EXHAUSTIVE,
        );
        ok &= check_fp(
            "sigmoid.exhaustive",
            pf::sweep_fingerprint(pf::sigmoid_f32, 1),
            pf::PROOF_SIGMOID_FP_EXHAUSTIVE,
        );
        println!("# duree_exhaustive_s={:.1}", t.elapsed().as_secs_f64());
    }

    println!("verdict={}", if ok { "PASS" } else { "FAIL" });
    if ok
    {
        ExitCode::SUCCESS
    }
    else
    {
        ExitCode::FAILURE
    }
}
