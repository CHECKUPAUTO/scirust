//! Preuve d'un **entraînement FP8 reproductible** : le même MLP portable et
//! la même recette « maîtres f32 + copies forward basse précision par
//! arrondi stochastique Philox » que `proof_lowprec_training` (bf16), mais
//! quantifiée en **FP8 E4M3** (spec OCP publique, 1+4+3 bits) — le format
//! poids usuel des recettes d'entraînement FP8 (E5M2 sert plutôt aux
//! gradients dans les recettes mixtes ; non traité ici, même principe).
//!
//! Ferme le dernier écart identifié après le volet 116 : bf16 avait sa
//! preuve d'entraînement, FP8 n'avait que les conversions/roundtrip
//! (`lowprec.rs`) sans preuve d'entraînement bout-en-bout. L'aléa de
//! quantification étant contre-basé (graine, pas, index de poids),
//! l'entraînement stochastique complet reste déterministe et bit-exact
//! inter-plates-formes.
//!
//! Même discipline que les autres binaires de preuve : lignes canoniques,
//! contexte `#`, exit 0 ⇔ PASS.

use scirust_core::autodiff::optim::{Adam, Optimizer};
use scirust_core::autodiff::reverse::{Tape, Tensor};
use scirust_core::lowprec::{Fp8Format, f32_to_fp8_stochastic, fp8_to_f32};
use scirust_core::nn::loss::Loss;
use scirust_core::nn::{CrossEntropyLoss, PcgEngine};
use scirust_core::philox::Philox4x32;
use scirust_core::portable_f32::{fnv1a_fold_bits, fnv1a_init};
use std::process::ExitCode;

const N_IN: usize = 32;
const N_HID: usize = 16;
const N_OUT: usize = 10;
const BATCH: usize = 8;
const STEPS: usize = 30;
const LR: f32 = 0.01;
const FMT: Fp8Format = Fp8Format::E4M3;

/// Empreinte attendue de la trajectoire de perte (x86-64).
const PROOF_LOSS_TRAJECTORY_FP: u64 = 0x9d51_f587_bc9d_5db4;
/// Empreinte attendue des CODES FP8 finaux des poids forward (x86-64).
const PROOF_FINAL_FP8_FP: u64 = 0xe55a_5fa4_691a_544c;

/// Quantifie un tenseur de poids en FP8 par arrondi stochastique — le flux
/// Philox est indexé par (pas, index du poids) : mêmes bits quel que soit
/// le découpage ou l'ordre de parcours.
fn quantize_sr(w: &[f32], rng: &Philox4x32, step: u32, offset: u64) -> Vec<u8> {
    w.iter()
        .enumerate()
        .map(|(i, &x)| f32_to_fp8_stochastic(x, FMT, rng, step, offset + i as u64))
        .collect()
}

fn main() -> ExitCode {
    println!("PROOF-FP8-TRAINING v1");
    println!(
        "# arch={} os={} family={}",
        std::env::consts::ARCH,
        std::env::consts::OS,
        std::env::consts::FAMILY
    );
    println!(
        "config=mlp{N_IN}x{N_HID}x{N_OUT} batch={BATCH} steps={STEPS} lr={LR} adam fp8-e4m3-sr-philox"
    );

    let mut rng = PcgEngine::new(2026_0710);
    let sr = Philox4x32::new(0x00fb_8e17);
    let mut w1v: Vec<f32> = (0..N_IN * N_HID).map(|_| rng.float() * 0.2 - 0.1).collect();
    let mut w2v: Vec<f32> = (0..N_HID * N_OUT)
        .map(|_| rng.float() * 0.2 - 0.1)
        .collect();

    let mut opt = Adam::new(LR);
    let mut loss_fp = fnv1a_init();
    let (mut q1, mut q2) = (Vec::new(), Vec::new());

    for step in 0..STEPS as u32
    {
        // copies forward FP8 (quantification stochastique des maîtres f32)
        q1 = quantize_sr(&w1v, &sr, step, 0);
        q2 = quantize_sr(&w2v, &sr, step, (N_IN * N_HID) as u64);
        let w1f: Vec<f32> = q1.iter().map(|&c| fp8_to_f32(c, FMT)).collect();
        let w2f: Vec<f32> = q2.iter().map(|&c| fp8_to_f32(c, FMT)).collect();

        let x: Vec<f32> = (0..BATCH * N_IN).map(|_| rng.float() * 2.0 - 1.0).collect();
        let mut onehot = vec![0.0f32; BATCH * N_OUT];
        for b in 0..BATCH
        {
            let label = ((rng.float() * N_OUT as f32) as usize).min(N_OUT - 1);
            onehot[b * N_OUT + label] = 1.0;
        }

        let tape = Tape::new();
        let w1 = tape.input(Tensor::from_vec(w1f, N_IN, N_HID));
        let w2 = tape.input(Tensor::from_vec(w2f, N_HID, N_OUT));
        let xv = tape.input(Tensor::from_vec(x, BATCH, N_IN));
        let target = tape.input(Tensor::from_vec(onehot, BATCH, N_OUT));

        let h = xv.matmul_portable(w1).relu();
        let logits = h.matmul_portable(w2);
        let loss = CrossEntropyLoss::new_portable().forward(&tape, logits, target);
        tape.backward(loss.idx());

        // les gradients (calculés sur les copies FP8) mettent à jour les
        // MAÎTRES f32 : on transfère les valeurs maîtres dans la tape avant
        // le pas d'Adam, puis on relit.
        tape.set_value(w1.idx(), Tensor::from_vec(w1v.clone(), N_IN, N_HID));
        tape.set_value(w2.idx(), Tensor::from_vec(w2v.clone(), N_HID, N_OUT));
        opt.step(&[w1.idx(), w2.idx()], &tape);

        loss_fp = fnv1a_fold_bits(loss_fp, tape.value(loss.idx()).data[0].to_bits());
        w1v = tape.value(w1.idx()).data;
        w2v = tape.value(w2.idx()).data;
    }

    let mut fp8_fp = fnv1a_init();
    for &c in q1.iter().chain(q2.iter())
    {
        fp8_fp = fnv1a_fold_bits(fp8_fp, c as u32);
    }

    let loss_ok = loss_fp == PROOF_LOSS_TRAJECTORY_FP;
    let fp8_ok = fp8_fp == PROOF_FINAL_FP8_FP;
    println!(
        "loss_trajectory.fp=0x{loss_fp:016x} attendu=0x{:016x} {}",
        PROOF_LOSS_TRAJECTORY_FP,
        if loss_ok { "OK" } else { "ECART" }
    );
    println!(
        "final_fp8_codes.fp=0x{fp8_fp:016x} attendu=0x{:016x} {}",
        PROOF_FINAL_FP8_FP,
        if fp8_ok { "OK" } else { "ECART" }
    );

    let ok = loss_ok && fp8_ok;
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
