//! Preuve d'un **entraînement basse précision reproductible** : le MLP
//! portable de `proof_portable_training`, mais avec des poids maîtres f32
//! quantifiés en **bf16 par arrondi stochastique Philox** à chaque pas
//! (recette d'entraînement basse précision standard : maîtres f32, forward
//! en bf16). L'aléa de quantification étant **contre-basé** (graine, pas,
//! index de poids), il est indépendant de tout découpage — l'entraînement
//! stochastique complet est déterministe et bit-exact inter-plates-formes.
//!
//! C'est la capacité de synthèse du volet 115 : personne (ni RepDL, qui
//! exclut les basses précisions, ni les frameworks grand public) n'offre
//! « entraînement bf16 à arrondi stochastique, bit-reproductible
//! cross-platform, sous contrat d'empreinte ».
//!
//! Même discipline que les autres binaires de preuve : lignes canoniques,
//! contexte `#`, exit 0 ⇔ PASS.

use scirust_core::autodiff::optim::{Adam, Optimizer};
use scirust_core::autodiff::reverse::{Tape, Tensor};
use scirust_core::lowprec::{bf16_to_f32, f32_to_bf16_stochastic};
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

/// Empreinte attendue de la trajectoire de perte (x86-64).
const PROOF_LOSS_TRAJECTORY_FP: u64 = 0xd6c1_3495_0dac_2ee0;
/// Empreinte attendue des CODES bf16 finaux des poids forward (x86-64).
const PROOF_FINAL_BF16_FP: u64 = 0x09c0_f6be_bbb0_ef71;

/// Quantifie un tenseur de poids en bf16 par arrondi stochastique — le flux
/// Philox est indexé par (pas, index du poids) : mêmes bits quel que soit
/// le découpage ou l'ordre de parcours.
fn quantize_sr(w: &[f32], rng: &Philox4x32, step: u32, offset: u64) -> Vec<u16> {
    w.iter()
        .enumerate()
        .map(|(i, &x)| f32_to_bf16_stochastic(x, rng, step, offset + i as u64))
        .collect()
}

fn main() -> ExitCode {
    println!("PROOF-LOWPREC-TRAINING v1");
    println!(
        "# arch={} os={} family={}",
        std::env::consts::ARCH,
        std::env::consts::OS,
        std::env::consts::FAMILY
    );
    println!(
        "config=mlp{N_IN}x{N_HID}x{N_OUT} batch={BATCH} steps={STEPS} lr={LR} adam bf16-sr-philox"
    );

    let mut rng = PcgEngine::new(2026_0710);
    let sr = Philox4x32::new(0x000b_16f8);
    let mut w1v: Vec<f32> = (0..N_IN * N_HID).map(|_| rng.float() * 0.2 - 0.1).collect();
    let mut w2v: Vec<f32> = (0..N_HID * N_OUT)
        .map(|_| rng.float() * 0.2 - 0.1)
        .collect();

    let mut opt = Adam::new(LR);
    let mut loss_fp = fnv1a_init();
    let (mut q1, mut q2) = (Vec::new(), Vec::new());

    for step in 0..STEPS as u32
    {
        // copies forward bf16 (quantification stochastique des maîtres f32)
        q1 = quantize_sr(&w1v, &sr, step, 0);
        q2 = quantize_sr(&w2v, &sr, step, (N_IN * N_HID) as u64);
        let w1f: Vec<f32> = q1.iter().map(|&c| bf16_to_f32(c)).collect();
        let w2f: Vec<f32> = q2.iter().map(|&c| bf16_to_f32(c)).collect();

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

        // les gradients (calculés sur les copies bf16) mettent à jour les
        // MAÎTRES f32 : on transfère les valeurs maîtres dans la tape avant
        // le pas d'Adam, puis on relit.
        tape.set_value(w1.idx(), Tensor::from_vec(w1v.clone(), N_IN, N_HID));
        tape.set_value(w2.idx(), Tensor::from_vec(w2v.clone(), N_HID, N_OUT));
        opt.step(&[w1.idx(), w2.idx()], &tape);

        loss_fp = fnv1a_fold_bits(loss_fp, tape.value(loss.idx()).data[0].to_bits());
        w1v = tape.value(w1.idx()).data;
        w2v = tape.value(w2.idx()).data;
    }

    let mut bf16_fp = fnv1a_init();
    for &c in q1.iter().chain(q2.iter())
    {
        bf16_fp = fnv1a_fold_bits(bf16_fp, c as u32);
    }

    let loss_ok = loss_fp == PROOF_LOSS_TRAJECTORY_FP;
    let bf16_ok = bf16_fp == PROOF_FINAL_BF16_FP;
    println!(
        "loss_trajectory.fp=0x{loss_fp:016x} attendu=0x{:016x} {}",
        PROOF_LOSS_TRAJECTORY_FP,
        if loss_ok { "OK" } else { "ECART" }
    );
    println!(
        "final_bf16_codes.fp=0x{bf16_fp:016x} attendu=0x{:016x} {}",
        PROOF_FINAL_BF16_FP,
        if bf16_ok { "OK" } else { "ECART" }
    );

    let ok = loss_ok && bf16_ok;
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
