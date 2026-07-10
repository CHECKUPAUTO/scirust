//! Preuve d'un **entraînement 100 % portable** : un MLP type MNIST
//! (32 → 16 → 10, sans biais) entraîné à la cross-entropy sur des données
//! synthétiques déterministes (PCG), où CHAQUE nœud du graphe est de la voie
//! portable — `matmul_portable` (GEMM f64 ordre fixe), ReLU (comparaisons),
//! `CrossEntropyLoss::new_portable()` (exp/ln sans libm), Adam (opérations
//! IEEE de base + `powi`/`sqrt`). L'empreinte FNV-1a de la trajectoire de
//! perte et celle des **poids finaux** sont comparées aux constantes commises
//! (calculées sur x86-64) : `verdict=PASS` ⇔ l'entraînement complet a produit
//! les mêmes bits sur cette machine que sur x86-64.
//!
//! Même discipline que `proof_portable_f32` : lignes canoniques (SHA-256
//! comparable entre machines), contexte préfixé `#`, code de sortie 0 ⇔ PASS.
//! Volets attendus : x86_64 (natif + Debian), Jetson/aarch64, CI QEMU.

use scirust_core::autodiff::optim::{Adam, Optimizer};
use scirust_core::autodiff::reverse::{Tape, Tensor};
use scirust_core::nn::loss::Loss;
use scirust_core::nn::{CrossEntropyLoss, PcgEngine};
use scirust_core::portable_f32::{fnv1a_fold_bits, fnv1a_init};
use std::process::ExitCode;

const N_IN: usize = 32;
const N_HID: usize = 16;
const N_OUT: usize = 10;
const BATCH: usize = 8;
const STEPS: usize = 30;
const LR: f32 = 0.01;

/// Empreinte attendue de la trajectoire de perte (bits de la perte à chaque
/// pas), calculée sur x86-64.
const PROOF_LOSS_TRAJECTORY_FP: u64 = 0x531f_63eb_5066_6b8a;
/// Empreinte attendue des poids finaux (w1 puis w2), calculée sur x86-64.
const PROOF_FINAL_WEIGHTS_FP: u64 = 0x4bbd_3d8d_c162_b305;

fn fold_slice(mut fp: u64, xs: &[f32]) -> u64 {
    for &x in xs
    {
        fp = fnv1a_fold_bits(fp, x.to_bits());
    }
    fp
}

fn main() -> ExitCode {
    println!("PROOF-PORTABLE-TRAINING v1");
    println!(
        "# arch={} os={} family={}",
        std::env::consts::ARCH,
        std::env::consts::OS,
        std::env::consts::FAMILY
    );
    println!("config=mlp{N_IN}x{N_HID}x{N_OUT} batch={BATCH} steps={STEPS} lr={LR} adam pcg");

    // Initialisation déterministe des poids (PCG seedé, intégralement entier
    // en interne ⇒ mêmes f32 sur toute plate-forme).
    let mut rng = PcgEngine::new(2026_0710);
    let mut w1v: Vec<f32> = (0..N_IN * N_HID).map(|_| rng.float() * 0.2 - 0.1).collect();
    let mut w2v: Vec<f32> = (0..N_HID * N_OUT)
        .map(|_| rng.float() * 0.2 - 0.1)
        .collect();

    let mut opt = Adam::new(LR);
    let mut loss_fp = fnv1a_init();

    for _step in 0..STEPS
    {
        // Batch synthétique déterministe : entrées dans [−1, 1), label = argmax
        // implicite dérivé du même flux PCG (distribution stable inter-runs).
        let x: Vec<f32> = (0..BATCH * N_IN).map(|_| rng.float() * 2.0 - 1.0).collect();
        let mut onehot = vec![0.0f32; BATCH * N_OUT];
        for b in 0..BATCH
        {
            let label = ((rng.float() * N_OUT as f32) as usize).min(N_OUT - 1);
            onehot[b * N_OUT + label] = 1.0;
        }

        // Tape éphémère par pas (pattern des optimiseurs du dépôt).
        let tape = Tape::new();
        let w1 = tape.input(Tensor::from_vec(w1v.clone(), N_IN, N_HID));
        let w2 = tape.input(Tensor::from_vec(w2v.clone(), N_HID, N_OUT));
        let xv = tape.input(Tensor::from_vec(x, BATCH, N_IN));
        let target = tape.input(Tensor::from_vec(onehot, BATCH, N_OUT));

        let h = xv.matmul_portable(w1).relu();
        let logits = h.matmul_portable(w2);
        let loss = CrossEntropyLoss::new_portable().forward(&tape, logits, target);
        tape.backward(loss.idx());
        opt.step(&[w1.idx(), w2.idx()], &tape);

        loss_fp = fnv1a_fold_bits(loss_fp, tape.value(loss.idx()).data[0].to_bits());
        w1v = tape.value(w1.idx()).data;
        w2v = tape.value(w2.idx()).data;
    }

    let mut weights_fp = fnv1a_init();
    weights_fp = fold_slice(weights_fp, &w1v);
    weights_fp = fold_slice(weights_fp, &w2v);

    let loss_ok = loss_fp == PROOF_LOSS_TRAJECTORY_FP;
    let weights_ok = weights_fp == PROOF_FINAL_WEIGHTS_FP;
    println!(
        "loss_trajectory.fp=0x{loss_fp:016x} attendu=0x{:016x} {}",
        PROOF_LOSS_TRAJECTORY_FP,
        if loss_ok { "OK" } else { "ECART" }
    );
    println!(
        "final_weights.fp=0x{weights_fp:016x} attendu=0x{:016x} {}",
        PROOF_FINAL_WEIGHTS_FP,
        if weights_ok { "OK" } else { "ECART" }
    );

    let ok = loss_ok && weights_ok;
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
